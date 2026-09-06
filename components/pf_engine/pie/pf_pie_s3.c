/* ESP32-S3 版: 重みステージング（内部 SRAM）、疎な経路の密化、起動時の自己テスト。
 *
 * 1. 重みステージング
 *    上流の weak な rf_stage_start / rf_stage_wait / rf_stage_drain（kernels_ref.c）を強いシンボルで置き換える。
 *    上流は PSRAM に置いた rf_arena のスロットへ memcpy するが、S3 の PSRAM（Quad 80 MHz）は遅いので、
 *    内部 SRAM に 200 KB のバッファを確保できればそこへ置く。スロットの相対配置は上流と同じ
 *    （dit.c が FC2 = FC1 + 4·D·D を前提にしている）。
 *
 * 2. 疎な経路を密な内積に
 *    S3 の QACC はレーン 20 bit（±2^19 で飽和）なので、fc2 や VAE の疎な gather を int32 で厳密に
 *    QACC へ溜めることはできない（esp-dl の S3 版 matmul は最後に ee.srcmb.s8.qacc でシフト付きに
 *    8 bit へ落とす設計で、量子化側で飽和を吸収している）。代わりに疎な経路を**密な内積**に置き換える。
 *    ゼロは和に寄与しないので、int32 の結果は上流の疎な経路と bit 一致する。
 *      - fc2: 重みを [K][O] → [O][K] に転置した複製を PSRAM に作り、FC2 スロットにはそれをステージする。
 *        --wrap した rf_axpy_acc16_sp が (idx, val) を 512 要素のベクトルに散らし直し、O 本の内積を取る
 *      - VAE の疎な層（flags bit 3）: 重み [3][3][C][O] を密な配置 [O][3][3][C] に転置した複製を作り、
 *        --wrap した rf_decode がその層をエンジン自身の密な畳み込み rf_conv3x3_i8_rows（内部で rf_dot_i8）に流す
 *    内積は rf_ops_pie.h の ee.vmulas.s8.accx（40 bit）で、自己テスト済み。 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "rf_model.h"
#include "rf_ops.h"

int rf_pie_enabled = 1;
int pf_dense_mask = 3; /* bit0 = fc2 を密化、bit1 = VAE を密化（シリアルの D で切り替え。切り分け用） */
int pf_vae_scalar;     /* 1 = VAE の間だけ PIE の内積を止めてスカラに（切り分け用） */

extern int64_t pf_stage_us; /* pf_par.c の集計 */
extern uint32_t pf_stage_calls;
extern size_t pf_stage_bytes;

/* ---- 内部 SRAM のステージングバッファ ----------------------------------------- */
#define PF_STAGE_BYTES (2 * 4 * RF_DIM * RF_DIM + 3 * RF_DIM * RF_DIM + RF_DIM * RF_DIM + 2 * RF_DIM * RF_PD)
static int8_t *s_stage;

static size_t slot_off(int slot) {
    switch (slot) {
    case RF_SLOT_FC1:  return 0;
    case RF_SLOT_FC2:  return (size_t)4 * RF_DIM * RF_DIM;
    case RF_SLOT_QKV:  return (size_t)8 * RF_DIM * RF_DIM;
    case RF_SLOT_PROJ: return (size_t)11 * RF_DIM * RF_DIM;
    case RF_SLOT_EMB:  return (size_t)12 * RF_DIM * RF_DIM;
    default:           return (size_t)12 * RF_DIM * RF_DIM + (size_t)RF_DIM * RF_PD;
    }
}

static int8_t *slot_ptr(int slot) { return s_stage ? s_stage + slot_off(slot) : rf_stage_slot(slot); }

/* ---- 転置した複製（PSRAM）---------------------------------------------------- */
typedef struct {
    const void *src;
    int8_t *T;
    size_t n;
} tcopy_t;
#define PF_T_MAX 48
static tcopy_t s_t[PF_T_MAX];
static int s_nt;
static int s_fc2_isT; /* FC2 スロットが今 [O][K] 配置か */

static tcopy_t *find_T(const void *src) {
    for (int i = 0; i < s_nt; i++)
        if (s_t[i].src == src) return &s_t[i];
    return NULL;
}

static tcopy_t *add_T(const void *src, size_t n) {
    if (s_nt >= PF_T_MAX) return NULL;
    int8_t *T = heap_caps_aligned_alloc(16, n, MALLOC_CAP_SPIRAM);
    if (!T) return NULL;
    tcopy_t *t = &s_t[s_nt++];
    t->src = src;
    t->T = T;
    t->n = n;
    return t;
}

/* fc2: W[K][O] → T[O][K] */
static void transpose_fc2(int8_t *T, const int8_t *W, int K, int O) {
    for (int k = 0; k < K; k++)
        for (int o = 0; o < O; o++) T[(size_t)o * K + k] = W[(size_t)k * O + o];
}

/* VAE 疎な層: W[9][C][O] → D[O][9][C]（密な畳み込みカーネルの配置） */
static void densify_conv(int8_t *D, const int8_t *W, int C, int O) {
    for (int tap = 0; tap < 9; tap++)
        for (int c = 0; c < C; c++)
            for (int o = 0; o < O; o++) D[((size_t)o * 9 + tap) * C + c] = W[((size_t)tap * C + c) * O + o];
}

static size_t s_dec_max;   /* 疎な層の密な重みの最大バイト数 */
static int16_t *s_zhwc;    /* unpatchify の作業域（4 KB、PSRAM） */

/* 起動時に 1 回。戻り値 = 作った複製の数 */
int pf_stage_prepare(const rf_model_t *m) {
    int made = 0;
    if (!s_stage) {
        s_stage = heap_caps_aligned_alloc(16, PF_STAGE_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        printf("PIE: DiT weight staging buffer %u B in %s\n", (unsigned)PF_STAGE_BYTES,
               s_stage ? "internal SRAM" : "(alloc failed -> rf_arena in PSRAM)");
    }
    for (uint32_t b = 0; b < m->depth; b++) {
        const int8_t *W = m->blk[b].Wfc2;
        if (find_T(W)) continue;
        tcopy_t *t = add_T(W, (size_t)4 * RF_DIM * RF_DIM);
        if (!t) break;
        transpose_fc2(t->T, W, 4 * RF_DIM, RF_DIM);
        made++;
    }
    for (uint32_t i = 0; i < m->n_dec; i++) {
        const rf_declayer_t *L = &m->dec[i];
        if (!((L->flags >> 3) & 1)) continue;
        size_t n = (size_t)9 * L->C * L->O;
        if (find_T(L->W)) continue;
        tcopy_t *t = add_T(L->W, n);
        if (!t) break;
        densify_conv(t->T, L->W, (int)L->C, (int)L->O);
        if (n > s_dec_max) s_dec_max = n;
        made++;
    }
    if (!s_zhwc) s_zhwc = heap_caps_aligned_alloc(16, sizeof(int16_t) * RF_ZHW * RF_ZHW * RF_ZCH, MALLOC_CAP_SPIRAM);
    printf("PIE: %d transposed / densified weight copies in PSRAM (dec max %u B)\n", made, (unsigned)s_dec_max);
    return made;
}

void rf_stage_start(int slot, const void *src, size_t n) {
    int64_t t0 = esp_timer_get_time();
    const void *from = src;
    int isT = 0;
    if (rf_pie_enabled && (pf_dense_mask & 1) && slot == RF_SLOT_FC2) {
        tcopy_t *t = find_T(src);
        if (t && t->n == n) {
            from = t->T;
            isT = 1;
        }
    }
    memcpy(slot_ptr(slot), from, n);
    if (slot == RF_SLOT_FC2) s_fc2_isT = isT;
    pf_stage_us += esp_timer_get_time() - t0;
    pf_stage_calls++;
    pf_stage_bytes += n;
}

const int8_t *rf_stage_wait(int slot) { return slot_ptr(slot); }
void rf_stage_drain(void) {}

/* ---- fc2: 疎な axpy を密な内積に ---------------------------------------------- */
void __real_rf_axpy_acc16_sp(const uint16_t *idx, const int8_t *val, int m, const int8_t *Wt, int O,
                             const int32_t *b, const int32_t *M, const uint8_t *s, int16_t *res);

/* 参照と同じ式: res[o] = sat16(res[o] + rq(b[o] + Σ_i val[i]·W[idx[i]][o], M[o], s[o]))。
 * ここでは h1 を密に戻して Σ_k h1[k]·T[o][k] を取る。ゼロの項は寄与しないので int32 の和は同じ */
void __wrap_rf_axpy_acc16_sp(const uint16_t *idx, const int8_t *val, int m, const int8_t *Wt, int O,
                             const int32_t *b, const int32_t *M, const uint8_t *s, int16_t *res) {
    const int K = 4 * RF_DIM;
    if (!rf_pie_enabled || !s_fc2_isT || Wt != slot_ptr(RF_SLOT_FC2) || O != RF_DIM) {
        __real_rf_axpy_acc16_sp(idx, val, m, Wt, O, b, M, s, res);
        return;
    }
    int8_t x[4 * RF_DIM] __attribute__((aligned(16)));
    memset(x, 0, sizeof x);
    for (int i = 0; i < m; i++) x[idx[i]] = val[i];
    for (int o = 0; o < O; o++) {
        int32_t acc = rf_dot_i8(Wt + (size_t)o * K, x, K, b[o]);
        res[o] = rf_sat16((int64_t)res[o] + rf_rq(acc, M[o], s[o]));
    }
}

/* ---- VAE: 疎な層を密な畳み込みに ---------------------------------------------- */
void __real_rf_decode(const rf_model_t *m, const int16_t z_tok[RF_TOKENS][RF_PD], uint8_t *img);

int64_t pf_vae_us;
int pf_vae_pie;

typedef struct {
    const int8_t *x;
    const rf_declayer_t *L;
    const int8_t *w;   /* 密な配置の重み（疎な層は内部 SRAM にステージした複製） */
    int H, W;
    int relu, up, u8;
    int8_t *dst;
} dctx_t;

static void conv_rows_dense(int y0, int y1, void *p) {
    dctx_t *c = p;
    rf_conv3x3_i8_rows(c->x, c->H, c->W, (int)c->L->C, c->w, c->L->b, c->L->M, c->L->s, (int)c->L->O, c->relu, 1,
                       c->up, c->u8, c->dst, NULL, y0, y1);
}

void __wrap_rf_decode(const rf_model_t *m, const int16_t z_tok[RF_TOKENS][RF_PD], uint8_t *img) {
    int64_t t0 = esp_timer_get_time();
    /* 疎な層の複製が全部そろい、ステージング先が内部 SRAM にあるときだけ */
    int ok = rf_pie_enabled && (pf_dense_mask & 2) && s_stage && s_zhwc && s_dec_max <= PF_STAGE_BYTES;
    for (uint32_t i = 0; ok && i < m->n_dec; i++)
        if (((m->dec[i].flags >> 3) & 1) && !find_T(m->dec[i].W)) ok = 0;
    if (!ok) {
        __real_rf_decode(m, z_tok, img);
        pf_vae_us = esp_timer_get_time() - t0;
        pf_vae_pie = 0;
        return;
    }
    /* unpatchify → requant（上流と同じ） */
    const int P = RF_PATCH, G = RF_ZHW / RF_PATCH;
    for (int y = 0; y < RF_ZHW; y++)
        for (int x = 0; x < RF_ZHW; x++)
            for (int c = 0; c < RF_ZCH; c++)
                s_zhwc[(y * RF_ZHW + x) * RF_ZCH + c] = z_tok[(y / P) * G + (x / P)][c * P * P + (y % P) * P + (x % P)];
    rf_requant_i16_to_i8(s_zhwc, RF_ZHW * RF_ZHW * RF_ZCH, m->M_zdec, m->s_zdec, rf_arena[0]);

    int H = RF_ZHW, W = RF_ZHW;
    const int8_t *in = rf_arena[0];
    int8_t *out = rf_arena[1];
    int saved_pie = rf_pie_enabled;
    if (pf_vae_scalar) rf_pie_enabled = 0;
    for (uint32_t i = 0; i < m->n_dec; i++) {
        const rf_declayer_t *L = &m->dec[i];
        dctx_t c;
        c.x = in;
        c.L = L;
        c.H = H;
        c.W = W;
        c.up = (int)(L->flags & 1);
        c.relu = (int)((L->flags >> 1) & 1);
        c.u8 = (int)((L->flags >> 2) & 1);
        c.dst = c.u8 ? (int8_t *)img : out;
        int wt = (int)((L->flags >> 3) & 1);
        if (wt) {
            tcopy_t *t = find_T(L->W);
            memcpy(s_stage, t->T, t->n); /* 密な配置の重みを内部 SRAM へ（DiT のステージング領域と時分割） */
            c.w = s_stage;
        } else {
            c.w = L->W;
        }
        int Ho = c.up ? 2 * H : H;
        rf_par_for(Ho, conv_rows_dense, &c);
        if (c.up) {
            H *= 2;
            W *= 2;
        }
        in = c.dst;
        out = (in == (const int8_t *)rf_arena[0]) ? rf_arena[1] : rf_arena[0];
    }
    rf_pie_enabled = saved_pie;
    pf_vae_us = esp_timer_get_time() - t0;
    pf_vae_pie = 1;
}

/* ---- 起動時の自己テスト ---------------------------------------------------
 * 参照実装（スカラ）と PIE を同じ乱数入力で比べる。落ちたら rf_pie_enabled = 0 にしてスカラに戻す。
 * バッファは PSRAM のヒープから借りる（内部 RAM を食わない）。戻り値 0 = すべて一致 */
static uint32_t s_rng = 0x12345678u;
static int32_t rnd(void) { s_rng = s_rng * 1664525u + 1013904223u; return (int32_t)(s_rng >> 8); }

int pf_pie_selftest(void) {
    int fails = 0;
    int8_t *w = heap_caps_aligned_alloc(16, 512 + 16, MALLOC_CAP_SPIRAM);
    int8_t *x = heap_caps_aligned_alloc(16, 512 + 16, MALLOC_CAP_SPIRAM);
    if (!w || !x) {
        printf("PIE selftest: alloc failed -> PIE disabled\n");
        rf_pie_enabled = 0;
        return 1;
    }
    /* 1. 内積: K ∈ {16, 32, 64, 128, 384, 512}、整列 / 非整列 x、極値（±127 で埋めた K=512） */
    const int Ks[] = {16, 32, 64, 128, 384, 512};
    for (int t = 0; t < 6; t++) {
        int K = Ks[t];
        for (int rep = 0; rep < 8; rep++) {
            for (int i = 0; i < K + 16; i++) {
                w[i] = (int8_t)(rep == 6 ? 127 : rep == 7 ? -127 : rnd() % 255 - 127);
                x[i] = (int8_t)(rep == 6 ? 127 : rep == 7 ? 127 : rnd() % 255 - 127);
            }
            for (int off = 0; off < 16; off += 5) {
                const int8_t *xp = x + off;
                int32_t ref = 12345;
                for (int k = 0; k < K; k++) ref += (int32_t)w[k] * xp[k];
                int32_t got = rf_dot_i8(w, xp, K, 12345);
                if (got != ref) {
                    if (fails < 5) printf("PIE selftest: dot K=%d rep=%d off=%d ref=%ld got=%ld\n", K, rep, off,
                                          (long)ref, (long)got);
                    fails++;
                }
            }
        }
    }
    /* 2. rf_rq（影のヘッダで書き換えた版）を汎用式と照合 */
    for (int rep = 0; rep < 4000; rep++) {
        uint8_t sh = (uint8_t)(1 + rnd() % 58);
        int32_t M = (int32_t)(rnd() ^ (rnd() << 12));
        int64_t acc = (rep & 1) ? (int64_t)(int32_t)(rnd() ^ (rnd() << 13))
                                : ((int64_t)(rnd() % 4096 - 2048) << 32) + (int32_t)rnd();
        if (rep % 7 == 0) acc = (rep & 8) ? INT32_MAX : INT32_MIN;
        if (rep % 11 == 0) M = (rep & 16) ? INT32_MAX : INT32_MIN + 1;
        int64_t p = acc * (int64_t)M;
        if (p > ((int64_t)1 << 61) || p < -((int64_t)1 << 61)) continue;
        int64_t ref = (p + ((int64_t)1 << (sh - 1))) >> sh;
        int64_t got = rf_rq(acc, M, sh);
        if (ref != got) {
            if (fails < 5) printf("PIE selftest: rq acc=%lld M=%ld s=%u ref=%lld got=%lld\n", (long long)acc,
                                  (long)M, sh, (long long)ref, (long long)got);
            fails++;
        }
    }
    /* 3. fc2 の密化: 疎な参照 vs 転置重み + 内積。K=512、O=128、非ゼロ m ∈ {1, 37, 256, 512} */
    {
        const int K = 4 * RF_DIM, O = RF_DIM;
        int8_t *Wkо = heap_caps_aligned_alloc(16, (size_t)K * O, MALLOC_CAP_SPIRAM);
        int8_t *T = heap_caps_aligned_alloc(16, (size_t)K * O, MALLOC_CAP_SPIRAM);
        uint16_t *idx = heap_caps_malloc(sizeof(uint16_t) * K, MALLOC_CAP_SPIRAM);
        int8_t *val = heap_caps_malloc(K, MALLOC_CAP_SPIRAM);
        int32_t *bb = heap_caps_malloc(sizeof(int32_t) * O, MALLOC_CAP_SPIRAM);
        int32_t *MM = heap_caps_malloc(sizeof(int32_t) * O, MALLOC_CAP_SPIRAM);
        uint8_t *ss = heap_caps_malloc(O, MALLOC_CAP_SPIRAM);
        int16_t *r0 = heap_caps_malloc(sizeof(int16_t) * O, MALLOC_CAP_SPIRAM);
        int16_t *r1 = heap_caps_malloc(sizeof(int16_t) * O, MALLOC_CAP_SPIRAM);
        if (Wkо && T && idx && val && bb && MM && ss && r0 && r1) {
            const int ms[] = {1, 37, 256, 512};
            for (int rep = 0; rep < 4; rep++) {
                int m = ms[rep];
                for (int i = 0; i < K * O; i++) Wkо[i] = (int8_t)(rnd() % 255 - 127);
                transpose_fc2(T, Wkо, K, O);
                /* 添字は重複なし・昇順（rf_compact_i8 が作る形）。重複があると参照は両方足すが密化版は上書きになる */
                {
                    int picked = 0;
                    for (int k = 0; k < K && picked < m; k++)
                        if ((K - k) <= (m - picked) || (rnd() % (K - k)) < (uint32_t)(m - picked)) idx[picked++] = (uint16_t)k;
                    for (int i = 0; i < m; i++) val[i] = (int8_t)(rnd() % 255 - 127);
                }
                for (int o = 0; o < O; o++) { bb[o] = rnd() % 200000 - 100000; MM[o] = (1 << 22) + rnd() % 100000; ss[o] = (uint8_t)(34 + rnd() % 8); r0[o] = r1[o] = (int16_t)(rnd() % 2000 - 1000); }
                __real_rf_axpy_acc16_sp(idx, val, m, Wkо, O, bb, MM, ss, r0);
                /* 密化版を直接（wrap の中身と同じ式） */
                int8_t xx[4 * RF_DIM] __attribute__((aligned(16)));
                memset(xx, 0, sizeof xx);
                for (int i = 0; i < m; i++) xx[idx[i]] = val[i];
                for (int o = 0; o < O; o++) {
                    int32_t acc = rf_dot_i8(T + (size_t)o * K, xx, K, bb[o]);
                    r1[o] = rf_sat16((int64_t)r1[o] + rf_rq(acc, MM[o], ss[o]));
                }
                if (memcmp(r0, r1, sizeof(int16_t) * O) != 0) {
                    if (fails < 5) printf("PIE selftest: fc2 dense m=%d mismatch\n", m);
                    fails++;
                }
            }
        } else {
            printf("PIE selftest: fc2 test alloc failed (skipped)\n");
        }
        heap_caps_free(Wkо); heap_caps_free(T); heap_caps_free(idx); heap_caps_free(val); heap_caps_free(bb);
        heap_caps_free(MM); heap_caps_free(ss); heap_caps_free(r0); heap_caps_free(r1);
    }
    /* 4. VAE の密化: 疎な式（スカラ）vs 密な配置 + rf_conv3x3_i8_rows。C=32, O=16, 4×4、up あり / なし */
    {
        const int C = 32, O = 16, H = 4, Wd = 4;
        int8_t *Ws = heap_caps_aligned_alloc(16, (size_t)9 * C * O, MALLOC_CAP_SPIRAM);
        int8_t *Dn = heap_caps_aligned_alloc(16, (size_t)9 * C * O, MALLOC_CAP_SPIRAM);
        int8_t *xin = heap_caps_aligned_alloc(16, (size_t)H * Wd * C, MALLOC_CAP_SPIRAM);
        int8_t *y0 = heap_caps_malloc((size_t)4 * H * Wd * O, MALLOC_CAP_SPIRAM);
        int8_t *y1 = heap_caps_malloc((size_t)4 * H * Wd * O, MALLOC_CAP_SPIRAM);
        int32_t *bb = heap_caps_malloc(sizeof(int32_t) * O, MALLOC_CAP_SPIRAM);
        int32_t *MM = heap_caps_malloc(sizeof(int32_t) * O, MALLOC_CAP_SPIRAM);
        uint8_t *ss = heap_caps_malloc(O, MALLOC_CAP_SPIRAM);
        if (Ws && Dn && xin && y0 && y1 && bb && MM && ss) {
            for (int up = 0; up < 2; up++) {
                for (int i = 0; i < 9 * C * O; i++) Ws[i] = (int8_t)(rnd() % 255 - 127);
                for (int i = 0; i < H * Wd * C; i++) xin[i] = (int8_t)((rnd() & 1) ? 0 : rnd() % 255 - 127); /* 半分ゼロ */
                for (int o = 0; o < O; o++) { bb[o] = rnd() % 20000 - 10000; MM[o] = (1 << 22) + rnd() % 100000; ss[o] = (uint8_t)(38 + rnd() % 4); }
                densify_conv(Dn, Ws, C, O);
                int Ho = up ? 2 * H : H, Wo = up ? 2 * Wd : Wd;
                /* 疎な式（vae_dec.c の conv_rows_sparse と同じ数学）をスカラで */
                for (int yo = 0; yo < Ho; yo++)
                    for (int xo = 0; xo < Wo; xo++)
                        for (int o = 0; o < O; o++) {
                            int32_t acc = bb[o];
                            for (int dy = 0; dy < 3; dy++) {
                                int yi = yo + dy - 1; if (up) yi >>= 1;
                                if (yi < 0 || yi >= H) continue;
                                for (int dx = 0; dx < 3; dx++) {
                                    int xi = xo + dx - 1; if (up) xi >>= 1;
                                    if (xi < 0 || xi >= Wd) continue;
                                    for (int ch = 0; ch < C; ch++)
                                        acc += (int32_t)xin[(yi * Wd + xi) * C + ch] * Ws[((dy * 3 + dx) * C + ch) * O + o];
                                }
                            }
                            int64_t v = rf_rq(acc, MM[o], ss[o]);
                            if (v < 0) v = 0;
                            y0[(yo * Wo + xo) * O + o] = rf_sat8(v);
                        }
                rf_conv3x3_i8_rows(xin, H, Wd, C, Dn, bb, MM, ss, O, 1, 1, up, 0, y1, NULL, 0, Ho);
                if (memcmp(y0, y1, (size_t)Ho * Wo * O) != 0) {
                    if (fails < 5) printf("PIE selftest: vae dense up=%d mismatch\n", up);
                    fails++;
                }
            }
        } else {
            printf("PIE selftest: vae test alloc failed (skipped)\n");
        }
        heap_caps_free(Ws); heap_caps_free(Dn); heap_caps_free(xin); heap_caps_free(y0); heap_caps_free(y1);
        heap_caps_free(bb); heap_caps_free(MM); heap_caps_free(ss);
    }
    /* 5. マイクロベンチ（正しさとは無関係）: 16 MAC チャンクあたりの ns */
    {
        int32_t sink = 0;
        int64_t t = esp_timer_get_time();
        for (int i = 0; i < 20000; i++) sink += rf_dot_i8(w, x, 128, i);
        double ns = (double)(esp_timer_get_time() - t) * 1000.0 / (20000.0 * 8);
        printf("PIE bench: dot K=128 %.1f ns/16MAC (sink %ld)\n", ns, (long)sink);
    }
    heap_caps_free(w);
    heap_caps_free(x);
    if (fails) {
        rf_pie_enabled = 0;
        printf("PIE selftest: %d FAIL -> PIE disabled, scalar fallback\n", fails);
    } else {
        printf("PIE selftest: all OK (dot 6K x8 x4off, rq x4000, fc2 dense x4, vae dense x2)\n");
    }
    return fails;
}
