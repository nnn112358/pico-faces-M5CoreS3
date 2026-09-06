/* ESP32-S3 PIE（Xtensa の 128 bit 整数 SIMD、`ee.*` 命令）による int8 内積カーネル。
 *
 * 上流 rf_ops.h の `#else`（参照実装）の手前に、影のヘッダ（make_shadow.py が生成）が
 *   #elif defined(RF_PIE_S3)
 *   #include "rf_ops_pie.h"
 * を差し込んで取り込む。ここで定義する 4 関数は参照実装と同じ署名・同じ結果（bit 一致）。
 *
 * 契約（rf_ops.h の "Width contract" と同じ）:
 *   rf_dot_i8(w, x, K, acc) = acc + Σ w[k]·x[k]（int32 の二の補数加算。モデル設計で |Σ| < 2^31）
 *   `ee.vmulas.s8.accx` は 16 レーンの s8×s8 の和を 40 bit の accx に溜める。途中で溢れず、
 *   `ee.srs.accx`（シフト 0）で下位 32 bit を取り出せば int32 の結果に等しい。
 *
 * 形は sanoTTS-jp（MIT）の saan_dot_i8_pie と同じ:
 *   - `ee.vmulas.s8.accx.ld.ip qu, as, 16, qx, qy` は「qx·qy を accx に足す」と「qu ← [as], as += 16」を
 *     1 命令でやる（積和はロード前の qx を使う）
 *   - `loopnez` は Xtensa のゼロオーバーヘッドループ
 *   - 最後の 1 組だけ併合しない `ee.vmulas.s8.accx` で締める（配列の 16 B 先を読まない）
 * ⚠️ `ee.vld.128.ip` は 16 バイト境界を要求する。影のヘッダで RF_ALIGN4 を aligned(16) にしたので
 *    エンジンの静的バッファと局所配列は 16 整列。整列していない x（attention の prow）は一時領域に複写する。
 *
 * rf_pie_enabled: 起動時の自己テスト（pf_pie_selftest）が参照実装との不一致を見つけたら 0 にする。 */
#ifndef RF_OPS_PIE_H
#define RF_OPS_PIE_H

#include <stdint.h>
#include <string.h>

#define RF_PIE_XTMP 512 /* 一時複写する x の最大長（K の最大は 4·RF_DIM = 512） */

extern int rf_pie_enabled;

static inline int rf_pie_al16(const void *p) { return ((uintptr_t)p & 15u) == 0; }

/* 両ポインタ 16 整列、k16 = K/16 ≥ 1。
 *
 * ⚠️ S3 の PIE は「QR へのロード → その使用」に 2 サイクルの遅延があり（esp-dl の資料）、
 *    もう一方のコアがメモリを混ませてロードが遅れると、間隔の短い使用は古い値を読んだ
 *    （CoreS3 実機で 2 コアのときだけ結果が実行ごとに変わり、1 コアなら毎回正しかった）。
 *    融合命令（ロード先 == MAC の入力）でも、ロードの直後に使う形でも起きた。
 *    ここでは融合命令を使わず、**ロードを 2 チャンク先行**させて、ロードと使用の間を
 *    常に 4 命令以上空ける。(q0,q1) と (q2,q3) を交互に使う。
 *   prologue: チャンク 0 → (q0,q1)、チャンク 1 → (q2,q3)
 *   loop（(k16−2)/2 回）: MAC(q0,q1); q0,q1 ← チャンク i+2 / MAC(q2,q3); q2,q3 ← チャンク i+3
 *   epilogue: 残り 2 チャンク（k16 偶数）か 3 チャンク（奇数。1 つだけ追加でロード）
 *   ロードは常にチャンク数の範囲内なので配列の先を読まない。k16 == 1 は別扱い */
static inline int32_t rf_pie_dot16(const int8_t *a, const int8_t *b, int k16) {
    int32_t out;
    const int8_t *pa = a, *pb = b;
    if (k16 == 1) {
        __asm__ volatile(
            "ee.zero.accx                    \n"
            "ee.vld.128.ip q0, %[pa], 16     \n"
            "ee.vld.128.ip q1, %[pb], 16     \n"
            "nop                             \n"
            "nop                             \n"
            "ee.vmulas.s8.accx q0, q1        \n"
            "ee.srs.accx %[out], %[sh], 0    \n"
            : [out] "=&a"(out), [pa] "+&a"(pa), [pb] "+&a"(pb)
            : [sh] "a"(0)
            : "memory");
        return out;
    }
    const int pairs = (k16 - 2) >> 1;
    const int odd = (k16 - 2) & 1;
    __asm__ volatile(
        "ee.zero.accx                    \n"
        "ee.vld.128.ip q0, %[pa], 16     \n"
        "ee.vld.128.ip q1, %[pb], 16     \n"
        "ee.vld.128.ip q2, %[pa], 16     \n"
        "ee.vld.128.ip q3, %[pb], 16     \n"
        "loopnez %[pairs], 1f            \n"
        "  ee.vmulas.s8.accx q0, q1      \n"
        "  ee.vld.128.ip q0, %[pa], 16   \n"
        "  ee.vld.128.ip q1, %[pb], 16   \n"
        "  ee.vmulas.s8.accx q2, q3      \n"
        "  ee.vld.128.ip q2, %[pa], 16   \n"
        "  ee.vld.128.ip q3, %[pb], 16   \n"
        "1:                              \n"
        "beqz %[odd], 2f                 \n"
        "  ee.vmulas.s8.accx q0, q1      \n"
        "  ee.vld.128.ip q0, %[pa], 16   \n"
        "  ee.vld.128.ip q1, %[pb], 16   \n"
        "  ee.vmulas.s8.accx q2, q3      \n"
        "  nop                           \n"
        "  ee.vmulas.s8.accx q0, q1      \n"
        "  j 3f                          \n"
        "2:                              \n"
        "  ee.vmulas.s8.accx q0, q1      \n"
        "  ee.vmulas.s8.accx q2, q3      \n"
        "3:                              \n"
        "ee.srs.accx %[out], %[sh], 0    \n"
        : [out] "=&a"(out), [pa] "+&a"(pa), [pb] "+&a"(pb)
        : [pairs] "a"(pairs), [odd] "a"(odd), [sh] "a"(0)
        : "memory");
    return out;
}

static inline int32_t rf_dot_i8(const int8_t *w, const int8_t *x, int K,
                                int32_t acc) {
    if (rf_pie_enabled && K >= 16 && (K & 15) == 0 && rf_pie_al16(w)) {
        if (rf_pie_al16(x)) return acc + rf_pie_dot16(w, x, K >> 4);
        if (K <= RF_PIE_XTMP) {
            int8_t tmp[RF_PIE_XTMP] __attribute__((aligned(16)));
            memcpy(tmp, x, (size_t)K);
            return acc + rf_pie_dot16(w, tmp, K >> 4);
        }
    }
    for (int k = 0; k < K; k++) acc += (int32_t)w[k] * x[k];
    return acc;
}

static inline void rf_dot2_i8(const int8_t *w0, const int8_t *w1,
                              const int8_t *x, int K, int32_t *a0,
                              int32_t *a1) {
    *a0 = rf_dot_i8(w0, x, K, *a0);
    *a1 = rf_dot_i8(w1, x, K, *a1);
}

/* a[0..7] += v · w[0..7]（疎な経路）。S3 の QACC はレーン 20 bit で溢れるのでスカラのまま */
static inline void rf_axpy8_i8(const int8_t *w, int32_t v, int32_t *a) {
    for (int j = 0; j < 8; j++) a[j] += v * (int32_t)w[j];
}

/* 2 行 × 2 トークン。x の 2 トークン目は x+K、w の 2 行目は w+K（呼び出し側の規約） */
static inline void rf_dot2x2_i8(const int8_t *w, const int8_t *x, int K,
                                int32_t *a00, int32_t *a01, int32_t *a10,
                                int32_t *a11) {
    *a00 = rf_dot_i8(w, x, K, *a00);
    *a01 = rf_dot_i8(w, x + K, K, *a01);
    *a10 = rf_dot_i8(w + K, x, K, *a10);
    *a11 = rf_dot_i8(w + K, x + K, K, *a11);
}

#endif /* RF_OPS_PIE_H */
