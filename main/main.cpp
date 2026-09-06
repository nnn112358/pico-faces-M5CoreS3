/* pico-faces on M5Stack CoreS3（ESP32-S3）
 *
 * 上流 cpldcpu/pico-faces の C99 推論エンジン（upstream/engine）で 128×128 RGB の顔画像を
 * 生成し、CoreS3 の 320×240 画面に 1.875 倍（240×240）で表示する。エンジンのソースは 1 バイトも変えていない。
 *
 * 構成:
 *   - モデル blob は app（.rodata）に埋め込み、flash から直接読む（S3 は PSRAM も flash も同じくらいの速さ）
 *   - 2 コア並列は pf_par.c（rf_par_for の置き換え）。生成タスクはコア 0 固定
 *   - 右 80 px のボタン: seed ±1 / ±10、class、cfg、1枚、10枚。USB シリアルからは `G <seed> [k] [class] [w] [count]`
 *
 * タッチ判定:
 *   M5Unified の wasClicked() は「押した位置から 8 px も動かず 500 ms 以内に離した」ときしか真にならないので、
 *   wasReleased()（離した瞬間）と base_x/base_y（押し始めの座標）で判定する。
 *   生成中も rf_par_for 経由（pf_ui_poll、40 ms 間隔）で読み、触れていれば印を立てる。
 *   1 枚生成ではその印を捨て、10 枚連続では次の画像に進まず「中断」にする。
 *
 * USB シリアル（115200 / USB Serial-JTAG）:
 *   G <seed> [k_steps] [class] [w] [count]   生成。class/w を省略すると上流の golden 規約
 *                                            （class = seed % n_cond, w = seed % (n_w+1) - 1）。
 *                                            count（既定 1）枚を seed から順に生成
 *   I                                        モデル情報
 *   M                                        メモリの使用状況（内部 SRAM / PSRAM）
 *   応答: OK seed=.. k=.. class=.. w=.. crc32=........ ms=..（1 枚ごと）
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <M5Unified.h>

#include "driver/usb_serial_jtag.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

extern "C" {
#include "rf_model.h"
#include "rf_ops.h"
void pf_par_init(void);
void pf_par_reset(void);
void pf_par_report(void);
#ifdef RF_PIE_S3
int pf_pie_selftest(void);   /* PIE カーネルを参照実装と照合（不一致なら PIE を切る） */
int pf_stage_prepare(const rf_model_t *m); /* ステージング先の確保と、fc2 / VAE の重みの転置複製 */
extern int64_t pf_vae_us;
extern int pf_vae_pie;
extern int pf_dense_mask;
#endif
extern int pf_par_single;
extern int pf_par_nopoll;
extern int pf_vae_scalar;
extern volatile uint8_t rf_progress;
extern volatile uint8_t rf_progress_total;
const int16_t *rf_z_state(void); /* dit.c: 直近の潜在 z_tok（64×32 int16）*/
}

static const char *TAG = "pico_faces";

extern const uint8_t model_bin_start[] asm("_binary_model_bin_start");
extern const uint8_t model_bin_end[] asm("_binary_model_bin_end");

/* ---- 画面レイアウト（320×240 横向き） ---------------------------------- */
#define PF_IMG_PX 240              /* 128 × 1.875 */
#define PF_ZOOM 1.875f
#define PF_PANEL_X 240
#define PF_PANEL_W 80
#define PF_STATUS_Y 224            /* 画像の下端に重ねる状態行（12 px）+ 進捗バー（4 px） */

static const char *k_class_names[] = {"F/no-smile", "F/smile", "M/no-smile", "M/smile", "null"};

typedef struct {
    uint64_t seed;
    int k_steps;
    int cond;
    int w_idx;
    int count;
    bool from_serial;
} pf_req_t;

static rf_model_t *s_model;          /* PSRAM */
static uint8_t *s_img;               /* PSRAM: 128×128×3 */
static QueueHandle_t s_q;
static SemaphoreHandle_t s_gfx;

static uint64_t s_seed = 3;
static int s_cond = 3;
static int s_w_idx = -1;
static int s_k = 8;

static pf_req_t s_cur;
static uint32_t s_last_ms;
static uint32_t s_last_crc;
static int s_batch_i, s_batch_n;
static bool s_busy;
static int64_t s_touch_ok_us;
static volatile bool s_touched_while_busy;

static int w_value(int w_idx) {
    if (w_idx < 0 || w_idx >= (int)s_model->n_w) return 0;
    return (int)(s_model->w_q8[w_idx] / 256);
}

/* ---- ボタン ------------------------------------------------------------- */
enum {
    B_SEED_M1, B_SEED_P1, B_SEED_M10, B_SEED_P10,
    B_CLASS, B_CFG,
    B_GEN1, B_GEN10,
    B_N,
    B_IMAGE = 100,
    B_OTHER = 101
};

typedef struct {
    int x, y, w, h;
} pf_rect_t;

#define ROW_H 34
#define ROW_Y(i) (16 + (i) * 38)
static const pf_rect_t k_btn[B_N] = {
    {PF_PANEL_X, ROW_Y(0), 38, ROW_H},        /* -1 */
    {PF_PANEL_X + 42, ROW_Y(0), 38, ROW_H},   /* +1 */
    {PF_PANEL_X, ROW_Y(1), 38, ROW_H},        /* -10 */
    {PF_PANEL_X + 42, ROW_Y(1), 38, ROW_H},   /* +10 */
    {PF_PANEL_X, ROW_Y(2), PF_PANEL_W, ROW_H}, /* class */
    {PF_PANEL_X, ROW_Y(3), PF_PANEL_W, ROW_H}, /* cfg */
    {PF_PANEL_X, ROW_Y(4), PF_PANEL_W, ROW_H}, /* 1 枚 */
    {PF_PANEL_X, ROW_Y(5), PF_PANEL_W, ROW_H}, /* 10 枚 */
};

static bool in_rect(const pf_rect_t &r, int x, int y) {
    return x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h;
}

static void draw_button(const pf_rect_t &r, const char *label, uint16_t fill, uint16_t fg) {
    auto &d = M5.Display;
    d.fillRoundRect(r.x, r.y, r.w, r.h, 5, fill);
    d.drawRoundRect(r.x, r.y, r.w, r.h, 5, TFT_DARKGREY);
    d.setTextDatum(middle_center);
    d.setTextColor(fg, fill);
    d.drawString(label, r.x + r.w / 2, r.y + r.h / 2);
}

/* ---- 描画 --------------------------------------------------------------- */
static void draw_status(const char *status, uint16_t color) {
    auto &d = M5.Display;
    d.startWrite();
    d.fillRect(0, PF_STATUS_Y, PF_IMG_PX, 12, TFT_BLACK);
    d.setFont(&fonts::lgfxJapanGothic_12);
    d.setTextDatum(top_left);
    d.setTextColor(color, TFT_BLACK);
    d.drawString(status, 2, PF_STATUS_Y);
    d.endWrite();
}

static void draw_panel(void) {
    auto &d = M5.Display;
    char buf[48];
    d.startWrite();
    d.fillRect(PF_PANEL_X, 0, PF_PANEL_W, d.height(), TFT_BLACK);
    d.setFont(&fonts::Font2);
    d.setTextDatum(top_left);
    d.setTextColor(TFT_WHITE, TFT_BLACK);
    snprintf(buf, sizeof buf, "seed %llu", (unsigned long long)s_seed);
    d.drawString(buf, PF_PANEL_X + 2, 0);
    d.setFont(&fonts::Font2);
    draw_button(k_btn[B_SEED_M1], "-1", TFT_DARKGREY, TFT_WHITE);
    draw_button(k_btn[B_SEED_P1], "+1", TFT_DARKGREY, TFT_WHITE);
    draw_button(k_btn[B_SEED_M10], "-10", TFT_DARKGREY, TFT_WHITE);
    draw_button(k_btn[B_SEED_P10], "+10", TFT_DARKGREY, TFT_WHITE);
    d.setFont(&fonts::Font0);
    snprintf(buf, sizeof buf, "cls%d %s", s_cond, s_cond < 5 ? k_class_names[s_cond] : "");
    draw_button(k_btn[B_CLASS], buf, TFT_DARKGREY, TFT_WHITE);
    if (s_w_idx < 0) snprintf(buf, sizeof buf, "cfg none K%d", s_k);
    else snprintf(buf, sizeof buf, "cfg w=%d K%d", w_value(s_w_idx), s_k);
    draw_button(k_btn[B_CFG], buf, TFT_DARKGREY, TFT_WHITE);
    d.setFont(&fonts::lgfxJapanGothic_16);
    draw_button(k_btn[B_GEN1], "1枚生成", 0x0320, TFT_WHITE);
    draw_button(k_btn[B_GEN10], "10枚連続", 0x0014, TFT_WHITE);
    d.endWrite();
}

static void draw_progress(void) {
    auto &d = M5.Display;
    const int x = 0, y = PF_STATUS_Y + 12, w = PF_IMG_PX, h = 4;
    int total = rf_progress_total ? rf_progress_total : 1;
    int done = rf_progress > total ? total : rf_progress;
    d.startWrite();
    d.fillRect(x, y, w * done / total, h, TFT_GREENYELLOW);
    d.fillRect(x + w * done / total, y, w - w * done / total, h, TFT_DARKGREY);
    d.endWrite();
}

/* 128×128 RGB (HWC) → 240×240（1.875 倍、最近傍）。
 * ⚠️ M5GFX の rgb888_t はメモリ上 b, g, r の順（colortype.hpp）。s_img（r, g, b）をそのまま cast すると
 *    R と B が入れ替わって青みがかる（実機で踏んだ）。メンバー代入で詰め直してから渡す */
static lgfx::rgb888_t *s_px; /* PSRAM: 128×128 */
static void draw_image(void) {
    for (int i = 0; i < RF_IMG_HW * RF_IMG_HW; i++) {
        s_px[i].r = s_img[i * 3 + 0];
        s_px[i].g = s_img[i * 3 + 1];
        s_px[i].b = s_img[i * 3 + 2];
    }
    auto &d = M5.Display;
    d.startWrite();
    d.pushImageRotateZoom(PF_IMG_PX / 2, PF_IMG_PX / 2, RF_IMG_HW / 2, RF_IMG_HW / 2, 0.0f, PF_ZOOM, PF_ZOOM,
                          RF_IMG_HW, RF_IMG_HW, s_px);
    d.endWrite();
}

extern "C" void rf_step_hook(void) {
    if (xSemaphoreTake(s_gfx, pdMS_TO_TICKS(50)) == pdTRUE) {
        draw_progress();
        xSemaphoreGive(s_gfx);
    }
}

/* ---- タッチ ------------------------------------------------------------- */
static int s_touch_ignore; /* 1 = タッチを完全に無視（切り分け用。D の bit5） */

static int poll_touch(void) {
    if (s_touch_ignore) return -1;
    M5.update();
    if (esp_timer_get_time() < s_touch_ok_us) return -1;
    int n = M5.Touch.getCount();
    for (int i = 0; i < n; i++) {
        auto t = M5.Touch.getDetail(i);
        if (!t.wasReleased()) continue;
        int x = t.base_x, y = t.base_y;
        ESP_LOGI(TAG, "touch release: began %d,%d ended %d,%d", x, y, (int)t.x, (int)t.y);
        if (x < PF_IMG_PX && y < PF_STATUS_Y) return B_IMAGE;
        for (int b = 0; b < B_N; b++)
            if (in_rect(k_btn[b], x, y)) return b;
        return B_OTHER;
    }
    return -1;
}

extern "C" void pf_ui_poll(void) {
    static int64_t last_us;
    if (!s_busy || s_touch_ignore) return;
    int64_t now = esp_timer_get_time();
    if (now - last_us < 40000) return;
    last_us = now;
    M5.update();
    if (now < s_touch_ok_us) return;
    int n = M5.Touch.getCount();
    for (int i = 0; i < n; i++) {
        auto t = M5.Touch.getDetail(i);
        if (t.isPressed() || t.wasReleased()) {
            if (!s_touched_while_busy) ESP_LOGI(TAG, "touch during generation at %d,%d", (int)t.x, (int)t.y);
            s_touched_while_busy = true;
        }
    }
}

static void drain_touch(void) {
    for (int i = 0; i < 3; i++) {
        M5.update();
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

/* ---- 生成 --------------------------------------------------------------- */
static void generate_one(const pf_req_t &r) {
    s_cur = r;
    s_busy = true;
    char buf[64];
    xSemaphoreTake(s_gfx, portMAX_DELAY);
    if (s_batch_n > 1) snprintf(buf, sizeof buf, "生成中 %d/%d seed %llu（触ると中断）", s_batch_i, s_batch_n,
                                (unsigned long long)r.seed);
    else snprintf(buf, sizeof buf, "生成中 seed %llu", (unsigned long long)r.seed);
    draw_status(buf, TFT_YELLOW);
    rf_progress = 0;
    draw_progress();
    xSemaphoreGive(s_gfx);

    pf_par_reset();
    int64_t t0 = esp_timer_get_time();
    rf_generate(s_model, r.seed, r.k_steps, r.cond, r.w_idx, s_img, NULL);
    s_last_ms = (uint32_t)((esp_timer_get_time() - t0) / 1000);
    s_last_crc = rf_crc32(s_img, (size_t)RF_IMG_HW * RF_IMG_HW * RF_IMG_CH);

    xSemaphoreTake(s_gfx, portMAX_DELAY);
    draw_image();
    draw_panel();
    draw_progress();
    xSemaphoreGive(s_gfx);

    uint32_t zcrc = rf_crc32((const uint8_t *)rf_z_state(), sizeof(int16_t) * RF_TOKENS * RF_PD);
    printf("OK seed=%llu k=%d class=%d w=%d crc32=%08" PRIx32 " ms=%" PRIu32 " z=%08" PRIx32 "\n", (unsigned long long)r.seed,
           r.k_steps, r.cond, w_value(r.w_idx), s_last_crc, s_last_ms, zcrc);
    ESP_LOGI(TAG, "seed %llu class %d k %d w %d: crc32 %08" PRIx32 " in %" PRIu32 " ms", (unsigned long long)r.seed,
             r.cond, r.k_steps, w_value(r.w_idx), s_last_crc, s_last_ms);
#ifdef RF_PIE_S3
    ESP_LOGI(TAG, "  DiT %lld ms, VAE decode %lld ms (%s)", (long long)(s_last_ms - pf_vae_us / 1000),
             (long long)(pf_vae_us / 1000), pf_vae_pie ? "dense/PIE" : "reference");
#endif
    pf_par_report();
    s_busy = false;
}

static void enqueue_from_panel(int count) {
    pf_req_t r;
    r.seed = s_seed;
    r.k_steps = s_k;
    r.cond = s_cond;
    r.w_idx = s_w_idx;
    r.count = count;
    r.from_serial = false;
    if (xQueueSend(s_q, &r, 0) == pdTRUE) s_seed += (uint64_t)count;
}

static void gen_task(void *arg) {
    (void)arg;
    pf_req_t req;
    for (;;) {
        if (xQueueReceive(s_q, &req, pdMS_TO_TICKS(20)) != pdTRUE) {
            int b = poll_touch();
            if (b < 0 || b == B_OTHER) continue;
            switch (b) {
            case B_SEED_M10: s_seed = s_seed >= 10 ? s_seed - 10 : 0; break;
            case B_SEED_M1: s_seed = s_seed >= 1 ? s_seed - 1 : 0; break;
            case B_SEED_P1: s_seed += 1; break;
            case B_SEED_P10: s_seed += 10; break;
            case B_CLASS: s_cond = (s_cond + 1) % (int)s_model->n_cond; break;
            case B_CFG:
                s_w_idx += 1;
                if (s_w_idx >= (int)s_model->n_w) s_w_idx = -1;
                break;
            case B_GEN1:
            case B_IMAGE: enqueue_from_panel(1); break;
            case B_GEN10: enqueue_from_panel(10); break;
            default: break;
            }
            xSemaphoreTake(s_gfx, portMAX_DELAY);
            draw_panel();
            xSemaphoreGive(s_gfx);
            continue;
        }
        int n = req.count > 0 ? req.count : 1;
        s_batch_n = n;
        bool cancelled = false;
        for (int i = 0; i < n; i++) {
            s_batch_i = i + 1;
            pf_req_t r = req;
            r.seed = req.seed + (uint64_t)i;
            s_touched_while_busy = false;
            generate_one(r);
            if (i + 1 < n) {
                int b = poll_touch();
                if (s_touched_while_busy || b >= 0) {
                    cancelled = true;
                    break;
                }
            }
        }
        s_touched_while_busy = false;
        drain_touch();
        char buf[64];
        if (cancelled) snprintf(buf, sizeof buf, "中断（%d/%d 枚）", s_batch_i, n);
        else if (n > 1) snprintf(buf, sizeof buf, "完了 %d 枚 seed %llu〜%llu", n, (unsigned long long)req.seed,
                                 (unsigned long long)(req.seed + n - 1));
        else snprintf(buf, sizeof buf, "完了 seed %llu  %.1f s", (unsigned long long)req.seed, s_last_ms / 1000.0);
        s_batch_n = 0;
        xSemaphoreTake(s_gfx, portMAX_DELAY);
        draw_status(buf, TFT_GREENYELLOW);
        xSemaphoreGive(s_gfx);
    }
}

/* ---- USB シリアルの行入力 ---------------------------------------------- */
static void console_task(void *arg) {
    (void)arg;
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    cfg.rx_buffer_size = 1024;
    cfg.tx_buffer_size = 1024;
    esp_err_t err = usb_serial_jtag_driver_install(&cfg);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "usb_serial_jtag_driver_install: %s", esp_err_to_name(err));
        vTaskDelete(NULL);
        return;
    }
    char line[96];
    int n = 0;
    for (;;) {
        uint8_t c;
        if (usb_serial_jtag_read_bytes(&c, 1, pdMS_TO_TICKS(100)) != 1) continue;
        if (c != '\n' && c != '\r') {
            if (n < (int)sizeof line - 1) line[n++] = (char)c;
            continue;
        }
        line[n] = 0;
        n = 0;
        if (line[0] == 'G') {
            char *e1, *e2, *e3, *e4, *e5;
            pf_req_t r;
            r.from_serial = true;
            r.seed = strtoull(line + 1, &e1, 0);
            r.k_steps = (int)strtol(e1, &e2, 0);
            if (!r.k_steps) r.k_steps = 4;
            long cv = strtol(e2, &e3, 0);
            r.cond = (e3 != e2) ? (int)cv : (int)(r.seed % s_model->n_cond);
            long wv = strtol(e3, &e4, 0);
            r.w_idx = -1;
            if (e4 != e3) {
                for (uint32_t j = 0; j < s_model->n_w; j++)
                    if (s_model->w_q8[j] == (uint32_t)(wv * 256)) r.w_idx = (int)j;
            } else if (e3 == e2 && s_model->n_w) {
                r.w_idx = (int)(r.seed % (s_model->n_w + 1)) - 1;
            }
            long cnt = strtol(e4, &e5, 0);
            r.count = (e5 != e4 && cnt > 0) ? (int)cnt : 1;
            if (xQueueSend(s_q, &r, 0) != pdTRUE) printf("BUSY\n");
        } else if (line[0] == 'D') {
#ifdef RF_PIE_S3
            int v = (int)strtol(line + 1, NULL, 0);
            pf_dense_mask = v & 3;
            pf_par_single = (v >> 2) & 1;
            pf_par_nopoll = (v >> 3) & 1;
            pf_vae_scalar = (v >> 4) & 1;
            s_touch_ignore = (v >> 5) & 1;
            printf("dense mask = %d (bit0 fc2, bit1 vae), single-core = %d, nopoll = %d, vae-scalar = %d, ignore-touch = %d\n",
                   pf_dense_mask, pf_par_single, pf_par_nopoll, pf_vae_scalar, s_touch_ignore);
#endif
        } else if (line[0] == 'X') {
            /* 直近の画像を 16 進で出す（1 行 = 1 画素行 128×3 バイト）。ホストで golden と比べるため */
            for (int y = 0; y < RF_IMG_HW; y++) {
                const uint8_t *row = s_img + (size_t)y * RF_IMG_HW * RF_IMG_CH;
                char hex[RF_IMG_HW * RF_IMG_CH * 2 + 1];
                for (int i = 0; i < RF_IMG_HW * RF_IMG_CH; i++) {
                    static const char d[] = "0123456789abcdef";
                    hex[i * 2] = d[row[i] >> 4];
                    hex[i * 2 + 1] = d[row[i] & 15];
                }
                hex[sizeof hex - 1] = 0;
                printf("X%03d %s\n", y, hex);
            }
            printf("XEND\n");
        } else if (line[0] == 'M') {
            const struct { const char *name; uint32_t caps; } kinds[] = {
                {"internal", MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT}, {"psram", MALLOC_CAP_SPIRAM}};
            for (auto &k : kinds) {
                printf("MEM %-8s total=%u free=%u min_free=%u largest=%u\n", k.name,
                       (unsigned)heap_caps_get_total_size(k.caps), (unsigned)heap_caps_get_free_size(k.caps),
                       (unsigned)heap_caps_get_minimum_free_size(k.caps),
                       (unsigned)heap_caps_get_largest_free_block(k.caps));
            }
        } else if (line[0] == 'I') {
            printf("pico-faces-cores3 K=%u dim=%u depth=%u cond=%u ch=%u n_w=%u blob=%u sys=%dMHz\n",
                   (unsigned)s_model->K, (unsigned)s_model->dim, (unsigned)s_model->depth,
                   (unsigned)s_model->n_cond, (unsigned)s_model->img_ch, (unsigned)s_model->n_w,
                   (unsigned)(model_bin_end - model_bin_start), CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
        } else if (line[0]) {
            printf("? (G <seed> [k] [class] [w] [count] | I | M)\n");
        }
    }
}

extern "C" void app_main(void) {
    auto cfg = M5.config();
    cfg.clear_display = true;
    cfg.internal_mic = false;
    cfg.internal_spk = false;
    M5.begin(cfg);
    auto &d = M5.Display;
    if (d.height() > d.width()) d.setRotation(1);
    d.fillScreen(TFT_BLACK);
    ESP_LOGI(TAG, "board %d, display %d x %d", (int)M5.getBoard(), (int)d.width(), (int)d.height());

    /* モデル: flash（.rodata）の blob をそのまま読む。エンジンは 4 バイト境界の配列を指すだけ */
    size_t blob_len = (size_t)(model_bin_end - model_bin_start);
    s_model = (rf_model_t *)heap_caps_calloc(1, sizeof(rf_model_t), MALLOC_CAP_SPIRAM);
    s_img = (uint8_t *)heap_caps_malloc((size_t)RF_IMG_HW * RF_IMG_HW * RF_IMG_CH, MALLOC_CAP_SPIRAM);
    s_px = (lgfx::rgb888_t *)heap_caps_malloc((size_t)RF_IMG_HW * RF_IMG_HW * sizeof(lgfx::rgb888_t), MALLOC_CAP_SPIRAM);
    if (!s_model || !s_img || !s_px) {
        ESP_LOGE(TAG, "PSRAM 確保に失敗");
        return;
    }
    int rc = rf_model_load(model_bin_start, blob_len, s_model);
    if (rc != 0) {
        ESP_LOGE(TAG, "rf_model_load: %d（blob と rf_cfg.h のモデルが食い違っていないか）", rc);
        d.setFont(&fonts::Font4);
        d.drawString("model load failed", 10, 10);
        return;
    }
    ESP_LOGI(TAG, "model: K=%u dim=%u depth=%u cond=%u n_w=%u blob=%u B (flash); internal free %u B, largest %u B",
             (unsigned)s_model->K, (unsigned)s_model->dim, (unsigned)s_model->depth, (unsigned)s_model->n_cond,
             (unsigned)s_model->n_w, (unsigned)blob_len,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));

    s_gfx = xSemaphoreCreateMutex();
    s_q = xQueueCreate(4, sizeof(pf_req_t));
    s_touch_ok_us = esp_timer_get_time() + 3000000;
#ifdef RF_PIE_S3
    {
        int64_t t = esp_timer_get_time();
        int f = pf_pie_selftest();
        ESP_LOGI(TAG, "PIE selftest: %s (%lld us)", f ? "FAILED -> scalar" : "OK", (long long)(esp_timer_get_time() - t));
        pf_stage_prepare(s_model);
    }
#endif
    pf_par_init();

    s_k = (int)s_model->K;
    s_cond = 3;
    s_w_idx = -1; /* cfg none（高速） */
    s_seed = 3;
    s_cur.seed = s_seed;
    s_cur.k_steps = s_k;
    s_cur.cond = s_cond;
    s_cur.w_idx = s_w_idx;
    draw_panel();
    enqueue_from_panel(1);

    xTaskCreatePinnedToCore(gen_task, "pf_gen", 20480, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(console_task, "pf_con", 6144, NULL, 4, NULL, 1);
}
