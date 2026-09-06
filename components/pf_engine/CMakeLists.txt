# pico-faces の推論エンジン（upstream/engine）を**ソース無改変で**コンパイルする。
#
# upstream/ は cpldcpu/pico-faces の git submodule。エンジンは移植可能な C99（整数演算のみ、
# malloc なし、静的 arena）で、RP2350 版ファームと同じファイルをそのまま使う。
#
# モデルごとの寸法（RF_DEPTH など）は upstream/checkpoints/<model>/rf_cfg.h にあり、
# rf_model.h が `#include <rf_cfg.h>` で **インクルードパスの先頭**から探す。
# したがって INCLUDE_DIRS の順序を変えてはいけない（既定の engine/include/rf_cfg.h は m1_gray 用）。
#
#   -DPF_MODEL=m3_decD_deep_full   高品質（既定。DiT 深さ 12、blob 4.02 MB）
#   -DPF_MODEL=m3_long_cfg         高速（DiT 深さ 8、blob 2.57 MB）
#   -DPF_PIE=0                     PIE（S3 の SIMD）を使わない参照実装に戻す（陰性対照 / bit 一致の確認用）
set(PF_UP "${CMAKE_CURRENT_LIST_DIR}/../../upstream")
if(NOT DEFINED PF_MODEL)
    set(PF_MODEL "m3_decD_deep_full")
endif()
set(PF_CFG_DIR "${PF_UP}/checkpoints/${PF_MODEL}")
if(NOT EXISTS "${PF_CFG_DIR}/rf_cfg.h")
    message(FATAL_ERROR "rf_cfg.h が無い: ${PF_CFG_DIR}\n  git submodule update --init を実行したか、-DPF_MODEL の綴りを確認すること")
endif()

# --- PIE: 上流の rf_ops.h に PIE 分岐を足した「影のヘッダ」を configure 時に生成し、
#     インクルードパスの先頭に置く（上流のソースは無改変。差分は pie/make_shadow.py の 3 箇所だけ）---
if(NOT DEFINED PF_PIE)
    set(PF_PIE 1)
endif()
set(PF_INC "${PF_CFG_DIR}" "${PF_UP}/engine/include")
set(PF_SRCS "${PF_UP}/engine/src/graph.c"
            "${PF_UP}/engine/src/dit.c"
            "${PF_UP}/engine/src/vae_dec.c"
            "${PF_UP}/engine/src/kernels_ref.c"
            "${PF_UP}/engine/src/prng.c")
if(PF_PIE)
    set(PF_SHADOW_DIR "${CMAKE_BINARY_DIR}/pf_pie")
    execute_process(
        COMMAND uv run --no-project python "${CMAKE_CURRENT_LIST_DIR}/pie/make_shadow.py"
                "${PF_UP}/engine/include/rf_ops.h" "${PF_SHADOW_DIR}"
        RESULT_VARIABLE PF_SHADOW_RC OUTPUT_VARIABLE PF_SHADOW_OUT ERROR_VARIABLE PF_SHADOW_ERR)
    if(NOT PF_SHADOW_RC EQUAL 0)
        message(FATAL_ERROR "影のヘッダの生成に失敗: ${PF_SHADOW_OUT}${PF_SHADOW_ERR}")
    endif()
    message(STATUS "pico-faces: PIE 有効（${PF_SHADOW_DIR}/rf_ops.h）")
    set(PF_INC "${PF_SHADOW_DIR}" "${CMAKE_CURRENT_LIST_DIR}/pie" ${PF_INC})
    list(APPEND PF_SRCS "${CMAKE_CURRENT_LIST_DIR}/pie/pf_pie_s3.c")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${PF_UP}/engine/include/rf_ops.h" "${CMAKE_CURRENT_LIST_DIR}/pie/make_shadow.py")
endif()

idf_component_register(
    SRCS ${PF_SRCS}
    INCLUDE_DIRS ${PF_INC}
    LDFRAGMENTS "linker.lf"
    REQUIRES ""
    PRIV_REQUIRES esp_timer
)

if(PF_PIE)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC RF_PIE_S3=1)
    # 疎な経路（fc2 / VAE デコーダ）は dit.c から呼ばれる関数をリンカの --wrap で差し替え、密な内積に流す。
    # 上流のソースは無改変。参照実装は __real_ として残り、フォールバックと自己テストに使う
    target_link_libraries(${COMPONENT_LIB} INTERFACE "-Wl,--wrap=rf_axpy_acc16_sp" "-Wl,--wrap=rf_decode")
endif()

# 上流の高速経路は Cortex-M33 の SMLAD 用。PIE を切ると参照カーネル（rf_ops.h の #else 側）になる。
# RF_HOT_IN_RAM は定義しない（.time_critical セクションは IDF のリンカスクリプトに無い）。
target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -Wno-unused-function)
