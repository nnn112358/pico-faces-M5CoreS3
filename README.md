# pico-faces on M5Stack CoreS3

[cpldcpu/pico-faces](https://github.com/cpldcpu/pico-faces) を **M5Stack CoreS3（ESP32-S3）** で動かすファームウェアです。
同じ移植の Tab5（ESP32-P4）版は [nnn112358/pico-faces-tab5](https://github.com/nnn112358/pico-faces-tab5) にあります。

## 元のリポジトリについて

pico-faces は、Raspberry Pi Pico 2（RP2350）のようなマイコンで顔画像を生成する小さな拡散モデルです。

- 潜在空間で動く拡散トランスフォーマー（DiT、約 2.4M パラメータ）と、潜在を 128×128 の RGB 画像に戻す VAE デコーダから成ります。
- 重みも計算もすべて int8 の整数演算で、推論エンジンは移植しやすい C99 で書かれています。
- 条件は「性別 × 笑顔」の 4 クラスと無条件の計 5 つで、classifier-free guidance（CFG）の強さも選べます。
- 生成は seed ごとに決定的で、同じ seed なら PC でもマイコンでも 1 bit も違わない画像になります。

この移植では、その推論エンジンを**無改変**のまま ESP-IDF でビルドし、CoreS3 の画面・タッチ・USB シリアルと、
ESP32-S3 の SIMD 命令（PIE、`ee.*`）による高速化を足しています。

## 使い方

### 用意するもの

- M5Stack CoreS3（16 MB flash、8 MB PSRAM）
- ESP-IDF v5.5（`~/esp/esp-idf` に展開してあるものを `idf.sh` が使います）
- `uv`（ビルド中に Python スクリプトを 1 本走らせます）

### ビルドと書き込み

```bash
git submodule update --init         # 上流を取る（約 300 MB。モデルの重みを含む）
./idf.sh build
./idf.sh -p /dev/ttyACM0 flash
```

### 操作

起動すると seed 3 で 1 枚生成して表示します（8 ステップ、male / smile、cfg none）。
画像は左に 240×240 で表示し、右 80 px のボタンで seed と条件を選んで生成します。

| ボタン | 動作 |
|---|---|
| `-1` `+1` `-10` `+10` | 次に生成する seed を増減（上に表示） |
| `cls N ...` | クラスを切り替え（F/no-smile → F/smile → M/no-smile → M/smile → null） |
| `cfg ...` | CFG を切り替え（none → w=4 → 6 → 8 → none） |
| **1枚生成** | 表示中の seed で 1 枚生成し、seed を 1 進める（左の画像をタップしても同じ） |
| **10枚連続** | seed から 10 枚を順に生成。途中で画面を触ると中断 |

seed と条件が同じなら毎回同じ顔になります（生成は決定的です）。

USB シリアル（115200）からは上流の Pico 版と同じ書式で指示できます。

```
G <seed> [k_steps] [class] [w] [count]   生成。class / w を省略すると上流の golden の規約。count 枚を seed から順に
I                                        モデル情報
M                                        メモリの使用量
```

応答は `OK seed=1 k=4 class=1 w=4 crc32=40c5e5a0 ms=6879 z=f0d5ec1a` のような 1 行です。
`crc32` が上流の golden（`G 1 4` → `40c5e5a0`）と一致すれば bit 単位で正しく動いています。`z` は潜在の CRC32 です。

⚠️ `cat` や `printf` でシリアルポートを開閉すると ESP32-S3 がリセットされることがあります。
`tools/serial_cmd.py` を使ってください。

```bash
uv run --no-project --with pyserial python tools/serial_cmd.py --boot 25 --wait 60 "G 1 4"
```

## 推論速度

CoreS3（ESP32-S3 240 MHz × 2 コア）で 1 枚を生成する時間です。既定モデル `m3_decD_deep_full`（DiT 深さ 12、blob 4.02 MB）。
PIE なし（`-DPF_PIE=0`）と PIE ありで画像は同じです（CRC32 が一致。連続 12 回の負荷試験でも同一）。

| 設定 | PIE なし | PIE あり | 倍率 | 参考: Tab5（ESP32-P4）の PIE 版 |
|---|---:|---:|---:|---:|
| K=4、cfg none | 17.0 s | **4.9 s** | 3.5× | 1.27 s |
| K=4、cfg w=4（golden の規約） | 27.9 s | **6.9 s** | 4.1× | 2.02 s |
| K=8、cfg none（起動時の既定） | 27.9 s | **6.9 s** | 4.1× | 2.03 s |
| K=8、cfg w=6 | 49.7 s | **10.9 s** | 4.6× | 3.52 s |

- 上流の RP2350 @300 MHz は K=4 w=4 で約 10 秒です。
- PIE ありでは、密な行列積を `ee.vmulas.s8.accx` の内積で計算するほか、疎な行列積（fc2）と VAE の疎な畳み込みも
  転置した重みで密な内積に置き換えています（ゼロは和に寄与しないので結果は同じ）。
  S3 の QACC はレーン 20 bit で溢れるため、esp-dl の S3 版 matmul のような QACC 累算は使っていません。
- 詳しくは [docs/details.md](docs/details.md)。

## 詳しい内容

構成、PIE の仕組みと実機で踏んだハザード、メモリの使用量、測定の内訳、切り分け用のシリアルコマンドは
[docs/details.md](docs/details.md) にまとめてあります。

## ライセンス

このリポジトリのコードと文書（`main/`、`components/`、`tools/`、`docs/`）は [MIT ライセンス](LICENSE)です。
S3 の内積カーネルは [sanoTTS-jp](https://github.com/ayutaz/sanoTTS-jp)（MIT）の `saan_dot_i8_pie` を出発点にし、
[esp-dl](https://github.com/espressif/esp-dl) の `esp32s3-pie-simd` の資料に従って書き直しました。

上流の cpldcpu/pico-faces（推論エンジンとモデルの重み。`upstream/` の submodule として参照）には
ライセンス表記がありません（2026-09-06 時点）。上流部分の利用条件は上流の作者に従ってください。
