# Lyra-2 セットアップ手順 (検証済み)

NVIDIA Lyra-2 (https://github.com/nv-tlabs/lyra) を Linux + conda 環境で構築するための、本ホスト (`/home/kojima/Lyra/lyra/Lyra-2`) で実際に動作確認した手順。
公式の `INSTALL.md` をベースに、conda チャネル衝突や不足ヘッダといったハマりどころを差し替えている。

本書および補助スクリプトは **`Lyra-2/Lyra2_setup/`** 配下にまとまっている:

```
Lyra-2/
├── Lyra2_setup/
│   ├── Setup.md              # 本書
│   ├── activate_lyra2.sh     # env 変数を source で一括投入
│   └── sample_check.py       # 動作確認スクリプト (cwd 非依存)
├── INSTALL.md                # 公式 (リファレンス用、本書とずれる箇所あり)
├── README.md                 # 公式モデル説明
├── assets/, checkpoints/, lyra_2/, ...
```

以下では `Lyra-2/` リポジトリルートを **`$LYRA2_ROOT`** と表記する (`activate_lyra2.sh` を source すると自動でこの env 変数が定義される)。本書では 2 ホストでの実機検証結果を併記しており、`LYRA2_ROOT` は前者で `/home/kojima/Lyra/lyra/Lyra-2`、後者で `/data1/kojima/WM/lyra/Lyra-2`。

## 1. ハードウェア / OS 要件

| 項目 | 検証構成 (A6000 ホスト) | 検証構成 (H200 ホスト) | 公式想定 |
|---|---|---|---|
| OS | Ubuntu (Linux 6.14) | Ubuntu (Linux 6.8) | Ubuntu 22.04 |
| GPU | NVIDIA RTX A6000 (48 GB) | NVIDIA H200 NVL (140 GB) | H100 80 GB |
| CUDA driver | 13.0 | 13.1 (driver 590.48) | 12.4+ |
| RAM | 125 GB | 503 GB | (大量推奨) |
| ディスク空き | 500 GB+ | 500 GB+ | チェックポイントだけで ~91 GB |

> **注意:** 14B Wan モデル本体が GPU 上で **約 46 GB** を要求するため、フル動画生成 (`lyra2_zoomgs_inference` / `lyra2_custom_traj_inference`) は **48 GB GPU では OOM** する。サブコンポーネント (MoGe / Depth Anything 3 / VAE) や、別途用意した動画からの 3D 再構成 (`vipe_da3_gs_recon`) は A6000 でも動作可能。**H200 NVL (140 GB)** であれば 14B Wan 本体も収まり、フル推論が可能。

## 2. リポジトリ取得 (submodule 必須)

```bash
cd /home/kojima/Lyra
git clone https://github.com/nv-tlabs/lyra.git
cd lyra
git submodule update --init --recursive --depth 1
```

`Lyra-2/lyra_2/_src/inference/{vipe,depth_anything_3}` が submodule として展開される。

## 3. conda 環境の作成

> 公式 INSTALL.md は env 作成 → `conda install gcc=13.3.0 ...` → `conda install cuda` の順だが、現行の conda-forge / nvidia チャネルでは `cuda` メタパッケージが Windows 専用依存を要求して **解決失敗** する。**1 コマンドで cuda-toolkit と Python ベースを同時にソルブさせると通る。**

```bash
conda create -n lyra2 \
  -c nvidia/label/cuda-12.8.0 -c conda-forge -y \
  python=3.10 pip cmake ninja libgl ffmpeg packaging cuda-toolkit=12.8

conda activate lyra2
conda install -c conda-forge -y eigen zlib
```

これで `gcc 13.4.0 / gxx 13.4.0 / cuda-toolkit 12.8` が同居した env が出来る (公式想定の gcc 13.3.0 ではなく、解決されるバージョンを採用)。

```bash
nvcc --version    # release 12.8 V12.8.61 が出れば OK
```

## 4. ビルド/実行用の env 変数 (毎セッション必要)

`Lyra2_setup/activate_lyra2.sh` を毎回 `source` する。
スクリプトは自身の置かれた場所から `LYRA2_ROOT` を計算するので、**別ホストや別パスへ Lyra-2 ごと丸ごと移しても書き換え不要**。

```bash
source /home/kojima/Lyra/lyra/Lyra-2/Lyra2_setup/activate_lyra2.sh
# => [lyra2] env activated.  LYRA2_ROOT=/home/kojima/Lyra/lyra/Lyra-2
```

スクリプトが行うこと:

- conda activate lyra2
- `CUDA_HOME` / `CPATH` / `LD_LIBRARY_PATH` 設定 (nvtx3 ヘッダ追加が肝)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `PYTHONPATH` に `$LYRA2_ROOT` を追加 (Lyra-2 本体は editable install されないため必須)

ビルド時のみ追加で必要なものは `activate_lyra2.sh` 内のコメントで案内している (`CC` / `CXX` / `MAX_JOBS` / `TORCH_CUDA_ARCH_LIST`)。新規ビルド時はコメントを外して export する。

## 5. PyTorch + Python 依存

source 後、`$LYRA2_ROOT` に移動して以下を順に実行する。

```bash
cd "$LYRA2_ROOT"

# 5.1 PyTorch (cu128 wheel)
pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128

# 動作確認
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# => 2.7.1+cu128 True NVIDIA RTX A6000

# 5.2 requirements.txt (--no-deps、deps は後段で個別)
pip install --no-deps -r requirements.txt

# 5.3 MoGe (deps 込み)
pip install "git+https://github.com/microsoft/MoGe.git"

# 5.4 Transformer Engine (PyTorch 拡張) — nvtx3 ヘッダが CPATH に必要
pip install --no-build-isolation "transformer_engine[pytorch]"

# 5.5 cuda_runtime → cudart シンボリックリンク (TE が cudart を探すため)
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
ln -sf "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"

# 5.6 --no-deps で取り残された transient deps を追加
#  (5.2 の `--no-deps` で tyro / zarr / fvcore / pytest 等の依存が落ちる。
#   sample_check.py や inference --help は通るが、`import tyro` / `import zarr`
#   が単独で失敗するため、後段で別の用途に詰まる前にここで補完しておく。)
pip install typeguard docstring-parser shtab asciitree iniconfig execnet yacs
```

## 6. Flash Attention 2.6.3 (ソースビルド)

`TORCH_CUDA_ARCH_LIST` は実 GPU に合わせて指定する (A6000 は `8.6`, H100/H200 などの Hopper は `9.0`)。
A6000 のみで使うなら 30〜45 分。複数アーキ生成するとさらに長い。

```bash
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST="8.6"      # A6000 / RTX 30xx
# export TORCH_CUDA_ARCH_LIST="9.0"    # H100 / H200 (Hopper)

pip install --no-build-isolation --no-binary :all: flash-attn==2.6.3
python -c "import flash_attn; print(flash_attn.__version__)"   # => 2.6.3
```

> 参考所要時間: H200 NVL ホスト + `MAX_JOBS=16` + `TORCH_CUDA_ARCH_LIST=9.0` で約 18 分 (single arch ビルド)。

## 7. vendored CUDA 拡張 (vipe / depth_anything_3)

```bash
cd "$LYRA2_ROOT"

# 7.1 hatchling 不足依存 (DA3 のビルド前に必要)
pip install pathspec pluggy trove-classifiers

# 7.2 vipe (editable)
USE_SYSTEM_EIGEN=1 pip install --no-build-isolation -e 'lyra_2/_src/inference/vipe'

# 7.3 depth_anything_3[gs] (gsplat 含む。numpy が一時的に <2 にダウングレードされる)
pip install --no-build-isolation -e 'lyra_2/_src/inference/depth_anything_3[gs]'

# 7.4 numpy を 2.x に戻す (Lyra-2 / rerun-sdk の要件)
pip install 'numpy>=2.0,<3'
```

> `depth-anything-3` 自体は `numpy<2` を宣言しているが、Lyra-2 ランタイムでの軽い検証では `numpy 2.2.6` で問題は出ていない。問題が出たら `numpy<2` に戻すこと。

## 8. インストール検証

```bash
cd "$LYRA2_ROOT"
python -c "
import torch, flash_attn, transformer_engine.pytorch, vipe_ext, depth_anything_3.api, moge.model.v1
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('all imports OK')
"

python -m lyra_2._src.inference.lyra2_zoomgs_inference --help | head
python -m lyra_2._src.inference.vipe_da3_gs_recon --help | head
```

## 9. チェックポイントのダウンロード (~91 GB)

`huggingface-cli` の `--local-dir` に `$LYRA2_ROOT` を指定する。リポジトリ側のレイアウト (`checkpoints/`) がそのまま展開される。

```bash
cd "$LYRA2_ROOT"
pip install huggingface_hub
huggingface-cli download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir .
```

ダウンロードされる構造:

| パス | サイズ | 用途 |
|---|---|---|
| `checkpoints/model/model/__*.distcp` | 64 GB | 14B Wan メイン (PyTorch 分散チェックポイント形式) |
| `checkpoints/recon/model.pt` | 13 GB | 3D 再構成モデル |
| `checkpoints/text_encoder/encoder.pth` | 11 GB | UMT5 テキストエンコーダ |
| `checkpoints/image_encoder/model.pth` | 2.3 GB | 画像エンコーダ |
| `checkpoints/lora/{detail_enhancer,realism_boost,dmd_distillation}.safetensors` | 1.1 GB | LoRA 群 |
| `checkpoints/vae/vae.pth` | 485 MB | VAE |

## 10. 動作確認サンプル

`Lyra2_setup/sample_check.py` を実行すると、依存 import → 画像読み込み → MoGe 単眼深度推定 → VAE state_dict 検証 までを通す軽量サンプル。`__file__` から `LYRA2_ROOT` を逆算して内部で `cd` するため、cwd / `PYTHONPATH` は activate スクリプト後ならどこからでも動く。

```bash
source /home/kojima/Lyra/lyra/Lyra-2/Lyra2_setup/activate_lyra2.sh
python "$LYRA2_ROOT/Lyra2_setup/sample_check.py"
# → outputs/sample_check/04_moge_depth.png が生成される
```

期待される出力 (抜粋):

```
Lyra-2 root: /home/kojima/Lyra/lyra/Lyra-2
============ Step 1: 基盤ライブラリの import 確認 ============
  torch       : 2.7.1+cu128
  flash_attn  : 2.6.3
  ...
  CUDA device : NVIDIA RTX A6000
  [GPU @ import 完了] 1.70 / 47.4 GiB used
============ Step 3: MoGe v1 で単眼深度推定 ============
  MoGe ロード: 18.28s
  depth shape : (720, 1280), range [0.304, 4.675]
  saved       : outputs/sample_check/04_moge_depth.png
============ 完了: Lyra-2 環境は正常に稼働中 ============
```

### H200 NVL (140 GB) ホストでの検証結果

別ホスト (`hawk`, Linux 6.8, `LYRA2_ROOT=/data1/kojima/WM/lyra/Lyra-2`) に上記手順をそのまま適用し、`TORCH_CUDA_ARCH_LIST="9.0"` で flash-attn / vipe / DA3 を再ビルド。
GPU 1 が他プロセス共用 (~37 GiB 占有済) のため、`CUDA_VISIBLE_DEVICES=1` で実行 (空いている GPU 番号を毎回確認)。

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
# 0, 88529 MiB, 54629 MiB     ← 別ジョブが占有
# 1, 37347 MiB, 105811 MiB    ← こちらに割当

source $LYRA2_ROOT/Lyra2_setup/activate_lyra2.sh
CUDA_VISIBLE_DEVICES=1 python "$LYRA2_ROOT/Lyra2_setup/sample_check.py"
```

実出力 (抜粋):

```
Lyra-2 root: /data1/kojima/WM/lyra/Lyra-2
============ Step 1: 基盤ライブラリの import 確認 ============
  torch       : 2.7.1+cu128
  flash_attn  : 2.6.3
  TE          : transformer_engine.pytorch OK
  vipe_ext    : OK (vipe_ext)
  DA3 api     : OK
  MoGeModel   : OK
  CUDA device : NVIDIA H200 NVL
  [GPU @ import 完了] 36.98 / 139.8 GiB used   # 共用 GPU の合算値
============ Step 2: サンプル画像の読み込み ============
  image     : assets/samples/04.png ((1280, 720))
  tensor    : (1, 3, 720, 1280) dtype=torch.float32 device=cuda:0
============ Step 3: MoGe v1 で単眼深度推定 ============
  MoGe ロード: 17.68s
  [GPU @ MoGe ロード後] 38.18 / 139.8 GiB used  # Lyra-2 自体の増分は ~1.2 GiB
  depth shape : (720, 1280), range [0.304, 4.675]
  mask valid  : 98.1%
  saved       : outputs/sample_check/04_moge_depth.png
============ Step 4: VAE エンコード/デコードのラウンドトリップ ============
  VAE checkpoint: checkpoints/vae/vae.pth (484.1 MiB)
  state_dict keys (head): ['encoder.conv1.weight', 'encoder.conv1.bias', 'encoder.downsamples.0.residual.0.gamma']
  total tensors: 194
============ 完了: Lyra-2 環境は正常に稼働中 ============
```

要点:

- **1 コマンド env 作成 (§3) と activate スクリプトの auto LYRA2_ROOT 検出** はホスト/パスを変えてもそのまま機能 (`/home/kojima/Lyra/lyra/...` → `/data1/kojima/WM/lyra/...`)。
- **flash-attn は sm_90 単独でビルド** (`TORCH_CUDA_ARCH_LIST="9.0"`, `MAX_JOBS=16`) で約 18 分。`flash_attn.__version__ == 2.6.3` を確認。
- **vipe_ext / depth_anything_3 / TE** は H200 環境でも追加対応なしで import 可。DA3 の `numpy<2` 宣言と Lyra-2 の `numpy>=2` 要件は既知の競合 (§7 末) で、`numpy 2.2.6` で軽量検証は問題なし。
- 共用 GPU 環境で Lyra-2 自身の VRAM 増分は MoGe 込みでも ~1.2 GiB。14B Wan 本体 (~46 GB) は 140 GB の H200 に余裕で収まる構成。
- **追加発見:** §5.2 の `pip install --no-deps -r requirements.txt` を素直に流すと、`tyro` / `zarr` / `fvcore` / `pytest` などが必要とする transient deps (`typeguard`, `docstring-parser`, `shtab`, `asciitree`, `iniconfig`, `execnet`, `yacs`) が抜ける。`sample_check.py` や `lyra2_zoomgs_inference --help` (argparse ベース) は通るので気付きにくいが、`import tyro` / `import zarr` 単独は失敗する。§5.6 で一括補完するよう手順を追加。`pip check` 残りの警告 (`decord` の platform tag、megatron-core の学習側依存) は実害なし。

## 11. フル推論 (H100 80 GB 想定)

```bash
cd "$LYRA2_ROOT"
python -m lyra_2._src.inference.lyra2_zoomgs_inference \
  --input_image_path assets/samples \
  --sample_id 4 \
  --experiment lyra2 \
  --checkpoint_dir checkpoints/model \
  --prompt_dir assets/samples \
  --output_path outputs/zoomgs \
  --num_frames_zoom_in 81 --num_frames_zoom_out 241 \
  --zoom_in_strength 0.5 --zoom_out_strength 1.5 \
  --use_dmd
```

`--use_dmd` で 4-step DMD 蒸留 LoRA を使い ~15× 高速化。RTX A6000 では本体ロード時点で OOM するため、80 GB GPU が必要。

3D 再構成 (Step 2):

```bash
python -m lyra_2._src.inference.vipe_da3_gs_recon \
  --input_video_path outputs/zoomgs/videos/4.mp4
```

## 12. トラブルシューティング

| 症状 | 原因 / 対処 |
|---|---|
| `conda install cuda` が `__win =* *` で失敗 | nvidia ラベルチャネルの依存衝突。本書 §3 のように 1 コマンドで env 作成と同時にソルブする |
| TE ビルドで `nvtx3/nvToolsExt.h: No such file` | torch wheel 同梱の nvtx3 が CPATH にない。`activate_lyra2.sh` を先に source する (`$SITE/nvidia/nvtx/include` が CPATH に入る) |
| DA3 editable install が `No module named 'pathspec'` | hatchling 依存が `--no-build-isolation` で揃わない。`pip install pathspec pluggy trove-classifiers` を先行 |
| flash-attn ビルドが遅い | `TORCH_CUDA_ARCH_LIST` を実 GPU のみに絞る (例: A6000 → `8.6`) |
| `lyra2_zoomgs_inference` が `to_empty(device="cuda")` で OOM | 14B Wan が 48 GB GPU に収まらない構造的制限。`--offload` フラグはコード上未配線なので効かない。80 GB H100 に持っていくか、サブコンポーネントだけを使う |
| `ModuleNotFoundError: No module named 'lyra_2'` | `activate_lyra2.sh` を source していない。または別シェルで `PYTHONPATH` が継承されていない |
| numpy 競合警告 (`depth-anything-3 requires numpy<2`) | Lyra-2 ランタイムは `numpy>=2.0` を要求するため >=2.0 を維持。互換性問題が出たら DA3 を別 env に分離 |
| `import tyro` で `No module named 'typeguard'` | §5.2 の `--no-deps` で tyro の transient deps が抜けている。§5.6 の `pip install typeguard docstring-parser shtab` を実行 |
| `import zarr` で `No module named 'asciitree'` | 同上。`pip install asciitree` (§5.6 で一括投入済み) |
| `import fvcore.nn` 関連で `yacs` 不足 | 同上。`pip install yacs` |
| `pytest` 起動で `iniconfig` / `execnet` 不足 | 同上。テスト実行を行うなら `pip install iniconfig execnet` |
| `pip check` で `decord 0.6.0 is not supported on this platform` | `import decord` 自体は通り `__version__ == 0.6.0`。pip の platform tag 警告で実害なし、無視してよい |
| `pip check` で megatron-core が `flask-restful / nltk / nvidia-modelopt / wandb` 不足 | これらは学習側依存。Lyra-2 の inference では未使用なので無視可 |

## 13. クイックリファレンス

```bash
# 環境を有効化 (毎セッション)
source /home/kojima/Lyra/lyra/Lyra-2/Lyra2_setup/activate_lyra2.sh

# 動作確認
python "$LYRA2_ROOT/Lyra2_setup/sample_check.py"

# ヘルプ
python -m lyra_2._src.inference.lyra2_zoomgs_inference --help
python -m lyra_2._src.inference.vipe_da3_gs_recon --help
```
