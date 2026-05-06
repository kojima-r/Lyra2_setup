#!/bin/bash
# Lyra-2 実行用環境変数。
# 使い方:  source /home/kojima/Lyra/lyra/Lyra-2/Lyra2_setup/activate_lyra2.sh
# 本スクリプト自身の場所から Lyra-2 リポジトリルートを推定して PYTHONPATH/cwd を整える。

# このファイルが置かれている Lyra2_setup/ の親ディレクトリ = Lyra-2 リポジトリルート
_LYRA2_SETUP_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export LYRA2_ROOT="$( dirname -- "$_LYRA2_SETUP_DIR" )"

source /home/kojima/miniconda3/etc/profile.d/conda.sh
conda activate lyra2

export CUDA_HOME=$CONDA_PREFIX
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
# nvtx3 ヘッダは torch wheel に同梱 (公式 INSTALL.md 不足分)
export CPATH="$CUDA_HOME/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:$SITE/nvidia/nvtx/include:$CPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Lyra-2 のモジュール (lyra_2.*) を import 可能にする。インラインビルドした
# vipe / depth_anything_3 は editable install 済みなので不要だが、Lyra-2 本体は
# パッケージ化されておらず PYTHONPATH 経由で読み込む必要がある。
export PYTHONPATH="${LYRA2_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ビルド時のみ追加で必要な変数 (普段は不要)。
#   export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
#   export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
#   export MAX_JOBS=8
#   export TORCH_CUDA_ARCH_LIST="8.6"

echo "[lyra2] env activated.  LYRA2_ROOT=$LYRA2_ROOT"
echo "[lyra2] e.g.  python \$LYRA2_ROOT/Lyra2_setup/sample_check.py"
echo "[lyra2]       (cd \$LYRA2_ROOT && python -m lyra_2._src.inference.lyra2_zoomgs_inference --help)"
