#!/bin/bash
# Lyra-2 example demo runner.
#
# example/ 配下の image01.jpg + demo*.txt を入力として
# lyra2_zoomgs_inference を回し、必要なら vipe_da3_gs_recon で 3DGS まで再構成する。
#
# 前提: 本スクリプトは Lyra-2/Lyra2_setup/ に置かれていること。
#       checkpoints/ 一式がダウンロード済み (Setup.md §9)。
#       14B Wan モデルを GPU に載せられる ~80GB VRAM があること (推奨: H100/H200)。
#
# 使い方:
#   bash Lyra2_setup/run_example.sh                    # 全プロンプトを順に実行
#   bash Lyra2_setup/run_example.sh demo01_01          # 指定したステムだけ実行
#   RUN_RECON=1 bash Lyra2_setup/run_example.sh        # 推論後に 3D 再構成も実行
#   CUDA_VISIBLE_DEVICES=1 bash Lyra2_setup/run_example.sh  # GPU 指定

set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
EXAMPLE_DIR="$SCRIPT_DIR/example"
IMAGE_SRC="$EXAMPLE_DIR/image01.jpg"

if [[ ! -f "$IMAGE_SRC" ]]; then
    echo "[run_example] ERROR: reference image not found: $IMAGE_SRC" >&2
    exit 1
fi

# conda env と LYRA2_ROOT, PYTHONPATH を整える
# shellcheck source=activate_lyra2.sh
source "$SCRIPT_DIR/activate_lyra2.sh"

# 必須チェックポイントの存在確認 (Setup.md §9 の huggingface-cli download が完了しているか)。
# 不足したまま Python に入ると、14B Wan のロード経路で deep stack のエラーになり原因把握が困難なため、
# シェル側で早めに弾く。
_REQUIRED_PATHS=(
    "checkpoints/model"
    "checkpoints/vae/vae.pth"
    "checkpoints/text_encoder/encoder.pth"
    "checkpoints/text_encoder/negative_prompt.pt"
    "checkpoints/image_encoder/model.pth"
    "checkpoints/lora/dmd_distillation.safetensors"
    "checkpoints/lora/realism_boost.safetensors"
    "checkpoints/lora/detail_enhancer.safetensors"
    "checkpoints/recon/model.pt"
)
_missing=()
for p in "${_REQUIRED_PATHS[@]}"; do
    [[ -e "$LYRA2_ROOT/$p" ]] || _missing+=("$p")
done
if [[ ${#_missing[@]} -gt 0 ]]; then
    echo "[run_example] ERROR: 以下のチェックポイントが見つかりません (Setup.md §9 を参照):" >&2
    for p in "${_missing[@]}"; do echo "  - $LYRA2_ROOT/$p" >&2; done
    echo "" >&2
    echo "ダウンロード例:" >&2
    echo "  cd \"\$LYRA2_ROOT\" && huggingface-cli download nvidia/Lyra-2.0 --include 'checkpoints/*' --local-dir ." >&2
    exit 1
fi

OUTPUT_ROOT="$LYRA2_ROOT/outputs/example_demo"
WORK_ROOT="$OUTPUT_ROOT/_inputs"
mkdir -p "$WORK_ROOT"

# 実行対象プロンプトの決定
if [[ $# -ge 1 ]]; then
    PROMPTS=("$@")
else
    PROMPTS=()
    for f in "$EXAMPLE_DIR"/demo*.txt; do
        [[ -f "$f" ]] || continue
        PROMPTS+=("$( basename "$f" .txt )")
    done
fi

if [[ ${#PROMPTS[@]} -eq 0 ]]; then
    echo "[run_example] ERROR: no demo*.txt prompts found in $EXAMPLE_DIR" >&2
    exit 1
fi

cd "$LYRA2_ROOT"

for stem in "${PROMPTS[@]}"; do
    prompt_txt="$EXAMPLE_DIR/${stem}.txt"
    if [[ ! -f "$prompt_txt" ]]; then
        echo "[run_example] WARN: prompt file not found, skipping: $prompt_txt" >&2
        continue
    fi

    # lyra2_zoomgs_inference は <basename>.jpg と同じ stem の .txt を要求するため、
    # プロンプトごとに作業ディレクトリを切って image01.jpg を ${stem}.jpg として symlink する。
    work_dir="$WORK_ROOT/$stem"
    mkdir -p "$work_dir"
    ln -sfn "$IMAGE_SRC"   "$work_dir/${stem}.jpg"
    ln -sfn "$prompt_txt"  "$work_dir/${stem}.txt"

    out_dir="$OUTPUT_ROOT/$stem"
    mkdir -p "$out_dir"

    echo
    echo "================================================================"
    echo "[run_example] stem=${stem}"
    echo "  image : $work_dir/${stem}.jpg  (-> $IMAGE_SRC)"
    echo "  prompt: $work_dir/${stem}.txt  (-> $prompt_txt)"
    echo "  output: $out_dir"
    echo "================================================================"

    python -m lyra_2._src.inference.lyra2_zoomgs_inference \
        --input_image_path "$work_dir" \
        --sample_id 0 \
        --experiment lyra2 \
        --checkpoint_dir checkpoints/model \
        --prompt_dir "$work_dir" \
        --output_path "$out_dir" \
        --num_frames_zoom_in 81 --num_frames_zoom_out 241 \
        --zoom_in_strength 0.5 --zoom_out_strength 1.5 \
        --use_dmd

    combined_video="$out_dir/videos/${stem}.mp4"
    echo "[run_example] combined video: $combined_video"

    if [[ "${RUN_RECON:-0}" == "1" ]]; then
        if [[ ! -f "$combined_video" ]]; then
            echo "[run_example] WARN: combined video missing, skipping recon for $stem" >&2
            continue
        fi
        echo "[run_example] === vipe_da3_gs_recon for ${stem} ==="
        python -m lyra_2._src.inference.vipe_da3_gs_recon \
            --input_video_path "$combined_video"
    fi
done

echo
echo "[run_example] Done. Outputs under: $OUTPUT_ROOT"
