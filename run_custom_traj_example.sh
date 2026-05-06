#!/bin/bash
# Lyra-2 custom-trajectory example runner.
#
# example_custom_traj/ 配下の image01.jpg + プロンプトを入力として、
# 1) make_trajectory.py で .npz の軌道ファイルを生成し、
# 2) lyra2_custom_traj_inference を回し、
# 必要なら 3) vipe_da3_gs_recon で 3DGS まで再構成する。
#
# 既定では以下 2 例を走らせる:
#   - orbit_horizontal_demo : 既存の "orbit_horizontal" 軌道 (内向き周回)
#   - orbit_outward_demo    : 本リポジトリで追加した "orbit_outward" 軌道
#                              (円周上を移動しながら各カメラが半径方向外側を向く)
#
# 前提:
#   - checkpoints/ 一式がダウンロード済み (Setup.md §9)
#   - 14B Wan モデルが GPU に載るだけの VRAM (推奨: H100/H200, ~80GB+)
#
# 使い方:
#   bash Lyra2_setup/run_custom_traj_example.sh
#       (両方のデモを順に実行)
#   bash Lyra2_setup/run_custom_traj_example.sh orbit_outward_demo
#       (指定ステムだけ実行)
#   RUN_RECON=1 bash Lyra2_setup/run_custom_traj_example.sh
#       (推論後に 3DGS 再構成も実行)
#   CUDA_VISIBLE_DEVICES=1 bash Lyra2_setup/run_custom_traj_example.sh
#       (GPU 指定)

set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
EXAMPLE_DIR="$SCRIPT_DIR/example_custom_traj"
IMAGE_SRC="$SCRIPT_DIR/example/image01.jpg"

if [[ ! -f "$IMAGE_SRC" ]]; then
    echo "[run_custom_traj] ERROR: reference image not found: $IMAGE_SRC" >&2
    exit 1
fi

# conda env と LYRA2_ROOT, PYTHONPATH を整える
# shellcheck source=activate_lyra2.sh
source "$SCRIPT_DIR/activate_lyra2.sh"

# 必須チェックポイントの存在確認 (run_example.sh と同じ)。
_REQUIRED_PATHS=(
    "checkpoints/model"
    "checkpoints/vae/vae.pth"
    "checkpoints/text_encoder/encoder.pth"
    "checkpoints/text_encoder/negative_prompt.pt"
    "checkpoints/image_encoder/model.pth"
    "checkpoints/lora/dmd_distillation.safetensors"
    "checkpoints/recon/model.pt"
)
_missing=()
for p in "${_REQUIRED_PATHS[@]}"; do
    [[ -e "$LYRA2_ROOT/$p" ]] || _missing+=("$p")
done
if [[ ${#_missing[@]} -gt 0 ]]; then
    echo "[run_custom_traj] ERROR: 以下のチェックポイントが見つかりません (Setup.md §9 を参照):" >&2
    for p in "${_missing[@]}"; do echo "  - $LYRA2_ROOT/$p" >&2; done
    exit 1
fi

OUTPUT_ROOT="$LYRA2_ROOT/outputs/example_custom_traj"
WORK_ROOT="$OUTPUT_ROOT/_inputs"
TRAJ_ROOT="$OUTPUT_ROOT/_trajectories"
mkdir -p "$WORK_ROOT" "$TRAJ_ROOT"

# (stem, trajectory_kind, traj_args...) のセット定義。
# traj_args は make_trajectory.py に渡す追加 CLI 引数。
declare -A DEMO_TRAJ_KIND=(
    [orbit_horizontal_demo]="orbit_horizontal"
    [orbit_outward_demo]="orbit_outward"
)
declare -A DEMO_TRAJ_ARGS=(
    # 既存軌道: orbit_horizontal は中心を見続けながら 0.6 rad ぶん右に弧を描く
    [orbit_horizontal_demo]="--strength 0.6 --direction right --center_depth 2.0"
    # 新規軌道: 円周上を 1 周 (360°) 旋回しつつ各位置で外向き
    [orbit_outward_demo]="--outward_radius 0.4 --outward_angle_deg 360 --outward_axis y --direction right"
)

# 実行対象ステムの決定
if [[ $# -ge 1 ]]; then
    STEMS=("$@")
else
    STEMS=(orbit_horizontal_demo orbit_outward_demo)
fi

NUM_FRAMES="${NUM_FRAMES:-161}"
RESOLUTION_H="${RESOLUTION_H:-480}"
RESOLUTION_W="${RESOLUTION_W:-832}"

cd "$LYRA2_ROOT"

for stem in "${STEMS[@]}"; do
    if [[ -z "${DEMO_TRAJ_KIND[$stem]:-}" ]]; then
        echo "[run_custom_traj] WARN: unknown stem '$stem' (no trajectory mapping); skip" >&2
        continue
    fi
    prompt_txt="$EXAMPLE_DIR/${stem}.txt"
    if [[ ! -f "$prompt_txt" ]]; then
        echo "[run_custom_traj] WARN: prompt not found: $prompt_txt; skip" >&2
        continue
    fi

    traj_kind="${DEMO_TRAJ_KIND[$stem]}"
    traj_extra="${DEMO_TRAJ_ARGS[$stem]}"

    # lyra2_custom_traj_inference は <basename>.jpg と同 stem の .txt を要求するので、
    # ステムごとに作業ディレクトリを切って image01.jpg / プロンプトを symlink する。
    work_dir="$WORK_ROOT/$stem"
    mkdir -p "$work_dir"
    ln -sfn "$IMAGE_SRC"  "$work_dir/${stem}.jpg"
    ln -sfn "$prompt_txt" "$work_dir/${stem}.txt"

    traj_dir="$TRAJ_ROOT/$stem"
    mkdir -p "$traj_dir"
    traj_npz="$traj_dir/${stem}.npz"

    out_dir="$OUTPUT_ROOT/$stem"
    mkdir -p "$out_dir"

    echo
    echo "================================================================"
    echo "[run_custom_traj] stem=${stem}  trajectory=${traj_kind}"
    echo "  image     : $work_dir/${stem}.jpg  (-> $IMAGE_SRC)"
    echo "  prompt    : $work_dir/${stem}.txt  (-> $prompt_txt)"
    echo "  trajectory: $traj_npz"
    echo "  output    : $out_dir"
    echo "================================================================"

    # 1) 軌道 .npz を生成
    # shellcheck disable=SC2086
    python "$SCRIPT_DIR/example_custom_traj/make_trajectory.py" \
        --output_path "$traj_npz" \
        --num_frames "$NUM_FRAMES" \
        --image_height "$RESOLUTION_H" \
        --image_width "$RESOLUTION_W" \
        --trajectory "$traj_kind" \
        $traj_extra

    # 2) lyra2_custom_traj_inference を実行 (DMD 4-step 蒸留で高速化)
    python -m lyra_2._src.inference.lyra2_custom_traj_inference \
        --input_image_path "$work_dir" \
        --sample_start_idx 0 --num_samples 1 \
        --prompt_dir "$work_dir" \
        --trajectory_path "$traj_npz" \
        --num_frames "$NUM_FRAMES" \
        --resolution "${RESOLUTION_H},${RESOLUTION_W}" \
        --experiment lyra2 \
        --checkpoint_dir checkpoints/model \
        --output_path "$out_dir" \
        --use_dmd

    # 出力動画の場所 (lyra2_custom_traj_inference は <stem>.mp4 を直接 output_path に置く)
    out_video="$out_dir/${stem}.mp4"
    echo "[run_custom_traj] generated video: $out_video"

    # 3) 必要なら 3DGS 再構成
    if [[ "${RUN_RECON:-0}" == "1" ]]; then
        if [[ ! -f "$out_video" ]]; then
            echo "[run_custom_traj] WARN: video missing; skip recon for $stem" >&2
            continue
        fi
        echo "[run_custom_traj] === vipe_da3_gs_recon for ${stem} ==="
        python -m lyra_2._src.inference.vipe_da3_gs_recon \
            --input_video_path "$out_video"
    fi
done

echo
echo "[run_custom_traj] Done. Outputs under: $OUTPUT_ROOT"
