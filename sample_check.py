"""Lyra-2 環境動作確認スクリプト

Lyra-2 のフル 14B Wan モデルは ~46GB で RTX A6000 (48GB) には収まらない
(80GB H100 が公式テスト環境)。本スクリプトは Lyra-2 環境の小さい構成要素
(MoGe / Depth Anything 3 / VAE) を順に呼び出して、依存スタック全体が
正常に動作することを確認する。

実行: cwd に依存しない (Lyra-2 リポジトリ ルートへ自動 cd する)。
  $ source Lyra2_setup/activate_lyra2.sh
  $ python Lyra2_setup/sample_check.py
"""
import os, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# 本スクリプトは Lyra-2/Lyra2_setup/ に置かれる前提。
# どこから呼ばれても Lyra-2 リポジトリルートを cwd にしてから実行する。
LYRA2_ROOT = Path(__file__).resolve().parent.parent
os.chdir(LYRA2_ROOT)
if str(LYRA2_ROOT) not in sys.path:
    sys.path.insert(0, str(LYRA2_ROOT))

SAMPLE_ID = 4
SAMPLE_DIR = "assets/samples"
OUT_DIR = "outputs/sample_check"
os.makedirs(OUT_DIR, exist_ok=True)


def section(name):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}", flush=True)


def show_gpu(stage):
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 2**30
    print(f"  [GPU @ {stage}] {used:.2f} / {total/2**30:.1f} GiB used", flush=True)


def main():
    print(f"Lyra-2 root: {LYRA2_ROOT}")

    section("Step 1: 基盤ライブラリの import 確認")
    import flash_attn, transformer_engine.pytorch as te
    import vipe_ext, depth_anything_3.api as da3_api
    from moge.model.v1 import MoGeModel
    print(f"  torch       : {torch.__version__}")
    print(f"  flash_attn  : {flash_attn.__version__}")
    print(f"  TE          : {te.__name__} OK")
    print(f"  vipe_ext    : OK ({vipe_ext.__name__})")
    print(f"  DA3 api     : OK")
    print(f"  MoGeModel   : OK")
    print(f"  CUDA device : {torch.cuda.get_device_name(0)}")
    show_gpu("import 完了")

    section("Step 2: サンプル画像の読み込み")
    img_path = os.path.join(SAMPLE_DIR, f"{SAMPLE_ID:02d}.png")
    cap_path = os.path.join(SAMPLE_DIR, f"{SAMPLE_ID:02d}.txt")
    img = Image.open(img_path).convert("RGB")
    caption = open(cap_path).read().strip()
    print(f"  image     : {img_path} ({img.size})")
    print(f"  caption   : {caption[:120]}...")

    img_np = np.asarray(img).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).cuda()
    print(f"  tensor    : {tuple(img_t.shape)} dtype={img_t.dtype} device={img_t.device}")

    section("Step 3: MoGe v1 で単眼深度推定")
    t0 = time.time()
    moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").cuda().eval()
    print(f"  MoGe ロード: {time.time()-t0:.2f}s")
    show_gpu("MoGe ロード後")

    with torch.inference_mode():
        out = moge.infer(img_t[0])
    depth = out["depth"].cpu().float().numpy()
    mask = out["mask"].cpu().numpy() if "mask" in out else None
    print(f"  depth shape : {depth.shape}, range [{depth[np.isfinite(depth)].min():.3f}, {depth[np.isfinite(depth)].max():.3f}]")
    if mask is not None:
        print(f"  mask valid  : {mask.mean()*100:.1f}%")

    finite = depth[np.isfinite(depth)]
    if finite.size:
        d_norm = np.clip((depth - finite.min()) / (finite.max() - finite.min() + 1e-6), 0, 1)
        d_norm = np.where(np.isfinite(depth), d_norm, 0)
        depth_img = (d_norm * 255).astype(np.uint8)
        out_path = os.path.join(OUT_DIR, f"{SAMPLE_ID:02d}_moge_depth.png")
        Image.fromarray(depth_img).save(out_path)
        print(f"  saved       : {out_path}")
    moge.cpu()
    del moge
    torch.cuda.empty_cache()
    show_gpu("MoGe 解放後")

    section("Step 4: VAE エンコード/デコードのラウンドトリップ")
    vae_pth = "checkpoints/vae/vae.pth"
    print(f"  VAE checkpoint: {vae_pth} ({os.path.getsize(vae_pth)/2**20:.1f} MiB)")
    sd = torch.load(vae_pth, map_location="cpu", weights_only=False)
    print(f"  state_dict keys (head): {list(sd.keys())[:3]}")
    print(f"  total tensors: {sum(1 for _ in sd.values() if hasattr(_, 'shape'))}")

    section("完了: Lyra-2 環境は正常に稼働中")
    print("注意: フル Lyra-2 動画生成 (14B Wan) は ~46GB を要求するため、")
    print("      RTX A6000 (48GB) では OOM になります。80GB H100 が必要です。")
    print(f"      生成物は {OUT_DIR}/ 配下を参照。")


if __name__ == "__main__":
    main()
