"""Generate camera trajectory .npz files for lyra2_custom_traj_inference.

Output format (consumed by `lyra2_custom_traj_inference.load_trajectory`):
  - w2c          : (N, 4, 4) world-to-camera matrices (frame 0 = identity, so the
                   first frame aligns with the input image: camera at world
                   origin, looking +Z).
  - intrinsics   : (N, 3, 3) per-frame intrinsics.
  - image_height : int (resolution the intrinsics refer to).
  - image_width  : int.

Supported trajectories:
  * Any name listed in `lyra_2._src.inference.camera_traj_utils.CAMERA_TRAJECTORY_CHOICES`
    (built via the shared `build_camera_trajectory()` helper, with `initial_w2c=I`).
  * `orbit_outward` (NEW): camera positions sit on a circle, and each camera's
    forward axis points radially outward from the orbit center. Useful for
    panoramic / outward-sweep video generation. Frame 0 sits at world origin
    looking +Z (matching the input image), and the orbit center is placed at
    `(0, 0, -outward_radius)` so that radial-outward = +Z at frame 0.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from lyra_2._src.inference.camera_traj_utils import (
    CAMERA_TRAJECTORY_CHOICES,
    build_camera_trajectory,
)
from lyra_2._src.inference.camera_utils import look_at_matrix


def make_orbit_outward(
    n_steps: int,
    radius: float,
    total_angle_rad: float,
    axis: str = "y",
    direction: str = "right",
    device: str = "cpu",
) -> torch.Tensor:
    """Outward-facing circular orbit.

    Positions sweep along a circle of given radius; orientations point radially
    outward from the orbit center. Orbit center is placed at (0, 0, -radius)
    so frame 0 ends up at the world origin with forward = +Z (identity w2c).

    axis='y' produces a horizontal sweep around +Y (camera moves in the X-Z
    plane). axis='x' produces a vertical sweep around +X (Y-Z plane).
    """
    if axis == "y":
        sweep_sign = 1.0 if direction == "right" else -1.0
    elif axis == "x":
        sweep_sign = 1.0 if direction == "up" else -1.0
    else:
        raise ValueError("axis must be 'x' or 'y'")

    orbit_center = torch.tensor([0.0, 0.0, -float(radius)], device=device)
    trajectory = []
    for i in range(n_steps):
        frac = 0.0 if n_steps <= 1 else i / (n_steps - 1)
        theta = sweep_sign * total_angle_rad * frac
        if axis == "y":
            forward = torch.tensor(
                [math.sin(theta), 0.0, math.cos(theta)], device=device
            )
        else:
            forward = torch.tensor(
                [0.0, math.sin(theta), math.cos(theta)], device=device
            )
        camera_pos = orbit_center + radius * forward
        target = camera_pos + forward
        w2c = look_at_matrix(camera_pos, target)
        trajectory.append(w2c)
    return torch.stack(trajectory)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a camera trajectory .npz for lyra2_custom_traj_inference"
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_frames", type=int, default=161)
    parser.add_argument("--image_height", type=int, default=480)
    parser.add_argument("--image_width", type=int, default=832)
    parser.add_argument(
        "--focal",
        type=float,
        default=701.077,
        help="Focal length (px) at the given resolution. Default matches the "
             "intrinsics produced by the Lyra-2 default 480x832 inference path "
             "(~60.7 deg HFOV).",
    )
    parser.add_argument(
        "--center_depth",
        type=float,
        default=2.0,
        help="Look-at distance used by the built-in trajectories. Ignored by "
             "`orbit_outward`.",
    )

    parser.add_argument(
        "--trajectory",
        required=True,
        choices=list(CAMERA_TRAJECTORY_CHOICES) + ["orbit_outward"],
    )
    parser.add_argument(
        "--direction",
        default="right",
        choices=["left", "right", "up", "down"],
    )
    parser.add_argument("--strength", type=float, default=0.5)

    parser.add_argument(
        "--outward_radius",
        type=float,
        default=0.5,
        help="orbit_outward only: circle radius in world units.",
    )
    parser.add_argument(
        "--outward_angle_deg",
        type=float,
        default=360.0,
        help="orbit_outward only: total sweep angle in degrees.",
    )
    parser.add_argument(
        "--outward_axis",
        default="y",
        choices=["x", "y"],
        help="orbit_outward only: 'y' = horizontal sweep, 'x' = vertical sweep.",
    )

    args = parser.parse_args()

    H, W = args.image_height, args.image_width
    K = torch.tensor(
        [
            [args.focal, 0.0, W / 2.0],
            [0.0, args.focal, H / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )

    if args.trajectory == "orbit_outward":
        w2cs = make_orbit_outward(
            n_steps=args.num_frames,
            radius=args.outward_radius,
            total_angle_rad=math.radians(args.outward_angle_deg),
            axis=args.outward_axis,
            direction=args.direction,
            device="cpu",
        )
    else:
        # The shared trajectory builders default to device="cuda" inside helper
        # functions, so push everything onto CUDA when available and pull the
        # result back to CPU for serialization.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        initial_w2c = torch.eye(4, device=device)
        K_dev = K.to(device)
        w2cs, _ = build_camera_trajectory(
            initial_w2c_44=initial_w2c,
            K_33=K_dev,
            center_depth=float(args.center_depth),
            video_len=int(args.num_frames),
            trajectory=args.trajectory,
            direction=args.direction,
            strength=float(args.strength),
        )
        w2cs = w2cs.detach().to("cpu")

    Ks = K.unsqueeze(0).expand(w2cs.shape[0], -1, -1).contiguous()

    np.savez(
        args.output_path,
        w2c=w2cs.numpy().astype(np.float32),
        intrinsics=Ks.numpy().astype(np.float32),
        image_height=np.int32(H),
        image_width=np.int32(W),
    )
    print(
        f"[make_trajectory] saved trajectory='{args.trajectory}' "
        f"frames={w2cs.shape[0]} resolution={H}x{W} -> {args.output_path}"
    )


if __name__ == "__main__":
    main()
