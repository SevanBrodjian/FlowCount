#!/usr/bin/env python3
"""
make_dots_dataset.py
--------------------
Generate a synthetic drifting-dots dataset for PyTorch.

Output: <out>.pt containing
    • frames : uint8 tensor  (T, 3, H, W)   with T = num_pairs + 1
    • disp   : int32 tensor (T-1,)          horizontal-displacement labels
"""

import argparse

import cv2
import torch
from tqdm import trange


def make_dataset(
    num_pairs: int,
    out_base: str,
    frame_size=(128, 128),
    entry_prob=0.05,
    vel_range=(1.0, 4.0),
    size_range=(2, 5),
    colour_range=((64, 255), (64, 255), (64, 255)),
    vert_irreg=0.3,
    seed=42,
):
    torch.manual_seed(seed)

    W, H = frame_size
    T = num_pairs + 1  # stored frames

    # ─── growable per-dot state ───────────────────────────────────────────────
    pos = torch.empty(0, 2)  # (N, 2) x,y
    vel = torch.empty(0)  # (N,)   +x velocity
    size = torch.empty(0, dtype=torch.int32)  # (N,)
    colour = torch.empty(0, 3, dtype=torch.uint8)  # (N, 3)
    age = torch.empty(0, dtype=torch.int32)  # (N,)

    # ─── preallocate dataset storage ──────────────────────────────────────────
    frames = torch.empty(T, 3, H, W, dtype=torch.uint8)
    disp = torch.empty(T - 1, dtype=torch.int32)

    # ─── helper functions ─────────────────────────────────────────────────────
    def spawn():
        """Possibly add a single new dot at x=0."""
        nonlocal pos, vel, size, colour, age
        if torch.rand(1).item() < entry_prob:
            # position
            new_pos = torch.tensor([[0.0, torch.rand(1).item() * H]])
            # velocity (+x)
            new_vel = torch.rand(1) * (vel_range[1] - vel_range[0]) + vel_range[0]
            # size
            new_size = torch.randint(
                size_range[0], size_range[1] + 1, (1,), dtype=torch.int32
            )
            # colour tensor (3,) uint8   — build from tensors, not ints
            new_colour = torch.cat(
                [
                    torch.randint(lo, hi + 1, (1,), dtype=torch.uint8)
                    for lo, hi in colour_range
                ]
            )
            # age
            new_age = torch.zeros(1, dtype=torch.int32)

            # concatenate to existing state
            pos = torch.cat([pos, new_pos], dim=0)
            vel = torch.cat([vel, new_vel], dim=0)
            size = torch.cat([size, new_size], dim=0)
            colour = torch.cat([colour, new_colour.unsqueeze(0)], dim=0)
            age = torch.cat([age, new_age], dim=0)

    def cull():
        """Remove dots whose x >= W (off the right edge)."""
        nonlocal pos, vel, size, colour, age
        keep = pos[:, 0] < W
        pos, vel, size, colour, age = (
            tensor[keep] for tensor in (pos, vel, size, colour, age)
        )

    # ─── main synthesis loop ──────────────────────────────────────────────────
    for t in trange(T, desc="Synthesising"):
        spawn()

        # 1) Label uses ONLY dots that were alive last frame
        if t > 0:
            disp[t - 1] = torch.sum(vel[age > 0]).round().int()

        # 2) Draw frame to NumPy BGR image
        frame_bgr = torch.zeros(H, W, 3, dtype=torch.uint8).numpy()
        for i in range(pos.shape[0]):
            x, y = int(pos[i, 0].item()), int(pos[i, 1].item())
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(
                    frame_bgr,
                    (x, y),
                    int(size[i].item()),
                    tuple(int(c) for c in colour[i].tolist()),  # BGR already
                    thickness=-1,
                )

        # 3) BGR→RGB with .copy() to ensure positive strides, then CHW
        frame_rgb = frame_bgr[:, :, ::-1].copy()  # now contiguous
        frame_t = torch.from_numpy(frame_rgb).permute(2, 0, 1)  # (3,H,W)
        frames[t].copy_(frame_t)

        # 4) Physics update
        pos[:, 0] += vel
        pos[:, 1] += (torch.rand_like(pos[:, 1]) - 0.5) * 2 * vert_irreg
        age += 1
        cull()

    # ─── save dataset ─────────────────────────────────────────────────────────
    torch.save({"frames": frames, "disp": disp}, f"{out_base}.pt")
    print(f"Saved → {out_base}.pt  (frames: {frames.shape}, disp: {disp.shape})")


# ─── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate drifting-dots dataset for PyTorch."
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        required=True,
        help="Number of (frameₜ, frameₜ₊₁) pairs",
    )
    parser.add_argument(
        "--out", default="drifting_dots_128", help="Base name for the output .pt file"
    )
    args = parser.parse_args()
    make_dataset(args.num_pairs, args.out)
