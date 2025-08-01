#!/usr/bin/env python
"""
Generate a drifting-dots video + displacement labels (PyTorch edition).
Author: you — adapted from original TF-free prototype.
"""

import cv2
import torch
from tqdm import tqdm


def generate_drifting_dots_clip(
    output_file_base_name: str = "drifting_dots",
    frame_size: tuple[int, int] = (128, 128),
    num_frames: int = 1_024,
    fps: int = 24,
    entry_probability: float = 0.05,
    dot_size_range: tuple[int, int] = (2, 5),
    dot_rgb_range: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = (
        (64, 255),
        (64, 255),
        (64, 255),
    ),
    dot_velocity_range: tuple[float, float] = (1.0, 4.0),
    vertical_irregularity: float = 0.30,
    seed: int = 42,
) -> None:
    """
    Writes two files:
        • <base>_X.mp4 : visual clip
        • <base>_Y.txt : displacement labels
    """
    torch.manual_seed(seed)

    W, H = frame_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid = cv2.VideoWriter(f"{output_file_base_name}_X.mp4", fourcc, fps, (W, H))

    # ─── per-dot state tensors ──────────────────────────────────────────────────
    pos = torch.empty(0, 2)       # (N, 2)  → x, y
    vel = torch.empty(0)          # (N,)    → positive x-velocity
    size = torch.empty(0, dtype=torch.int32)
    colour = torch.empty(0, 3, dtype=torch.uint8)
    age = torch.empty(0, dtype=torch.int32)  # frames since birth

    labels = []  # list of (t-1, t, displacement_px)

    # helpers ------------------------------------------------------------------
    def spawn_new_dots():
        nonlocal pos, vel, size, colour, age
        if torch.rand(1).item() < entry_probability:
            n_new = 1  # one at a time keeps visual density reasonable
            # x=0, random y
            new_pos = torch.stack(
                (
                    torch.zeros(n_new),
                    torch.rand(n_new) * H,
                ),
                dim=-1,
            )
            new_vel = torch.rand(n_new) * (
                dot_velocity_range[1] - dot_velocity_range[0]
            ) + dot_velocity_range[0]
            new_size = torch.randint(
                dot_size_range[0], dot_size_range[1] + 1, (n_new,), dtype=torch.int32
            )
            new_colour = torch.stack(
                [
                    torch.randint(lo, hi + 1, (n_new,), dtype=torch.uint8)
                    for lo, hi in dot_rgb_range
                ],
                dim=-1,
            )
            new_age = torch.zeros(n_new, dtype=torch.int32)

            # concatenate onto existing state
            pos = torch.cat([pos, new_pos], dim=0)
            vel = torch.cat([vel, new_vel], dim=0)
            size = torch.cat([size, new_size], dim=0)
            colour = torch.cat([colour, new_colour], dim=0)
            age = torch.cat([age, new_age], dim=0)

    def cull_out_of_frame():
        nonlocal pos, vel, size, colour, age
        keep_mask = pos[:, 0] < W  # x still in bounds
        pos, vel, size, colour, age = (
            tensor[keep_mask] for tensor in (pos, vel, size, colour, age)
        )

    # ─── main frame loop ────────────────────────────────────────────────────────
    for t in tqdm(range(num_frames), desc="Generating video"):

        # 1. Possibly spawn new arrival(s) on the left
        spawn_new_dots()

        # 2. Compute displacement label before moving outliers away
        if t > 0:
            displacement_px = torch.sum(vel[age > 0]).round().int().item()
            labels.append((t - 1, t, displacement_px))

        # 3. Render current frame
        frame = torch.zeros((H, W, 3), dtype=torch.uint8)

        # draw dots
        for i in range(pos.shape[0]):
            x_i, y_i = int(pos[i, 0].item()), int(pos[i, 1].item())
            if 0 <= x_i < W and 0 <= y_i < H:
                cv2.circle(
                    frame.numpy(),  # cv2 needs ndarray
                    (x_i, y_i),
                    int(size[i].item()),
                    tuple(int(c) for c in colour[i].tolist()),
                    thickness=-1,
                )

        vid.write(frame.numpy())

        # 4. Update physics
        pos[:, 0] += vel                         # horizontal drift
        pos[:, 1] += (torch.rand_like(pos[:, 1]) - 0.5) * 2 * vertical_irregularity
        age += 1

        # 5. Prune dots that exited
        cull_out_of_frame()

    vid.release()

    # ─── write labels ----------------------------------------------------------
    with open(f"{output_file_base_name}_Y.txt", "w") as f:
        for row in labels:
            f.write(" ".join(map(str, row)) + "\n")


if __name__ == "__main__":
    generate_drifting_dots_clip()
