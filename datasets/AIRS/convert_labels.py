# From 6 columns to 9

import os
import math
import shutil

def convert_xywhr_to_xyxyxyxy(cls, cx, cy, w, h, angle):
    """Convert center-based OBB to 4-point (xyxyxyxy) format."""
    angle = float(angle)
    w = float(w)
    h = float(h)
    cx = float(cx)
    cy = float(cy)

    hw, hh = w / 2, h / 2
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # corners relative to center
    corners = [
        (-hw, -hh),
        ( hw, -hh),
        ( hw,  hh),
        (-hw,  hh)
    ]

    # rotate & shift to absolute coords
    pts = []
    for x, y in corners:
        xr = cos_a * x - sin_a * y + cx
        yr = sin_a * x + cos_a * y + cy
        pts.extend([xr, yr])

    return [cls] + pts

# base labels directory
base_dir = "/home/jupyter-dai7591/yolo/datasets/AIRS/labels"

# for each split, define source & target
splits = ["train", "val", "test"]
for split in splits:
    src_dir = os.path.join(base_dir, split)
    dst_dir = os.path.join(base_dir, f"{split}_xyxy")
    os.makedirs(dst_dir, exist_ok=True)

    for fname in os.listdir(src_dir):
        if not fname.endswith(".txt"):
            continue

        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)

        with open(src_path, "r") as f_in, open(dst_path, "w") as f_out:
            for line in f_in:
                parts = line.strip().split()
                # only convert lines with exactly 6 values
                if len(parts) == 6:
                    cls, cx, cy, w, h, angle = parts
                    conv = convert_xywhr_to_xyxyxyxy(cls, cx, cy, w, h, angle)
                    f_out.write(" ".join(map(str, conv)) + "\n")
                else:
                    # copy unchanged or warn
                    f_out.write(line)
                    if parts and len(parts) != 9:
                        print(f"[!] {split}/{fname}: unexpected {len(parts)} cols, copied as-is.")

print("✅ Conversion complete. New folders: train_xyxy/, val_xyxy/, test_xyxy/")