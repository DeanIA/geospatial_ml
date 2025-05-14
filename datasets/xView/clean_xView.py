#!/usr/bin/env python3
import sys
from pathlib import Path
from PIL import Image

# adjust these if your paths differ
ROOT = Path("/home/jupyter-dai7591/yolo/datasets/xView")
for split in ("train", "val"):
    img_dir = ROOT / "images" / split
    lbl_dir = ROOT / "labels" / split

    for img_path in img_dir.glob("*.jpg"):
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as e:
            print(f"[SKIP] cannot open {img_path}: {e}", file=sys.stderr)
            continue

        if w < 32 or h < 32:
            (lbl_dir / f"{img_path.stem}.txt").unlink(missing_ok=True)
            img_path.unlink()
            print(f"Removed tiny tile: {split}/{img_path.name}")