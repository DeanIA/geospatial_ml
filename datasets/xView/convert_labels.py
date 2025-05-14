#!/usr/bin/env python3
from pathlib import Path


def hbb_to_obb(xc, yc, w, h):
    """
    Convert a normalized horizontal bounding box (center-based) to
    a normalized oriented bounding box (corner-based).
    Returns [x1, y1, x2, y2, x3, y3, x4, y4].
    """
    x1, y1 = xc - w / 2, yc - h / 2
    x2, y2 = xc - w / 2, yc + h / 2
    x3, y3 = xc + w / 2, yc + h / 2
    x4, y4 = xc + w / 2, yc - h / 2
    return [x1, y1, x2, y2, x3, y3, x4, y4]


def convert_hbb_dir_to_obb(input_dir: str, output_dir: str):
    """
    Recursively reads all *.txt in input_dir (YOLO HBB) and writes
    corresponding OBB files under output_dir, preserving subdirectory structure.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    # Iterate recursively
    for txt_file in in_path.rglob('*.txt'):
        # Determine relative path and output destination
        rel = txt_file.relative_to(in_path)
        dest = out_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        lines = txt_file.read_text().splitlines()
        obb_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = parts[0]
            try:
                xc, yc, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            coords = hbb_to_obb(xc, yc, w, h)
            # Clamp and format
            coords = [max(0.0, min(c, 1.0)) for c in coords]
            coord_str = ' '.join(f'{c:.6f}' for c in coords)
            obb_lines.append(f'{cls_id} {coord_str}')

        # Write only if any boxes
        if obb_lines:
            dest.write_text("\n".join(obb_lines) + "\n")


if __name__ == '__main__':
    # Adjust these paths
    convert_hbb_dir_to_obb(
        input_dir='datasets/xView/labels',
        output_dir='datasets/xView/labels_new'
    )