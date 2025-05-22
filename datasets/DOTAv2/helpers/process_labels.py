from pathlib import Path
from PIL import Image
import re

def patch_labels(
    original_label_dir,
    cropped_image_dir,
    output_label_dir
):
    original_label_dir = Path(original_label_dir)
    cropped_image_dir = Path(cropped_image_dir)
    output_label_dir = Path(output_label_dir)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    # Pattern to extract: name__<ignored>__x0___y0
    pattern = re.compile(r"^(?P<name>[^_]+)__.*?__(?P<x0>\d+)___(?P<y0>\d+)$")

    for crop_img in sorted(cropped_image_dir.glob("**/*.*")):
        if crop_img.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        stem = crop_img.stem
        m = pattern.match(stem)
        if not m:
            print(f"[WARN] Unexpected crop image name: {crop_img.name}")
            continue
        orig_name = m.group('name')
        x0 = int(m.group('x0'))
        y0 = int(m.group('y0'))

        orig_label_file = original_label_dir / f"{orig_name}.txt"
        if not orig_label_file.exists():
            print(f"[WARN] No label for {orig_name}")
            continue

        # Load crop image to get its actual size
        with Image.open(crop_img) as im:
            w_crop, h_crop = im.size

        # Define crop bounds based on actual crop size
        x_min, y_min = x0, y0
        x_max, y_max = x0 + w_crop, y0 + h_crop

        lines = orig_label_file.read_text().splitlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            # Expect: x1 y1 x2 y2 x3 y3 x4 y4 class_id
            if len(parts) != 9:
                continue
            coords = list(map(float, parts[:8]))
            class_id = parts[8]

            # Check if the box intersects this crop window
            xs = coords[0::2]
            ys = coords[1::2]
            if max(xs) < x_min or min(xs) >= x_max or \
               max(ys) < y_min or min(ys) >= y_max:
                continue

            # Adjust coordinates to crop origin and normalize
            norm_coords = []
            for xi, yi in zip(xs, ys):
                adj_x = max(0, min(xi - x_min, w_crop - 1))
                adj_y = max(0, min(yi - y_min, h_crop - 1))
                norm_coords.append(f"{adj_x / w_crop:.6f}")
                norm_coords.append(f"{adj_y / h_crop:.6f}")

            # class first, then 8 normalized coords
            new_lines.append(f"{class_id} " + " ".join(norm_coords))

        # Write out only if we have labels for this crop
        if new_lines:
            out_file = output_label_dir / f"{stem}.txt"
            out_file.write_text("\n".join(new_lines) + "\n")


if __name__ == "__main__":
    # Replace these paths as needed
    patch_labels(
        original_label_dir="datasets/DOTAv2/labels/train",
        cropped_image_dir="datasets/DOTAv2/processed/images/train",
        output_label_dir="datasets/DOTAv2/processed/labels/train"
    )
    patch_labels(
        original_label_dir="datasets/DOTAv2/labels/val",
        cropped_image_dir="datasets/DOTAv2/processed/images/val",
        output_label_dir="datasets/DOTAv2/processed/labels/val"
    )