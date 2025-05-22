from pathlib import Path
from shapely.geometry import Polygon
from shapely.validation import make_valid

splits = ['train', 'val']
labels_base = Path("datasets/DOTAv2/labels")

for split in splits:
    label_dir = labels_base / split
    for label_file in label_dir.glob("*.txt"):
        print(f"Checking {label_file}")  # Add this line
        fixed_lines = []
        changed = False
        with label_file.open("r") as fin:
            for line in fin:
                parts = line.strip().split()
                if len(parts) != 9:
                    fixed_lines.append(line)
                    continue
                coords = list(map(float, parts[:8]))
                points = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
                polygon = Polygon(points)
                if not polygon.is_valid:
                    polygon = make_valid(polygon)
                    changed = True
                    # Use the first 4 exterior coords for the fixed polygon
                    new_points = list(polygon.exterior.coords)[:4]
                    coords_fixed = [str(coord) for pt in new_points for coord in pt]
                    # If less than 8 coords, pad with zeros
                    coords_fixed = coords_fixed[:8] + ['0'] * (8 - len(coords_fixed))
                    fixed_line = ' '.join(coords_fixed + [parts[8]]) + '\n'
                    fixed_lines.append(fixed_line)
                else:
                    fixed_lines.append(line)
        if changed:
            with label_file.open("w") as fout:
                fout.writelines(fixed_lines)
            print(f"Fixed invalid polygons in {label_file}")


