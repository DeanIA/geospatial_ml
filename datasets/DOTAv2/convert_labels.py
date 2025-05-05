import sys
from pathlib import Path

# Updated class list in the specified order
DOTA_CLASSES = [
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge",
    "large vehicle", "small vehicle", "helicopter", "roundabout",
    "soccer ball field", "swimming pool", "container crane",
    "airport", "helipad"
]
CLASS_MAP = {name: idx for idx, name in enumerate(DOTA_CLASSES)}

def convert_labels(input_base: Path, output_base: Path):
    splits = ['train', 'val', 'test']
    for split in splits:
        in_dir = input_base / split
        out_dir = output_base / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for txt_path in in_dir.glob('*.txt'):
            out_path = out_dir / txt_path.name
            with txt_path.open('r') as fin, out_path.open('w') as fout:
                for line in fin:
                    parts = line.strip().split()
                    # Expect: 8 coords, 1+ tokens for class name, 1 difficulty flag
                    if len(parts) < 10:
                        print(f"[WARN] malformed line in {txt_path}: {line.strip()}", file=sys.stderr)
                        continue
                    coords = parts[:8]
                    class_name = ' '.join(parts[8:-1]).replace('-', ' ')
                    diff_flag  = parts[-1]
                    if class_name not in CLASS_MAP:
                        print(f"[WARN] unknown class '{class_name}' in {txt_path}", file=sys.stderr)
                        continue
                    class_id = CLASS_MAP[class_name]
                    # write: x1 y1 ... x4 y4 class_id
                    fout.write(' '.join(coords + [str(class_id)]) + '\n')

if __name__ == '__main__':
    base      = Path('datasets/DOTAv2/labels')
    clean_dir = base.parent / 'labels_clean'
    convert_labels(base, clean_dir)
    print("Done! Clean labels are in:", clean_dir)