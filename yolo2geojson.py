#!/usr/bin/env python3
import json
from pathlib import Path
from PIL import Image
from shapely.geometry import Polygon, mapping, shape
import rasterio.transform as riotrans

def convert_one(image_path, txt_path, in_geojson, out_geojson):
    # 1) Load footprint GeoJSON and extract polygon
    gj = json.load(open(in_geojson))
    footprint = shape(gj["features"][0]["geometry"])
    minx, miny, maxx, maxy = footprint.bounds

    # 2) Build pixel→world Affine
    width, height = Image.open(image_path).size
    transform = riotrans.from_bounds(minx, miny, maxx, maxy, width, height)

    # 3) Parse YOLO-OBB coords and convert
    features = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue  # skip malformed lines
            cls_id = int(parts[0])
            vals   = list(map(float, parts[1:]))
            # normalized → pixel
            pix_coords = [(vals[i] * width, vals[i+1] * height)
                          for i in range(0, 8, 2)]
            # pixel → geo
            geo_coords = [transform * pt for pt in pix_coords]
            # close the ring
            if geo_coords[0] != geo_coords[-1]:
                geo_coords.append(geo_coords[0])

            feature = {
                "type": "Feature",
                "properties": {"class_id": cls_id},
                "geometry": mapping(Polygon(geo_coords))
            }
            features.append(feature)

    # 4) Write out GeoJSON
    out_fc = {"type": "FeatureCollection", "features": features}
    with open(out_geojson, "w") as f:
        json.dump(out_fc, f, indent=2)
    print(f"[✓] {out_geojson} ({len(features)} boxes)")

def main(images_dir, labels_dir, geojson_dir, out_dir):
    images_dir  = Path(images_dir)
    labels_dir  = Path(labels_dir)
    geojson_dir = Path(geojson_dir)
    out_dir     = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for txt_path in labels_dir.glob("*.txt"):
        stem = txt_path.stem

        # find matching image
        img_path = None
        for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            print(f"[!] No image for {stem}, skipping.")
            continue

        # find matching original GeoJSON
        in_geo = geojson_dir / f"{stem}.geojson"
        if not in_geo.exists():
            print(f"[!] No geojson for {stem}, skipping.")
            continue

        # output path
        out_geo = out_dir / f"{stem}_bboxes.geojson"
        convert_one(img_path, txt_path, in_geo, out_geo)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Convert YOLO-OBB .txt labels → GeoJSON polygons"
    )
    p.add_argument("--images-dir",  required=True, help="Directory of your images")
    p.add_argument("--labels-dir",  required=True, help="Directory of YOLO .txt files")
    p.add_argument("--geojson-dir", required=True,
                   help="Directory of original footprint .geojson files")
    p.add_argument("--out-dir",     required=True, help="Where to write new GeoJSONs")
    args = p.parse_args()
    main(args.images_dir, args.labels_dir, args.geojson_dir, args.out_dir)