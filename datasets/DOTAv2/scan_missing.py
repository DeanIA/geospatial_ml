import os
import glob


label_folder = "datasets/DOTAv2/labels/train"
image_folder = "datasets/DOTAv2/images/train"
image_exts = (".png", ".jpg", ".jpeg", ".tif")

# Count label files
label_files = glob.glob(os.path.join(label_folder, "*.txt"))
print(f"Number of label files in train: {len(label_files)}")

# Count image files
image_files = []
for ext in image_exts:
    image_files.extend(glob.glob(os.path.join(image_folder, f"*{ext}")))
print(f"Number of image files in train: {len(image_files)}")

label_folder = "datasets/DOTAv2/labels/val"
image_folder = "datasets/DOTAv2/images/val"
image_exts = (".png", ".jpg", ".jpeg", ".tif")

# Count label files
label_files = glob.glob(os.path.join(label_folder, "*.txt"))
print(f"Number of label files in val: {len(label_files)}")

# Count image files
image_files = []
for ext in image_exts:
    image_files.extend(glob.glob(os.path.join(image_folder, f"*{ext}")))
print(f"Number of image files in val: {len(image_files)}")

# def scan_missing_images(label_folder, image_dirs, exts=(".png", ".jpg", ".jpeg", ".tif")):
#     missing = []
#     for label_file in glob.glob(os.path.join(label_folder, "*.txt")):
#         base = os.path.splitext(os.path.basename(label_file))[0]
#         found = False
#         for image_dir in image_dirs:
#             for ext in exts:
#                 img_path = os.path.join(image_dir, base + ext)
#                 if os.path.exists(img_path):
#                     found = True
#                     break
#             if found:
#                 break
#         if not found:
#             missing.append(label_file)
#     if missing:
#         print("Missing images for the following label files:")
#         for m in missing:
#             print(m)
#     else:
#         print("No missing images found.")
#     return missing

# scan_missing_images("datasets/DOTAv2/labels/train", ["datasets/DOTAv2/images/train"])