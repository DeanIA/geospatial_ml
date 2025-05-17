
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" # To run on GPU 1

#device = int(os.environ["LOCAL_RANK"]) # To run on both GPU

#import torch
#import torch.distributed as dist
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

# Init process group for DDP
#if not dist.is_initialized():
#    dist.init_process_group(backend="nccl")
#torch.cuda.set_device(local_rank)
#print(f"Rank {dist.get_rank()} training on device {local_rank}")

from ultralytics import YOLO

model = YOLO("./yolox/xView/weights/last.pt")
model.train(
    data="datasets/DOTAv2/DOTAv2.yml",
    project="yolox",
    name="DOTAv2",

    imgsz=640,
    batch=14,
    epochs=50,
    patience=15,

    cache='disk',
    device='0',
    workers=6,

    multi_scale=True,
    close_mosaic=10,
    dropout=0.1,
    val=True,
    plots=False,
    
    # Color changes
    hsv_h=0.015,    # hue shift
    hsv_s=0.7,      # saturation variation
    hsv_v=0.4,      # brightness changes

    # Geometric transforms
    degrees=25.0,       # random rotation ±45°
    translate=0.10,     # ±10% translation
    scale=0.50,         # scale range [0.5–1.5]
    perspective=0.001,  # slight perspective warp

    # Flips
    flipud=0.10,  # 10% vertical flips
    fliplr=0.50,  # 50% horizontal flips

    # Multi-image
    mosaic=0.1,  # always apply mosaic
)