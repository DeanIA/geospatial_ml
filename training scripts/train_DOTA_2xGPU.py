# to run:
# torchrun --nproc_per_node=2 /home/jupyter-dai7591/yolo/training\ scripts/train_DOTA.py

import os
import torch
import torch.distributed as dist
from ultralytics import YOLO

def main():
    # Init process group for DDP
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    print(f"Rank {dist.get_rank()} training on device {local_rank}")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    model = YOLO("yolo11x-obb.pt")
    model.train(
        data="datasets/DOTAv2/DOTAv2.yml",
        project="yolox",
        name="DOTAv2",

        imgsz=640,
        batch=8,
        epochs=100,

        cache='disk',
        device='1',
        workers=6,

        multi_scale=True,
        close_mosaic=10,
        dropout=0.1,
        val=False,
        plots=True,
                # Color jitter
        hsv_h=0.015,    # hue shift
        hsv_s=0.7,      # saturation variation
        hsv_v=0.4,      # brightness changes

        # Geometric transforms
        degrees=45.0,       # random rotation ±45°
        translate=0.10,     # ±10% translation
        scale=0.50,         # scale range [0.5–1.5]
        shear=10.0,         # shear ±10°
        perspective=0.001,  # slight perspective warp

        # Flips
        flipud=0.10,  # 10% vertical flips
        fliplr=0.50,  # 50% horizontal flips

        # Multi-image augmentations
        mosaic=0.1,  # always apply mosaic
    )

if __name__ == "__main__":
    main()