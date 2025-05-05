# train.py
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

    model = YOLO("yolon/DOTA1.5 to xView22/weights/best.pt")
    model.train(
        data="datasets/AIRS/airs.yaml",
        project="yolon",
        name="DOTA1.5 to xView to AIRS",
        imgsz=640,
        batch=16,
        epochs=100,
        cache='disk',
        device='0,1',
        amp=True,
        workers=6,
        multi_scale=True,
        close_mosaic=10,
        optimizer="auto",
        lr0=0.01,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0,
        dfl=1.5,
        dropout=0,
        val=False,
        plots=True
    )

if __name__ == "__main__":
    main()