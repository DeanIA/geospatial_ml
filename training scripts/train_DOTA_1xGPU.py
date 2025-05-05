#!/usr/bin/env python3
import os
from ultralytics import YOLO

import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model = YOLO("yolo11x-obb.pt")
model.train(
    data="datasets/DOTAv2/DOTAv2.yml",
    project="yolox",
    name="DOTAv2",

    imgsz=640,
    batch=8,
    epochs=100,

    cache="disk",
    workers=6,
    device='1',           # this is now the original GPU 1

    multi_scale=True,
    close_mosaic=10,
    val=True,
    plots=True,

    # Color jitter
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    # Geometric
    degrees=45.0,
    translate=0.10,
    scale=0.50,
    shear=10.0,
    perspective=0.001,

    # Flips
    flipud=0.10,
    fliplr=0.50,

    # Mosaic
    mosaic=0.1,
)