#!/usr/bin/env python3
import os
import torch
from ultralytics import YOLO
import torch

os.environ["CUDA_VISIBLE_DEVICES"]=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments=True 

model = YOLO("yolox/xView_1xGPU8/weights/last.pt")

model.train(
    data="datasets/xView/xview_yolo.yaml",
    project="yolox",
    name="xView_1xGPU",
    resume=True,

    imgsz=640,
    batch=14,
    epochs=100,
    patience=15,
    
    cache="disk",
    device='cuda:0',
    workers=6,

    # evaluation + plotting
    val=True,
    plots=False,

    # disable the AMP all‑close check
    amp=False,

    # augmentations
    multi_scale=False,
    close_mosaic=10,
    dropout=0.1,

    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=45.0, translate=0.05, scale=0.25,
    shear=10.0, perspective=0.001,
    flipud=0.10, fliplr=0.50,
    mosaic=0.1,
)