from ultralytics import YOLO
model = YOLO("./yolox/xView2/weights/last.pt")
model.train(
    data="datasets/xView/xview_yolo.yaml",
    project="yolox",
    name="xView",

    imgsz=640,
    batch=16,
    epochs=350,
    patience=10,

    cache='disk',
    #device=str(local_rank),
    device='0,1',
    workers=8,
    save=True,                           
    save_period=5,

    multi_scale=True,
    close_mosaic=10,
    dropout=0.1,
    val=True,
    plots=False,
    #amp=True,

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

