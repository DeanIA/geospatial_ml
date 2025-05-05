#To train DOTA dataset, we split original DOTA images with high-resolution into images with 1024x1024 resolution in multiscale way. 
#This preprocessing step is crucial for efficient training as the original images can be extremely large.


from ultralytics.data.split_dota import split_test, split_trainval

# split train and val set, with labels.
split_trainval(
    data_root="datasets/DOTAv2/",
    save_dir="datasets/DOTAv2/processed",
    crop_size=1024  # Base tile dimension in pixels  
    gap=200,
    rates=[0.5, 1.0, 1.5],  # multiscale
)