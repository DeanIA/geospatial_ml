#To train DOTA dataset, we split original DOTA images with high-resolution into images with 1024x1024 resolution in multiscale way. 
#This preprocessing step is crucial for efficient training as the original images can be extremely large.


from ultralytics.data.split_dota import split_test, split_trainval

# split train and val set, with labels.
split_trainval(
    data_root="datasets/DOTAv2/",
    save_dir="datasets/DOTAv2/processed",
    rates=[0.5, 1.0, 1.5],  # multiscale
    gap=500,
)