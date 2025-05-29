# Delving Deep into Engagement Prediction of Short Videos

This is an official implementation of “Delving Deep into Engagement Prediction of Short Videos” with PyTorch, accepted by ECCV 2024.

## Get Started
This inference code is provided for the baseline of VQualA 2025 EVQA-SnapUGC: Engagement prediction for short videos Challenge @ ICCV 2025

### Installation
```
ffmpeg
```

```
conda env create -f environment.yaml
```

### Pretrained Models
* download the pretrained models in [Google Drive](https://drive.google.com/drive/folders/19_s6Z4R-iTaQHkRWFRn2Aby1FOy2cHes?usp=share_link) or [Baiduyun](https://pan.baidu.com/s/11HxcmxVOjX4-FRRlfKy3zA?pwd=3miu) set in checkpoints/


## Validation:
Please download the validation videos in `dataset/test_videos/`.

### Validation code of 10 samples:
```
python3 test_SnapUGC_baseline.py --videos_dir dataset/test_videos/ --csv_file dataset/val_data_sample.csv
```

### Validation code of all samples:
```
python3 test_SnapUGC_baseline.py --videos_dir dataset/test_videos/ --csv_file dataset/val_data.csv
```



### Citation
If our work is useful for your research, please consider citing:

```bibtex
@InProceedings{li2024Delving,
    author = {Li, Dasong and Li, Wenjie and Lu, Baili and Li, Hongsheng and Ma, Sizhuo and Krishnan, Gurunandan and Wang, Jian},
    title = {Delving Deep into Engagement Prediction of Short Videos},
    booktitle = {ECCV},
    year = {2024}
}
```


## Acknowledgement

In this project, we use parts of codes in:
- [Basicsr](https://github.com/XPixelGroup/BasicSR)
- [mPLUG-2](https://github.com/X-PLUG/mPLUG-2)
- [Efficientnet-V2](https://github.com/da2so/efficientnetv2)
- [Resnet3d](https://github.com/kenshohara/3D-ResNets-PyTorch/)
- [UVQ](https://github.com/google/uvq)
