# MIRNet
 super-resolution image-denoising image-enhancement
 ## Installation
 For installing, follow these intructions
 ```
 conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
 ```
 
## Image Denoising

```
python test_denoising_sidd.py
```

## Image Super-resolution

```
python test_super_resolution.py --save_images --scale 3
python test_super_resolution.py --save_images --scale 4
```

## Image Enhancement

```
python test_enhancement.py --save_images --input_dir ./datasets/lol/ --result_dir ./results/enhancement/lol/ --weights ./pretrained_models/enhancement/model_lol.pth
```
```
python test_enhancement.py --save_images --input_dir ./datasets/fivek_sample_images/ --result_dir ./results/enhancement/fivek/ --weights ./pretrained_models/enhancement/model_fivek.pth
```
