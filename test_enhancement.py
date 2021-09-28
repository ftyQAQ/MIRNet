"""
## Learning Enriched Features for Real Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## ECCV 2020
## https://arxiv.org/abs/2003.06792
"""

import numpy as np
import os
import argparse
from tqdm import tqdm
import dxchange
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import scipy.io as sio
from networks.MIRNet_model import MIRNet
from dataloaders.data_rgb import DataLoaderVal_denoising_sidd_tiff
import utils
from skimage import img_as_ubyte

parser = argparse.ArgumentParser(description='Image Enhancement using MIRNet')
# parser.add_argument('--input_dir', default='./datasets/lol/', type=str, help='Directory of validation images')
# parser.add_argument('--result_dir', default='./results/enhancement/lol/', type=str, help='Directory for results')
# parser.add_argument('--weights', default='./pretrained_models/enhancement/model_lol.pth', type=str, help='Path to weights')

parser.add_argument('--input_dir', default='G:/MIRNet/MIRNet-master/data_enhance_nmc/tomo_or_re/fbp_1/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='G:/MIRNet/MIRNet-master/data_enhance_nmc/tomo_or_re/fbp_1_enhance_lol/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/enhancement/model_lol.pth', type=str, help='Path to weights')

parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save Enahnced images in the result directory')

args = parser.parse_args(['--save_images'])


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

test_dataset = DataLoaderVal_denoising_sidd_tiff(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=0, drop_last=False)

model_restoration = MIRNet()

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()

model_restoration=nn.DataParallel(model_restoration)

model_restoration.eval()
result_dir0 = args.result_dir+'result0/'
result_dir1 = args.result_dir + 'result1/'
result_dir2 = args.result_dir + 'result2/'
with torch.no_grad():
    psnr_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_noisy = data_test[0].cuda()
        filenames = data_test[1]
        rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = torch.clamp(rgb_restored,0,1)

        rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        if args.save_images:
            for batch in range(len(rgb_noisy)):
                rgb_restored_batch = rgb_restored[batch]
                dxchange.write_tiff(rgb_restored_batch[:, :, 0], result_dir0 + filenames[batch][:-4] + 'tiff')
                dxchange.write_tiff(rgb_restored_batch[:, :, 1], result_dir1 + filenames[batch][:-4] + 'tiff')
                dxchange.write_tiff(rgb_restored_batch[:, :, 2], result_dir2 + filenames[batch][:-4] + 'tiff')

