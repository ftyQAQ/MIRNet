

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
import cv2
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import dxchange
import scipy.io as sio
from networks.MIRNet_model import MIRNet
from dataloaders.data_rgb import DataLoaderVal_denoising_sidd_tiff,DataLoaderVal_denoising_sidd_png
import utils
from skimage import img_as_ubyte

def denoise_test_main_tiff(input_dir,result_dir):
    parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
    parser.add_argument('--input_dir', default='',type=str, help='Directory of validation images')
    parser.add_argument('--result_dir', default='',type=str, help='Directory for results')
    parser.add_argument('--weights', default='./pretrained_models/denoising/model_denoising.pth',
                        type=str, help='Path to weights')
    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--bs', default=16, type=int, help='Batch size for dataloader')
    parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

    args = parser.parse_args(['--input_dir=%s'%input_dir,'--result_dir=%s'%result_dir,'--save_images'])

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    utils.mkdir(args.result_dir)

    test_dataset = DataLoaderVal_denoising_sidd_tiff(args.input_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=0, drop_last=False)

    model_restoration = MIRNet()

    utils.load_checkpoint(model_restoration, args.weights)
    print("===>Testing using weights: ", args.weights)

    model_restoration.cuda()

    model_restoration = nn.DataParallel(model_restoration)

    model_restoration.eval()

    result_dir0 = args.result_dir+'result0/'
    result_dir1 = args.result_dir + 'result1/'
    result_dir2 = args.result_dir + 'result2/'

    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            rgb_noisy = data_test[0].cuda()
            # ttt =  data_test[0].cpu().detach().numpy()
            filenames = data_test[1]
            rgb_restored = model_restoration(rgb_noisy)
            rgb_restored = torch.clamp(rgb_restored, 0, 1)
            rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            if args.save_images:
                for batch in range(len(rgb_noisy)):
                    rgb_restored_batch = rgb_restored[batch]
                    dxchange.write_tiff(rgb_restored_batch[:,:,0], result_dir0 + filenames[batch][:-4] + 'tiff')
                    dxchange.write_tiff(rgb_restored_batch[:, :, 1], result_dir1 + filenames[batch][:-4] + 'tiff')
                    dxchange.write_tiff(rgb_restored_batch[:, :, 2], result_dir2+ filenames[batch][:-4] + 'tiff')
                    # denoised_img = img_as_ubyte(rgb_restored[batch])
                    # utils.save_img(args.result_dir + filenames[batch][:-4] + '.png', denoised_img)

def denoise_test_main_png(input_dir,result_dir):
    parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
    parser.add_argument('--input_dir', default='',type=str, help='Directory of validation images')
    parser.add_argument('--result_dir', default='',type=str, help='Directory for results')
    parser.add_argument('--weights', default='./pretrained_models/denoising/model_denoising.pth',
                        type=str, help='Path to weights')
    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--bs', default=16, type=int, help='Batch size for dataloader')
    parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

    args = parser.parse_args(['--input_dir=%s'%input_dir,'--result_dir=%s'%result_dir,'--save_images'])

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    utils.mkdir(args.result_dir)

    test_dataset = DataLoaderVal_denoising_sidd_png(args.input_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=0, drop_last=False)

    model_restoration = MIRNet()

    utils.load_checkpoint(model_restoration, args.weights)
    print("===>Testing using weights: ", args.weights)

    model_restoration.cuda()

    model_restoration = nn.DataParallel(model_restoration)

    model_restoration.eval()

    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            rgb_noisy = data_test[0].cuda()
            # ttt =  data_test[0].cpu().detach().numpy()
            filenames = data_test[1]
            rgb_restored = model_restoration(rgb_noisy)
            rgb_restored = torch.clamp(rgb_restored, 0, 1)
            rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            if args.save_images:
                for batch in range(len(rgb_noisy)):
                    dxchange.write_tiff(rgb_restored[batch],args.result_dir + filenames[batch][:-4] + '.tiff')
                    # denoised_img = img_as_ubyte(rgb_restored[batch])
                    # utils.save_img(args.result_dir + filenames[batch][:-4] + '.png', denoised_img)

def rename_out(result_dir,rename_dir):
    files = os.listdir(result_dir)
    for i_na, file in enumerate(files):
        p = cv2.imread(result_dir+file)[:,:,2]
        dxchange.write_tiff(p,rename_dir+'%s.tiff'%(file.split(".")[0]))

def test_re_denoise():
    names = ['test']
    # names = ['FBP_1','FBP_2','gridrec_1','gridrec_2']
    for name in names:
        print(name)
        input_dir = r'G:\MIRNet\MIRNet-master\datasets\data_LiNi\out_rename_re/'+name+'/'
        result_dir = r'G:\MIRNet\MIRNet-master\datasets\data_LiNi\out_rename_re/' + name + '_denoise_out/'
        # rename_dir = r'G:\MIRNet\MIRNet-master\datasets\data_LiNi\out_rename_re/' + name + '_denoise_out_rename/'
        denoise_test_main(input_dir,result_dir)
        # rename_out(result_dir,rename_dir)

def fty_make_3_to_1():
    path_input = r'G:\MIRNet\MIRNet-master\data_LiNi_8349\EM_1_enhance_fivek/'
    path_out = r'G:\MIRNet\MIRNet-master\data_LiNi_8349\EM_1_enhance_fivek31/'
    files = os.listdir(path_input)
    for i_na, file in enumerate(files):
        tomos = dxchange.read_tiff(path_input+file)
        for i_channel in range(3):
            dxchange.write_tiff(tomos[:,:,i_channel], path_out + '%s_%s.tiff' % (file.split(".")[0],i_channel))


if __name__ == '__main__':
    input_dir = r'G:\SloMo\Adobe240fps\CT_um\tutorial\tomo_ringt_255_shift/'
    result_dir = r'G:\SloMo\Adobe240fps\CT_um\tutorial\tomo_ringt_255_shift_denoise/'
    denoise_test_main_tiff(input_dir,result_dir)


    # names = ['FBP_1_png', 'FBP_1_png_denoise_out', 'FBP_2_png', 'FBP_2_png_denoise_out',
    #          'gridrec_1_png','gridrec_1_png_denoise_out','gridrec_2_png','gridrec_2_png_denoise_out']
    # for name in names:
    #     print(name)
    #     result_dir = r'G:\MIRNet\MIRNet-master\datasets\data_LiNi\out_rename_re\%s/'%name
    #     rename_dir = r'G:\MIRNet\MIRNet-master\datasets\data_LiNi\out_rename_re\%s_rename/'%name
    #     rename_out(result_dir, rename_dir)
    # test_re_denoise()