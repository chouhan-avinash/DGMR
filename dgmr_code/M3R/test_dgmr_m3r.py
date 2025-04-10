import os
from tqdm import tqdm
import itertools
import torch
import torch.nn as nn
import argparse
import numpy as np
import csv
from metrics import PSNR, SSIM, SAM, MAE, get_SAM_with_landcover, get_MAE_with_landcover

from dataloader_m3r import get_filelist, ValDataset

def test(CR_net, opts):

    test_filelist = get_filelist(opts.test_list_filepath)
    
    test_data = ValDataset(opts, test_filelist)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=opts.batch_sz, shuffle=False)


    with torch.no_grad():
        for inputs in test_dataloader:

            #inference code

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for training')

    parser.add_argument('--input_data_folder', type=str, default='/mm-cr/M3M-CR/test')
    parser.add_argument('--input_data_folder1', type=str, default='/mm-cr/M3M-CR/test')
    parser.add_argument('--test_list_filepath', type=str, default='/mm-cr/M3M-CR/test.csv')
    parser.add_argument('--is_load_SAR', type=bool, default=True)
    parser.add_argument('--is_upsample_SAR', type=bool, default=True) 
    parser.add_argument('--is_load_landcover', type=bool, default=False)
    parser.add_argument('--is_upsample_landcover', type=bool, default=False) 
    parser.add_argument('--lc_level', type=str, default='2') 
    parser.add_argument('--is_load_cloudmask', type=bool, default=True)
    parser.add_argument('--load_size', type=int, default=300)
    parser.add_argument('--crop_size', type=int, default=160)
    parser.add_argument('--model_train_size', type=int, default=160)

    opts = parser.parse_args()

    from dgmr_m3r import MRNet
    CR_net = MRNet(6,4,1.0,16,256).cuda()
   
    path = ''
    print(path)
    checkpoint = torch.load(path)
    CR_net.load_state_dict(checkpoint['network'])

    CR_net.eval()

    test(CR_net, opts)
    
if __name__ == "__main__":
    main()