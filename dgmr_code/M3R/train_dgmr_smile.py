import os
import argparse

##===================================================##
##********** Configure training settings ************##
##===================================================##
parser=argparse.ArgumentParser()
parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for training')

parser.add_argument('--input_data_folder', type=str, default='/smile-cr/TrainData/TrainData')
parser.add_argument('--train_list_filepath', type=str, default='/smile-cr/train_256.csv')
parser.add_argument('--val_list_filepath', type=str, default='/smile-cr/val_256.csv')
parser.add_argument('--is_load_SAR', type=bool, default=True)
parser.add_argument('--is_upsample_SAR', type=bool, default=True) # only useful when is_load_SAR = True
parser.add_argument('--is_load_landcover', type=bool, default=False)
parser.add_argument('--is_upsample_landcover', type=bool, default=False) # only useful when is_load_landcover = True
parser.add_argument('--lc_level', type=str, default='2')  # only useful when is_load_landcover = True
parser.add_argument('--is_load_cloudmask', type=bool, default=True)
parser.add_argument('--load_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=256)

parser.add_argument('--optimizer', type=str, default='Adam', help = 'Adam')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of optimizer')
parser.add_argument('--lr_step', type=int, default=5, help='lr decay rate')
parser.add_argument('--lr_start_epoch_decay', type=int, default=10, help='epoch to start lr decay')
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--val_freq', type=int, default=2)
parser.add_argument('--log_iter', type=int, default=10)
parser.add_argument('--save_model_dir', type=str, default='/m3-cr/dgmr_smile/', help='directory used to store trained networks')
parser.add_argument('--save_model_dir1', type=str, default='/ckpt/m3-cr/dgmr_smile/ddpm/', help='directory used to store trained networks')

parser.add_argument('--gpu_ids', type=str, default='2')

opts = parser.parse_args()

##===================================================##
##****************** choose gpu *********************##
##===================================================##
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids

import sys
import torch
import numpy as np
from dataloader_smile import get_filelist, TrainDataset, ValDataset
from dgmr_smile import DGMR
from generic_dgmr import Generic_Train
from model_base import print_options, seed_torch

print_options(opts)
##===================================================##
##*************** Create dataloader *****************##
##===================================================##
seed_torch()

train_filelist = get_filelist(opts.train_list_filepath)
val_filelist = get_filelist(opts.val_list_filepath)

train_data = TrainDataset(opts, train_filelist)
val_data = ValDataset(opts, val_filelist)
print("Train set: %d, Val set: %d" % (len(train_data), len(val_data)))

train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=opts.batch_sz,shuffle=True, num_workers=4, drop_last=True)
val_dataloader = torch.utils.data.DataLoader(dataset=val_data, batch_size=opts.batch_sz,shuffle=False, num_workers=4)

##===================================================##
##****************** Create model *******************##
##===================================================##
model=DGMR(opts)

model.load_checkpoint(14)
##===================================================##
##**************** Train the network ****************##
##===================================================##
Generic_Train(model, opts, train_dataloader, val_dataloader).train()
