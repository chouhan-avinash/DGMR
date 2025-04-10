import os

import sys
import torch
import argparse
import numpy as np
from dgmr import *
from generic_train_test import *
import torch as t
import numpy as np
import os 
from metrics import *



parser=argparse.ArgumentParser()
parser.add_argument('--batch_sz', type=int, default=8, help='batch size used for training')

parser.add_argument('--load_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--input_data_folder', type=str, default='')
parser.add_argument('--is_use_cloudmask', type=bool, default=True)
parser.add_argument('--cloud_threshold', type=float, default=0.2) 
parser.add_argument('--data_list_filepath', type=str, default='')

parser.add_argument('--optimizer', type=str, default='Adam', help = 'Adam, SGD')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of optimizer')
parser.add_argument('--lr_step', type=int, default=5, help='lr decay rate')
parser.add_argument('--lr_start_epoch_decay', type=int, default=0, help='epoch to start lr decay')
parser.add_argument('--max_epochs', type=int, default=31)
parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--log_freq', type=int, default=100)
parser.add_argument('--save_model_dir', type=str, default='/dgmr/', help='directory used to store trained networks')
parser.add_argument('--save_model_dir1', type=str, default='/ddpm/', help='directory used to store trained networks')

parser.add_argument('--is_test', type=bool, default=False)

parser.add_argument('--load_pretrained_model', type=bool, default=True)
parser.add_argument('--pretrained_model', type=str, default='/test/')

parser.add_argument('--gpu_ids', type=str, default='0')

opts = parser.parse_args()
print_options(opts)


seed_torch()


from dataload_new_128 import SEN12MSCR_train


dir_SEN12MSCR   = ''
sen12mscr       = SEN12MSCR_train(dir_SEN12MSCR, split='train', region='all')

train_dataloader = torch.utils.data.DataLoader(dataset=sen12mscr, batch_size=opts.batch_sz, shuffle=True, num_workers=8,pin_memory=True)


model=DGMR(opts)


class Train(Generic_train_test):
	def decode_input(self, data):
		return data

Train(model, opts, train_dataloader).train()

