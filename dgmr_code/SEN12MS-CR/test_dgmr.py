import os
import torch
import argparse
import itertools
from metrics import *
from tqdm import tqdm
import math
from dgmr import *
import ttach as tta
torch.pi = math.pi 

##########################################################
import torch
from dataload_new import SEN12MSCR
def test(CR_net, opts,file):

	dir_SEN12MSCR   = ''
	sen12mscr       = SEN12MSCR(dir_SEN12MSCR, split='test', region='all')
	dataloader      = torch.utils.data.DataLoader(sen12mscr,batch_size=opts.batch_sz,shuffle=False)


	for pdx, patch in enumerate(dataloader):

		cloudy_data = patch['input']['S2']#inputs['cloudy_data']#.cuda()
		cloudfree_data = patch['target']['S2']#.cuda()
		SAR_data = patch['input']['S1']#.cuda()
		file_name = patch['input']['S2 path']

        #inference code	
	
def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--tpath', type=str, default='./dgmr_18.txt')
    parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for training')

    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--input_data_folder', type=str, default='/') 
    parser.add_argument('--data_list_filepath', type=str, default='/')

    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--is_use_cloudmask', type=bool, default=False) 
    parser.add_argument('--cloud_threshold', type=float, default=0.2) 

    opts = parser.parse_args()


    CR_net = MRNet(15,13,1.0,16,256).cuda()

    checkpoint = torch.load(opts.path)


    CR_net.load_state_dict(checkpoint['network'])

    CR_net.eval()
    for _,param in CR_net.named_parameters():
        param.requires_grad = False
    with open(opts.tpath, "w") as f:
         test(CR_net, opts,f)
    print(opts.path)

if __name__ == "__main__":
    main()
    