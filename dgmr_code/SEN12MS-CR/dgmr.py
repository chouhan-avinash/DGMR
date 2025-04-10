import torch
import torch.nn as nn

from model_base import *



from metrics import *

from torch.optim import lr_scheduler
import torchvision
import torch.nn as nn
from collections import OrderedDict
import math
from typing import Union, Tuple, Optional

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torch.nn.init as init

import math
BatchNorm2d = nn.BatchNorm2d
from cdgm import CDGM

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
        bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)
		
def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)






class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
class WAB(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, C,H,W = x.shape
        #assert L == H * W, "input feature has wrong size"

        shortcut = x
        #x = self.norm1(x)
        x = x.contiguous().view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, C,H , W)

        # FFN
        x = shortcut + self.drop_path(x)
        #x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels=256,alpha=0.1):
        super(ResBlock, self).__init__()
        m = OrderedDict()
        m['conv1']=nn.Conv2d(in_channels, out_channels,kernel_size=3,bias=False,stride=1,padding=1)
        m['relu1']=nn.ReLU(True)
        m['conv2']=nn.Conv2d(out_channels,out_channels, kernel_size=3,bias=False,stride=1,padding=1)
        self.net = nn.Sequential(m)
        self.relu= nn.Sequential(nn.ReLU(True))
        self.alpha = alpha
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
        
    def forward(self,x):
        out =self.net(x)
        
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.alpha*out + x
        return out
    
class ResBlock_A(nn.Module):
    def __init__(self,in_channels,out_channels=256,alpha=0.1):
        super(ResBlock_A, self).__init__()
        m = OrderedDict()
        m['conv1']=nn.Conv2d(in_channels, out_channels,kernel_size=3,bias=False,stride=1,padding=1)
        m['relu1']=nn.ReLU(True)
        m['conv2']=nn.Conv2d(out_channels,out_channels, kernel_size=3,bias=False,stride=1,padding=1)
        self.net = nn.Sequential(m)
        self.relu= nn.Sequential(nn.ReLU(True))
        self.alpha = alpha

        self.tr = WAB(dim=out_channels ,input_resolution=[128,128],num_heads=4,window_size=8,shift_size=8 // 2)
    def forward(self,x):
        out =self.net(x)
        out = self.tr(out) 

        return out
		
def get_normalized_difference(channel1, channel2):
    subchan = channel1 - channel2
    sumchan = channel1 + channel2
    sumchan[sumchan == 0] = 0.001  
    return subchan / sumchan
class MRNet(nn.Module):
    def __init__(self,in_channels,out_channels,alpha=0.1,num_layers = 16 , feature_sizes = 256):
        super(MRNet,self).__init__()
        m= []
        m.append(nn.Conv2d(in_channels,out_channels=feature_sizes,kernel_size=3,bias=True,stride = 1 ,padding=1))
        m.append(nn.ReLU(True))
        for i in range(num_layers):
            m.append(ResBlock(feature_sizes,feature_sizes,alpha))
        
        self.csa = ResBlock_A(feature_sizes,feature_sizes,alpha)
        self.fin_org = nn.Conv2d(feature_sizes,13,kernel_size=3, bias=True,stride=1,padding=1)
        
        self.fin_sr = nn.Conv2d(feature_sizes,2,kernel_size=3, bias=True,stride=1,padding=1)
        self.net = nn.Sequential(*m)
        self.rec_or = nn.Sequential(
        ResBlock(feature_sizes,feature_sizes,alpha),
        ResBlock(feature_sizes,feature_sizes,alpha),
        ResBlock(feature_sizes,feature_sizes,alpha),
        )
        self.rec = nn.Sequential(
        ResBlock(feature_sizes,feature_sizes,alpha),
        ResBlock(feature_sizes,feature_sizes,alpha),
        ResBlock(feature_sizes,feature_sizes,alpha),
        )
        self.feat = nn.Conv2d(feature_sizes,13,kernel_size=3, bias=True,stride=1,padding=1)		

        self.opt_sa = WAB(dim=13 ,input_resolution=[128,128],num_heads=1,window_size=8,shift_size=8 // 2) #
        self.sar_sa = WAB(dim=2 ,input_resolution=[128,128],num_heads=1,window_size=8,shift_size=8 // 2) # 

        print("******^^^^^^^^^^^^^^*******")
    
    def forward(self, x):
        B, C, H, W = x.shape
        x1_opt, x1_sar = x[:, :13, :, :], x[:, 13:15, :, :]			
        op1 = self.net(x)
        op1 = self.csa(op1) + op1
        op11 = self.rec_or(op1)
        op2 = self.fin_org(op11) 
        op2_up = self.opt_sa(op2) +  x1_opt 
		
        op2_up = op2_up #+ op2
        recon = self.rec(op1 ) 
        recon1= self.fin_sr(recon)
        recon2_up= self.sar_sa(recon1)+ x1_sar
		
        recon2_up = recon2_up #+ recon1
        feat = self.feat(op1)		

        return op2_up, recon2_up,feat 
import random

# Function to apply the same random crop to two tensors (with different channels)
def random_crop_same_location(tensor1, crop_height, crop_width):
    _, _, h, w = tensor1.shape

    start_h = random.randint(0, h - crop_height)
    start_w = random.randint(0, w - crop_width)
    

    tensor1_cropped = tensor1[:, :, start_h:start_h + crop_height, start_w:start_w + crop_width]

    
    return tensor1_cropped#, tensor2_cropped		


dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def gaussian_kernel(size=5, device=dev, channels=3, sigma=1, dtype=torch.float):
    # Create Gaussian Kernel. In Numpy
    interval  = (2*sigma +1)/(size)
    ax = np.linspace(-(size - 1)/ 2., (size-1)/2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx)+ np.square(yy)) / np.square(sigma))
    kernel /= np.sum(kernel)
    # Change kernel to PyTorch. reshapes to (channels, 1, size, size)
    kernel_tensor = torch.as_tensor(kernel, dtype=dtype)
    kernel_tensor = kernel_tensor.repeat(channels, 1 , 1, 1)
    kernel_tensor.to(device)
    return kernel_tensor

def gaussian_conv2d(x, g_kernel, dtype=torch.float):

    channels = g_kernel.shape[0]
    padding = g_kernel.shape[-1] // 2 # Kernel size needs to be odd number
    if len(x.shape) != 4:
        raise IndexError('Expected input tensor to be of shape: (batch, depth, height, width) but got: ' + str(x.shape))
    y = F.conv2d(x, weight=g_kernel, stride=1, padding=padding, groups=channels)
    return y

def downsample(x):
    # Downsamples along  image (H,W). Takes every 2 pixels. output (H, W) = input (H/2, W/2)
    return x[:, :, ::2, ::2]

def create_laplacian_pyramid(x, kernel, levels):
    upsample = torch.nn.Upsample(scale_factor=2) # Default mode is nearest: [[1 2],[3 4]] -> [[1 1 2 2],[3 3 4 4]]
    pyramids = []
    current_x = x
    for level in range(0, levels):
        gauss_filtered_x = gaussian_conv2d(current_x, kernel.to(dev))
        down = downsample(gauss_filtered_x)

        laplacian = current_x - upsample(down) 
        pyramids.append(laplacian)
        current_x = down 
    pyramids.append(current_x)
    return pyramids

class LaplacianPyramidLoss(torch.nn.Module):
    def __init__(self, max_levels=3, channels=13, kernel_size=5, sigma=1, device=torch.device('cpu'), dtype=torch.float):
        super(LaplacianPyramidLoss, self).__init__()
        self.max_levels = max_levels
        self.kernel = gaussian_kernel(size=kernel_size, channels=channels, sigma=sigma, dtype=dtype)
    
    def forward(self, x, target):
        x, target = x.float(), target.float()
        input_pyramid = create_laplacian_pyramid(x, self.kernel, self.max_levels)
        target_pyramid = create_laplacian_pyramid(target, self.kernel, self.max_levels)
        return sum(torch.nn.functional.l1_loss(x,y) for x, y in zip(input_pyramid, target_pyramid))
		
from focal_frequency_loss import FocalFrequencyLoss as FFL
ffl = FFL(loss_weight=1.0, alpha=1.0)
class DGMR(ModelBase):
    def __init__(self, opts):
        super(DGMR, self).__init__()
        self.opts = opts
        print("%%%%%%%%%%DGMR%%%%%%%%%%%%%%")
        
        

        self.net_G = MRNet(15,13,1.0,16,256).cuda()
        self.print_networks(self.net_G)


        beta_schedule = {"train": {"schedule": "sigmoid","n_timestep": 2000,"linear_start": 1e-06,"linear_end": 0.01},
                        "test": {
                            "schedule": "sigmoid",
                            "n_timestep": 1000,
                            "linear_start": 0.0001,
                            "linear_end": 0.09}}
        self.diffusion = CDGM(13,15, 13, beta_schedule)
		
        self.diffusion.cuda()     

        self.diffusion.set_new_noise_schedule(phase="train")
        self.diffusion.set_loss()		

        if self.opts.load_pretrained_model:
            checkpoint = torch.load('')
            self.net_G.load_state_dict(checkpoint['network'],strict=False)



        
        # Parallel training
        if len(self.opts.gpu_ids) > 1:
            print("Parallel training!")
            self.net_G = nn.DataParallel(self.net_G)

        # initialize optimizers
        if self.opts.optimizer == 'SGD':
            self.optimizer_G = torch.optim.SGD(self.net_G.parameters(), lr=opts.lr, momentum=0.9)
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer_G, step_size=self.opts.lr_step, gamma=0.1)
            self.optimizer_G1 = torch.optim.SGD(self.diffusion.parameters(), lr=2e-5, momentum=0.9)
            self.lr_scheduler1 = lr_scheduler.StepLR(self.optimizer_G1, step_size=self.opts.lr_step, gamma=0.1)			
        elif self.opts.optimizer == 'Adam':
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=opts.lr)
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer_G, step_size=self.opts.lr_step, gamma=0.5)
            self.optimizer_G1 = torch.optim.Adam(self.diffusion.parameters(), lr=2e-5)
            self.lr_scheduler1 = lr_scheduler.StepLR(self.optimizer_G1, step_size=self.opts.lr_step, gamma=0.5)            
        self.loss_fn=LaplacianPyramidLoss(channels=13)#nn.L1Loss()
        self.loss_fn1=LaplacianPyramidLoss(channels=2)#nn.L1Loss()
        self.loss_fn_mse =nn.MSELoss()
                        
    def set_input(self, _input):
        inputs = _input
        self.cloudy_data = inputs['input']['S2'].cuda()
        self.cloudfree_data = inputs['target']['S2'].cuda()
        self.SAR_data = inputs['input']['S1'].cuda()
        self.cloud_mask = inputs['input']['masks'].cuda()
        
    def forward(self):
        data = torch.cat([self.cloudy_data, self.SAR_data],1)
        pred_CloudFree_data, sar_data,feat = self.net_G(data)
        return pred_CloudFree_data,sar_data,feat

    def optimize_parameters(self,epoch):
                
        pred_Cloudfree_data,sar_data_p,feat = self.forward()
        if (torch.isnan(self.cloudy_data).any() or torch.isinf(self.cloudy_data).any()) and (torch.isnan(self.SAR_data).any() or torch.isinf(self.SAR_data).any()):
            self.optimizer_G.zero_grad()
            self.optimizer_G1.zero_grad()
            print("NaN encountered his batch.")
            del loss_diff, self.loss_G1
            torch.cuda.empty_cache()
            self.loss_G = 0.0
            print("return ", self.loss_G)
            self.loss_G.backward()
            self.optimizer_G.step()
            self.optimizer_G1.step()
            return self.loss_G
        self.pred = pred_Cloudfree_data
        index_exp = self.cloudfree_data
        index_output = pred_Cloudfree_data
        index_actual = self.cloudy_data
        cloud_m = self.cloud_mask
        cloud_m_s2 = torch.cat((torch.unsqueeze(cloud_m,1),torch.unsqueeze(cloud_m,1),torch.unsqueeze(cloud_m,1),
        torch.unsqueeze(cloud_m,1),torch.unsqueeze(cloud_m,1),torch.unsqueeze(cloud_m,1),
        torch.unsqueeze(cloud_m,1),torch.unsqueeze(cloud_m,1),torch.unsqueeze(cloud_m,1),
        torch.unsqueeze(cloud_m,1),torch.unsqueeze(cloud_m,1),torch.unsqueeze(cloud_m,1),
        torch.unsqueeze(cloud_m,1)
        ),dim=1)
		

        sar_or = self.SAR_data
        


        diff1 = torch.abs(index_actual - index_exp)*(1-cloud_m_s2)
        diff2 = torch.abs(index_actual - index_output)*(1-cloud_m_s2)
        cloud_m_stack= torch.cat((torch.unsqueeze(cloud_m,1),torch.unsqueeze(cloud_m,1)),dim=1)

        ssim_loss = 1 - SSIM(pred_Cloudfree_data, index_exp)
        ssim_loss1 = 1 - SSIM_1(sar_data_p, sar_or,cloud_m_stack)

        
        training_images = pred_Cloudfree_data*cloud_m_s2 + index_exp*(1-cloud_m_s2)
        training_images1 = pred_Cloudfree_data*cloud_m_s2 + index_actual*(1-cloud_m_s2)
		
        self.loss_G1 = self.loss_fn(pred_Cloudfree_data, index_exp)  +self.loss_fn1(sar_data_p*cloud_m_stack,sar_or*cloud_m_stack)*0.5 +(ssim_loss+ssim_loss1*0.5)*.6+ nn.L1Loss()(diff1,diff2)*0.01 + ffl(pred_Cloudfree_data, index_exp) + ffl(sar_data_p*cloud_m_stack,sar_or*cloud_m_stack)*0.5	
        index_exp_down, comb1_down,sr1,sr2 = index_exp, pred_Cloudfree_data,sar_or, sar_data_p 
        data_diff = torch.cat([feat,sr1],1)


        batch,_,_,_= index_exp_down.shape

        
        loss_diff = self.diffusion(y_0 = index_exp_down, y_cond = data_diff)
        loss_diff = loss_diff.mean()
		
        if (torch.isnan(loss_diff) or torch.isinf(loss_diff)) and (torch.isnan(self.loss_G1) or torch.isinf(self.loss_G1)):

           self.optimizer_G.zero_grad()
           self.optimizer_G1.zero_grad()
           print("NaN or Inf encountered in loss; skipping this batch.")

           del loss_diff, self.loss_G1
           torch.cuda.empty_cache()
    
           # Return a zero tensor for loss
           self.loss_G = torch.tensor(0.0, requires_grad=False, device="cuda")  # Prevents .backward() calls on this
           print("return", self.loss_G.item())

           return self.loss_G
        elif torch.isnan(loss_diff) or torch.isinf(loss_diff):
            del loss_diff
            torch.cuda.empty_cache()
            self.loss_G = self.loss_G1
            self.optimizer_G.zero_grad()
            self.optimizer_G1.zero_grad()
            self.loss_G.backward()
            self.optimizer_G.step()
            self.optimizer_G1.step()  
            return self.loss_G.item()
        elif torch.isnan(self.loss_G1) or torch.isinf(self.loss_G1):
            del loss_G1
            torch.cuda.empty_cache()
            self.loss_G = loss_diff
            self.optimizer_G.zero_grad()
            self.optimizer_G1.zero_grad()
            self.loss_G.backward()
            self.optimizer_G.step()
            self.optimizer_G1.step()  
            return self.loss_G.item()
        else:
            self.loss_G = loss_diff*.1 + self.loss_G1		
            self.optimizer_G.zero_grad()
            self.optimizer_G1.zero_grad()
            self.loss_G.backward()
            self.optimizer_G.step()
            self.optimizer_G1.step()   	
            return self.loss_G.item()			




    def get_current_scalars(self):
        losses = {}
        losses['PSNR_train']=PSNR(self.pred, self.cloudfree_data)
        return losses

    def save_checkpoint(self, epoch):
        self.save_network(self.net_G, self.optimizer_G, epoch, self.lr_scheduler, self.opts.save_model_dir)
        self.save_network(self.diffusion, self.optimizer_G1, epoch, self.lr_scheduler1, self.opts.save_model_dir1)
def random_crop_same_location1(tensor1, tensor2, tensor3, tensor4,crop_height, crop_width):
    _, _, h, w = tensor1.shape
    _, _, h2, w2 = tensor2.shape
    
    # Ensure both tensors have the same spatial dimensions (height, width)
    assert (h, w) == (h2, w2), "Both tensors must have the same height and width"
    
    # Randomly select the starting points for the crop
    start_h = random.randint(0, h - crop_height)
    start_w = random.randint(0, w - crop_width)
    
    # Crop both tensors at the same location
    tensor1_cropped = tensor1[:, :, start_h:start_h + crop_height, start_w:start_w + crop_width]
    tensor2_cropped = tensor2[:, :, start_h:start_h + crop_height, start_w:start_w + crop_width]

    tensor3_cropped = tensor3[:, :, start_h:start_h + crop_height, start_w:start_w + crop_width]
    tensor4_cropped = tensor4[:, :, start_h:start_h + crop_height, start_w:start_w + crop_width]
    
    return tensor1_cropped, tensor2_cropped, tensor3_cropped,tensor4_cropped		
	
def normalize_to_01_channelwise(image_tensor):

    if image_tensor.dtype != torch.float32:
        image_tensor = image_tensor.float()
    
    # Find min and max values for each channel (B, C, 1, 1)
    min_val = image_tensor.amin(dim=(2, 3), keepdim=True)
    max_val = image_tensor.amax(dim=(2, 3), keepdim=True)

    # Avoid division by zero
    normalized_tensor = (image_tensor - min_val) / (max_val - min_val).clamp(min=1e-8)
    
    return normalized_tensor