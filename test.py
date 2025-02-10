import os
import torchvision
import torch
import torchvision
import torchvision.transforms as transforms


filepath = "" //provide model path
model = torch.load(filepath)
print('MODEL LOADED')
//run inference on input images using the loaded model.
