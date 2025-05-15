#!/usr/bin/python3

import argparse
import shutil
import sys
import os

import cv2
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from datasets import ImageDataset
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
folder_path = "parameters"
output_dir = 'output'

def pad_image(image, patch_size=(256, 256)):
    _, H, W = image.size()
    patch_h, patch_w = patch_size
    pad_h = (patch_h - H % patch_h) % patch_h
    pad_w = (patch_w - W % patch_w) % patch_w
    padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
    return padded_image


def split_image(image, patch_size=(256, 256)):
    _, H, W = image.size()
    patch_h, patch_w = patch_size
    patches = image.unfold(1, patch_h, patch_h).unfold(2, patch_w, patch_w)
    patches = patches.contiguous().view(-1, 1, patch_h, patch_w)
    return patches


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_patches(patches, model):
    processed_patches = []
    for i in range(patches.size(0)):
        input_patch = patches[i, :, :, :].unsqueeze(0)
        output_patch = model(input_patch.to(device))
        processed_patches.append(output_patch.squeeze(0))
    processed_patches = torch.stack(processed_patches, dim=0)
    return processed_patches


def combine_patches(patches, original_size, padded_size, patch_size=(256, 256)):
    _, H, W = padded_size
    patch_h, patch_w = patch_size
    patches = patches.view(H // patch_h, W // patch_w, 1, patch_h, patch_w)
    patches = patches.permute(2, 0, 3, 1, 4).contiguous()
    combined_image = patches.view(1, H, W)

    _, original_h, original_w = original_size
    combined_image = combined_image[:, :original_h, :original_w]
    return combined_image


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, required=True, help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Networks
netG_B2A = Generator(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_B2A.cuda()


netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [transforms.ToTensor(),
               transforms.Normalize(0.5, 0.5)]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'),
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

generator_files = []

for filename in os.listdir(folder_path):
    if 'netG_B2A.pth' in filename:
        generator_files.append(filename)


with torch.no_grad():

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    for generator_file in generator_files:
        file_path = os.path.join(folder_path, generator_file)

        netG_B2A.load_state_dict(torch.load(file_path))
        for i, batch in enumerate(dataloader):
            # Set model input
            real_B = batch['B'].squeeze(0)
            real_B_name = batch['B_name'][0]

            ori_real_B_shape = real_B.size()
            real_B = pad_image(real_B, patch_size=(256, 256))
            real_B_listimg = split_image(real_B, patch_size=(256, 256))

            fake_B_processed = process_patches(real_B_listimg, netG_B2A)
            fake_B = combine_patches(fake_B_processed, ori_real_B_shape, real_B.size(), patch_size=(256, 256))

            fake_B = 0.5 * (fake_B + 1.0)
            #fake_B = (fake_B > 0.5).float()

            # Save image files
            output_subdir = os.path.join(output_dir, generator_file[:-14])
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            save_image(fake_B, os.path.join(output_subdir, real_B_name))

            sys.stdout.write('\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))
            sys.stdout.flush()

            del real_B_listimg, fake_B_processed, fake_B
            torch.cuda.empty_cache()

    sys.stdout.write('\n')
