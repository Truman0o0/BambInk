import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.mode = mode
        self.files_A = sorted(glob.glob(root + mode + 'A' + '/*.*'))
        self.files_B = sorted(glob.glob(root + mode + 'B' + '/*.*'))

    def __getitem__(self, index):
        img_A_path = self.files_A[index % len(self.files_A)]
        item_A = self.transform(Image.open(img_A_path).convert("L"))

        if self.unaligned:
            img_B_path = self.files_B[random.randint(0, len(self.files_B) - 1)]
        else:
            img_B_path = self.files_B[index % len(self.files_B)]

        item_B = self.transform(Image.open(img_B_path).convert("L"))

        if self.mode == 'test':
            return {'A': item_A, 'B': item_B, 'A_name': os.path.basename(img_A_path), 'B_name': os.path.basename(img_B_path)}
        else:
            return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
