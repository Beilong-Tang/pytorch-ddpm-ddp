from torch.utils.data.dataset import Dataset
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from PIL import Image

def get_cifar_dataset(noise_scp, img_scp):

    if noise_scp is None or img_scp is None:
        return CIFAR10(
            root='./data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    else:
        return ImageNoisePairDataset(img_scp, noise_scp)


def read_scp(scp_path):
    res = {}
    with open(scp_path, "r") as f:
        for l in f.readlines():
            l = l.replace("\n", "")
            uttid, val = l.split(" ")
            res[uttid] = val
    return res

class ImageNoisePairDataset(Dataset):

    def __init__(self, img_scp, noise_scp):
        
        uttid_img = read_scp(img_scp)
        uttid_noise = read_scp(noise_scp)

        transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        to_tensor = transforms.ToTensor()

        self.uttid_img = uttid_img
        self.uttid_noise = uttid_noise  
        self.transform = transform
        self.to_tensor = to_tensor

    def __len__(self):
        return len(self.uttid_img)

    def __getitem__(self, index):
        _key = list(self.uttid_img.keys())[index]
        _img_path = self.uttid_img[_key]
        _noise_path = self.uttid_noise[_key]
        # [3, H, W]
        img = Image.open(_img_path)
        img = self.to_tensor(img) # [C,H,W]
        img = self.transform(img)
        noise = torch.from_numpy(np.load(_noise_path)) # [C, H, W]

        return img, noise
