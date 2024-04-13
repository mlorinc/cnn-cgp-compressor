#source: https://www.kaggle.com/code/thevinhdoan/full-pipeline-for-inference-and-evaluation

import torch
import os
import torch.nn.functional as F

from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset

class ImagenetTestDataset(Dataset):
    def __init__(self, path: str, transform):
        assert path.split("/")[-1] == "test"
        super().__init__()
        
        self.path = path
        self.img_names = sorted(os.listdir(self.path))
        self.transform = transform
    
    def __getitem__(self, idx):
        img_path = self.path + "/" + self.img_names[idx]
        image = read_image(img_path, ImageReadMode.RGB)
        return self.transform(image)
    
    def __len__(self):
        return len(self.img_names)