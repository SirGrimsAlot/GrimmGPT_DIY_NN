import pandas as pd
import torch
import os
from PIL import Image


class IntelDataSet(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return self.len(self.annotations) #get dataset size
    
    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0]) #get image path by taking the image name of curr image to get  
        image = Image.open(image_path).convert("RGB")
        image_label = int(self.annotations.iloc[index, 1]) #get image label by taking second column of the curr row 
        
        if self.transform:
            image = self.transform(image) #apply transform on image if transform is given 

        return image, image_label