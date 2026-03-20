import pandas as pd
import torch
import os
import cv2


class IntelDataSet(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_image_dir="", transform=None):
        self.annotation_df = pd.read_csv(csv_file)
        self.root_image_dir = root_image_dir
        self.transform = transform
        
    def __len__(self):
        return self.len(self.annotation_df)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.root_image_dir, self.annotation_df.iloc[index, 0]) #take actual path
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        class_index = self.annotation_df.iloc[index, 1]
        return image, class_index