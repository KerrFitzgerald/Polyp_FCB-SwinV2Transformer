import glob
import pandas as pd
import torch
import numpy as np
from PIL import Image
from skimage.io import imread
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import albumentations as A
import cv2
import os


def create_csv_from_folder(folder_path, csv_name):
    IMAGES_PATH = folder_path + 'images/'
    
    all_images = glob.glob(IMAGES_PATH + "*.png")
    
    data = {
        'image_name': [],
    }
    
    for image_path in all_images:
        filename = os.path.basename(image_path)
        
        data['image_name'].append(filename)
        
    df = pd.DataFrame(data)
    df.to_csv(folder_path + csv_name + '.csv', index=False)
    print("CSV file has been created at", folder_path + 'data.csv')

    
def load_data_to_model(img_size, folder_path, csv_path):
    df = pd.read_csv(csv_path)
    images_folder = folder_path + '/images/'
    masks_folder  = folder_path + '/masks/'
    all_ids = df['image_name'].tolist()
    #print(all_ids)

    X = np.zeros((len(all_ids), img_size, img_size, 3), dtype=np.float32)
    Y = np.zeros((len(all_ids), img_size, img_size), dtype=np.uint8)
    IDs = []
    
    for n, id_ in tqdm(enumerate(all_ids)):
        #IDs.append(id_).split("\\")[-1])
        IDs.append(id_)
        image_path = images_folder+id_
        mask_path = image_path.replace("images", "masks")

        image = imread(image_path)
        mask_ = imread(mask_path)

        mask = np.zeros((img_size, img_size), dtype=np.bool_)

        pillow_image = Image.fromarray(image)

        image = np.array(pillow_image)

        X[n] = image / 255

        pillow_mask = Image.fromarray(mask_)

        # First, resize the mask, convert it to grayscale (assuming RGB channels are equal)
        mask_ = np.array(pillow_mask.convert("L"))

        # Then, threshold the mask
        mask = (mask_ >= 127).astype(np.uint8)

        # Finally, add the mask to the training array
        Y[n] = mask
    
    Y = np.expand_dims(Y, axis=-1)
    
    return IDs, X, Y


class Polyp_Dataset(Dataset):
    def __init__(self, IDs, X, Y, geo_transform=None, color_transform=None):
        self.IDs = IDs
        self.X = X
        self.Y = Y
        self.geo_transform = geo_transform
        self.color_transform = color_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        image = self.X[index]
        mask = self.Y[index]

        if self.geo_transform:
            augmented = self.geo_transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        if self.color_transform:
            augmented = self.color_transform(image=image)
            image = augmented['image']
        
        #print(np.unique(image))    
        image = torch.from_numpy(image)
        #print(np.unique(mask))
        mask  = torch.from_numpy(mask)
        image = image.permute(2, 0, 1)
        mask  = mask.permute(2, 0, 1)

        return self.IDs[index], image, mask
