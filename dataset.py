import os
import random
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from preprocessing import preprocess_fundus_image

class RFMiDDataset(Dataset):
    def __init__(self, csv_path, img_dir, indices=None, target_size=(256, 256), is_train=False):
        """
        PyTorch Dataset for RFMiD.
        
        Args:
            csv_path (str): Path to the labels CSV file.
            img_dir (str): Directory containing the images.
            indices (list, optional): Subset of indices to use (for train/val splits).
            target_size (tuple): Image size (H, W) for input to model.
            is_train (bool): Whether to apply data augmentation.
        """
        self.img_dir = img_dir
        self.target_size = target_size
        self.is_train = is_train
        
        # Load labels CSV
        self.df = pd.read_csv(csv_path)
        
        # If indices are provided, filter the dataframe
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
            
        # Target labels start from Disease_Risk (index 1) to the end (index 46)
        # Total columns in df: 47 (ID, Disease_Risk, + 45 diseases)
        self.label_cols = self.df.columns[1:].tolist()
        self.num_classes = len(self.label_cols)
        
    def __len__(self):
        return len(self.df)
        
    def _augment(self, img):
        """
        Applies numpy/OpenCV based data augmentations.
        """
        # Random Horizontal Flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            
        # Random Vertical Flip
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
            
        # Random 90-degree Rotation
        if random.random() > 0.5:
            rot = random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
            img = cv2.rotate(img, rot)
            
        return img
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['ID']
        
        # Read and preprocess the image (CLAHE + MSR + resize)
        img_name = f"{int(img_id)}.png"
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            # preprocessing.py resizes image to target_size, applies CLAHE and MSR
            img = preprocess_fundus_image(img_path, target_size=self.target_size)
        except Exception as e:
            # Fallback in case of missing files or error
            # Create a dummy image
            img = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            print(f"Warning: Failed to load image {img_path}, using dummy image. Error: {e}")
            
        # Apply data augmentations if training
        if self.is_train:
            img = self._augment(img)
            
        # Normalization: scale to [0, 1] and standardize (ImageNet mean/std)
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_normalized = img.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img_normalized = (img_normalized - mean) / std
        
        # HWC to CHW representation
        img_tensor = img_normalized.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_tensor)
        
        # Extract labels
        # y contains Disease_Risk and all 45 disease category columns (Total 46 elements)
        labels = row[self.label_cols].values.astype(np.float32)
        labels_tensor = torch.from_numpy(labels)
        
        return img_tensor, labels_tensor

class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, base_dir, split="train", target_size=(256, 256), is_train=False):
        """
        PyTorch Dataset for archive 2 / retino (Diabetic Retinopathy binary classification).
        
        Args:
            base_dir (str): Path to the base directory of the dataset (e.g. archive 2/retino)
            split (str): 'train', 'valid', or 'test'
            target_size (tuple): Image size (H, W) for input to model.
            is_train (bool): Whether to apply data augmentation.
        """
        self.base_dir = base_dir
        self.split = split
        self.target_size = target_size
        self.is_train = is_train
        
        self.dr_dir = os.path.join(base_dir, split, "DR")
        self.nodr_dir = os.path.join(base_dir, split, "No_DR")
        
        self.image_paths = []
        # labels: [Disease_Risk, DR]
        # For DR: [1.0, 1.0]
        # For No_DR: [0.0, 0.0]
        self.labels = []
        
        # Load DR images
        if os.path.exists(self.dr_dir):
            for f in os.listdir(self.dr_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(self.dr_dir, f))
                    self.labels.append([1.0, 1.0])
                    
        # Load No_DR images
        if os.path.exists(self.nodr_dir):
            for f in os.listdir(self.nodr_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(self.nodr_dir, f))
                    self.labels.append([0.0, 0.0])
                    
        self.label_cols = ["Disease_Risk", "DR"]
        self.num_classes = len(self.label_cols)
        
    def __len__(self):
        return len(self.image_paths)
        
    def _augment(self, img):
        # Random Horizontal Flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            
        # Random Vertical Flip
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
            
        # Random 90-degree Rotation
        if random.random() > 0.5:
            rot = random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
            img = cv2.rotate(img, rot)
            
        return img
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # preprocessing.py resizes image to target_size, applies CLAHE and MSR
            parent_dir = os.path.dirname(self.base_dir)
            cache_dir = os.path.join(parent_dir, "preprocessed_cache", self.split)
            img = preprocess_fundus_image(img_path, target_size=self.target_size, cache_dir=cache_dir)
        except Exception as e:
            img = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            print(f"Warning: Failed to load image {img_path}, using dummy image. Error: {e}")
            
        # Apply data augmentations if training
        if self.is_train:
            img = self._augment(img)
            
        # Normalization: scale to [0, 1] and standardize (ImageNet mean/std)
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_normalized = img.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img_normalized = (img_normalized - mean) / std
        
        # HWC to CHW
        img_tensor = img_normalized.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_tensor)
        
        labels = np.array(self.labels[idx], dtype=np.float32)
        labels_tensor = torch.from_numpy(labels)
        
        return img_tensor, labels_tensor

if __name__ == "__main__":
    # Test script for datasets
    archive2_dir = "/Users/apple/Library/Mobile Documents/com~apple~CloudDocs/paper/archive 2/retino"
    if os.path.exists(archive2_dir):
        print("Testing DiabeticRetinopathyDataset loading...")
        dataset = DiabeticRetinopathyDataset(base_dir=archive2_dir, split="valid", is_train=True)
        print(f"Dataset initialized. Length: {len(dataset)}, Classes: {dataset.num_classes}")
        if len(dataset) > 0:
            img, lbl = dataset[0]
            print(f"Sample 0: Image Tensor shape: {img.shape}, Data Type: {img.dtype}")
            print(f"Sample 0: Labels shape: {lbl.shape}, Data Type: {lbl.dtype}")
            print(f"Sample 0: Label vector: {lbl}")
    else:
        csv_p = "/Users/apple/Library/Mobile Documents/com~apple~CloudDocs/paper/Evaluation_Set/RFMiD_Validation_Labels.csv"
        img_d = "/Users/apple/Library/Mobile Documents/com~apple~CloudDocs/paper/Evaluation_Set/Validation"
        
        print("Testing RFMiDDataset loading...")
        dataset = RFMiDDataset(csv_path=csv_p, img_dir=img_d, is_train=True)
        print(f"Dataset initialized. Length: {len(dataset)}, Classes: {dataset.num_classes}")
        
        img, lbl = dataset[0]
        print(f"Sample 0: Image Tensor shape: {img.shape}, Data Type: {img.dtype}")
        print(f"Sample 0: Labels shape: {lbl.shape}, Data Type: {lbl.dtype}")
        print(f"Sample 0: Label vector: {lbl}")

