import os
import json
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
import random

class TinyCocoDataset(Dataset):
    def __init__(self, root="dataset", num_images=100, transform=None, is_train=True, test_split_ratio=0.2):
        self.root = root
        self.transform = transform
        self.images_dir = os.path.join(root, "images")
        self.is_train = is_train
        
        # Load annotations
        with open(os.path.join(root, "annotations.json"), 'r') as f:
            self.annotations = json.load(f)
            
        # Group annotations by image
        self.image_annotations = {}
        for ann in self.annotations:
            img_name = ann['image']
            if img_name not in self.image_annotations:
                self.image_annotations[img_name] = []
            self.image_annotations[img_name].append(ann)
            
        # Get unique image names
        self.image_names = list(self.image_annotations.keys())
        if num_images < len(self.image_names):
            self.image_names = self.image_names[:num_images]
            
        # Split into train and test
        random.seed(42)  # For reproducibility
        random.shuffle(self.image_names)
        split_idx = int(len(self.image_names) * (1 - test_split_ratio))
        if is_train:
            self.image_names = self.image_names[:split_idx]
        else:
            self.image_names = self.image_names[split_idx:]
            
        # Category mapping
        self.category_map = {1: 0, 3: 1, 18: 2}  # person:0, car:1, dog:2
        
    def __len__(self):
        return len(self.image_names)
        
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        anns = self.image_annotations[img_name]
        
        # Convert annotations to boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            # Convert [x,y,w,h] to [x1,y1,x2,y2]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.category_map[ann['category_id']])
            
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
            
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels
        }
