import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# for bvac
# DATA_DIR = "/workspace/Bisenetv2"

# # for windows
# DATA_DIR = "D:\\Hair_Segmentation\\Data"

# for wsl
DATA_DIR = "/mnt/d/Hair_Segmentation/Data"
#(H, W)
INPUT_SIZE = (192, 160)

class HairSegmentationDataset(Dataset):
    def __init__(self, dataset_name, root_dir=DATA_DIR, target_size=INPUT_SIZE):
        self.img_dir = os.path.join(root_dir, dataset_name, "images")
        self.ann_dir = os.path.join(root_dir, dataset_name, "annotations")
        self.new_mask_dir = os.path.join(root_dir, dataset_name, "numpys")
        self.target_size = target_size

        self.json_files = sorted([f for f in os.listdir(self.ann_dir) if f.endswith(".json")])

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def pad_image(self, img, target_size):
        img_padded = cv2.resize(img, (target_size[1],target_size[0]), interpolation=cv2.INTER_AREA)
        return img_padded
    
    def __len__(self):
        return len(self.json_files)
    
    def __getitem__(self, index):
        json_file = self.json_files[index]
        base_name = os.path.basename(json_file).replace(".json", "")

        img_path = os.path.join(self.img_dir, base_name)
        mask_path = os.path.join(self.ann_dir, base_name + "_hair_parse_mask.data")
        json_path = os.path.join(self.ann_dir, base_name + ".json")
        b_name = base_name.replace(".jpg", "")
        new_mask_path = os.path.join(self.new_mask_dir, b_name + ".npy")

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"File missing: {img_path} or {mask_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mask_h = data["huoshanAlgDetai"]["hairParseAlgRet"]["height"]
        mask_w = data["huoshanAlgDetai"]["hairParseAlgRet"]["width"]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.pad_image(image, INPUT_SIZE)

        with open(mask_path, "rb") as f:
            raw_data = f.read()
        mask = np.frombuffer(raw_data, dtype=np.uint8).reshape((mask_h, mask_w))
        mask = self.pad_image(mask, INPUT_SIZE)

        image = self.img_transform(image)
        mask = torch.tensor(mask, dtype=torch.long)
        binary_mask = (mask > 127).to(dtype=mask.dtype)
        
        new_mask = np.load(new_mask_path)
        new_mask = self.pad_image(new_mask, INPUT_SIZE)
        new_mask = torch.tensor(new_mask, dtype=torch.long)

        #img: (C, H, W), mask: (H, W)
        return image, new_mask