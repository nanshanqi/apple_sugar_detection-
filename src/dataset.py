import os
import re
import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional, Union
from scipy.linalg import lstsq
from scipy.signal import savgol_filter
import random

from src.utils import SpectralPreprocessor



def load_spectrum_table(xls_path: str):
    df = pd.read_excel(xls_path)
    df.columns = [str(c).strip() for c in df.columns]
    id_col = df.columns[0]
    sugar_col = df.columns[1]
    spec_cols = df.columns[2:]

    def parse_id(s):
        s = str(s).strip()
        m = re.match(r'^\s*(\d+)[_\-](\d+)[_\-](\d+)\s*$', s)
        if not m:
            raise ValueError(f"Unrecognized ID format: {s}")
        return tuple(map(int, m.groups()))

    sid, cid, _ = zip(*df[id_col].map(parse_id))
    df["sid"] = sid
    df["cid"] = cid

    for c in spec_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, id_col, sugar_col, list(spec_cols)

class AppleSugarDataset(Dataset):
    def __init__(
        self,
        img_dir=None,
        df=None,
        sugar_col=None,
        spec_cols=None,
        views=1,
        spec_pp=None,
        transform=None,
        train=True,
        default_img_size=(224, 224),
    ):
        self.img_dir = img_dir
        self.df = df
        self.transform = transform
        self.train = train
        self.views = views
        self.default_img_size = default_img_size

        self.sugar = self.df[sugar_col].values
        self.specs = self.df[spec_cols].values
        self.sid = self.df["sid"].values
        self.cid = self.df["cid"].values

        if spec_pp:
            self.specs = spec_pp(self.specs)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(self.specs[idx], np.float32):
            spectrum = torch.FloatTensor([self.specs[idx]])
        else:
            spectrum = torch.FloatTensor(self.specs[idx])
        sugar = torch.FloatTensor([self.sugar[idx]])
        sid = str(self.sid[idx])
        cid = str(self.cid[idx])

        image_tensor = None
        if self.img_dir is not None:
            if self.views == 1:
                pattern = os.path.join(self.img_dir, f"{sid}_{cid}_1.*")
                matching_files = glob.glob(pattern)
                try:
                    if not matching_files:
                        image_tensor = torch.zeros(3, *self.default_img_size, dtype=torch.float32)
                        print(f"Warning: No image found for {sid}_{cid}_1 in {self.img_dir}")
                    else:
                        img_path = matching_files[0]
                        image = Image.open(img_path).convert("RGB")
                        if self.transform:
                            image_tensor = self.transform(image)
                        else:
                            image_tensor = torch.FloatTensor(np.array(image) / 255.0).permute(2, 0, 1)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    image_tensor = torch.zeros(3, *self.default_img_size, dtype=torch.float32)
                    
            else:
                image_tensor = torch.zeros((self.views, 3, *self.default_img_size), dtype=torch.float32)
                for view in range(1, self.views + 1):
                    pattern = os.path.join(self.img_dir, f"{sid}_{cid}_{view}.*")
                    matching_files = glob.glob(pattern)
                    
                    try:
                        if not matching_files:
                            print(f"Warning: No image found for {sid}_{cid}_{view} in {self.img_dir}")
                            pass
                        else:
                            img_path = matching_files[0]
                            image = Image.open(img_path).convert("RGB")
                            if self.transform:
                                img_tensor = self.transform(image)
                            else:
                                img_tensor = torch.FloatTensor(np.array(image) / 255.0).permute(2, 0, 1)
                            image_tensor[view - 1] = img_tensor
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")

        sample = {
            "spectrum": spectrum.unsqueeze(0),
            "sugar": sugar,
            "sid": torch.LongTensor([int(sid)]),
            "cid": torch.LongTensor([int(cid)]),
        }

        if image_tensor is not None:
            sample["image"] = image_tensor

        return sample

# def split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
#     assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为1"
    
#     # 确保可重复性
#     df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
#     n = len(df)
#     train_end = int(n * train_ratio)
#     val_end = train_end + int(n * val_ratio)
    
#     train_df = df.iloc[:train_end]
#     val_df = df.iloc[train_end:val_end]
#     test_df = df.iloc[val_end:]
    
#     return train_df, val_df, test_df
def split_data(df, train_ratio=0.7, val_ratio=0.3, test_ratio=0, random_seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为1"

    n = len(df)
    indexs = np.arange(0, n)
    np.random.seed(random_seed)
    np.random.shuffle(indexs)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[indexs[:train_end], :]
    val_df = df.iloc[indexs[train_end:val_end], :]
    test_df = df.iloc[indexs[val_end:], :]

    return train_df, val_df, test_df
def get_data_loaders(
    img_dir,
    excel_path,
    batch_size=32,
    img_size=224,
    num_workers=8,
    random_seed=42,
    train_ratio=0.7,
    val_ratio=0.3,
    test_ratio=0,
    views=1,
    spec_preprocess_steps=["snv"],
    train_augment=True
):
    # 加载并分割数据
    df, _, sugar_col, spec_cols = load_spectrum_table(excel_path)
    train_df, val_df, test_df = split_data(df, train_ratio, val_ratio, test_ratio, random_seed)
    
    # 光谱预处理
    spec_pp = lambda x: SpectralPreprocessor.pipeline(x, steps=spec_preprocess_steps)
    
    # 图像变换
    if train_augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = AppleSugarDataset(
        img_dir=img_dir,
        df=train_df,
        sugar_col=sugar_col,
        spec_cols=spec_cols,
        views=views,
        spec_pp=spec_pp,
        transform=train_transform,
        train=True,
        default_img_size=(img_size, img_size)
    )
    
    val_dataset = AppleSugarDataset(
        img_dir=img_dir,
        df=val_df,
        sugar_col=sugar_col,
        spec_cols=spec_cols,
        views=views,
        spec_pp=spec_pp,
        transform=test_transform,
        train=False,
        default_img_size=(img_size, img_size))
    
    test_dataset = AppleSugarDataset(
        img_dir=img_dir,
        df=test_df,
        sugar_col=sugar_col,
        spec_cols=spec_cols,
        views=views,
        spec_pp=spec_pp,
        transform=test_transform,
        train=False,
        default_img_size=(img_size, img_size))
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True)
    
    return train_loader, val_loader, test_loader

# 使用示例
if __name__ == "__main__":
    # 参数设置
    img_dir = "path/to/images"
    excel_path = "path/to/data.xlsx"
    batch_size = 32
    img_size = 224
    random_seed = 42
    views = 3  # 使用3个视角
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(
        img_dir=img_dir,
        excel_path=excel_path,
        batch_size=batch_size,
        img_size=img_size,
        random_seed=random_seed,
        views=views,
        spec_preprocess_steps=["snv", "detrend"]  # 光谱预处理步骤
    )
    
    # 测试数据加载
    sample = next(iter(train_loader))
    print("Batch shapes:")
    print(f"Spectrum: {sample['spectrum'].shape}")
    print(f"Sugar: {sample['sugar'].shape}")
    if 'image' in sample:
        print(f"Images: {sample['image'].shape}")  # (batch, views, C, H, W) 或 (batch, C, H, W)