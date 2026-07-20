import os
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import zoom

class ACDC_dataloder(Dataset):
    """
    针对 TopoMamba 优化的 ACDC 数据集加载器支持加载原图、Scribble 标注和全标签(用于验证)
    """
    def __init__(self, base_dir, split='train', transform=None):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        self.sample_list = []
        
        # 数据集路径设定
        self.data_path = os.path.join(base_dir, split)
        
        # 简单划分：ACDC Tiny 样本较少，这里直接读取目录下所有文件
        if os.path.exists(self.data_path):
            self.sample_list = sorted([
            f for f in os.listdir(self.data_path) 
            if f.endswith('.h5')
        ])
        
        print(f"TopoMamba Loader: {split} mode, found {len(self.sample_list)} slices.")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case_name = self.sample_list[idx]
        h5f = h5py.File(os.path.join(self.data_path, case_name), 'r')
        
        # 1. 读取数据
        image = h5f['image'][:]      # [H, W]
        label = h5f['label'][:]      # [H, W] 全监督标签（验证用）
        scribble = h5f['scribble'][:] # [H, W] 弱监督涂鸦（训练用）
        
        # 2. 归一化处理
        image = (image - image.min()) / (image.max() - image.min())
        image = image.astype(np.float32)

        sample = {'image': image, 'label': label.astype(np.uint8), 'scribble': scribble.astype(np.uint8)}

        # 3. 针对 TopoMamba 的数据增强
        # 注意：TopoMamba 建议使用全图缩放而非随机裁剪，以保持拓扑连通性
        if self.transform:
            sample = self.transform(sample)

        # 4. 转换为 Tensor 格式
        # 确保 image 是 [1, H, W]
        sample['image'] = torch.from_numpy(sample['image']).unsqueeze(0)
        sample['label'] = torch.from_numpy(sample['label']).long()
        sample['scribble'] = torch.from_numpy(sample['scribble']).long()

        return sample

# --- 辅助增强类示例 ---
class TopoResize(object):
    """统一缩放至固定大小，不改变拓扑形状"""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, scribble = sample['image'], sample['label'], sample['scribble']
        h, w = image.shape
        # 使用 zoom 进行重采样
        image = zoom(image, (self.output_size[0] / h, self.output_size[1] / w), order=3)
        label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)
        scribble = zoom(scribble, (self.output_size[0] / h, self.output_size[1] / w), order=0)
        
        return {'image': image, 'label': label, 'scribble': scribble}