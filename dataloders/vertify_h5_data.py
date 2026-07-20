import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def verify_acdc_h5(data_dir, num_samples=3):
    """
    验证 H5 文件中图像与涂鸦的重合情况及尺寸
    :param data_dir: 存放 .h5 文件的目录路径
    :param num_samples: 随机查看的样本数量
    """
    # 1. 获取目录下所有 h5 文件
    h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
    
    if not h5_files:
        print(f"错误: 在路径 {data_dir} 下没找到 .h5 文件")
        return

    # 随机挑选样本
    selected_files = random.sample(h5_files, min(num_samples, len(h5_files)))

    for file_name in selected_files:
        file_path = os.path.join(data_dir, file_name)
        
        with h5py.File(file_path, 'r') as f:
            # 2. 读取数据
            # 假设 key 分别为 'image', 'label', 'scribble'
            image = f['image'][:]
            label = f['label'][:]
            scribble = f['scribble'][:]
            
            # 3. 打印基本信息
            print(f"\n文件名: {file_name}")
            print(f"  Image 尺寸: {image.shape}, 数据类型: {image.dtype}")
            print(f"  Label 尺寸: {label.shape}")
            print(f"  Scribble 尺寸: {scribble.shape}")
            print(f"  Scribble 包含的类别 ID: {np.unique(scribble)}")

            # 4. 可视化检查
            plt.figure(figsize=(15, 5))

            # 子图1: 原始图像
            plt.subplot(1, 3, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            plt.axis('off')

            # 子图2: 全标签 (GT)
            plt.subplot(1, 3, 2)
            plt.title("Ground Truth (Label)")
            plt.imshow(label, cmap='jet')
            plt.axis('off')

            # 子图3: 图片与 Scribble 叠加
            plt.subplot(1, 3, 3)
            plt.title("Image + Scribble Overlay")
            plt.imshow(image, cmap='gray')
            
            # 将 scribble 中的 4 (未标注) 设为透明，只显示 0,1,2,3 的线条
            mask_scribble = np.ma.masked_where(scribble == 4, scribble)
            plt.imshow(mask_scribble, cmap='autumn', alpha=0.9, interpolation='nearest')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    # 修改为你电脑上实际的 ACDC 数据路径
    # 注意：Windows 路径建议使用 r"" 原始字符串
    target_path = r"data\ACDC\acdc_tiny_test" 
    
    verify_acdc_h5(target_path)