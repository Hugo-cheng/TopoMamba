import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def visualize_single_pair(img_path, label_path, save_path):
    """
    专门处理一对：2K 原图 + 640x480 掩码
    """
    if not os.path.exists(img_path) or not os.path.exists(label_path):
        print(f"错误: 找不到输入文件 \nImage: {img_path}\nLabel: {label_path}")
        return

    # 1. 读取 2K 原图
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_2k, w_2k = img.shape[:2]
    print(f"原图读取成功: {img_path} (尺寸: {w_2k}x{h_2k})")

    # 2. 读取并处理掩码 (支持文件夹或单文件)
    full_mask_small = np.zeros((480, 640), dtype=np.uint32)
    
    if os.path.isdir(label_path):
        mask_files = [f for f in os.listdir(label_path) if f.endswith(('.png', '.tif'))]
        for idx, mf in enumerate(mask_files, 1):
            m = cv2.imread(os.path.join(label_path, mf), cv2.IMREAD_UNCHANGED)
            if m is not None:
                if len(m.shape) == 3: m = m[:,:,0]
                full_mask_small[m > 0] = idx
    else:
        m = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if len(m.shape) == 3: m = m[:,:,0]
        full_mask_small = m.astype(np.uint32)
    
    print(f"掩码读取完成，包含细胞数: {np.max(full_mask_small)}")

    # 3. 核心步骤：将 640x480 放大到 2K
    # 转换为 uint16 (支持 0-65535 个 ID)，OpenCV 对此类型的 INTER_NEAREST 支持良好
    full_mask_small_ready = full_mask_small.astype(np.uint16)

    full_mask_2k = cv2.resize(
        full_mask_small_ready, 
        (w_2k, h_2k), 
        interpolation=cv2.INTER_NEAREST
    )

    # 4. 配色逻辑 (固定种子确保多次运行颜色一致)
    max_id = np.max(full_mask_2k)
    if max_id > 0:
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(int(max_id) + 1, 3))
        colors[0] = [0, 0, 0] # 背景黑
        color_mask_2k = colors[full_mask_2k].astype(np.uint8)
        
        # 5. 叠加显示
        overlay = cv2.addWeighted(img, 0.7, color_mask_2k, 0.3, 0)

        # 6. 绘制白色轮廓 (2K 下线宽设为 2-3 效果最好)
        mask_binary = (full_mask_2k > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    else:
        overlay = img
        print("警告: 掩码为空")

    # 7. 极速保存
    # 不使用 plt 封装，直接用 opencv 保存 2K 图片速度最快
    save_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, save_bgr)
    print(f"可视化结果已极速保存至: {save_path}")

if __name__ == "__main__":
    # --- 每次只需修改这两个路径 ---
    IMAGE_FILE = r"H:/Research/Datasets/Mamba/data/nnUNet_raw/Dataset703_NeurIPSCell/imagesTr/cell_00001_0000.png"
    LABEL_FILE = r"H:/Research/Datasets/Mamba/data/nnUNet_raw/Dataset703_NeurIPSCell/labelsTr/cell_00001.png"

    OUTPUT_NAME = "cell_vis_final.png"

    visualize_single_pair(IMAGE_FILE, LABEL_FILE, OUTPUT_NAME)