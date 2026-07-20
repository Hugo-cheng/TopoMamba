import cv2
import numpy as np
from skimage.morphology import skeletonize
import os

def process_scribble(image_path, mask_path, output_path, alpha=0.6):
    """
    输入原图和全标注掩码，输出叠加了Scribble的结果图
    :param image_path: 原图路径 (.png)
    :param mask_path: 掩码路径 (.png)
    :param output_path: 输出结果路径 (.png)
    :param alpha: 原图融合权重 (0-1)
    """
    # 1. 读取图像
    # 如果原图是带Alpha通道的PNG，也统一转为BGR处理
    img = cv2.imread(image_path)
    # 掩码通常是单通道灰度图或索引图
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"错误: 无法加载 {image_path} 或 {mask_path}")
        return

    # 2. 准备画布
    h, w = mask.shape
    scribble_overlay = img.copy()
    
    # 3. 遍历掩码中的每个类别 (跳过背景 0)
    unique_ids = np.unique(mask)
    
    for label_id in unique_ids:
        if label_id == 0: continue  # 假设0是背景
        
        # 提取当前类别的二值掩码
        binary_mask = np.zeros_like(mask)
        binary_mask[mask == label_id] = 1
        
        # 4. 骨架化提取 (生成物体中心线)
        # skeletonize 需要布尔型输入
        skeleton = skeletonize(binary_mask.astype(bool))
        
        # 将骨架转回OpenCV格式并适当加粗，模拟手绘笔触
        skeleton_uint8 = (skeleton.astype(np.uint8)) * 255
        kernel = np.ones((3, 3), np.uint8)
        dilated_scribble = cv2.dilate(skeleton_uint8, kernel, iterations=1)
        
        # 5. 分配随机颜色 (或者根据 ID 分配固定颜色)
        # 这里使用简单的哈希生成颜色
        np.random.seed(label_id)
        color = np.random.randint(0, 255, 3).tolist()
        
        # 在结果图上着色
        scribble_overlay[dilated_scribble > 0] = color

    # 6. 图像融合 (Alpha Blending)
    # 将原图和带有Scribble的图像按比例混合
    final_result = cv2.addWeighted(img, alpha, scribble_overlay, 1 - alpha, 0)

    # 7. 保存结果
    cv2.imwrite(output_path, final_result)
    print(f"处理完成！已保存至: {output_path}")

# --- 执行示例 ---
if __name__ == "__main__":
    # 请确保这两个文件在当前目录下，或者输入绝对路径
    input_img = "H:/Research/Datasets/Mamba/data/nnUNet_raw/Dataset703_NeurIPSCell/imagesTr/cell_00001_0000.png"
    input_mask = "H:/Research/Datasets/Mamba/data/nnUNet_raw/Dataset703_NeurIPSCell/labelsTr/cell_00001.png"

    output_res = "output_scribble_result.png"
    
    # 如果文件存在则执行
    if os.path.exists(input_img) and os.path.exists(input_mask):
        process_scribble(input_img, input_mask, output_res, alpha=0.5)
    else:
        print("请检查输入文件名是否正确。")