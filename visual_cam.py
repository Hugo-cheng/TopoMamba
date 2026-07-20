import argparse
import os
import re
import importlib
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from scipy.ndimage import zoom
from tqdm import tqdm

# ==========================================
#  Grad-CAM 核心类 (针对医学图像分割优化)
# ==========================================
class MedGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            # 获取反向传播的梯度
            self.gradients = grad_output[0].detach()

        self.handlers.append(self.target_layer.register_forward_hook(forward_hook))
        self.handlers.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor, class_idx, target_size=None):
        """
        生成热力图
        class_idx: 1:RV, 2:Myo, 3:LV
        target_size: (H, W) 原始图像尺寸，用于修复尺寸不匹配报错
        """
        self.model.zero_grad()
        # 即使在 eval 模式下也必须启用梯度以计算 CAM
        with torch.enable_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            # 对目标类别的空间平均预测分数求梯度
            score = probs[:, class_idx, :, :].mean()
            score.backward()

        # GAP (全局平均池化) 计算通道权重
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # 将权重作用于特征图
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        
        # 归一化到 [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # 关键修复：将热力图 resize 到原始图像尺寸 (W, H)
        if target_size is not None:
            return cv2.resize(cam, (target_size[1], target_size[0]))
        return cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))

    def release(self):
        for h in self.handlers:
            h.remove()

# ==========================================
# 图像叠加与保存工具
# ==========================================
def save_heatmap_overlay(image, cam_mask, save_path, alpha=0.4):
    """
    将彩色热力图叠加在灰度原图上
    alpha: 融合权重，值越大热力图越明显
    """
    # 归一化原图用于显示
    img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    img_rgb = np.stack([img_norm]*3, axis=-1)
    
    # 生成伪彩色热力图 (JET 颜色映射)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # 线性融合: (1-alpha)*原图 + alpha*热力图
    overlay = img_rgb * (1 - alpha) + heatmap * alpha
    overlay = (overlay / (overlay.max() + 1e-8) * 255).astype(np.uint8)
    
    # OpenCV 写入需转回 BGR
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# ==========================================
# 主执行函数
# ==========================================
def run_cam_visualization(FLAGS):
    # 创建保存目录
    snapshot_dir = os.path.dirname(FLAGS.pth_path)
    save_base_path = os.path.join(snapshot_dir, "cam_visualizations")
    os.makedirs(save_base_path, exist_ok=True)

    # 1. 加载模型
    print(f"正在加载模型: {FLAGS.model_type}")
    if FLAGS.model_type == 'BiS_Mamba':
        # 确保当前目录下有 network/topomamba.py
        model_module = importlib.import_module("network.topomamba")
        net = getattr(model_module, 'BiS_Mamba')(in_chns=1, class_num=FLAGS.num_classes).cuda()
    else:
        model_module = importlib.import_module("network.unet")
        net = getattr(model_module, 'UNet')(in_chns=1, class_num=FLAGS.num_classes).cuda()
    
    # 加载权重
    checkpoint = torch.load(FLAGS.pth_path)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.eval()

    # 2. 锁定目标层 (Decoder 的最后一个上采样块)
    try:
        # BiS_Mamba 的 decoder 属性
        target_layer = net.decoder.up4 if hasattr(net, 'decoder') else net.up4
        cam_extractor = MedGradCAM(net, target_layer)
        print(f"成功锁定目标层: {target_layer}")
    except Exception as e:
        print(f"错误: 无法找到目标层。请检查模型定义。{e}")
        return

    # 3. 扫描数据并生成
    all_h5 = [f for f in os.listdir(FLAGS.root_path) if f.endswith('.h5')]
    cases_dict = {}
    for f in all_h5:
        match = re.match(r'(patient\d+_frame\d+)', f)
        if match:
            cid = match.group(1)
            cases_dict.setdefault(cid, []).append(f)

    print(f"开始为 {len(cases_dict)} 个病例生成热力图...")

    for cid, slices in tqdm(cases_dict.items()):
        # 排序切片
        slices.sort(key=lambda x: int(re.findall(r'slice_(\d+)', x)[0]))
        
        for f_name in slices:
            with h5py.File(os.path.join(FLAGS.root_path, f_name), 'r') as h5f:
                img_slice = h5f['image'][:]
                h, w = img_slice.shape
                
                # 模型输入缩放 (256x256)
                img_input = zoom(img_slice, (256/h, 256/w), order=1)
                tensor_input = torch.from_numpy(img_input).unsqueeze(0).unsqueeze(0).float().cuda()
                
                # 创建保存切片的子目录
                slice_name = f_name.replace('.h5', '')
                vis_dir = os.path.join(save_base_path, cid, slice_name)
                os.makedirs(vis_dir, exist_ok=True)

                # 为 ACDC 三个前景类别生成热力图
                for idx, name in enumerate(['RV', 'Myo', 'LV'], 1):
                    # 传入 target_size=(h, w) 彻底解决广播错误
                    cam_mask = cam_extractor.generate(tensor_input, class_idx=idx, target_size=(h, w))
                    save_path = os.path.join(vis_dir, f"cam_{name}.png")
                    save_heatmap_overlay(img_slice, cam_mask, save_path)

    cam_extractor.release()
    print(f"\n所有热力图已保存至: {save_base_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='./data/acdc_tiny_test/val', help='H5数据目录')
    parser.add_argument('--pth_path', type=str, required=True, help='权重文件路径')
    parser.add_argument('--model_type', type=str, default='BiS_Mamba', choices=['BiS_Mamba', 'UNet'])
    parser.add_argument('--num_classes', type=int, default=4)
    
    args = parser.parse_args()
    run_cam_visualization(args)