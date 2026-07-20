import argparse
import os
import re
import importlib
import h5py
import numpy as np
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
from skimage import measure

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/acdc_tiny_test/val', help='存放.h5和配套.nii.gz的文件夹')
parser.add_argument('--pth_path', type=str, default='./model/ACDC/TopoMamba/EMA_Strict_Finetune/best_model.pth', help='权重路径')
parser.add_argument('--num_classes', type=int, default=4, help='类别数')
parser.add_argument('--save_prediction', type=bool, default=True, help='是否保存结果')

def calculate_metric_2d(pred, gt, spacing):
    """在 2D 维度计算 Dice, HD95, ASD"""
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        # 仅取 x, y 轴的 spacing
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing[:2])
        asd = metric.binary.asd(pred, gt, voxelspacing=spacing[:2])
        return dice, hd95, asd
    else:
        return 0.0, 50.0, 20.0

def calculate_topology_2d(pred, gt):
    """计算 2D 拓扑 Betti-0 误差"""
    def get_b0(mask):
        if mask.sum() == 0: return 0
        return measure.label(mask).max() 
    t_err = 0
    for i in range(1, 4): # 对 RV, Myo, LV 分别计算
        t_err += abs(get_b0(pred == i) - get_b0(gt == i))
    return t_err

def save_individual_png(image, mask, save_path):
    """
    强制执行配色标准:
    1: RV -> 蓝色
    2: Myo -> 黄色
    3: LV -> 红色
    """
    # 定义颜色列表：索引0是背景(透明)，1是蓝色，2是黄色，3是红色
    colors = ['none', 'blue', 'yellow', 'red']
    custom_cmap = ListedColormap(colors)
    
    plt.figure(figsize=(6, 6))
    # 显示底层灰度原图
    plt.imshow(image, cmap='gray')
    
    if mask is not None:
        # 对 Mask 进行掩模处理，使标签 0 (背景) 变为透明
        masked_mask = np.ma.masked_where(mask == 0, mask)
        # 叠加颜色图层，alpha=0.5 保证能看到下面的原图纹理
        plt.imshow(masked_mask, cmap=custom_cmap, vmin=0, vmax=3, alpha=0.5, interpolation='nearest')
    
    plt.axis('off')
    # 使用 bbox_inches='tight' 确保没有白边
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()

def test_single_case(case_id, slice_files, net, test_save_path, FLAGS):
    # 读取物理元数据
    org_img_path = os.path.join(FLAGS.root_path, f"{case_id}.nii.gz")
    if os.path.exists(org_img_path):
        org_img_itk = sitk.ReadImage(org_img_path)
        spacing = org_img_itk.GetSpacing() 
    else:
        org_img_itk = None
        spacing = (1.0, 1.0, 10.0)

    # 排序切片
    slice_files.sort(key=lambda x: int(re.findall(r'slice_(\d+)', x)[0]))
    
    with h5py.File(os.path.join(FLAGS.root_path, slice_files[0]), 'r') as f:
        h, w = f['image'].shape
    d = len(slice_files)
    
    # 初始化 3D 拼合容器
    full_image = np.zeros((d, h, w))
    full_label = np.zeros((d, h, w))
    full_pred = np.zeros((d, h, w))
    case_2d_metrics = []

    for i, f_name in enumerate(slice_files):
        with h5py.File(os.path.join(FLAGS.root_path, f_name), 'r') as h5f:
            img_slice = h5f['image'][:]
            lab_slice = h5f['label'][:]
            full_image[i] = img_slice
            full_label[i] = lab_slice
            
            # 2D 推理
            img_res = zoom(img_slice, (256/h, 256/w), order=1)
            input_t = torch.from_numpy(img_res).unsqueeze(0).unsqueeze(0).float().cuda()
            with torch.no_grad():
                out = net(input_t)
                pred_out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0).cpu().numpy()
                pred_res = zoom(pred_out, (h/256, w/256), order=0)
                full_pred[i] = pred_res
            
            # 计算切片指标
            m1 = calculate_metric_2d(pred_res==1, lab_slice==1, spacing)
            m2 = calculate_metric_2d(pred_res==2, lab_slice==2, spacing)
            m3 = calculate_metric_2d(pred_res==3, lab_slice==3, spacing)
            t_err = calculate_topology_2d(pred_res, lab_slice)
            case_2d_metrics.append((m1, m2, m3, t_err))

            # --- 保存单独的 PNG ---
            if FLAGS.save_prediction:
                vis_dir = os.path.join(test_save_path, "visual_pngs", case_id, f"slice_{i:02d}")
                os.makedirs(vis_dir, exist_ok=True)
                # 1. 纯原图
                save_individual_png(img_slice, None, os.path.join(vis_dir, "raw_img.png"))
                # 2. 原图 + GT
                save_individual_png(img_slice, lab_slice, os.path.join(vis_dir, "gt_overlay.png"))
                # 3. 原图 + Prediction
                save_individual_png(img_slice, pred_res, os.path.join(vis_dir, "pred_overlay.png"))

    # --- 保存 3D NIfTI ---
    if FLAGS.save_prediction:
        for arr, suffix in zip([full_image, full_pred, full_label], ["_img", "_pred", "_gt"]):
            itk_out = sitk.GetImageFromArray(arr.astype(np.float32))
            if org_img_itk:
                itk_out.CopyInformation(org_img_itk)
            else:
                itk_out.SetSpacing(spacing)
            sitk.WriteImage(itk_out, os.path.join(test_save_path, f"{case_id}{suffix}.nii.gz"))

    return case_2d_metrics

def Inference(FLAGS):
    snapshot_dir = os.path.dirname(FLAGS.pth_path)
    test_save_path = os.path.join(snapshot_dir, "test_results_final")
    os.makedirs(test_save_path, exist_ok=True)

    # 动态加载网络
    model_module = importlib.import_module("network.unet")
    net = getattr(model_module, 'UNet')(in_chns=1, class_num=FLAGS.num_classes).cuda()
    
    checkpoint = torch.load(FLAGS.pth_path)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.eval()

    # 扫描 H5 并分组
    all_h5 = [f for f in os.listdir(FLAGS.root_path) if f.endswith('.h5')]
    cases_dict = {}
    for f in all_h5:
        match = re.match(r'(patient\d+_frame\d+)', f)
        if match:
            cid = match.group(1)
            cases_dict.setdefault(cid, []).append(f)

    print(f"识别到 {len(cases_dict)} 个病例，开始 2D 测试与 3D 保存...")
    all_metrics = []
    for cid, s_list in tqdm(cases_dict.items()):
        all_metrics.extend(test_single_case(cid, s_list, net, test_save_path, FLAGS))

    # 最终汇总统计
    results = np.mean([np.array([m[0], m[1], m[2]]) for m in all_metrics], axis=0)
    avg_topo = np.mean([m[3] for m in all_metrics])
    
    print(f"\n=================== 测试结果统计 ===================")
    print(f"平均 Dice: {np.mean(results[:, 0]):.4f}")
    print(f"平均 HD95: {np.mean(results[:, 1]):.2f}")
    print(f"平均 Betti 误差: {avg_topo:.4f}")
    print(f"可视化与 3D 文件已存至: {test_save_path}")

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    Inference(FLAGS)