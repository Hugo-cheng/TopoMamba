import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import pandas as pd
from torchmetrics.classification import MultilabelAccuracy


catagory_list = pd.read_excel('slice_classification_ACDC.xlsx')
catagory_list.set_index('slice', inplace=True)
catagory_list = catagory_list.astype(bool)
test_accuracy = MultilabelAccuracy(num_labels=4).cuda()


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if gt.sum() == 0 and pred.sum() == 0:
        return np.nan, np.nan
    elif gt.sum() == 0 and pred.sum() > 0:
        return 0, 0
    elif gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        if pred.sum() == 0:
            hd95 = np.nan
        else:
            hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    # 转换为 numpy
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    
    # --- 核心修改：强制对齐为 3D 逻辑 ---
    # 即使是 (256, 208)，也处理成 (1, 256, 208)
    if len(image.shape) == 2:
        image = image[np.newaxis, ...]
    if len(label.shape) == 2:
        label = label[np.newaxis, ...]
    
    # 现在的 image.shape[0] 对应切片数（即使只有1层）
    prediction = np.zeros_like(label)
    
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        
        # 缩放至网络输入的 patch_size
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        
        net.eval()
        with torch.no_grad():
            # 获取模型预测
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            
            # 缩放回原始尺寸
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            
            # 此时 prediction 已经是 (Slices, H, W)，所以 prediction[ind] 形状为 (H, W)
            # 正好可以放入形状为 (H, W) 的 pred
            prediction[ind] = pred
            
    # --- 后续指标计算 ---
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


@torch.no_grad()
def test_single_volume_CAM(image, label, net, classes, patch_size=[256, 256], epoch=None, model_type=None):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                output = net(input, ep=epoch, model_type=model_type)
                out_aux1, out_aux2, cls_output = output
                out_aux1_soft = torch.softmax(out_aux1, dim=1)
                out_aux2_soft = torch.softmax(out_aux2, dim=1)
                out = torch.argmax(((torch.min(out_aux1_soft, out_aux2_soft) > 0.5) * \
                                    (0.5 * out_aux1_soft + 0.5 * out_aux2_soft)), dim=1).squeeze(0)

                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred

    else:
        input = torch.from_numpy(image).float().cuda()

        net.eval()
        with torch.no_grad():
            out_aux1, out_aux2 = net(input)[0], net(input)[1]
            out_aux1_soft = torch.softmax(out_aux1, dim=1)
            out_aux2_soft = torch.softmax(out_aux2, dim=1)
            out = torch.argmax(((torch.min(out_aux1_soft, out_aux2_soft) > 0.5) * \
                                (0.5 * out_aux1_soft + 0.5 * out_aux2_soft)), dim=1).squeeze(0)

            out = out.cpu().detach().numpy()
            prediction = zoom(
                out, (x / patch_size[0], y / patch_size[1]), order=0)
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))

    return metric_list

import gudhi as gd

def calculate_betti_error(pred, gt):
    """
    计算预测和真实标签之间的 Betti 数误差 (Beta 0 和 Beta 1)
    """
    def get_betti(mask):
        if mask.sum() == 0:
            return 0, 0
        # 使用 GUDHI 计算 2D 图像的持久同调
        # 这里我们将 mask 转换为点云或像素网格
        dims = mask.shape
        # 创建 CubicalComplex 处理像素网格
        cc = gd.CubicalComplex(dimensions=dims, top_dimensional_cells=mask.flatten())
        persistence = cc.persistence()
        # 统计持久性足够长的特征（在 mask 为 1 的区域）
        b0 = len([p for p in persistence if p[0] == 0 and p[1][1] == float('inf')])
        b1 = len([p for p in persistence if p[0] == 1 and p[1][1] == float('inf')])
        return b0, b1

    p_b0, p_b1 = get_betti(pred)
    g_b0, g_b1 = get_betti(gt)
    
    return abs(p_b0 - g_b0), abs(p_b1 - g_b1)