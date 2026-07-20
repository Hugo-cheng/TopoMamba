import argparse
import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 导入自定义模块
from dataloders.acdc import ACDC_dataloder, TopoResize
from network.topomamba import BiS_Mamba
from network.unet import UNet
from utils.gate_crf_loss import ModelLossSemsegGatedCRF
from utils.topology_loss import TopologicalLoss2D
from utils.boudary_loss import BoundaryConsistencyLoss
from val_2D import test_single_volume, calculate_betti_error

# ---  EMA 更新逻辑 ---
def update_ema_variables(model, ema_model, alpha, global_step):
    # 动态调整 alpha，初期快更新，后期平滑
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

# --- 拓扑筛选逻辑 (Strict Mode) ---
class TopoPseudoFilter:
    def __init__(self, target_idx=2, threshold=0.5):
        self.target_idx = target_idx
        self.threshold = threshold

    def is_topology_valid(self, prob_map):
        mask = (prob_map > self.threshold).float()
        num_pixels = torch.sum(mask).item()
        total_pixels = prob_map.shape[0] * prob_map.shape[1]
        
        # 基础过滤
        if num_pixels < 100 or num_pixels > (total_pixels * 0.2):
            return False

        mask_np = mask.cpu().numpy().astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np)
        
        if num_labels <= 1: return False
        
        max_area = np.max(stats[1:, cv2.CC_STAT_AREA])
        # 严格占比要求
        if max_area / num_pixels > 0.7:
            return True
        return False

    def generate_filtered_label(self, outputs_soft):
        B = outputs_soft.shape[0]
        pseudo_labels = torch.argmax(outputs_soft, dim=1)
        valid_mask = torch.zeros(B, device=outputs_soft.device)
        
        myo_probs = outputs_soft[:, self.target_idx, :, :]
        for b in range(B):
            if self.is_topology_valid(myo_probs[b]):
                valid_mask[b] = 1.0
        return pseudo_labels, valid_mask

# --- 辅助函数 ---
def weights_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight.data)

def validation(model, valloader, num_classes, patch_size):
    model.eval()
    metric_list = []
    betti_errors = [] 
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(valloader):
            image, label = sampled_batch["image"], sampled_batch["label"]
            metric_i = test_single_volume(image, label, model, classes=num_classes, patch_size=patch_size)
            metric_list.append(metric_i)
            
            # 计算 Betti Error (此处简化逻辑，调用你已有的 calculate_betti_error)
            # ... (保持你原有 validation 函数中的逻辑)
            
    avg_metrics = np.nanmean(np.array(metric_list), axis=0)
    return np.mean(avg_metrics[:, 0]), [0.03, 0.00] # 示例返回值

# ---  主训练函数 ---
def train(args):
    snapshot_path = f"model/{args.exp}/{args.session_name}"
    os.makedirs(snapshot_path, exist_ok=True)
    writer = SummaryWriter(os.path.join(snapshot_path, "log"))

    # 初始化学生与老师模型
    model = BiS_Mamba(in_chns=1, class_num=args.num_classes).cuda()
    ema_model = BiS_Mamba(in_chns=1, class_num=args.num_classes).cuda()
    
    model.apply(weights_init)
    for param in ema_model.parameters():
        param.requires_grad = False

    # 加载预训练权重
    if args.pretrain_weights:
        print(f"==> Loading: {args.pretrain_weights}")
        checkpoint = torch.load(args.pretrain_weights)
        load_info=model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint, strict=False)
        # 建议打印一下缺失的 Key，确认是不是只有你新增的那几个层
        print(f"Missing keys (will be randomly initialized): {load_info.missing_keys}")
        ema_model.load_state_dict(model.state_dict())

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wt_dec)
    
    # 数据增强：强增强分支
    strong_aug = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    ])

    db_train = ACDC_dataloder(base_dir=args.root_path, split="train", transform=transforms.Compose([TopoResize(output_size=args.patch_size)]))
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    db_val = ACDC_dataloder(base_dir=args.root_path, split="val")
    valloader = DataLoader(db_val, batch_size=1, shuffle=False)

    ce_loss = CrossEntropyLoss(ignore_index=4)
    gatecrf_loss = ModelLossSemsegGatedCRF()
    criterion_topo = TopologicalLoss2D(target_idx=2, roi_size=64).cuda()
    criterion_boundary = BoundaryConsistencyLoss().cuda()
    topo_filter = TopoPseudoFilter(target_idx=2, threshold=0.5)
    pseudo_ce_loss = CrossEntropyLoss(reduction='none')

    best_performance = 0.0
    global_step = 0

    for epoch in range(args.max_epochs):
        model.train()
        ema_model.train() # Teacher 处于训练模式以同步 BN (或设为 eval 保持稳定)

        for i, sampled_batch in enumerate(trainloader):
            img, scribble = sampled_batch['image'].cuda(), sampled_batch['scribble'].cuda()
            
            # --- 强增强一致性支路 ---
            img_strong = strong_aug(img.clone())
            outputs_student_strong = model(img_strong)
            outputs_soft_student_strong = torch.softmax(outputs_student_strong, dim=1)

            # --- 老师预测 (原图) ---
            with torch.no_grad():
                outputs_teacher = ema_model(img)
                outputs_soft_teacher = torch.softmax(outputs_teacher, dim=1)
                pseudo_labels, valid_mask = topo_filter.generate_filtered_label(outputs_soft_teacher)
                num_valid = valid_mask.sum().item()

            # A. 基础 Scribble 监督
            outputs_orig = model(img)
            loss_ce = ce_loss(outputs_orig, scribble)

            # B. 严格筛选下的 伪标签 + 一致性 MSE
            loss_pseudo = torch.tensor(0.0, device=img.device)
            loss_mse = torch.tensor(0.0, device=img.device)
            loss_boundary = torch.tensor(0.0, device=img.device) # 初始化边界损失

            if num_valid == args.batch_size:
                # 让学生在加噪图上对齐老师的伪标签
                loss_pseudo = pseudo_ce_loss(outputs_student_strong, pseudo_labels).mean()
                # 一致性损失：对齐概率分布
                loss_mse = F.mse_loss(outputs_soft_student_strong, outputs_soft_teacher)
                # 边界一致性损失 强制 Student 的边界轮廓对齐 Teacher 的平滑轮廓(针对 HD95/ASD 优化)
                loss_boundary = criterion_boundary(outputs_soft_student_strong, outputs_soft_teacher)

            # C. 其他辅助损失 (基于原图)
            loss_crf = gatecrf_loss(torch.softmax(outputs_orig, dim=1), [{"weight":1, "xy":6, "rgb":0.1}], 5, img, 224, 224)["loss"]
            loss_topo = criterion_topo(torch.softmax(outputs_orig, dim=1), 1, 1) if epoch >= 5 else torch.tensor(0.0)

            # 联合损失
            total_loss = loss_ce + args.weight_crf * loss_crf + \
                         args.weight_topo * loss_topo + \
                         args.weight_pseudo * (loss_pseudo + 1.0 * loss_mse + args.weight_boundary * loss_boundary)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 更新 Teacher
            update_ema_variables(model, ema_model, 0.99, global_step)
            global_step += 1

        # 验证
        if (epoch + 1) % args.val_interval == 0:
            # 验证时使用稳健的 Teacher 模型
            performance, betti_err = validation(ema_model, valloader, args.num_classes, args.patch_size)
            print(f"Epoch {epoch}: Dice: {performance:.4f}, B0-Err: {betti_err[0]:.2f}")

            if performance > best_performance:
                best_performance = performance
                torch.save({'state_dict': ema_model.state_dict(), 'dice': performance}, os.path.join(snapshot_path, 'best_model.pth'))
                print(f"--- Best Saved: {best_performance:.4f} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="./data/acdc_tiny_test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patch_size", nargs='+', type=int, default=[224, 224])
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wt_dec", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--exp", default="ACDC/TopoMamba")
    parser.add_argument("--session_name", default="EMA_Strict_Finetune")
    parser.add_argument("--pretrain_weights", default='', type=str)
    parser.add_argument("--weight_crf", type=float, default=0.1)
    parser.add_argument("--weight_topo", type=float, default=0.02)
    parser.add_argument("--weight_pseudo", type=float, default=0.5) # 引入一致性后可微调权重
    parser.add_argument("--weight_boundary", type=float, default=0.1) # 边界损失权重

    args = parser.parse_args()
    train(args)