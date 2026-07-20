import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryConsistencyLoss(nn.Module):
    """
    边界一致性损失 (Boundary Consistency Loss)
    用于弱监督学习中，强制 Student 模型在增强图像上的边界与 Teacher 模型在原图上的边界保持一致。
    """
    def __init__(self, reduction='mean'):
        super(BoundaryConsistencyLoss, self).__init__()
        self.reduction = reduction
        # 定义拉普拉斯算子用于提取边缘特征
        self.register_buffer('laplacian_kernel', torch.tensor(
            [[-1, -1, -1], 
             [-1,  8, -1], 
             [-1, -1, -1]], dtype=torch.float32).reshape(1, 1, 3, 3))

    def get_boundary_map(self, x):
        """
        x: [B, C, H, W] - 概率分布图 (Softmax 后)
        """
        B, C, H, W = x.shape
        # 将通道合并到 Batch 维进行统一卷积
        x_grouped = x.reshape(B * C, 1, H, W)
        boundary = F.conv2d(x_grouped, self.laplacian_kernel, padding=1)
        return torch.abs(boundary).reshape(B, C, H, W)

    def forward(self, pred_student, pred_teacher):
        """
        pred_student: Student 模型的预测概率图 [B, C, H, W]
        pred_teacher: Teacher 模型的预测概率图 [B, C, H, W]
        """
        # 1. 提取双方的边界特征
        s_boundary = self.get_boundary_map(pred_student)
        t_boundary = self.get_boundary_map(pred_teacher)

        # 2. 计算边界之间的一致性损失 (MSE)
        # 排除背景通道 (Channel 0)，重点关注 RV, Myo, LV
        loss = F.mse_loss(s_boundary[:, 1:, ...], t_boundary[:, 1:, ...], reduction=self.reduction)
        
        return loss