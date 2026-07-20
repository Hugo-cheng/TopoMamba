import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch.cuda.amp import autocast
from .unet import Encoder, UpBlock

# --- 空间注意力模块 (Spatial Attention) ---
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 使用大核卷积获取更广的感受野，从而更好地识别解剖结构的轮廓
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿着通道维度提取平均池化和最大池化特征
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        res = self.conv(res)
        return self.sigmoid(res)

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
    
    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C, H, W = x.shape
        L = H * W
        x_flat = x.reshape(B, C, L).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, H, W)
        return out

# --- 优化后的双流 Mamba 模块 ---
class DS_TMambaBlock(nn.Module):
    def __init__(self, channels_enc, channels_dec, d_state=16, expand=2):
        super().__init__()
        self.dim = channels_enc
        
        # 1. 解剖流空间对齐与投影
        self.align_conv = nn.Sequential(
            nn.Conv2d(channels_dec, channels_enc, kernel_size=1),
            nn.BatchNorm2d(channels_enc),
            nn.SiLU()
        )
        
        # 2. 空间注意力机制：利用解剖流生成权重图，压制远端噪声
        self.spatial_attn = SpatialAttention()
        
        # 3. 局部精修模块：使用深度可分离卷积补足 Mamba 的局部感应偏置
        self.local_refine = nn.Sequential(
            nn.Conv2d(channels_enc, channels_enc, kernel_size=3, padding=1, groups=channels_enc),
            nn.BatchNorm2d(channels_enc),
            nn.SiLU()
        )
        
        # 4. 拓扑制导参数生成层
        self.to_topo_gate = nn.Sequential(
            nn.Linear(channels_enc, channels_enc),
            nn.Sigmoid()
        )
        
        # 5. 核心 Mamba 引擎
        self.mamba = Mamba(
            d_model=channels_enc,
            d_state=d_state,
            expand=expand
        )
        
        self.norm = nn.LayerNorm(channels_enc)
        self.final_gate = nn.Conv2d(channels_enc, 1, kernel_size=1)

    @autocast(enabled=False)
    def forward(self, x_enc, x_dec):
        if x_enc.dtype == torch.float16: x_enc = x_enc.type(torch.float32)
        if x_dec.dtype == torch.float16: x_dec = x_dec.type(torch.float32)

        B, C, H, W = x_enc.shape
        L = H * W

        # Step 1: 解剖流上采样对齐
        x_dec_up = F.interpolate(x_dec, size=(H, W), mode='bilinear', align_corners=True)
        x_dec_aligned = self.align_conv(x_dec_up)
        
        # Step 2: 空间注意力过滤 (消灭远距离孤立点)
        attn_map = self.spatial_attn(x_dec_aligned)
        x_enc_focused = x_enc * attn_map
        
        # Step 3: 局部精修 (平滑边缘锯齿)
        x_enc_refined = self.local_refine(x_enc_focused)
        
        # Step 4: 生成拓扑引导隐变量 (基于解剖流)
        s_anatomy = x_dec_aligned.reshape(B, C, L).transpose(-1, -2)
        topo_gate = self.to_topo_gate(s_anatomy)
        
        # Step 5: 空间细节流序列化并应用 Mamba
        s_spatial = x_enc_refined.reshape(B, C, L).transpose(-1, -2)
        s_spatial = self.norm(s_spatial)
        
        s_spatial_guided = s_spatial * topo_gate
        mamba_out = self.mamba(s_spatial_guided)
        
        # Step 6: 还原维度并进行最终融合
        out_mamba = mamba_out.transpose(-1, -2).reshape(B, C, H, W)
        spatial_gate = torch.sigmoid(self.final_gate(x_dec_aligned))
        out = x_enc * spatial_gate + out_mamba
        
        return out

class Decoder_TM(nn.Module):
    def __init__(self, params):
        super(Decoder_TM, self).__init__()
        self.ft_chns = params['feature_chns']
        self.n_class = params['class_num']

        # 实例化三个 DS-TMamba 模块
        self.tm_skip3 = DS_TMambaBlock(self.ft_chns[3], self.ft_chns[4])
        self.tm_skip2 = DS_TMambaBlock(self.ft_chns[2], self.ft_chns[3])
        self.tm_skip1 = DS_TMambaBlock(self.ft_chns[1], self.ft_chns[2])

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0, x1, x2, x3, x4 = feature
        x3_pure = self.tm_skip3(x_enc=x3, x_dec=x4)
        x = self.up1(x4, x3_pure)
        x2_pure = self.tm_skip2(x_enc=x2, x_dec=x)
        x = self.up2(x, x2_pure)
        x1_pure = self.tm_skip1(x_enc=x1, x_dec=x)
        x = self.up3(x, x1_pure)
        x = self.up4(x, x0)
        return self.out_conv(x)

class BiS_Mamba(nn.Module):
    def __init__(self, in_chns, class_num):
        super(BiS_Mamba, self).__init__()
        self.params = {
            'in_chns': in_chns,
            'feature_chns': [16, 32, 64, 128, 256],
            'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
            'class_num': class_num,
            'bilinear': False,
        }
        self.encoder = Encoder(self.params)
        self.decoder = Decoder_TM(self.params)
        self.mamba_bottleneck = MambaLayer(dim=256)

    def forward(self, x):
        features = self.encoder(x)
        features[-1] = self.mamba_bottleneck(features[-1])
        output = self.decoder(features)
        return output