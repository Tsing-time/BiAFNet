import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F


import numpy as np
from mmcv.cnn import ConvModule
from mmdet.models import BaseRoIExtractor
from mmdet.registry import MODELS




@MODELS.register_module()
class EnhanceSimpleHRoIExtractor(BaseRoIExtractor):

    def __init__(self,
                 direction,
                 #out_channels = 256,
                 use_shape=False,#默认不使用特征增强
                 use_attention=None,  # 新增参数控制各层注意力使用
                 conv_cfg=None,
                 init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
                 **kwargs):
        super(EnhanceSimpleHRoIExtractor, self).__init__(
            init_cfg=init_cfg, **kwargs)
        assert direction in ('top_down', 'bottom_up')

        self.direction = direction
        #self.out_channels = out_channels
        self.use_shape = use_shape
        #self.num_inputs = 4 #len(self.featmap_strides)
        #self.sobel_x1, self.sobel_y1 = get_sobel(self.out_channels, self.out_channels)

        self.LightEnhance = SafeSpectralEnhance()

        # 初始化注意力使用配置
        if use_attention is None:
            use_attention = [True] * len(self.featmap_strides)
        assert len(use_attention) == len(self.featmap_strides), \
            "use_attention长度需与特征层数一致"
        self.use_attention = use_attention

        # 为需要注意力的层创建卷积模块
        self.atts = nn.ModuleList()
        for use_att in self.use_attention:
            if use_att:
                self.atts.append(
                    ConvModule(
                        self.out_channels,
                        self.out_channels,
                        1,
                        conv_cfg=conv_cfg,
                        act_cfg=None,
                        bias=False))  # 关闭偏置减少参数
            else:
                # 占位模块，实际不会使用
                self.atts.append(nn.Identity())

    def forward(self, feats, rois, roi_scale_factor=None):

        # 类型转换和初始化检查
        rois = rois.type_as(feats[0])
        assert len(feats) == len(self.featmap_strides)

        # 初始化ROI特征容器
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *self.roi_layers[0].output_size)

        if rois.size(0) == 0:
            return roi_feats

        # ROI缩放处理
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        # 提取各层基础特征
        ori_roi_feats = [self.roi_layers[i](feat, rois) for i, feat in enumerate(feats)]

        # 确定特征处理顺序
        if self.direction == 'top_down':
            indices = reversed(range(len(self.featmap_strides)))  # 顶层到底层
        else:
            indices = range(len(self.featmap_strides))           # 底层到顶层

        # 分层特征聚合
        for i in indices:
            current_feat = ori_roi_feats[i]
            if self.use_attention[i]:
                # 注意力加权融合
                hid = roi_feats + current_feat
                att = self.atts[i](hid).sigmoid()
                roi_feats = roi_feats + current_feat * att
            else:
                # 直接特征相加
                roi_feats = roi_feats + current_feat

        # 特征增强
        if self.use_shape :
            #roi_feats_ = run_sobel(self.sobel_x1, self.sobel_y1, roi_feats )
            #roi_feats =  roi_feats_ + roi_feats
            roi_feats = self.LightEnhance(roi_feats)

        return roi_feats


class SafeSpectralEnhance(nn.Module):
    def __init__(self, in_channels=256, enhance_ratio=0.1, clamp_threshold=1e4):
        super().__init__()
        self.enhance_ratio = enhance_ratio  # 增强强度系数
        self.clamp_threshold = clamp_threshold  # 输入截断阈值

        # 安全频域滤波器（跨通道共享）
        self.filter = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),  # 输入为幅度谱均值
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

        # 输出稳定层
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)  # 通道级归一化

    def safe_fft(self, x):
        """带数值保护的FFT处理"""
        # 输入截断防止溢出
        x = torch.clamp(x, min=-self.clamp_threshold, max=self.clamp_threshold)

        # FFT变换及异常值处理
        fft = torch.fft.fft2(x)
        fft_real = torch.nan_to_num(fft.real)  # 替换NaN/Inf
        fft_imag = torch.nan_to_num(fft.imag)
        return torch.complex(fft_real, fft_imag)

    def forward(self, x):
        """输入x: [B,C,H,W], 输出同形状"""
        B, C, H, W = x.shape

        # Step 1: 安全FFT变换
        fft = self.safe_fft(x)  # [B,C,H,W]

        # Step 2: 计算幅度谱（带平滑处理）
        magnitude = torch.abs(fft) + 1e-6  # 避免除零
        magnitude_norm = magnitude / (magnitude.amax(dim=(2, 3), keepdim=True) + 1e-6)

        # Step 3: 生成频域掩码（通道共享）
        mask = self.filter(magnitude_norm.mean(dim=1, keepdim=True))  # [B,1,H,W]
        mask = torch.clamp(mask, 0.0, 2.0)  # 限制增强幅度

        # Step 4: 高频增强与逆变换
        enhanced_fft = fft * (1.0 + self.enhance_ratio * mask)
        restored = torch.fft.ifft2(enhanced_fft).real

        # Step 5: 残差连接与归一化
        restored = self.norm(restored.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # LayerNorm
        return x + restored * 0.5  # 形状不变

'''
from thop import profile
import time

def prepare_inputs():
    # 创建符合尺寸要求的模拟输入张量
    out_1 = torch.randn(1, 256, 208, 272)  # [batch, channels, height, width]
    out_2 = torch.randn(1, 256, 104, 136)
    out_3 = torch.randn(1, 256, 52, 68)
    out_4 = torch.randn(1, 256, 26, 34)


    # 组合成元组
    return (out_1, out_2, out_3,out_4)

# 步骤3: 计算参数量和FLOPs
def calculate_params_flops(model, inputs):
    flops, params = profile(model, inputs=(inputs,), verbose=False)
    print(f"参数量: {params / 1e6:.2f} M")
    print(f"FLOPs: {flops / 1e9:.2f} G")

# 步骤4: 测量FPS
def measure_fps(model, inputs, warmup=10, test_iter=100):
    device = next(model.parameters()).device

    # 预热
    for _ in range(warmup):
        _ = model(inputs)

    # 同步GPU操作
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 时间测量
    start_time = time.time()
    for _ in range(test_iter):
        _ = model(inputs)

    # 同步GPU操作
    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start_time
    fps = test_iter / elapsed
    print(f"FPS: {fps:.2f} (测试次数: {test_iter}, 耗时: {elapsed:.4f}秒)")


if __name__ == "__main__":
    # 创建模型和输入
    model = EnhanceSimpleHRoIExtractor(roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
            direction='bottom_up',
            use_attention=[True,True, True, True],
            use_shape=True)
    inputs = prepare_inputs()

    # 将模型设置为评估模式
    model.eval()

    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = tuple(inp.to(device) for inp in inputs)

    print("设备:", device)

    # 计算指标
    calculate_params_flops(model, inputs)
    measure_fps(model, inputs)


'''