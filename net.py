import ipdb
import torch
import torch.nn as nn

from mmcv.cnn import ConvModule
from mmdet.registry import MODELS

from mmengine.model import BaseModule, caffe2_xavier_init#, constant_init

import torch.nn.functional as F


@MODELS.register_module()
class BiAF_net(BaseModule):

    def __init__(self,
                 in_channels,
                 init_cfg=dict(type='Caffe2Xavier', layer='Conv2d')):
        super(BiAF_net, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = in_channels

        self.att3 = ConvModule(
                self.out_channels * 3,
                self.out_channels,
                1,
                conv_cfg=None,
                act_cfg=None)


        self.att2 = ConvModule(
            self.out_channels ,
            self.out_channels,
            1,
            conv_cfg=None,
            act_cfg=None)


        #self.UP =  nn.ConvTranspose2d(self.in_channels, self.out_channels,  kernel_size=3, stride=2, padding=1)
        self.DOWN = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gate = DynamicGatedFusion(in_channels=self.in_channels, out_channels=self.out_channels)
        self.DAFM = DAFM()



        self.init_weights()
    def init_weights(self):
        for att in [self.att3, self.att2]:
            nn.init.normal_(att.conv.weight, mean=0, std=0.01)
            nn.init.zeros_(att.conv.bias)
        if hasattr(self.DAFM, "init_weights"):
            self.DAFM.init_weights()
        if hasattr(self.gate, "init_weights"):
            self.gate.init_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Depthwise卷积初始化
                if m.groups == m.in_channels:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # Pointwise卷积初始化
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, inputs):
        fusion_1 = []
        for i in range(len(inputs)):
            fusion_1.append(inputs[i])

        #分成双路
        fusion_13_up = F.interpolate(fusion_1[-1], size=(fusion_1[3].shape[2],fusion_1[3].shape[3]), mode="bilinear", align_corners=False)
        fusion_13 = torch.cat((fusion_13_up, fusion_1[3],self.DOWN(fusion_1[2])), dim=1)
        att_13 = self.att3(fusion_13).sigmoid()
        fusion_1[3] = fusion_1[3] * att_13 + fusion_1[3]
        #fusion_1[3] = self.atten(fusion_13_up, fusion_1[3], self.DOWN(fusion_1[2]))


        fusion_12_up = F.interpolate(fusion_1[2], size=(fusion_1[1].shape[2], fusion_1[1].shape[3]), mode="bilinear",align_corners=False)
        fusion_11 = torch.cat((fusion_12_up, fusion_1[1], self.DOWN(fusion_1[0])), dim=1)
        att_11 = self.att3(fusion_11).sigmoid()
        fusion_1[1] = fusion_1[1] * att_11 + fusion_1[1]
        #替换成------
        #fusion_1[1] = self.atten(fusion_12_up,fusion_1[1],self.DOWN(fusion_1[0]))

        #上半部分

        fusion_23 ,fusion_33= self.DAFM(fusion_1[3])
        fusion_1[3] = self.gate(fusion_1[3],fusion_23,fusion_33) + fusion_33

        # 上分支向上融合_终
        fusion_34 = self.DOWN(fusion_1[3])
        att_34 = self.att2(fusion_34).sigmoid()
        fusion_1[4] = fusion_1[4] + fusion_34 * att_34

        #上分支向下融合_1
        fusion_32 = F.interpolate(fusion_1[3], size=(fusion_1[2].shape[2], fusion_1[2].shape[3]), mode="bilinear", align_corners=False)
        att_32 = self.att2(fusion_32).sigmoid()
        fusion_1[2] = fusion_1[2] + fusion_32 * att_32


        #下半部分_1
        fusion_21_context, fusion_21_channel= self.DAFM(fusion_1[1])


        #上分支向下融合_2--->进入下分支
        fusion_22 = F.interpolate(fusion_1[2],size=(fusion_1[1].shape[2], fusion_1[1].shape[3]), mode="bilinear", align_corners=False)
        att_22 = self.att2(fusion_22).sigmoid()
        fusion_31 = fusion_21_channel + fusion_22 * att_22

        #下半部分_2
        fusion_1[1] = self.gate(fusion_31,fusion_21_context,fusion_21_channel) + fusion_1[1]

        #下分支向下融合_终
        fusion_30 = F.interpolate(fusion_1[1], size=(fusion_1[0].shape[2],fusion_1[0].shape[3]),mode="bilinear", align_corners=False)
        att_30 = self.att2(fusion_30).sigmoid()
        fusion_1[0] = fusion_1[0] + fusion_30 * att_30

        # 下分支向上融合_终
        fusion_42 = self.DOWN(fusion_1[1])
        att_42 = self.att2(fusion_42).sigmoid()
        fusion_1[2] =  fusion_42 * att_42 + fusion_1[2]


        fusion_outs=[]
        for i in range(len(inputs)):
            fusion_outs.append(inputs[i] + fusion_1[i])


        return tuple(fusion_outs)


class DynamicGatedFusion(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(DynamicGatedFusion, self).__init__()


        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(),
            nn.Linear(out_channels // 4, 3),
            nn.Softmax(dim=-1)
        )

    def init_weights(self):
        """初始化所有子模块和卷积层的权重。"""
        # 初始化 ConvModule 中的卷积层
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2, x3):
        # batch_size, channels, height, width = x2.shape

        f1 = self.conv1(x1)
        f2 = self.conv2(x2)
        f3 = self.conv3(x3)

        # ipdb.set_trace()
        f_sum = f1 + f2 + f3

        weights = self.gate_fc(f_sum)

        weights_0 = weights[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1, 1]
        weights_0 = weights_0.expand(-1, f1.size(1), f1.size(2), f1.size(3))  # 扩展到 [batch_size, 256, height, width]

        # 加权融合
        fused_feature = weights_0 * f1 + weights[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, f1.size(1),
                                                                                                        f1.size(2),
                                                                                                        f1.size(
                                                                                                            3)) * f2 + weights[
                                                                                                                       :,
                                                                                                                       2].unsqueeze(
            -1).unsqueeze(-1).unsqueeze(-1).expand(-1, f1.size(1), f1.size(2), f1.size(3)) * f3

        #  ipdb.set_trace()
        return fused_feature



class DAFM(nn.Module):
    def __init__(self, in_channels=256, reduction=1, conv_cfg=None):
        super(DAFM, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction


        self.corssConv = CrossConv()
        self.CAB = ContextAggregation(in_channels=256)
        self.ChannelAttention = ChannelAttention(in_channels=256)

        self.conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv_tri = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.in_channels)
            )


        self.init_weights()


    def init_weights(self):
        # 初始化 CAB 模块
        if hasattr(self.CAB, "init_weights"):
            self.CAB.init_weights()

        # 初始化 ChannelAttention 模块
        if hasattr(self.ChannelAttention, "init_weights"):
            self.ChannelAttention.init_weights()

        # 初始化深度可分离卷积和其他自定义模块
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def forward(self, x):

        ##分支
        Cab = self.CAB(x)

        # 分支
        cross_out = self.corssConv(x)
    #    w = self.w(cross_out).sigmoid()

        #聚合
        out_1 = Cab + cross_out
        out_1 = self.conv(out_1)
        out_1 = out_1 + x

        #通道
        out_2 = self.ChannelAttention(out_1)
        out_2 = self.conv_tri(out_2)
        out_2 = out_2 + out_1


        return  out_1,out_2


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # 全局平均池化：将 H x W 的特征图降为 1 x 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全局最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 两层全连接层，使用 reduction_ratio 缩减通道数后再扩展回去
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        # 激活函数
        self.sigmoid = nn.Sigmoid()

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """初始化权重."""
        nn.init.kaiming_normal_(self.fc[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc[2].weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入的 batch size 和通道数

        # 通过全局平均池化和全局最大池化生成两个通道注意力特征
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        # 将两个注意力特征相加并通过 sigmoid 函数得到最终的注意力权重

        a = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        out = a * x

        # 将原始特征与通道注意力相乘，得到加权后的特征
        return x + out

class CrossConv(nn.Module):
    def __init__(self, in_channels=256):
        super(CrossConv, self).__init__()
        self.in_channels = in_channels

        self.conv_tri = self.conv_tri = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.in_channels)
            )

        # 假设要分配一半通道到垂直卷积，另一半到水平卷积
        self.vertical_channels = in_channels // 2
        self.horizontal_channels = in_channels - self.vertical_channels

        # 垂直条形卷积 (height方向卷积)
        self.vertical_conv = nn.Conv2d(self.vertical_channels, self.vertical_channels, kernel_size=(3, 1),
                                       padding=(1, 0), groups=self.vertical_channels, bias=False)

        # 水平条形卷积 (width方向卷积)
        self.horizontal_conv = nn.Conv2d(self.horizontal_channels, self.horizontal_channels, kernel_size=(1, 3),
                                         padding=(0, 1), groups=self.horizontal_channels, bias=False)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 使用标准的 Xavier 初始化
        for m in [self.vertical_conv, self.horizontal_conv]:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        n, c, h, w = x.size()

        # 切分输入特征图的通道
        vertical_input = x[:, :self.vertical_channels, :, :]  # 垂直卷积输入
        horizontal_input = x[:, self.vertical_channels:, :, :]  # 水平卷积输入

        # 通过垂直条形卷积
        vertical_output = self.vertical_conv(vertical_input)
        vertical_output = self.horizontal_conv(vertical_output)

        # 通过水平条形卷积
        horizontal_output = self.horizontal_conv(horizontal_input)
        horizontal_output = self.vertical_conv(horizontal_output)

        # 拼接结果
        out = torch.cat([vertical_output, horizontal_output], dim=1)

        out = self.conv_tri(out)
        return out

class ContextAggregation(nn.Module):
    def __init__(self, in_channels, reduction=1, conv_cfg=None):
        super(ContextAggregation, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)

        conv_params = dict(kernel_size=1, conv_cfg=conv_cfg, act_cfg=None)
        #self.a = ConvModule(in_channels, 1, **conv_params)
        self.k = ConvModule(in_channels, 1, **conv_params)


        self.init_weights()
    def init_weights(self):
        #for m in ( self.k):
            #caffe2_xavier_init(m.conv)
        caffe2_xavier_init(self.k.conv)


    def forward(self, x):
        n, c = x.size(0), self.inter_channels
        k = self.k(x).view(n, 1, -1, 1).softmax(2)
        v = x.view(n, 1, c, -1)
        y = torch.matmul(v, k).view(n, c, 1, 1)

        return x + y





