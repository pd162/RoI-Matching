import torch
import torch.nn as nn
from torchvision.models import resnet18
from typing import Any, Callable, List, Optional, Type, Union
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1
from segmentation_models_pytorch.base import modules as md
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by the number of heads"

        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.out_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x1, x2):
        bs, c, h, w = x1.shape

        # 将输入展平为 (bs, h * w, c)
        x1_flat = x1.view(bs, c, -1).permute(0, 2, 1)  # (bs, h*w, c)
        x2_flat = x2.view(bs, c, -1).permute(0, 2, 1)  # (bs, h*w, c)

        # 计算 Q, K, V
        query = self.query_projection(x1_flat)  # (bs, h*w, embed_dim)
        key = self.key_projection(x2_flat)      # (bs, h*w, embed_dim)
        value = self.value_projection(x2_flat)  # (bs, h*w, embed_dim)

        # 将维度转换为 (bs, h*w, num_heads, head_dim)
        query = query.view(bs, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, num_heads, h*w, head_dim)
        key = key.view(bs, -1, self.num_heads, self.head_dim).transpose(1, 2)      # (bs, num_heads, h*w, head_dim)
        value = value.view(bs, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, num_heads, h*w, head_dim)

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (bs, num_heads, h*w, h*w)
        attention_probs = torch.softmax(attention_scores, dim=-1)  # (bs, num_heads, h*w, h*w)

        # 计算注意力输出
        attention_output = torch.matmul(attention_probs, value)  # (bs, num_heads, h*w, head_dim)

        # 连接头并通过输出投影
        attention_output = attention_output.transpose(1, 2).contiguous().view(bs, -1, self.embed_dim)  # (bs, h*w, embed_dim)
        output = self.out_projection(attention_output)  # (bs, h*w, embed_dim)

        # 重新展平为 (bs, c, h, w)
        return output.permute(0, 2, 1).view(bs, self.embed_dim, h, w)  # (bs, embed_dim, h, w)



class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        out = []
        x = self.layer1(x)
        out.append(x)
        x = self.layer2(x)
        out.append(x)
        x = self.layer3(x)
        out.append(x)
        x = self.layer4(x)
        out.append(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return out

    def forward(self, x):
        return self._forward_impl(x)


class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, ks, stride=1, norm=True, res=False):
        super(ConvBNReLU, self).__init__()
        if norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=ks, padding=ks // 2, stride=stride, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(True))
        else:
            self.conv = nn.Conv2d(in_c, out_c, kernel_size=ks, padding=ks // 2, stride=stride, bias=False)
        self.res = res

    def forward(self, x):
        if self.res:
            return (x + self.conv(x))
        else:
            return self.conv(x)

class FUSE1(nn.Module):
    def __init__(self, in_channels_list=(64, 128, 256, 512)):
        super(FUSE1, self).__init__()
        self.c31 = ConvBNReLU(in_channels_list[2], in_channels_list[2], 1)
        self.c32 = ConvBNReLU(in_channels_list[3], in_channels_list[2], 1)
        self.c33 = ConvBNReLU(in_channels_list[2], in_channels_list[2], 3)

        self.c21 = ConvBNReLU(in_channels_list[1], in_channels_list[1], 1)
        self.c22 = ConvBNReLU(in_channels_list[2], in_channels_list[1], 1)
        self.c23 = ConvBNReLU(in_channels_list[1], in_channels_list[1], 3)

        self.c11 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 1)
        self.c12 = ConvBNReLU(in_channels_list[1], in_channels_list[0], 1)
        self.c13 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 3)

    def forward(self, x):
        x, x1, x2, x3 = x
        h, w = x2.shape[-2:]
        x2 = self.c33(F.interpolate(self.c32(x3), size=(h, w)) + self.c31(x2))
        h, w = x1.shape[-2:]
        x1 = self.c23(F.interpolate(self.c22(x2), size=(h, w)) + self.c21(x1))
        h, w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1), size=(h, w)) + self.c11(x))
        return x, x1, x2, x3


class FUSE2(nn.Module):
    def __init__(self, in_channels_list=(64, 128, 256)):
        super(FUSE2, self).__init__()

        self.c21 = ConvBNReLU(in_channels_list[1], in_channels_list[1], 1)
        self.c22 = ConvBNReLU(in_channels_list[2], in_channels_list[1], 1)
        self.c23 = ConvBNReLU(in_channels_list[1], in_channels_list[1], 3)

        self.c11 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 1)
        self.c12 = ConvBNReLU(in_channels_list[1], in_channels_list[0], 1)
        self.c13 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 3)

    def forward(self, x):
        x, x1, x2 = x
        h, w = x1.shape[-2:]
        x1 = self.c23(F.interpolate(self.c22(x2), size=(h, w), mode='bilinear', align_corners=True) + self.c21(x1))
        h, w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1), size=(h, w), mode='bilinear', align_corners=True) + self.c11(x))
        return x, x1, x2


class FUSE3(nn.Module):
    def __init__(self, in_channels_list=(64, 128)):
        super(FUSE3, self).__init__()

        self.c11 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 1)
        self.c12 = ConvBNReLU(in_channels_list[1], in_channels_list[0], 1)
        self.c13 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 3)

    def forward(self, x):
        x, x1 = x
        h, w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1), size=(h, w), mode='bilinear', align_corners=True) + self.c11(x))
        return x, x1



class DecoderBlock(nn.Module):
    def __init__(self, cin, cadd, cout, ):
        super().__init__()
        self.cin = (cin + cadd)
        self.cout = cout
        self.conv1 = md.Conv2dReLU(self.cin, self.cout, kernel_size=3, padding=1, use_batchnorm=True)
        self.conv2 = md.Conv2dReLU(self.cout, self.cout, kernel_size=3, padding=1, use_batchnorm=True)

    def forward(self, x1, x2=None):
        x1 = F.interpolate(x1, scale_factor=2.0, mode="nearest")
        if x2 is not None:
            x1 = torch.cat([x1, x2], dim=1)
        x1 = self.conv1(x1[:, :self.cin])
        x1 = self.conv2(x1)
        return x1


class MID(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()
        encoder_channels = encoder_channels[1:][::-1]
        self.in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        self.add_channels = list(encoder_channels[1:]) + [64]
        self.out_channels = decoder_channels
        self.fuse1 = FUSE1()
        self.fuse2 = FUSE2()
        self.fuse3 = FUSE3()
        decoder_convs = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.add_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.add_channels[layer_idx]
                    skip_ch = self.add_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.add_channels[layer_idx - 1]
                decoder_convs[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch)
        decoder_convs[f"x_{0}_{len(self.in_channels) - 1}"] = DecoderBlock(self.in_channels[-1], 0,
                                                                           self.out_channels[-1])
        self.decoder_convs = nn.ModuleDict(decoder_convs)

    def forward(self, *features):
        decoder_features = {}
        features = self.fuse1(features)[::-1]
        decoder_features["x_0_0"] = self.decoder_convs["x_0_0"](features[0], features[1])
        decoder_features["x_1_1"] = self.decoder_convs["x_1_1"](features[1], features[2])
        decoder_features["x_2_2"] = self.decoder_convs["x_2_2"](features[2], features[3])
        decoder_features["x_2_2"], decoder_features["x_1_1"], decoder_features["x_0_0"] = self.fuse2(
            (decoder_features["x_2_2"], decoder_features["x_1_1"], decoder_features["x_0_0"]))
        decoder_features["x_0_1"] = self.decoder_convs["x_0_1"](decoder_features["x_0_0"],
                                                                torch.cat((decoder_features["x_1_1"], features[2]), 1))
        decoder_features["x_1_2"] = self.decoder_convs["x_1_2"](decoder_features["x_1_1"],
                                                                torch.cat((decoder_features["x_2_2"], features[3]), 1))
        decoder_features["x_1_2"], decoder_features["x_0_1"] = self.fuse3(
            (decoder_features["x_1_2"], decoder_features["x_0_1"]))
        decoder_features["x_0_2"] = self.decoder_convs["x_0_2"](decoder_features["x_0_1"], torch.cat(
            (decoder_features["x_1_2"], decoder_features["x_2_2"], features[3]), 1))
        return self.decoder_convs["x_0_3"](
            torch.cat((decoder_features["x_0_2"], decoder_features["x_1_2"], decoder_features["x_2_2"]), 1))

class Matcher(nn.Module):
    def __init__(self):
        super(Matcher, self).__init__()
        self.backbone1 = ResNet(BasicBlock, [2, 2, 2, 2])  # ToDo: 加预训练权重
        self.backbone2 = ResNet(BasicBlock, [2, 2, 2, 2])
        self.attn = CrossAttention(embed_dim=512, num_heads=4)
        # self.decoder = MID(encoder_channels=(64, 128, 256, 512), decoder_channels=(256, 128, 64, 32))
        self.seg_head = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=32.),
            nn.Conv2d(512, 64, kernel_size=3, padding=3 // 2, stride=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 2, kernel_size=1)
        )

    def forward(self, x1, x2, mask):
        feat_1 = self.backbone1(x1 * mask)
        # feat_1 = self.decoder(*feat_1)
        feat_2 = self.backbone2(x2)
        # feat_2 = self.decoder(*feat_2)
        attn_score = self.attn(feat_1[-1], feat_2[-1])
        pred = self.seg_head(attn_score)
        return pred


