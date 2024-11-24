import torch
import torch.nn as nn
from torchvision.models import resnet18
from typing import Any, Callable, List, Optional, Type, Union
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1, load_state_dict_from_url, model_urls
from segmentation_models_pytorch.base import modules as md
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import MultiheadAttention


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
        self.conv1 = nn.Conv2d(6, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
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


class Decoder(nn.Module):
    def __init__(self, input_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_channels, 128, kernel_size=4, stride=2,
                                        padding=1)  # Output: (bs, 128, height*2, width*2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,
                                        padding=1)  # Output: (bs, 64, height*4, width*4)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2,
                                        padding=1)  # Output: (bs, 32, height*8, width*8)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2,
                                        padding=1)  # Output: (bs, 2, height*16, width*16)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # 使用 He 初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv4(x)
        return x


class UpBlock(nn.Module):
    """Upsample block for DRRG and TextSnake."""

    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()

        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)

        self.conv1x1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1x1(x))
        x = F.relu(self.conv3x3(x))
        x = self.deconv(x)
        return x


class FPN_UNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ):
        super(FPN_UNet, self).__init__()
        assert len(in_channels) == 4
        assert isinstance(out_channels, int)

        blocks_out_channels = [out_channels] + [
            min(out_channels * 2**i, 256) for i in range(4)
        ]
        blocks_in_channels = [blocks_out_channels[1]] + [
            in_channels[i] + blocks_out_channels[i + 2] for i in range(3)
        ] + [in_channels[3]]

        self.up4 = nn.ConvTranspose2d(
            blocks_in_channels[4],
            blocks_out_channels[4],
            kernel_size=4,
            stride=2,
            padding=1)
        self.up_block3 = UpBlock(blocks_in_channels[3], blocks_out_channels[3])
        self.up_block2 = UpBlock(blocks_in_channels[2], blocks_out_channels[2])
        self.up_block1 = UpBlock(blocks_in_channels[1], blocks_out_channels[1])
        self.up_block0 = UpBlock(blocks_in_channels[0], blocks_out_channels[0])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # 使用 He 初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                # 使用 He 初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        c2, c3, c4, c5 = x

        x = F.relu(self.up4(c5))

        x = torch.cat([x, c4], dim=1)
        x = F.relu(self.up_block3(x))

        x = torch.cat([x, c3], dim=1)
        x = F.relu(self.up_block2(x))

        x = torch.cat([x, c2], dim=1)
        x = F.relu(self.up_block1(x))

        x = self.up_block0(x)
        # the output should be of the same height and width as backbone input
        return x

class AddCoords(nn.Module):
    def __init__(self, with_r=True):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        batch_size, _, x_dim, y_dim = input_tensor.size()
        xx_c, yy_c = torch.meshgrid(torch.arange(x_dim, dtype=input_tensor.dtype),
                                    torch.arange(y_dim, dtype=input_tensor.dtype))
        xx_c = xx_c.to(input_tensor.device) / (x_dim - 1) * 2 - 1
        yy_c = yy_c.to(input_tensor.device) / (y_dim - 1) * 2 - 1
        xx_c = xx_c.expand(batch_size, 1, x_dim, y_dim)
        yy_c = yy_c.expand(batch_size, 1, x_dim, y_dim)
        ret = torch.cat((input_tensor, xx_c, yy_c), dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_c - 0.5, 2) + torch.pow(yy_c - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret


class Matcher_wdq(nn.Module):
    def __init__(self):
        super(Matcher_wdq, self).__init__()
        self.backbone1 = ResNet(BasicBlock, [2, 2, 2, 2])
        # state_dict = load_state_dict_from_url(model_urls['resnet18'],
        #                                       progress=True)
        # self.backbone1.load_state_dict(state_dict, strict=False)
        self.backbone2 = ResNet(BasicBlock, [2, 2, 2, 2])
        # state_dict = load_state_dict_from_url(model_urls['resnet18'],
        #                                       progress=True)
        # self.backbone2.load_state_dict(state_dict, strict=False)
        self.fpn1 = FPN_UNet(
            in_channels=[64, 128, 256, 512],
            out_channels=256
        )
        self.fpn2 = FPN_UNet(
            in_channels=[64, 128, 256, 512],
            out_channels=256
        )
        self.cross_attn_1 = MultiheadAttention(embed_dim=256, num_heads=8, batch_first=False)
        self.cross_attn_2 = MultiheadAttention(embed_dim=256, num_heads=8, batch_first=False)
        self.addcoords_1 = AddCoords()
        # self.addcoords_2 = AddCoords()
        self.seg_head = Decoder(input_channels=256)
        self.strides = [4, 8, 16, 32]

    def forward(self, x1, x2, mask):
        feat_1 = self.backbone1(self.addcoords_1(x1))
        feat_2 = self.backbone2(self.addcoords_1(x2))
        mask = (mask != 0).any(dim=1).to(torch.float)
        feat_1_mask = [f1 * F.max_pool2d(mask.unsqueeze(1), stride, stride)
                  for (f1, stride) in zip(feat_1, self.strides)]

        feat_1_fpn = self.fpn1(feat_1)  # 4000M
        feat_1_mask = self.fpn1(feat_1_mask)
        feat_2 = self.fpn2(feat_2)  # 4000M

        bs, c, h, w = feat_2.shape
        new_h, new_w = h // 16, w // 16
        y, x = torch.meshgrid(torch.arange(new_h), torch.arange(new_w))
        y = y.float() * 16 + torch.randint(0, 4, (new_h, new_w)).float()  # 随机选择 4x4 网格中的 y 坐标
        x = x.float() * 16 + torch.randint(0, 4, (new_h, new_w)).float()  # 随机选择 4x4 网格中的 x 坐标
        y = 2 * y / (h - 1) - 1  # 归一化到 [-1, 1]
        x = 2 * x / (w - 1) - 1  # 归一化到 [-1, 1]

        # 扩展 x, y 坐标为 (bs, new_h, new_w, 2)
        grid = torch.stack((x, y), dim=-1)  # (new_h, new_w, 2)
        grid = grid.unsqueeze(0).expand(bs, -1, -1, -1).cuda()  # (bs, new_h, new_w, 2)

        feat_1_fpn = F.grid_sample(feat_1_fpn, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        feat_1_mask = F.grid_sample(feat_1_mask, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        feat_2 = F.grid_sample(feat_2, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        B, C, H, W = feat_1_fpn.shape
        tensor1_flat = feat_1_fpn.view(B, C, -1).permute(2, 0, 1)  # 变成 (H*W, B, C)
        tensor2_flat = feat_1_mask.view(B, C, -1).permute(2, 0, 1)  # 变成 (H*W, B, C)

        tensor1_flat, _ = self.cross_attn_1(tensor2_flat, tensor1_flat, tensor1_flat)

        tensor2_flat = feat_2.view(B, C, -1).permute(2, 0, 1)
        attention_output, attention_scores = self.cross_attn_2(tensor2_flat, tensor1_flat, tensor1_flat)  # try
        attention_output = attention_output.permute(1, 2, 0).reshape(B, C, H, W)

        pred = self.seg_head(attention_output)
        return pred



class Matcher(nn.Module):
    def __init__(self):
        super(Matcher, self).__init__()
        self.backbone1 = ResNet(BasicBlock, [2, 2, 2, 2])
        # state_dict = load_state_dict_from_url(model_urls['resnet18'],
        #                                       progress=True)
        # self.backbone1.load_state_dict(state_dict, strict=False)


        self.backbone_mask = ResNet(BasicBlock, [2, 2, 2, 2])
        # state_dict = load_state_dict_from_url(model_urls['resnet18'],
        #                                       progress=True)
        # self.backbone_mask.load_state_dict(state_dict, strict=False)


        self.backbone2 = ResNet(BasicBlock, [2, 2, 2, 2])
        # state_dict = load_state_dict_from_url(model_urls['resnet18'],
        #                                       progress=True)
        # self.backbone2.load_state_dict(state_dict)
        self.seg_head = Decoder(input_channels=256)
        # self.seg_head = nn.Sequential(
        #     nn.UpsamplingBilinear2d(scale_factor=16),
        #     nn.Conv2d(256, 64, kernel_size=3, padding=3 // 2, stride=1, bias=False),
        #     nn.BatchNorm2d(64), nn.ReLU(True),
        #     nn.Conv2d(64, 2, kernel_size=1)
        # )
        self.strides = [4, 8, 16, 32]
        self.fpn1 = FPN_UNet(
            in_channels=[64, 128, 256, 512],
            out_channels=256
        )
        self.fpn2 = FPN_UNet(
            in_channels=[64, 128, 256, 512],
            out_channels=256
        )
        self.self_attn = MultiheadAttention(embed_dim=256, num_heads=8, batch_first=False)
        self.cross_attn = MultiheadAttention(embed_dim=256, num_heads=8, batch_first=False)
        self.addcoords = AddCoords()

    def forward(self, x1, x2, mask):
        feat_1 = self.backbone1(self.addcoords(x1))
        feat_mask = self.backbone_mask(self.addcoords(mask.type_as(x1)))
        feat_1 = [f1 + fm for (f1, fm) in zip(feat_1, feat_mask)]  # follow alpha clip
        # feat_1 = [f1 * F.max_pool2d(mask.sum(dim=1).unsqueeze(1), stride, stride)
        #           for (f1, stride) in zip(feat_1, self.strides)]
        feat_2 = self.backbone2(self.addcoords(x2))

        # ToDo: 占用显存 可能使用方法不太对 先不用了 但是理论上上限高
        # ToDo: grid sampling
        # ToDo: Problem: 没有学到跨区域的信息
        # feat_1 = F.avg_pool2d(self.fpn1(feat_1), 16, 16)  # need to init fpn
        # feat_2 = F.avg_pool2d(self.fpn2(feat_2), 16, 16)
        feat_1 = self.fpn1(feat_1)  # 4000M
        feat_2 = self.fpn2(feat_2)  # 4000M

        bs, c, h, w = feat_1.shape
        new_h, new_w = h // 16, w // 16
        y, x = torch.meshgrid(torch.arange(new_h), torch.arange(new_w))
        y = y.float() * 16 + torch.randint(0, 4, (new_h, new_w)).float()  # 随机选择 4x4 网格中的 y 坐标
        x = x.float() * 16 + torch.randint(0, 4, (new_h, new_w)).float()  # 随机选择 4x4 网格中的 x 坐标
        y = 2 * y / (h - 1) - 1  # 归一化到 [-1, 1]
        x = 2 * x / (w - 1) - 1  # 归一化到 [-1, 1]

        # 扩展 x, y 坐标为 (bs, new_h, new_w, 2)
        grid = torch.stack((x, y), dim=-1)  # (new_h, new_w, 2)
        grid = grid.unsqueeze(0).expand(bs, -1, -1, -1).cuda()  # (bs, new_h, new_w, 2)

        feat_1 = F.grid_sample(feat_1, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        feat_2 = F.grid_sample(feat_2, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        # attn_score = self.attn(feat_1, feat_2)
        # 将输入展平为 (H*W, B, C)
        B, C, H, W = feat_1.shape
        tensor1_flat = feat_1.view(B, C, -1).permute(2, 0, 1)  # 变成 (H*W, B, C)
        tensor2_flat = feat_2.view(B, C, -1).permute(2, 0, 1)  # 变成 (H*W, B, C)

        tensor2_flat, _ = self.self_attn(tensor2_flat, tensor2_flat, tensor2_flat)
        attention_output, attention_scores = self.cross_attn(tensor1_flat, tensor2_flat, tensor2_flat)
        attention_output = attention_output.permute(1, 2, 0).reshape(B, C, H, W)
        # attention_map = attention_scores.mean(dim=1).view(B, H, W)
        # attn_score = self.interactive_attention(feat_1, feat_2)
        pred = self.seg_head(attention_output)
        return pred