import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from torchvision.models.resnet import ResNet, Bottleneck
from torchvision._internally_replaced_utils import load_state_dict_from_url

MODEL_URL = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)
        if x.shape[1] != y.shape[1]:  # for the uplayers.
            return y

        return self.relu(x+y)


class FeatureEncoderBasic(nn.Module):
    def __init__(self, output_dim=128, norm_fn='group'):
        super(FeatureEncoderBasic, self).__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # top-down feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 96, stride=2)
        self.out_cov2 = nn.Conv2d(96, 96, kernel_size=1)
        self.layer3 = self._make_layer(96, 128, stride=2)
        self.out_cov3 = nn.Conv2d(128, 128, kernel_size=1)
        self.layer4 = self._make_layer(128, 160, stride=2)
        self.out_cov4 = nn.Conv2d(160, 160, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_dim, dim, stride=1):
        layer1 = ResidualBlock(in_dim, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
            if y is not None:
                y = torch.cat(y, dim=0)

        x = self.conv1(x)
        if y is not None:
            x = x + self.conv1a(y)
        x = self.norm1(x)
        x = self.relu1(x)

        # enc_out1 -> 1/2 resolution, enc_out2 -> 1/4 resolution
        # enc_out3 -> 1/8 resolution, enc_out4 -> 1/16 resolution

        x = self.layer1(x)
        x = self.layer2(x)
        enc_out2 = self.out_cov2(x)
        x = self.layer3(x)
        enc_out3 = self.out_cov3(x)
        x = self.layer4(x)
        enc_out4 = self.out_cov4(x)

        if is_list:
            enc_out4 = torch.split(enc_out4, [batch_dim, batch_dim], dim=0)
            enc_out3 = torch.split(enc_out3, [batch_dim, batch_dim], dim=0)
            enc_out2 = torch.split(enc_out2, [batch_dim, batch_dim], dim=0)

        return [enc_out4, enc_out3, enc_out2]


class FeatureEncoderCCMR(nn.Module):
    def __init__(self, output_dim=128, norm_fn='group', four_scales=False):
        super(FeatureEncoderCCMR, self).__init__()
        self.four_scales = four_scales
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # top-down feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 96, stride=2)
        self.layer3 = self._make_layer(96, 128, stride=2)
        self.layer4 = self._make_layer(128, 160, stride=2)
        self.conv2 = nn.Conv2d(160, 160, kernel_size=1)

        # bottom-up feature consolidation
        self.up_layer2 = self._make_layer(160 + 128, 128, stride=1)
        self.after_up_layer2_conv = nn.Conv2d(128, 128, kernel_size=1)
        self.up_layer1 = self._make_layer(128 + 96, 96, stride=1)
        self.after_up_layer1_conv = nn.Conv2d(96, 96, kernel_size=1)
        if self.four_scales:
            self.up_layer0 = self._make_layer(96 + 64, 64, stride=1)
            self.after_up_layer0_conv = nn.Conv2d(64, 64, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_dim, dim, stride=1):
        layer1 = ResidualBlock(in_dim, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
            if y is not None:
                y = torch.cat(y, dim=0)

        x = self.conv1(x)
        if y is not None:
            x = x + self.conv1a(y)
        x = self.norm1(x)
        x = self.relu1(x)

        # enc_out1 -> 1/2 resolution, enc_out2 -> 1/4 resolution
        # enc_out3 -> 1/8 resolution, enc_out4 -> 1/16 resolution

        enc_out1 = x = self.layer1(x)
        enc_out2 = x = self.layer2(x)
        enc_out3 = x = self.layer3(x)

        x = self.layer4(x)
        enc_out4 = x = self.conv2(x)

        # uplayer2 (1/16 -> 1/8 resolution)
        x = TF.resize(x, enc_out3.shape[-2:])
        x = torch.cat((x, enc_out3), dim=1)
        up2_out = x = self.after_up_layer2_conv(self.up_layer2(x))

        # uplayer1 (1/8 -> 1/4 resolution)
        x = TF.resize(x, enc_out2.shape[-2:])
        x = torch.cat((x, enc_out2), dim=1)
        up1_out = x = self.after_up_layer1_conv(self.up_layer1(x))

        if self.four_scales:
            # uplayer0 (1/4 -> 1/2 resolution)
            x = TF.resize(x, enc_out1.shape[-2:])
            x = torch.cat((x, enc_out1), dim=1)
            up0_out = x = self.after_up_layer0_conv(self.up_layer0(x))

        if is_list:
            enc_out4 = torch.split(enc_out4, [batch_dim, batch_dim], dim=0)
            up2_out = torch.split(up2_out, [batch_dim, batch_dim], dim=0)
            up1_out = torch.split(up1_out, [batch_dim, batch_dim], dim=0)
            if self.four_scales:
                up0_out = torch.split(up0_out, [batch_dim, batch_dim], dim=0)

        return [enc_out4, up2_out, up1_out, up0_out] if self.four_scales else [enc_out4, up2_out, up1_out]


class ContextEncoderBasic(nn.Module):
    def __init__(self, output_dim=512, channels=[64, 96, 128, 192, 256], norm_fn='group', four_scales=False):
        super(ContextEncoderBasic, self).__init__()
        self.four_scales = four_scales
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels[0])
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(channels[0])
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(channels[0])
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # top-down feature extraction
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(channels[0], channels[1], stride=1)
        if self.four_scales:
            self.out_cov1 = nn.Conv2d(channels[1], output_dim, kernel_size=1)
        self.layer2 = self._make_layer(channels[1], channels[2], stride=2)
        self.out_cov2 = nn.Conv2d(channels[2], output_dim, kernel_size=1)
        self.layer3 = self._make_layer(channels[2], channels[3], stride=2)
        self.out_cov3 = nn.Conv2d(channels[3], output_dim, kernel_size=1)
        self.layer4 = self._make_layer(channels[3], channels[4], stride=2)
        self.out_cov4 = nn.Conv2d(channels[4], output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_dim, dim, stride=1):
        layer1 = ResidualBlock(in_dim, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
            if y is not None:
                y = torch.cat(y, dim=0)

        x = self.conv1(x)
        if y is not None:
            x = x + self.conv1a(y)
        x = self.norm1(x)
        x = self.relu1(x)

        # enc_out1 -> 1/2 resolution, enc_out2 -> 1/4 resolution
        # enc_out3 -> 1/8 resolution, enc_out4 -> 1/16 resolution

        x = self.layer1(x)
        if self.four_scales:
            enc_out1 = self.out_cov1(x)
        x = self.layer2(x)
        enc_out2 = self.out_cov2(x)
        x = self.layer3(x)
        enc_out3 = self.out_cov3(x)
        x = self.layer4(x)
        enc_out4 = self.out_cov4(x)

        if is_list:
            enc_out4 = torch.split(enc_out4, [batch_dim, batch_dim], dim=0)
            enc_out3 = torch.split(enc_out3, [batch_dim, batch_dim], dim=0)
            enc_out2 = torch.split(enc_out2, [batch_dim, batch_dim], dim=0)
            if self.four_scales:
                enc_out1 = torch.split(enc_out1, [batch_dim, batch_dim], dim=0)

        return [enc_out4, enc_out3, enc_out2, enc_out1] if self.four_scales else [enc_out4, enc_out3, enc_out2]


class ContextEncoderFPN(nn.Module):
    def __init__(self, output_dim=512, channels=[64, 96, 128, 192, 256], norm_fn='batch'):
        super(ContextEncoderFPN, self).__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels[0])
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(channels[0])
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(channels[0])
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(channels[0], channels[1], stride=1)
        self.layer2 = self._make_layer(channels[1], channels[2], stride=2)
        self.layer3 = self._make_layer(channels[2], channels[3], stride=2)
        self.layer4 = self._make_layer(channels[3], channels[4], stride=2)
        self.conv2 = nn.Conv2d(channels[4], output_dim, kernel_size=1)

        self.conv1x1_l3 = nn.Conv2d(channels[3], output_dim, kernel_size=1)
        self.up_layer2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1) # self.conv_l3

        self.conv1x1_l2 = nn.Conv2d(channels[2], output_dim, kernel_size=1)
        self.up_layer1 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1) # self.conv_l2

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_dim, dim, stride=1):
        layer1 = ResidualBlock(in_dim, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # return [enc_out3, up2_out]
        enc_out1 = x = self.layer1(x)
        enc_out2 = x = self.layer2(x)
        enc_out3 = x = self.layer3(x)

        x = self.layer4(x)
        enc_out4 = x = self.conv2(x)

        #uplayer2:
        enc_out4_resized = TF.resize(enc_out4, enc_out3.shape[-2:])
        enc3_1x1_out = self.conv1x1_l3(enc_out3)
        up2_out = self.up_layer2(enc3_1x1_out + enc_out4_resized)

        #uplayer1:
        up2_out_resized = TF.resize(up2_out, enc_out2.shape[-2:])
        enc2_1x1_out = self.conv1x1_l2(enc_out2)
        up1_out = self.up_layer1(enc2_1x1_out + up2_out_resized)

        return [enc_out4, up2_out, up1_out]


class ContextEncoderCCMR(nn.Module):
    def __init__(self, output_dim=512, channels=[64, 96, 128, 192, 256], norm_fn='group'):
        super(ContextEncoderCCMR, self).__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels[0])
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(channels[0])
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(channels[0])
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # top-down feature extraction
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(channels[0], channels[1], stride=1)
        self.layer2 = self._make_layer(channels[1], channels[2], stride=2)
        self.layer3 = self._make_layer(channels[2], channels[3], stride=2)
        self.layer4 = self._make_layer(channels[3], channels[4], stride=2)
        self.conv2 = nn.Conv2d(channels[4], output_dim, kernel_size=1)

        # bottom-up feature consolidation
        self.up_layer2 = self._make_layer(output_dim + channels[3], channels[3], stride=1)
        self.after_up_layer2_conv = nn.Conv2d(channels[3], output_dim, kernel_size=1)
        self.up_layer1 = self._make_layer(output_dim + channels[2], channels[2], stride=1)
        self.after_up_layer1_conv = nn.Conv2d(channels[2], output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_dim, dim, stride=1):
        layer1 = ResidualBlock(in_dim, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
            if y is not None:
                y = torch.cat(y, dim=0)

        x = self.conv1(x)
        if y is not None:
            x = x + self.conv1a(y)
        x = self.norm1(x)
        x = self.relu1(x)

        # enc_out1 -> 1/2 resolution, enc_out2 -> 1/4 resolution
        # enc_out3 -> 1/8 resolution, enc_out4 -> 1/16 resolution

        enc_out1 = x = self.layer1(x)
        enc_out2 = x = self.layer2(x)
        enc_out3 = x = self.layer3(x)

        x = self.layer4(x)
        enc_out4 = x = self.conv2(x)

        # uplayer2 (1/16 -> 1/8 resolution)
        x = TF.resize(x, enc_out3.shape[-2:])
        x = torch.cat((x, enc_out3), dim=1)
        up2_out = x = self.after_up_layer2_conv(self.up_layer2(x))

        # uplayer1 (1/8 -> 1/4 resolution)
        x = TF.resize(x, enc_out2.shape[-2:])
        x = torch.cat((x, enc_out2), dim=1)
        up1_out = x = self.after_up_layer1_conv(self.up_layer1(x))

        if is_list:
            enc_out4 = torch.split(enc_out4, [batch_dim, batch_dim], dim=0)
            up2_out = torch.split(up2_out, [batch_dim, batch_dim], dim=0)
            up1_out = torch.split(up1_out, [batch_dim, batch_dim], dim=0)

        return [enc_out4, up2_out, up1_out]


class ContextEncoderResnet(ResNet):
    def __init__(self, output_dim=512):
        super(ContextEncoderResnet, self).__init__(Bottleneck, [3, 4, 6, 3], norm_layer=nn.BatchNorm2d)
        state_dict = load_state_dict_from_url(MODEL_URL)
        self.load_state_dict(state_dict)

        # bottom-up feature consolidation
        self.out_conv1 = nn.Conv2d(256, output_dim, kernel_size=1)
        self.out_conv2 = nn.Conv2d(512, output_dim, kernel_size=1)
        self.out_conv3 = nn.Conv2d(1024, output_dim, kernel_size=1)

    def _forward_impl(self, x):
        # See note [TorchScript super()]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)   # output: 1/4  resolution,  256 channels
        enc_out1 = self.out_conv1(x)
        x = self.layer2(x)   # output: 1/8  resolution,  512 channels
        enc_out2 = self.out_conv2(x)
        x = self.layer3(x)   # output: 1/16 resolution, 1024 channels
        enc_out3 = self.out_conv3(x)

        return [enc_out3, enc_out2, enc_out1]

    def forward(self, x):
        """ Input img, Output [1/16, 1/8, 1/4, 1/2] feature map """
        return self._forward_impl(x)


class ContextEncoderResnetFeatureAggregation(ResNet):
    def __init__(self, output_dim=512):
        super(ContextEncoderResnetFeatureAggregation, self).__init__(Bottleneck, [3, 4, 6, 3], norm_layer=nn.BatchNorm2d)
        state_dict = load_state_dict_from_url(MODEL_URL)
        self.load_state_dict(state_dict)

        # bottom-up feature consolidation
        self.up_conv4 = nn.Conv2d(2048, output_dim, kernel_size=3, padding=1)

        self.up_conv3 = nn.Conv2d(output_dim + 1024, output_dim, kernel_size=3, padding=1)
        self.out_conv3 = nn.Conv2d(output_dim, output_dim, kernel_size=1)

        self.up_conv2 = nn.Conv2d(output_dim + 512, output_dim, kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=1)

        self.up_conv1 = nn.Conv2d(output_dim + 256, output_dim, kernel_size=3, padding=1)
        self.out_conv1 = nn.Conv2d(output_dim, output_dim, kernel_size=1)

    def _forward_impl(self, x):
        # See note [TorchScript super()]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        enc_out1 = x = self.layer1(x)   # output: 1/4  resolution,  256 channels
        enc_out2 = x = self.layer2(x)   # output: 1/8  resolution,  512 channels
        enc_out3 = x = self.layer3(x)   # output: 1/16 resolution, 1024 channels
        x = self.layer4(x)              # output: 1/32 resolution, 2048 channels

        x = self.relu(self.up_conv4(x))

        # uplayer3 (1/32 -> 1/16 resolution)
        x = TF.resize(x, enc_out3.shape[-2:])
        x = torch.cat((x, enc_out3), dim=1)
        x = self.relu(self.up_conv3(x))
        up3_out = x = self.out_conv3(x)

        # uplayer2 (1/16 -> 1/8 resolution)
        x = TF.resize(x, enc_out2.shape[-2:])
        x = torch.cat((x, enc_out2), dim=1)
        x = self.relu(self.up_conv2(x))
        up2_out = x = self.out_conv2(x)

        # uplayer1 (1/8 -> 1/4 resolution)
        x = TF.resize(x, enc_out1.shape[-2:])
        x = torch.cat((x, enc_out1), dim=1)
        x = self.relu(self.up_conv1(x))
        up1_out = x = self.out_conv1(x)

        return [up3_out, up2_out, up1_out]

    def forward(self, x):
        """ Input img, Output [1/16, 1/8, 1/4, 1/2] feature map """
        return self._forward_impl(x)
