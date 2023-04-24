import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url

# 预权重下载链接
model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    # 'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet50': 'file:///E:centernet-pytorch-main/centernet-pytorch-main/model_data/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

# 3x3卷积定义
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

# 1×1卷积定义
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# block,该BasicBlock在resNet18,resnet34用到
class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# # block,该Bottleneck在resNet50,resnet101用到
class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def ScalaNet(channel_in, channel_out, size):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=size, stride=size),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        nn.AvgPool2d(4, 4)
        )

class SepConv(nn.Module):
    '''
    这个卷积名字起得花里胡哨的，其实总结起来就是输入通道每个通道一个卷积得到和输入通道数相同的特征图，然后再使用若干个1*1的卷积聚合每个特征图的值得到输出特征图。
    假设我们输入通道是16，输出特征图是32，并且使用3*3的卷积提取特征，那么第一步一共需要16*3*3个参数，第二步需要32*16*1*1个参数，一共需要16*3*3+32*16*1*1=656个参数。
    '''

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        # 3*3 s=2 p=1
        # 1*1 s=1 p=1
        # BN
        # Relu
        # 3*3 s=1 p=1
        # 1*1 s=1 p=0
        # BN
        # Relu
        self.op = nn.Sequential(
            # 分组卷积，这里的分组数=输入通道数，那么每个group=channel_in/channel_in=1个通道，就是每个通道进行一个卷积
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            # affine 设为 True 时，BatchNorm 层才会学习参数 gamma 和 beta，否则不包含这两个变量，变量名是 weight 和 bias。
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            # 分组卷积
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        '''
        x-->conv_3x3_s2(分组卷积)-->conv_1x1-->bn-->relu-->conv_3x3(分组卷积)-->conv_1x1-->bn-->relu-->out
        '''
        return self.op(x)

def dowmsampleBottleneck(channel_in, channel_out, stride=2):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        )

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        # 空洞卷积定义
        self.dilation = 1
        # 是否用空洞卷积代替步长，如果不采用空洞卷积，均为False
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups  # 分组卷积分组数
        self.base_width = width_per_group  # 卷积宽度
        # conv1与原始Resnet不同，原始Resnet为7x7卷积
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        # bn层
        self.bn1 = norm_layer(self.inplanes)
        # relu激活函数
        self.relu = nn.ReLU(inplace=True)
        # 最大池化，不过在forward中没有用到
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])  # 尺寸不变
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])  # 尺寸减半
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])  # 尺寸减半
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])  # 尺寸减半
        '''
        此处和原Resnet不同，原Resnet这里是自适应平均池化，然后接一个全连接层。
        scala层的作用是对特征层的H，W做缩放处理，因为要和深层网络中其他Bottleneck输出特征层之间做loss
        '''
        self.scala1 = nn.Sequential(
            # 输入通道64*4=256，输出通道128*4=512
            SepConv(  # 尺寸减半
                channel_in=64 * block.expansion,
                channel_out=128 * block.expansion
            ),
            # 输入通道128*4=512， 输出通道256*4=1024
            SepConv(  # 尺寸减半
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion
            ),
            # 输入通道256*4=1024，输出通道512*4=2048
            SepConv(  # 尺寸减半
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion
            ),
            # 平均池化
            nn.AvgPool2d(4, 4)
        )
        self.scala2 = nn.Sequential(
            # 输入通道128*4=512，输出通道1024
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion,
            ),
            # 输入通道256*4=1024，输出通道512*4=2048
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            # 平均池化
            nn.AvgPool2d(4, 4)
        )
        self.scala3 = nn.Sequential(
            # 输入通道256*4=1024，输出通道512*4=2048
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            # 平均池化
            nn.AvgPool2d(4, 4)
        )
        # 平均池化
        self.scala4 = nn.AvgPool2d(4, 4)

        self.attention1 = nn.Sequential(
            SepConv(  # 尺寸减半
                channel_in=64 * block.expansion,  # 256
                channel_out=64 * block.expansion  # 256
            ),  # 比输入前大两个像素
            nn.BatchNorm2d(64 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 恢复原来尺寸
            nn.Sigmoid()
        )

        self.attention2 = nn.Sequential(
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=128 * block.expansion
            ),
            nn.BatchNorm2d(128 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.attention3 = nn.Sequential(
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=256 * block.expansion
            ),
            nn.BatchNorm2d(256 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)
        self.fc4 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # 残差边采用1x1卷积升维条件，即当步长不为1或者输入通道数不等于输出通道数的时候
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        # layers用来存储每个当前残差层的所有残差块
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        # 仅在第一个bottleneck采用1x1进行升维，其他的bottleneck是直接输入和输出相加
        return nn.Sequential(*layers)

    def forward(self, x):
        # 以x = (1,3,224,224)为例
        feature_list = []
        x = self.conv1(x)  # get 1,64,224,224
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)  # conv2_x  输出256通道  1,256,224,224

        fea1 = self.attention1(x)  # 输出通道为256 224,224
        fea1 = fea1 * x
        feature_list.append(fea1)

        x = self.layer2(x)  # conv3_x  1,512,112,112

        fea2 = self.attention2(x)  # 512,112,112
        fea2 = fea2 * x
        feature_list.append(fea2)

        x = self.layer3(x)  # conv4_x 1,1024,56,56

        fea3 = self.attention3(x)  # 1024,56,56
        fea3 = fea3 * x
        feature_list.append(fea3)

        x = self.layer4(x)  # conv5_x  最深层网络 1,2048,28,28
        feature_list.append(x)

        # feature_list[0].shape is [1,256 224,224] scala1 shape is [1,2048,7,7] view is [1,7*7*2048]
        out1_feature = self.scala1(feature_list[0]).view(x.size(0), -1)  # # 得到新的特征图 对应到论文中的Bottleneck1
        # feature_list[1].shape is [1,512,112,112], scala2 shape is [1,2048,7,7] view is [1,7*7*2048]
        out2_feature = self.scala2(feature_list[1]).view(x.size(0), -1)  # 得到新的特征图 对应到论文中的Bottleneck2
        # feature_list[2].shape is [1,1024,56,56],scala3 shape is [1,2048,7,7] view is [1,7*7*2048]
        out3_feature = self.scala3(feature_list[2]).view(x.size(0), -1)  # 得到新的特征图 对应到论文中的Bottleneck3
        # feature_list[3].shape is [1,2048,28,28],scala4 shape is [1,2048,7,7], view is [1,2048*7*7]
        out4_feature = self.scala4(feature_list[3]).view(x.size(0), -1)  # conv5_x  最深层网络

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)
        # 返回的特征层分别是经过全连接和不仅过全连接的
        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

