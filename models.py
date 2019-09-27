import math

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter

from config import device, num_classes

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, use_se=True, im_size=112):
        self.inplanes = 64
        self.use_se = use_se
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()

        if im_size == 112:
            self.fc = nn.Linear(512 * 7 * 7, 512)
        else:  # 224
            self.fc = nn.Linear(512 * 14 * 14, 512)
        self.bn3 = nn.BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)

        return x


def resnet18(args, **kwargs):
    model = ResNet(IRBlock, [2, 2, 2, 2], use_se=args.use_se, **kwargs)
    if args.pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(args, **kwargs):
    model = ResNet(IRBlock, [3, 4, 6, 3], use_se=args.use_se, **kwargs)
    if args.pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(args, **kwargs):
    model = ResNet(IRBlock, [3, 4, 6, 3], use_se=args.use_se, im_size=args.im_size, **kwargs)
    if args.pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(args, **kwargs):
    model = ResNet(IRBlock, [3, 4, 23, 3], use_se=args.use_se, **kwargs)
    if args.pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(args, **kwargs):
    model = ResNet(IRBlock, [3, 8, 36, 3], use_se=args.use_se, **kwargs)
    if args.pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class ArcMarginModel(nn.Module):
    def __init__(self, args):
        super(ArcMarginModel, self).__init__()

        self.weight = Parameter(torch.FloatTensor(num_classes, args.emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = args.easy_margin
        self.m = args.margin_m
        self.s = args.margin_s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


from torchvision import models


class FrameDetectionModel(nn.Module):
    def __init__(self):
        super(FrameDetectionModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove linear layer (since we're not doing classification)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, 8)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        x = self.resnet(images)  # [N, 2048, 1, 1]
        x = x.view(-1, 2048)  # [N, 2048]
        x = self.fc(x)
        x = self.sigmoid(x)  # [N, 8]
        return x


class FaceAttributeModel(nn.Module):
    def __init__(self):
        super(FaceAttributeModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, 17)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, images):
        x = self.resnet(images)  # [N, 2048, 1, 1]
        x = x.view(-1, 2048)  # [N, 2048]
        x = self.fc(x)
        reg = self.sigmoid(x[:, :5])  # [N, 8]
        expression = self.softmax(x[:, 5:8])
        gender = self.softmax(x[:, 8:10])
        glasses = self.softmax(x[:, 10:13])
        race = self.softmax(x[:, 13:17])
        return reg, expression, gender, glasses, race


class FaceExpressionModel(nn.Module):
    def __init__(self):
        super(FaceExpressionModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.dropout = nn.Dropout(0.2)
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(2048, 7)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, images):
        x = self.resnet(images)  # [N, 2048, 1, 1]
        x = self.dropout(x)
        x = self.avgpool(x)
        x = x.view(-1, 2048)  # [N, 2048]
        x = self.fc(x)
        out = self.softmax(x)
        return out


class EastModel(nn.Module):
    def __init__(self, args):
        super(EastModel, self).__init__()

        if args.network == 'r18':
            self.resnet = resnet18(args)
        elif args.network == 'r34':
            self.resnet = resnet34(args)
        elif args.network == 'r50':
            self.resnet = resnet50(args)
        elif args.network == 'r101':
            self.resnet = resnet101(args)
        else:  # args.network == 'r152':
            self.resnet = resnet152(args)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(in_channels=3072, out_channels=128, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=640, out_channels=64, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(in_channels=320, out_channels=64, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)

        self.conv7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)

        self.conv8 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)

        self.conv9 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.unpool1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.unpool3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, images):
        _, f = self.resnet(images)

        h = f[3]  # bs 2048 w/32 h/32
        g = (self.unpool1(h))  # bs 2048 w/16 h/16
        c = self.conv1(torch.cat((g, f[2]), 1))
        c = self.bn1(c)
        c = self.relu(c)

        h = self.conv2(c)  # bs 128 w/16 h/16
        h = self.bn2(h)
        h = self.relu(h)
        g = self.unpool2(h)  # bs 128 w/8 h/8
        c = self.conv3(torch.cat((g, f[1]), 1))
        c = self.bn3(c)
        c = self.relu(c)

        h = self.conv4(c)  # bs 64 w/8 h/8
        h = self.bn4(h)
        h = self.relu(h)
        g = self.unpool3(h)  # bs 64 w/4 h/4
        c = self.conv5(torch.cat((g, f[0]), 1))
        c = self.bn5(c)
        c = self.relu(c)

        h = self.conv6(c)  # bs 32 w/4 h/4
        h = self.bn6(h)
        h = self.relu(h)
        g = self.conv7(h)  # bs 32 w/4 h/4
        g = self.bn7(g)
        g = self.relu(g)

        score = self.conv8(g)  # bs 1 w/4 h/4
        score = self.sigmoid(score)
        geo_map = self.conv9(g)
        geo_map = self.sigmoid(geo_map) * 512
        angle_map = self.conv10(g)
        angle_map = self.sigmoid(angle_map)
        angle_map = (angle_map - 0.5) * math.pi / 2

        geo = torch.cat((geo_map, angle_map), 1)  # bs 5 w/4 w/4
        return score, geo
