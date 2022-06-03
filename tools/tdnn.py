import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from tools.pooling import AttentiveStatsPool, MlutiheadAttentiveStatsPool, MlutiheadAttentiveStatsPool3


import os, math
from pytorch_model_summary import summary

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class SEBlock(nn.Module):

    def __init__(self, channels, bottleneck=128):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):
    
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width      = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1   = nn.BatchNorm1d(width*scale)
        self.nums  = scale - 1

        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns   = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3   = nn.BatchNorm1d(planes)
        self.relu  = nn.ReLU()
        self.width = width
        self.se    = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i==0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i==0:
                out = sp
            else:
                out = torch.cat((out,sp), 1)

        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool1d(kernel_size=pooling_r, stride=pooling_r), 
                    nn.Conv1d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv1d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )

    def forward(self, x):
        identity = x
        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4
        return out

class SCBottleneck(nn.Module):
    """SCNet SCBottleneck
    """
    expansion = 4
    pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, bottleneck_width=32,
                 avd=False, dilation=1, is_first=False,
                 norm_layer=nn.BatchNorm1d):
        super(SCBottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality

        self.conv1_a = nn.Conv1d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv1d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)
        self.avd = avd and (stride > 1 or is_first)

        if self.avd:
            self.avd_layer = nn.AvgPool1d(3, stride, padding=1)
            stride = 1

        self.k1 = nn.Sequential(
                    nn.Conv1d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        groups=cardinality, bias=False),
                    norm_layer(group_width),
                    )

        self.scconv = SCConv(
            group_width, group_width, stride=stride,
            padding=dilation, dilation=dilation,
            groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer)

        self.conv3 = nn.Conv1d(
            group_width * 2, planes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        self.se = SEBlock(planes)

    def forward(self, x):
        residual = x

        out_a= self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        out_a = self.k1(out_a)
        out_b = self.scconv(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        if self.avd:
            out_a = self.avd_layer(out_a)
            out_b = self.avd_layer(out_b)

        x = torch.cat([out_a, out_b], dim=1)
        
        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se(out)
        out += residual
        out = self.relu(out)

        return out


class AFM(nn.Module):
    def __init__(self, planes, reduction_ratio=8) -> None:
        super(AFM, self).__init__()

        self.local_attention = nn.Sequential(
            nn.Conv1d(2*planes, int(planes/reduction_ratio), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(int(planes/reduction_ratio)),
            nn.ReLU(inplace=True),
            nn.Conv1d(int(planes/reduction_ratio), planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(planes),
        )

        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(2*planes, int(planes/reduction_ratio), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(int(planes/reduction_ratio)),
            nn.ReLU(inplace=True),
            nn.Conv1d(int(planes/reduction_ratio), planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(planes),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2):

        x  = torch.cat((input1, input2), dim=1)
        x_local = self.local_attention(x)
        x_global= self.global_attention(x)
        s  = self.sigmoid(x_local+x_global)
        x  = input1.mul(s) + input2.mul(1-s)
        return x


class TDNN(nn.Module):

    def __init__(self, channel, feature_dim, context=True, **kwargs):
        super(TDNN, self).__init__()
        
        self.pooling_way = pooling_way
        self.context     = context

        self.conv1 = nn.Conv1d(feature_dim, channel, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1  = nn.BatchNorm1d(channel)

        self.layer1 = Bottle2neck(channel, channel, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(channel, channel, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(channel, channel, kernel_size=3, dilation=4, scale=8)
        

        self.layer5 = nn.Conv1d(3*channel, channel*3, kernel_size=1)
        
        cat_channel = channel*3
        
        self.pooling = AttentiveStatsPool(cat_channel, 128, context=True)
        self.bn5 = nn.BatchNorm1d(cat_channel * 2)
        self.fc6 = nn.Linear(cat_channel * 2, 256)


    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1+x)
        x3 = self.layer3(x2+x1+x)

        x = self.layer5(torch.cat((x1,x2,x3), dim=1))
        x = self.relu(x)
        x = self.bn5(self.pooling(x))
        x = self.fc6(x)
        
        return x


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class ChannelClassifier(nn.Module):
    def __init__(self, enc_dim, nclasses, lambda_=0.05, ADV=True):
        super(ChannelClassifier, self).__init__()
        self.adv = ADV
        if self.adv:
            self.grl = GradientReversal(lambda_)
        self.classifier = nn.Sequential(nn.Linear(enc_dim, enc_dim // 2),
                                        nn.Dropout(0.3),
                                        nn.ReLU(),
                                        nn.Linear(enc_dim // 2, nclasses),
                                        nn.ReLU())

    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

    def forward(self, x):
        if self.adv:
            x = self.grl(x)
        return self.classifier(x)



if __name__ =="__main__":
    
    os.environ["CUDA_VISABLE_DEVICE"] = "1"
    print(summary(TDNN(channel=512, feature_dim=60, pooling_way="ASP",context=True), torch.randn((32,60,400)), show_input=False))
