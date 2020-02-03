import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import os

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

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


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
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
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

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
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class FSSD(nn.Module):

    def fuse_module(self, in_ch, size):
        fm = [
            nn.Conv2d(in_ch, 256, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        ]
        return fm

    def fe_module(self, in_ch, out_ch):
        fe = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        return fe
    
    def pred_module(self, in_ch):
        pm =[
            nn.Conv2d(in_ch, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),            
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),            
            nn.ReLU(inplace=True),            
            nn.Conv2d(256, 1024, kernel_size=1, padding=0),
            nn.Conv2d(in_ch, 1024, kernel_size=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        ]
        return pm

    def __init__(self, phase, cfg, use_pred_module=False):
        super(FSSD, self).__init__()

        self.phase = phase
        self.num_classes = cfg['num_classes']
        self.cfg = cfg


        kwargs = {}
        kwargs['groups'] = 32
        kwargs['width_per_group'] = 4
        # 512->256->128->64->32->16        
        base = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs) 

        # Fusion Module
        size = 64
        self.fm0 = nn.ModuleList(self.fuse_module(512, size))          # layer2  # 64 (OUT), 512
        self.fm1 = nn.ModuleList(self.fuse_module(1024, size))         # layer3  # 32 (OUT), 1024
        self.fm2 = nn.ModuleList(self.fuse_module(2048, size))         # layer4  # 16 (OUT), 2048
        self.fm = [self.fm0, self.fm1, self.fm2]
        self.fm_bn = nn.BatchNorm2d(256*3, affine=True)


        source = [512, 1024, 512, 256, 256, 256, 256]
        box = cfg['box']
        
        # Feature Extractor
        self.fe0 = nn.ModuleList(self.fe_module(256*3, source[0]))
        self.fe1 = nn.ModuleList(self.fe_module(source[0], source[1]))
        self.fe2 = nn.ModuleList(self.fe_module(source[1], source[2]))
        self.fe3 = nn.ModuleList(self.fe_module(source[2], source[3]))
        self.fe4 = nn.ModuleList(self.fe_module(source[3], source[4]))
        self.fe5 = nn.ModuleList(self.fe_module(source[4], source[5]))
        self.fe6 = nn.ModuleList(self.fe_module(source[5], source[6])) # check, k4, p1
        self.fe = [self.fe0, self.fe1, self.fe2, self.fe3, self.fe4, self.fe5, self.fe6]

        # prediction module
        self.use_pred_module = use_pred_module
        if self.use_pred_module:
            self.pm0 = nn.ModuleList(self.pred_module(source[0]))
            self.pm1 = nn.ModuleList(self.pred_module(source[1]))
            self.pm2 = nn.ModuleList(self.pred_module(source[2]))
            self.pm3 = nn.ModuleList(self.pred_module(source[3]))
            self.pm4 = nn.ModuleList(self.pred_module(source[4]))
            self.pm5 = nn.ModuleList(self.pred_module(source[5]))
            self.pm6 = nn.ModuleList(self.pred_module(source[6]))
            self.pm = [self.pm0, self.pm1, self.pm2, self.pm3, self.pm4, self.pm5, self.pm6]
            source = [1024, 1024, 1024, 1024, 1024, 1024, 1024]

            
        loc_layers = [
            nn.Conv2d(source[0], box[0]*4, kernel_size=3, padding=1),
            nn.Conv2d(source[1], box[1]*4, kernel_size=3, padding=1),
            nn.Conv2d(source[2], box[2]*4, kernel_size=3, padding=1),
            nn.Conv2d(source[3], box[3]*4, kernel_size=3, padding=1),
            nn.Conv2d(source[4], box[4]*4, kernel_size=3, padding=1),
            nn.Conv2d(source[5], box[5]*4, kernel_size=3, padding=1),
            nn.Conv2d(source[6], box[6]*4, kernel_size=3, padding=1),
        ]
            
        conf_layers = [
            nn.Conv2d(source[0], box[0]*self.num_classes, kernel_size=3, padding=1),
            nn.Conv2d(source[1], box[1]*self.num_classes, kernel_size=3, padding=1),
            nn.Conv2d(source[2], box[2]*self.num_classes, kernel_size=3, padding=1),
            nn.Conv2d(source[3], box[3]*self.num_classes, kernel_size=3, padding=1),
            nn.Conv2d(source[4], box[4]*self.num_classes, kernel_size=3, padding=1),
            nn.Conv2d(source[5], box[5]*self.num_classes, kernel_size=3, padding=1),
            nn.Conv2d(source[6], box[6]*self.num_classes, kernel_size=3, padding=1)
        ]
        
        # NEED TO CHECK THIS
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.size = size

        # SSD network
        self.resnext = base
        self.loc = nn.ModuleList(loc_layers)
        self.conf = nn.ModuleList(conf_layers)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(self.num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        x = self.resnext.conv1(x)       # 256
        x = self.resnext.bn1(x)
        x = self.resnext.relu(x)
        x = self.resnext.maxpool(x)     # 128
        x = self.resnext.layer1(x)
        x = self.resnext.layer2(x)      # 64
        sources.append(x)
        x = self.resnext.layer3(x)      # 32
        sources.append(x)
        x = self.resnext.layer4(x)      # 16
        sources.append(x)

        upsampled = list()
        for k, v in enumerate(self.fm):
            x = sources[k]
            for l in v:
                x = l(x)
                x = F.interpolate(x, 64, mode='bilinear', align_corners=False)
            upsampled.append(x)
        fused_feature = torch.cat(upsampled, 1)

        x = self.fm_bn(fused_feature)

        feature_maps = list()
        for l in self.fe[0]:
            x = l(x)
        feature_maps.append(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        for v in self.fe[1:-1]:
            for l in v:
                x = l(x)
            feature_maps.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        for l in self.fe[-1]:
            x = l(x)
        feature_maps.append(x)            

        # apply multibox head to source layers
        if self.use_pred_module:
            for (x, p, l, c) in zip(feature_maps, self.pm, self.loc, self.conf):
                xs = p[7](x)
                for i in range(0, 6):
                    x = p[i](x)
                x = xs + p[6](x)
                x = p[8](x)
                x = p[9](x)
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        else:
            for (x, l, c) in zip(feature_maps, self.loc, self.conf):
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # print('CF', conf.shape)        

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors                
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            for c in torch.load(base_file, map_location=lambda storage, loc: storage):
                print(c)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

def build_ssd(phase, cfg, use_pred_module=False):
    if phase != "test" and phase != "train" and phase != 'stat':
        print("ERROR: Phase: " + phase + " not recognized")
        return
    return FSSD(phase, cfg, use_pred_module)
