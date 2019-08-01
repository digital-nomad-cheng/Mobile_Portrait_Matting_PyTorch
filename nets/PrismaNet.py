"""
Implementation of PrismaNet which I found on the blog of Prisma.
I improve the model by adding skip connection to it
# Todo: change all the ReLU layers to ReLU6
"""

import re
import math
import argparse

import torch
from torch import nn
from torch.nn import functional as F
import torch.onnx
from torchsummary import summary

def conv_bn(inp, oup, stride, relu6=True):
    if relu6:
        relu = nn.ReLU6(inplace=True)
    else:
        relu = nn.ReLU(inplace=True)
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        relu
    )

class _Interpolate(nn.Module):
    """Utility function for using nn.functional.interpolate() in nn.Sequential()"""
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=False):
        super(_Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
                                                            
    def forward(self, x): 
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x

class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, num_layers=2, bias=False):
        super(SeparableConv2D, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias))
        self.layers.append(nn.BatchNorm2d(in_channels))
        self.layers.append(nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias))
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_layers-1):
            self.layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, dilation, groups=out_channels, bias=bias))
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.Conv2d(out_channels, out_channels, 1, 1, 0, 1, 1, bias=bias))
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU(inplace=True))
    
    def forward(self, x):
        for i in range(5):
            x = self.layers[i](x)
        
        # add skip connection here
        count = 0
        for i in range(5, len(self.layers)):    
            if i % 5 == 0:
                x0 = x
            x= self.layers[i](x)
            count += 1
            if count == 5:
                x = x0 + x
                count = 0
        
        return x

class PrismaNet(nn.Module):
    
    def __init__(self):
        super(PrismaNet, self).__init__()
        
        self.conv1 = conv_bn(3,  16, 1, True)
        self.separable_block1 = SeparableConv2D(16, 16, num_layers=2)
        self.conv2 = conv_bn(16, 32, 2, True)
        self.separable_block2 = SeparableConv2D(32, 32, num_layers=2)
        self.conv3 = conv_bn(32, 64, 2, True)
        self.separable_block3 = SeparableConv2D(64, 64, num_layers=2)
        self.conv4 = conv_bn(64, 128, 2, True)
        self.separable_block4 = SeparableConv2D(128, 128, num_layers=4)
        self.conv5 = conv_bn(128, 128, 2, True)
        self.separable_block5 = SeparableConv2D(128, 128, num_layers=12)
        self.up = _Interpolate()
       
        self.separable_block6 = SeparableConv2D(128, 64, num_layers=6)
        self.separable_block7 = SeparableConv2D(64, 32, num_layers=6)
        self.separable_block8 = SeparableConv2D(32, 16, num_layers=6)
        self.separable_block9 = SeparableConv2D(16, 16, num_layers=6)

        self.conv_last = nn.Conv2d(16, 2, 3, 1, padding=1) 
        self.softmax = nn.Softmax(dim=-1)
        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.separable_block1(x)
        
        b1 = self.conv2(x)
        b1 = self.separable_block2(b1)

        b2 = self.conv3(b1)
        b2 = self.separable_block3(b2)

        b3 = self.conv4(b2)
        b3 = self.separable_block4(b3)

        b4 = self.conv5(b3)
        b4 = self.separable_block5(b4)
        
        up1 = self.up(b4)
        up1 = torch.add(up1, b3)
        up1 = self.separable_block6(up1)

        up2 = self.up(up1)
        up2 = torch.add(up2, b2)
        up2 = self.separable_block7(up2)

        up3 = self.up(up2)
        up3 = torch.add(up3, b1)
        up3 = self.separable_block8(up3)

        up4 = self.up(up3)
        up4 = torch.add(up4, x)
        up4 = self.separable_block9(up4)

        up4 = self.conv_last(up4)
        up4 = up4.permute(0, 2, 3, 1)
        output = self.softmax(up4)
        
        return output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == "__main__":
    model = PrismaNet()
    summary(model.cuda(), (3, 256, 256))
  
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_path', type=str, default='models/PrismaNet2.pth', help='input model path')
    args = parser.parse_args()
    input_model_path = args.input_model_path
    output_model_path = re.sub(r'pth$', 'onnx', input_model_path)
    
    model.load_state_dict(torch.load(input_model_path))
    torch.onnx.export(model, torch.randn(1, 3, 256, 256).cuda(), output_model_path)

