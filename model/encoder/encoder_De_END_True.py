import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
from model.encoder.base_encoder import BaseEncoder


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)

class UpConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, scale_factor=2):
        super(UpConvBNReLU, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.up(x)
        return self.relu(self.bn(self.conv(x)))


class Encoder(BaseEncoder):
    def __init__(self, config: HiDDenConfiguration,input_size):
        super(Encoder, self).__init__(config,input_size)
        in_feature_dim = config.message_length * 2
        output_channels = self.input_size
        alpha=0.1

        self.alpha = alpha
        self.linear = LinearBlock(in_feature_dim, (self.H // 8) * (self.W // 8))
        self.conv1 = ConvBNRelu(1, 64)
        self.upconv1 = UpConvBNReLU(64, 64)
        self.upconv2 = UpConvBNReLU(64, 64)
        self.upconv3 = UpConvBNReLU(64, 64)
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, Io,concatenated_feature ):
        batch_size = concatenated_feature.shape[0]
        # print(concatenated_feature.shape)
        x = self.linear(concatenated_feature)
        x = x.view(batch_size, 1, self.H // 8, self.W // 8)
        x = self.conv1(x)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        R = self.final_conv(x)
        
        # Generate watermarked image
        R = self.alpha * R
        return R 