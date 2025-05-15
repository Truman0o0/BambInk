import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            ODConv2d(in_features, in_features, kernel_size=3, stride=1, padding=1,
                     reduction=0.0625),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            ODConv2d(in_features, in_features, kernel_size=3, stride=1, padding=1,
                     reduction=0.0625),
            nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)
        self.edge_enhancer = EdgeEnhancer(
                in_dim=in_features,
                norm=nn.InstanceNorm2d,
                act=nn.ReLU
            )

    def forward(self, x):
        out = self.edge_enhancer(x) + self.conv_block(x)
        return out


class Generator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    

class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        # Channel attention
        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)

        # Filter attention
        self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)

        # Spatial attention
        if kernel_size == 1:
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        return torch.sigmoid(self.channel_fc(x) / self.temperature)

    def get_filter_attention(self, x):
        return torch.sigmoid(self.filter_fc(x) / self.temperature)

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(
            x.size(0), 1, 1, self.kernel_size, self.kernel_size)
        return torch.sigmoid(spatial_attention / self.temperature)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return (self.get_channel_attention(x),
                self.get_filter_attention(x),
                self.func_spatial(x),
                1.0)
    

class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, reduction=0.0625):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.attention = Attention(in_planes, out_planes, kernel_size,
                                   groups=groups, reduction=reduction)

        self.weight = nn.Parameter(torch.randn(out_planes, in_planes // groups,
                                               kernel_size, kernel_size))
        self._initialize_weights()

        self._forward_impl = self._forward_impl_pw1x if kernel_size == 1 else self._forward_impl_common

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        channel_att, filter_att, spatial_att, _ = self.attention(x)

        x = x * channel_att

        batch_size, _, height, width = x.size()

        x = x.view(1, -1, height, width)

        spatial_att = spatial_att.view(-1, 1, 1, self.kernel_size, self.kernel_size)

        aggregate_weight = self.weight.unsqueeze(0) * spatial_att  # [B, O, I, K, K]
        aggregate_weight = aggregate_weight.view(-1, self.in_planes // self.groups,
                                                 self.kernel_size, self.kernel_size)

        output = F.conv2d(x, aggregate_weight, None, self.stride, self.padding,
                          self.dilation, self.groups * batch_size)

        output = output.view(batch_size, self.out_planes,
                             output.size(-2), output.size(-1))

        return output * filter_att

    def _forward_impl_pw1x(self, x):
        channel_att, filter_att, _, _ = self.attention(x)
        x = x * channel_att
        output = F.conv2d(x, self.weight, None, self.stride, self.padding,
                          self.dilation, self.groups)
        return output * filter_att

    def forward(self, x):
        return self._forward_impl(x)
    

class Self_Attn(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.in_dim = in_dim
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        
        proj_query = self.query_conv(x).view(B, C // 8, -1).permute(0, 2, 1)
        
        proj_key = self.key_conv(x).view(B, C // 8, -1)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        proj_value = self.value_conv(x).view(B, C, -1)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        out = self.gamma * out + x
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, conv_dim=64):
        super(Discriminator, self).__init__()

        self.conv1 = spectral_norm(nn.Conv2d(in_channels, conv_dim, kernel_size=4, stride=2, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1))
        self.conv3 = spectral_norm(nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=4, stride=2, padding=1))
        self.attn = Self_Attn(conv_dim * 4)
        self.conv4 = spectral_norm(nn.Conv2d(conv_dim * 4, conv_dim * 8, kernel_size=4, stride=1, padding=1))
        self.conv5 = nn.Conv2d(conv_dim * 8, 1, kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.leaky_relu(self.conv1(x)) 
        out = self.leaky_relu(self.conv2(out))  
        out = self.leaky_relu(self.conv3(out)) 
        out = self.attn(out) 
        out = self.leaky_relu(self.conv4(out)) 
        out = self.conv5(out)
        out = torch.mean(out, dim=[2, 3])
        return out


if __name__ == '__main__':
    model = Generator(1)
    a = torch.randn(2, 1, 256, 256)
    print(model(a).shape)