import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Upsampler(nn.Sequential):
    def __init__(self, in_channels, scale, bn=False, act=False, bias=True):
        m = []
        if (scale % 2 == 0) & (scale > 0):  # sclae 为 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(in_channels, 4 * in_channels, kernel_size=3, padding=1, bias=bias))
                # (B, C * K * K, H, W) -> (B, C, H*K, W*K)
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(in_channels))
                if act:
                    m.append(nn.ReLU())
        elif scale == 3:
            m.append(nn.Conv2d(in_channels, 9*in_channels, kernel_size=3, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(in_channels))
            if act:
                m.append(nn.ReLU())
        else:
            raise NotImplementedError
        
        super(Upsampler, self).__init__(*m)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, bias=True, bn=False, res_scale=1):
        super().__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias))
            if bn: m.append(nn.BatchNorm2d(in_channels))
            if i == 0: m.append(nn.ReLU())
        
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
    
    def forward(self, x):
        res = self.body(x).mul(*self.res_scale)
        res += x
        return res

class EDSR(nn.Module):
    def __init__(self, in_channels, out_channels=3, scale_factor=4, 
                 hidden_channels=64, n_blocks=16, kernel_size=3):
        super().__init__()

        # define head module
        m_head = [nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, 
                            padding=kernel_size // 2)]
        
        # define body module
        m_body = []
        for i in range(n_blocks):
            m_body.append(
                ResBlock(hidden_channels, kernel_size=kernel_size, res_scale=1)
            )
        m_body.append(nn.Conv2d(hidden_channels, hidden_channels, 
                                kernel_size=kernel_size, padding=kernel_size//2))
        
        # define tail module, scale_factor意味着输入大小需要翻多少倍
        m_tail = [
            Upsampler(hidden_channels, scale=scale_factor),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        ]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x

class Encoder(nn.Module):
    def __init__(self, in_channel_1, in_channel_2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel_1, in_channel_1 // 2, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channel_2, in_channel_2 // 2, kernel_size=1, bias=False)

        self.relu = nn.ReLU()

        self.last_conv = nn.Sequential(
            nn.Conv2d((in_channel_1 + in_channel_2) // 2, 256, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1)
        )

        self._init_weight()
    
    def forward(self, low_level_x, high_level_x, scale_factor=1):
        """scale_factor: 最后的特征是low_level_x的几倍
        """
        low_level_x = self.conv1(low_level_x)
        low_level_x = self.relu(low_level_x)
        # 将 high-level 特征上采样到 low-level.size * scale_factor
        high_level_x = self.conv2(high_level_x)
        high_level_x = self.relu(high_level_x)

        high_level_x = F.interpolate(high_level_x, size=[i * scale_factor for i in low_level_x.size()[-2:]],
                                     mode='bilinear', align_corners=True)
        if scale_factor > 1:
            low_level_x = F.interpolate(low_level_x, size=[i * scale_factor for i in low_level_x.size()[-2:]],
                                        mode='bilinear', align_corners=True)
        
        x = torch.cat([high_level_x, low_level_x], dim=1)
        x = self.last_conv(x)

        return x
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SRNet(nn.Module):
    """输出的特征大小要和输入图片的大小相同，最后和原始图片进行L1_Loss计算
    """
    def __init__(self, channel_1, channel_2, scale_factor=1) -> None:
        super().__init__()
        self.encoder = Encoder(channel_1, channel_2)

        self.scale_factor = scale_factor # 用于 encoder, 输出特征是low_level的多少倍分辨率
        self.edsr = EDSR(in_channels=64, out_channels=3, scale_factor=8)

    def forward(self, low_level_x, high_level_x):
        x_sr = self.encoder(low_level_x, high_level_x, self.scale_factor)

        x_sr_up = self.edsr(x_sr)

        return x_sr_up