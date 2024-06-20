import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor, einsum
from einops import rearrange
import einops
from .ts_resnet import TwoStreamResNet

from mmcv.cnn import ConvModule
from mmdet.models.utils import make_divisible
from mmrotate.registry import MODELS

class Involution(nn.Module):
    """ Involution 对每一个空间位置都是不同的kernel, 无法改变通道维度
        Convolution 对每一个通道位置都是不同的kernel
    """
    def __init__(self, channels, kernel_size=7, dilation=1, stride=1, 
                 group_channels=16, reduce_ratio=4) -> None:
        super().__init__()
        assert not (channels % group_channels or channels % reduce_ratio)

        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride

        # 每组多少个通道
        self.group_channels = group_channels
        self.groups = channels // group_channels
        # reduce channels
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, channels // reduce_ratio, 1),
            nn.BatchNorm2d(channels // reduce_ratio),
            nn.ReLU()
        )
        # span channels
        self.span = nn.Conv2d(
            channels // reduce_ratio,
            self.groups * kernel_size ** 2, 1)
        self.down_sample = nn.AvgPool2d(stride) if stride != 1 else nn.Identity()
        self.unfold = nn.Unfold(kernel_size, dilation=dilation, padding=dilation*(kernel_size-1) // 2, stride=stride)
    
    def forward(self, x):
        # generate involution kernel: (B, G*K*K, H, W)
        weight_matrix = self.span(self.reduce(self.down_sample(x)))
        b, _, h, w = weight_matrix.shape

        # unfold input (b, C*K*K, h, w)
        x_unfolded = self.unfold(x)
        # (b, G, C//G, K*K, h, w)
        x_unfolded = x_unfolded.reshape(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        #  (b,G*K*K,h,w) -> (b,G,1,K*K,h,w)
        weight_matrix = weight_matrix.reshape(b, self.groups, 1, self.kernel_size**2, h, w)
        # (b, G, C//G, h, w)
        mul_add = (weight_matrix * x_unfolded).sum(dim=3)
        # (b, C, H, W)
        out = mul_add.reshape(b, self.channels, h, w)
        return out 

class Attention(nn.Module):
    def __init__(self, d_model, d_k, d_v, heads, sr_ratio,
                 attn_drop=.1, resid_drop=.1) -> None:
        super().__init__()
        assert d_model % heads == 0
        self.d_model = d_model
        self.d_k = d_model // heads
        self.d_v = d_model // heads
        self.heads = heads

        self.scale = self.d_v ** -0.5

        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, d_model*2)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(resid_drop)

        self.out = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.sr_ratio = sr_ratio

        # Spatial Reduction 实现等同于一个卷积层
        if sr_ratio > 1:
            self.spatial_reduce = nn.Conv2d(d_model, d_model, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(d_model)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        # (B, heads, H*W, C//heads)
        q = self.q(x).reshape(B, -1, self.heads, C // self.heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.spatial_reduce(x_).reshape(B, C, -1).permute(0, 2, 1)  # (B, h*w, C)
            kv = self.kv(x_).reshape(B, -1, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, H*W, h*w)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        out = self.out(out)

        return out
    
class InceptionBottleneck(nn.Module):
    def __init__(self, in_channels,
                 out_channels = None, 
                 kernel_sizes = [3, 3, 5, 5],
                 dilations = [1, 8, 2, 3],
                 expansion = 1.0,
                 sr_ratio=1,
                 ratio=16,
                 add_identity = True) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(expansion * out_channels), 8)
        assert len(kernel_sizes) == len(dilations)
        self.num_branch = len(kernel_sizes) + 1
        self.pre_conv = ConvModule(in_channels, hidden_channels, kernel_size=1, 
                                   norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))

        self.branches = nn.ModuleList()

        for i in range(len(kernel_sizes)):
            self.branches.append(
                nn.Sequential(
                    Involution(hidden_channels,
                               kernel_size=kernel_sizes[i],
                               dilation=dilations[i]),
                    nn.BatchNorm2d(hidden_channels),
                    nn.SiLU()
                ))
        
        self.attn = Attention(hidden_channels, hidden_channels, hidden_channels, 
                              heads=8, sr_ratio=sr_ratio)
        self.add_identity = add_identity and in_channels == out_channels
        self.post_conv = ConvModule(hidden_channels * self.num_branch, out_channels, kernel_size=1,
                                    norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.f1 = nn.Conv2d(self.num_branch * hidden_channels, 
                            self.num_branch * hidden_channels // ratio,
                            kernel_size=1, bias=False)
        self.act = nn.SiLU()
        self.f2 = nn.Conv2d(self.num_branch * hidden_channels // ratio,
                            out_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.final_conv = ConvModule(out_channels, out_channels, kernel_size=1,
                                     norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))

    def forward(self, x):
        B, C, H, W = x.shape
        short_cut = x
        x = self.pre_conv(x)
        out = []
        for i in range(self.num_branch-1):
            out.append(self.branches[i](x))
        out.append(self.attn(x))
        out = torch.cat(out, dim=1)  # (B, K * hidden_channels, H, W)
        tmp = self.post_conv(out)
        if self.add_identity:
            tmp = tmp + short_cut
        # channel attention
        avg_out = self.f2(self.act(self.f1(self.avg_pool(out))))
        max_out = self.f2(self.act(self.f1(self.max_pool(out))))
        out = self.sigmoid(avg_out+max_out)
        out = out * tmp
        out = self.final_conv(out)

        return out

class LowPassModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6)) -> None:
        super().__init__()
        self.stages = nn.ModuleList([self._make_stages(size) for size in sizes])
        self.act = nn.SiLU()
        ch = in_channels // 4
        self.channel_splits = [ch, ch, ch, ch]
    
    def _make_stages(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(prior)
    
    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        feats = torch.split(feats, self.channel_splits, dim=1)
        priors = [F.interpolate(input=self.stages[i](feats[i]), size=(h, w), mode='bilinear', align_corners=True) for i in range(4)]
        feats = torch.cat(priors, 1)

        return feats

class FilterModule(nn.Module):
    def __init__(self, in_channels, head=8, window={3:2, 5:3, 7:3}) -> None:
        super().__init__()
        head_channel = in_channels // head
        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            pad_size = cur_window // 2
            cur_conv = nn.Conv2d(
                cur_head_split * head_channel,
                cur_head_split * head_channel,
                kernel_size=cur_window,
                padding=pad_size,
                groups=cur_head_split * head_channel
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        # head_split 要和 num_heads 数量对应
        self.channel_splits = [x * head_channel for x in self.head_splits]
        self.low_pass = LowPassModule(in_channels)
    
    def forward(self, q, v, size):
        B, h, N, Ch = q.shape
        H, W = size
        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img = rearrange(v, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W)
        low_freq = self.low_pass(v_img)
        # Split according to the channels
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        high_freq = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        high_freq = torch.cat(high_freq, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        low_freq = rearrange(low_freq, "B (h Ch) H W -> B h (H W) Ch", h=h)
        high_freq = rearrange(high_freq, "B (h Ch) H W -> B h (H W) Ch", h=h)

        dynamic_filter = q * high_freq + low_freq

        return dynamic_filter

class FrequencyFilter(nn.Module):
    def __init__(self, d_model, 
                 num_heads=8, 
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.1,
                 proj_drop=0.1
                 ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = d_model // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

        self.filter_dynamic = FilterModule(d_model, head=num_heads, window={3:2, 5:3, 7:3})
    
    def forward(self, x):
        # (B, C, H, W)
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        # Generate QKV
        # (3, B, num_heads, H*W, C // heads)
        qkv = self.qkv(x).reshape(B, H*W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Factorized attention
        k_softmax = k.softmax(dim=2)
        k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)  # (B, num_heads, C//heads, C//heads)
        factor_attn = einsum("b h n k, b h k v -> b h n v", q, k_softmax_T_dot_v)  # (B, num_heads, H*W, C//heads)

        # Convolutional relative position encoding.
        crpe = self.filter_dynamic(q, v, size=(H, W))  # (B, num_heads, H*W, C // heads)
        # Merge and shape
        x = self.scale * factor_attn + crpe
        x = x.transpose(1, 2).reshape(B, H*W, C)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x

class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h,sr_ratio, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model  =  channel
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(CrossAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.num_heads = h

        self.scale = self.d_v ** -0.5

        # key, query, value projections for all heads
        self.q = nn.Linear(d_model, d_model)
        
        self.kv_rgb = nn.Linear(d_model, d_model * 2)
        self.kv_ir = nn.Linear(d_model, d_model * 2)

        self.attn_drop_rgb = nn.Dropout(attn_pdrop)
        self.attn_drop_ir = nn.Dropout(attn_pdrop)


        self.proj_rgb = nn.Linear(d_model, d_model)
        self.proj_ir = nn.Linear(d_model, d_model)



        self.proj_drop_rgb = nn.Dropout(resid_pdrop)
        self.proj_drop_ir = nn.Dropout(resid_pdrop)

        # self.kv = nn.Linear(d_model, d_model * 2)
        self.out_rgb = nn.Conv2d(d_model*2, d_model, kernel_size=1, stride=1)
        self.out_ir = nn.Conv2d(d_model*2, d_model, kernel_size=1, stride=1)

        self.sr_ratio = sr_ratio
        # 实现上这里等价于一个卷积层
        if sr_ratio > 1:
            self.sr_rgb = nn.Conv2d(d_model, d_model, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_rgb = nn.LayerNorm(d_model)


        if sr_ratio > 1:
            self.sr_ir = nn.Conv2d(d_model, d_model, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_ir = nn.LayerNorm(d_model)



        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        B, N ,C= x.shape

        h=int(math.sqrt(N//2))
        w=h


        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
     #   token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat

        x_rgb,x_ir =  torch.split(x,N//2,dim=1)	# 按单位长度切分，可以使用一个列表

        x = x_rgb
        if self.sr_ratio > 1:

            x_ = x.permute(0, 2, 1).reshape(B, C, h,w)    
            x_ = self.sr_rgb(x_).reshape(B, C, -1).permute(0, 2, 1) # 这里x_.shape = (B, N/R^2, C)
            x_ = self.norm_rgb(x_)
            kv_rgb = self.kv_rgb(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv_rgb = self.kv_rgb(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)



        x = x_ir
        if self.sr_ratio > 1:

            x_ = x.permute(0, 2, 1).reshape(B, C, h,w)    
            x_ = self.sr_ir(x_).reshape(B, C, -1).permute(0, 2, 1) # 这里x_.shape = (B, N/R^2, C)
            x_ = self.norm_ir(x_)
            kv_ir = self.kv_ir(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv_ir = self.kv_ir(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            
        k_rgb, v_rgb = kv_rgb[0], kv_rgb[1]
        k_ir, v_ir = kv_ir[0], kv_ir[1]

        attn_rgb = (q @ k_rgb.transpose(-2, -1)) * self.scale
        attn_rgb = attn_rgb.softmax(dim=-1)
        attn_rgb = self.attn_drop_rgb(attn_rgb)

        attn_ir = (q @ k_ir.transpose(-2, -1)) * self.scale
        attn_ir = attn_ir.softmax(dim=-1)
        attn_ir = self.attn_drop_ir(attn_ir)

        x_rgb = (attn_rgb @ v_rgb).transpose(1, 2).reshape(B, N, C)
        x_rgb = self.proj_rgb(x_rgb)
        out_rgb = self.proj_drop_rgb(x_rgb)


        x_ir = (attn_ir @ v_ir).transpose(1, 2).reshape(B, N, C)
        x_ir = self.proj_ir(x_ir)
        out_ir = self.proj_drop_ir(x_ir)

        out_rgb_1,out_rgb_2 =  torch.split(out_rgb,N//2,dim=1)	# 按单位长度切分，可以使用一个列表
        out_ir_1,out_ir_2 =  torch.split(out_ir,N//2,dim=1)	# 按单位长度切分，可以使用一个列表

        out_rgb_1_ = out_rgb_1.permute(0, 2, 1).reshape(B, C, h,w)    
        out_rgb_2_ = out_rgb_2.permute(0, 2, 1).reshape(B, C, h,w)    

        out_ir_1_ = out_ir_1.permute(0, 2, 1).reshape(B, C, h,w)    
        out_ir_2_ = out_ir_2.permute(0, 2, 1).reshape(B, C, h,w)  

        out_rgb = self.out_rgb(torch.cat([out_rgb_1_, out_rgb_2_], dim=1) ) # concat
        out_ir = self.out_ir(torch.cat([out_ir_1_, out_ir_2_], dim=1))  # concat


        out_rgb=out_rgb.view(B, C, -1).permute(0, 2, 1)
        out_ir=out_ir.view(B, C, -1).permute(0, 2, 1)

        out = torch.cat([out_rgb, out_ir], dim=1)  # concat


        return out

class CrossBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop,sr_ratio):
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)

        self.sa = CrossAttention(d_model, d_k, d_v, h,sr_ratio, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )
    
    def forward(self, x):
        bs, nx, c = x.size()
        x = x + self.sa(self.ln_input(x))

        x = x + self.mlp(self.ln_output(x))
        return x

class FeatureExtract(nn.Module):
    def __init__(self, d_model, sr_ratio, vert_anchors, horz_anchors,
                 n_layer, kernerl_sizes=[3, 3, 5, 5], 
                 dilations = [1, 8, 2, 3], expansion=0.5,
                 heads=8, block_exp=4, embed_drop=0.1,
                 attn_drop=0.1, resid_drop=0.1) -> None:
        super().__init__()
        self.n_embed = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # self.rgb_bottle = InceptionBottleneck(d_model, d_model, 
        #                                       kernel_sizes=kernerl_sizes,
        #                                       dilations=dilations,
        #                                       expansion=expansion,
        #                                       sr_ratio=sr_ratio)
        # self.inf_bottle = InceptionBottleneck(d_model, d_model,
        #                                       kernel_sizes=kernerl_sizes,
        #                                       dilations=dilations,
        #                                       expansion=expansion,
        #                                       sr_ratio=sr_ratio)

        self.rgb_filter = FrequencyFilter(d_model, num_heads=heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=resid_drop)
        self.inf_filter = FrequencyFilter(d_model, num_heads=heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=resid_drop)
        # positional embedding parameter
        self.pos_emb1 = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embed))
        self.pos_emb2 = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embed))
        self.trans_blocks = nn.Sequential(*[CrossBlock(d_model, d_k, d_v, heads, 
                                                                    block_exp, attn_drop, resid_drop, sr_ratio) for i in range(n_layer)])
        
         # decoder head
        self.ln_f = nn.LayerNorm(self.n_embed)
        # regularization
        self.drop = nn.Dropout(embed_drop)

        # avgpool
        self.avg_pool_rgb = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        self.avg_pool_inf = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, x):
        """x : tuple(Tensor)
        """
        rgb_feat = x[0]  # (B, C, H, W)
        inf_feat = x[1]
        assert rgb_feat.shape == inf_feat.shape
        B, C, H, W = rgb_feat.shape

        # ------------------ Frequency Feature Extract ---------------------
        # rgb_feat = self.rgb_bottle(rgb_feat)
        # inf_feat = self.inf_bottle(inf_feat)
        rgb_feat = self.rgb_filter(rgb_feat)
        inf_feat = self.inf_filter(inf_feat)
        # ------------------ Avg Pooling -----------------------------------
        rgb_feat = self.avg_pool_rgb(rgb_feat)
        inf_feat = self.avg_pool_inf(inf_feat)

        # Transformer
        rgb_embeddings = rgb_feat.view(B, C, -1)  # (B, C, H*W)
        inf_embeddings = inf_feat.view(B, C, -1)
        rgb_embeddings = rgb_embeddings.permute(0, 2, 1).contiguous()
        inf_embeddings = inf_embeddings.permute(0, 2, 1).contiguous()

        x_rgb = self.drop(self.pos_emb1 + rgb_embeddings)
        x_inf = self.drop(self.pos_emb1 + inf_embeddings)

        x = torch.cat([x_rgb, x_inf], dim=1)  # (B, 2*H*W, C)

        x = self.trans_blocks(x)
        # decoder head
        x = self.ln_f(x)  # (B, 2*H*W, C)
        x = x.view(B, 2, self.vert_anchors, self.horz_anchors, self.n_embed)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(B, self.n_embed, self.vert_anchors, self.horz_anchors)
        inf_fea_out = x[:, 1, :, :, :].contiguous().view(B, self.n_embed, self.vert_anchors, self.horz_anchors)

        # Interpolate Unsample
        rgb_fea_out = F.interpolate(rgb_fea_out, size=(H, W), mode='bilinear', align_corners=True)
        inf_fea_out = F.interpolate(inf_fea_out, size=(H, W), mode='bilinear', align_corners=True)

        return rgb_fea_out, inf_fea_out

@MODELS.register_module()
class FMCFNet(TwoStreamResNet):
    def __init__(self, 
                 dims_in=[512, 1024, 2048],
                 sr_ratio=[3, 2, 1],
                 n_layers=[6, 6, 6],
                 num_heads=[8, 8, 8],
                 vert_ahchors=[40, 20, 10],
                 horz_anchors=[40, 20, 10],
                 **kwargs):
        super(FMCFNet, self).__init__(**kwargs)
        assert len(dims_in) == len(sr_ratio), "Please ensure the parameter of FFE has the same length"
        self.dims_in = dims_in
        self.sr_ratio = sr_ratio
        self.n_layers = n_layers
        self.num_heads = num_heads
        # add FFE
        self.ffes = nn.ModuleList()
        for i in range(len(self.out_indices)):
            self.ffes.append(FeatureExtract(
                dims_in[i],
                sr_ratio=sr_ratio[i],
                vert_anchors=vert_ahchors[i],
                horz_anchors=horz_anchors[i],
                n_layer=n_layers[i],
                heads=num_heads[i],
            ))
    
    def forward(self, vis_x, lwir_x):
        # resnet part
        if self.deep_stem:
            vis_x = self.vis_stem(vis_x)
            lwir_x = self.lwir_stem(lwir_x)
        else:
            vis_x = self.vis_conv1(vis_x)
            vis_x = self.vis_norm1(vis_x)
            vis_x = self.relu(vis_x)

            lwir_x = self.lwir_conv1(lwir_x)
            lwir_x = self.lwir_norm1(lwir_x)
            lwir_x = self.relu(lwir_x)
        vis_x = self.maxpool(vis_x)
        lwir_x = self.maxpool(lwir_x)

        outs = []

        for i in range(self.num_stages):
            # resnet
            vis_layer_name = self.vis_res_layers[i]
            vis_res_layer = getattr(self, vis_layer_name)
            vis_x = vis_res_layer(vis_x)

            lwir_layer_name = self.lwir_res_layers[i]
            lwir_res_layer = getattr(self, lwir_layer_name)
            lwir_x = lwir_res_layer(lwir_x)

            # FMCFNet
            if i in self.out_indices:
                rgb_x, inf_x = self.ffes[i-1]((vis_x, lwir_x))
                vis_x = vis_x + rgb_x
                lwir_x = lwir_x + inf_x
                out = vis_x + lwir_x
                outs.append(out)
        
        return tuple(outs)
    
