import torch
from torch import nn
from torch.nn import functional as F
from .attention import SelfAttention, CrossAttention


from IPython import embed
import hotline


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        with hotline.annotate('Linear'):
            x = self.linear_1(x)
        with hotline.annotate('Silu'):
            x = F.silu(x)
        with hotline.annotate('Linear'):
            x = self.linear_2(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        residue = feature

        with hotline.annotate('groupnorm'):
            feature = self.groupnorm_feature(feature)
        with hotline.annotate('silu'):
            feature = F.silu(feature)
        with hotline.annotate('conv'):
            feature = self.conv_feature(feature)
        with hotline.annotate('silu'):
            time = F.silu(time)
        with hotline.annotate('linear'):
            time = self.linear_time(time)
        with hotline.annotate('groupnorm'):
            merged = feature + time.unsqueeze(-1).unsqueeze(-1)
            merged = self.groupnorm_merged(merged)
        with hotline.annotate('silu'):
            merged = F.silu(merged)
        with hotline.annotate('conv'):
            merged = self.conv_merged(merged)
            x = self.residual_layer(residue)
        return merged + x

class AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        residue_long = x

        with hotline.annotate('groupnorm'):
            x = self.groupnorm(x)
        with hotline.annotate('conv'):
            x = self.conv_input(x)
        with hotline.annotate('layernorm'):
            n, c, h, w = x.shape
            x = x.view((n, c, h * w))   # (n, c, hw)
            x = x.transpose(-1, -2)  # (n, hw, c)
            residue_short = x
            x = self.layernorm_1(x)
        with hotline.annotate('attention'):
            x = self.attention_1(x)
        with hotline.annotate('layernorm'):
            x += residue_short
            residue_short = x
            x = self.layernorm_2(x)
        with hotline.annotate('attention'):
            x = self.attention_2(x, context)
        with hotline.annotate('layernorm'):
            x += residue_short
            residue_short = x
            x = self.layernorm_3(x)
        with hotline.annotate('linear'):
            x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        with hotline.annotate('gelu'):
            x = x * F.gelu(gate)
        with hotline.annotate('linear'):
            x = self.linear_geglu_2(x)
        with hotline.annotate('conv'):
            x += residue_short
            x = x.transpose(-1, -2)  # (n, c, hw)
            x = x.view((n, c, h, w))    # (n, c, h, w)
            x = self.conv_output(x)
        return x + residue_long

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        with hotline.annotate('interpolate'):
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        with hotline.annotate('conv'):
            x = self.conv(x)
        return x

class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                with hotline.annotate('AttentionBlock'):
                    x = layer(x, context)
            elif isinstance(layer, ResidualBlock):
                with hotline.annotate('ResidualBlock'):
                    x = layer(x, time)
            else:
                with hotline.annotate(layer._get_name()):
                    x = layer(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(ResidualBlock(320, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(640, 640), AttentionBlock(8, 80)),
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(ResidualBlock(640, 1280), AttentionBlock(8, 160)),
            SwitchSequential(ResidualBlock(1280, 1280), AttentionBlock(8, 160)),
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(ResidualBlock(1280, 1280)),
            SwitchSequential(ResidualBlock(1280, 1280)),
        ])
        self.bottleneck = SwitchSequential(
            ResidualBlock(1280, 1280),
            AttentionBlock(8, 160),
            ResidualBlock(1280, 1280),
        )
        self.decoders = nn.ModuleList([
            SwitchSequential(ResidualBlock(2560, 1280)),
            SwitchSequential(ResidualBlock(2560, 1280)),
            SwitchSequential(ResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
            SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
            SwitchSequential(ResidualBlock(1920, 1280), AttentionBlock(8, 160), Upsample(1280)),
            SwitchSequential(ResidualBlock(1920, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(1280, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(960, 640), AttentionBlock(8, 80), Upsample(640)),
            SwitchSequential(ResidualBlock(960, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(640, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(640, 320), AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        skip_connections = []
        for i, layers in enumerate(self.encoders):
            with hotline.annotate(f'Encoder{i}'):
                x = layers(x, context, time)
                skip_connections.append(x)

        with hotline.annotate('Bottleneck'):
            x = self.bottleneck(x, context, time)

        for i, layers in enumerate(self.decoders):
            with hotline.annotate(f'Decoder{i}'):
                x = torch.cat((x, skip_connections.pop()), dim=1)
                x = layers(x, context, time)

        return x


class FinalLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        with hotline.annotate('GroupNorm'):
            x = self.groupnorm(x)
        with hotline.annotate('Silu'):
            x = F.silu(x)
        with hotline.annotate('Conv'):
            x = self.conv(x)
        return x

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = FinalLayer(320, 4)
    
    def forward(self, latent, context, time):
        with hotline.annotate('Time Embedding'):
            time = self.time_embedding(time)
        with hotline.annotate('UNet'):
            output = self.unet(latent, context, time)
        with hotline.annotate('Final'):
            output = self.final(output)
            return output