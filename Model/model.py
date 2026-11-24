import torch
import torch.nn.functional as F
from einops import rearrange
import torch.nn as nn
import torch.nn.init
import numpy as np
import torch.nn.init as init
from einops import rearrange


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_qx = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_kx = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_vx = nn.Linear(dim, dim_head * heads, bias=False)

        self.to_qy = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_ky = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_vy = nn.Linear(dim, dim_head * heads, bias=False)

        self.rescalem = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescale2 = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescalef = nn.Parameter(torch.ones(heads, 1, 1))

        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim
        self.weights = nn.Parameter(torch.tensor([0.6, 0.2, 0.2]))  # [A_xx, A_xy, A_yy]

        self.gate_q = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 1), nn.Sigmoid()
        )
        self.gate_k = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 1), nn.Sigmoid()
        )

    def forward(self, x, y):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """

        b, h, w, c = y.shape
        x = x.reshape(b, h * w, c)
        y = y.reshape(b, h * w, c)

        x_q = self.to_qx(x)
        x_k = self.to_kx(x)
        x_v = self.to_vx(x)
        qm, km, vm = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                         (x_q, x_k, x_v))

        # q: b,heads,hw,c

        qm = F.normalize(qm, dim=-1, p=2)
        km = F.normalize(km, dim=-1, p=2)

        attnm = (km @ qm.transpose(-2, -1))  # x 自注意力
        attnm = attnm * self.rescalem
        attnm = attnm.softmax(dim=-1)

        y_q = self.to_qy(y)
        y_k = self.to_ky(y)
        y_v = self.to_vy(y)
        qy, ky, vy = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                         (y_q, y_k, y_v))
        # q: b,heads,hw,c

        k2 = F.normalize(ky, dim=-1, p=2)
        q2 = F.normalize(qy, dim=-1, p=2)
        attn2 = (k2 @ qm.transpose(-2, -1))  # cross_attn  x
        attn2 = attn2 * self.rescale2
        attn2 = attn2.softmax(dim=-1)

        # === Step2: 生成 gating 系数 ===
        gx = x.mean(dim=1)  # (B,C)
        gy = y.mean(dim=1)  # (B,C)
        g_in = torch.cat([gx, gy], dim=-1)  # (B,2C)

        alpha_q = self.gate_q(g_in)  # (B,1)，控制 Q 融合
        alpha_k = self.gate_k(g_in)  # (B,1)，控制 K 融合

        # === Step3: 融合 Q 和 K ===
        alpha_q = self.gate_q(g_in).unsqueeze(1).unsqueeze(2)  # (B,1,1,1)
        alpha_k = self.gate_k(g_in).unsqueeze(1).unsqueeze(2)

        Q_f = alpha_q * qm + (1 - alpha_q) * qy  # (B,N,inner_dim)
        K_f = alpha_k * km + (1 - alpha_k) * ky  # (B,N,inner_dim)

        attn_f = (K_f @ Q_f.transpose(-2, -1))
        attn_f = self.rescalef * attn_f
        attn_f = attn_f.softmax(dim=-1)

        alpha, beta, gamma = F.softmax(self.weights, dim=0)
        x = alpha * (attnm @ vm) + beta * (attn2 @ vy) + gamma * (attn_f @ vm)  # b,heads,d,hw

        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(y_v.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class MSA_Spectral(nn.Module):
    def __init__(self, dim, dim_head, heads):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_qm = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_km = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_vm = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_q2 = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k2 = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v2 = nn.Linear(dim, dim_head * heads, bias=False)

        self.rescalem = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescale2 = nn.Parameter(torch.ones(heads, 1, 1))

        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim
        self.weights = nn.Parameter(torch.tensor([0.6, 0.3, 0.1]))  # [A_xx, A_xy, A_yy]

    def forward(self, x, y):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)

        y = y.reshape(b, h * w, c)

        q_inpm = self.to_qm(x)
        k_inpm = self.to_km(x)
        v_inpm = self.to_vm(x)
        qm, km, vm = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                         (q_inpm, k_inpm, v_inpm))
        vm = vm
        # q: b,heads,hw,c
        qm = qm.transpose(-2, -1)
        km = km.transpose(-2, -1)
        vm = vm.transpose(-2, -1)
        qm = F.normalize(qm, dim=-1, p=2)
        km = F.normalize(km, dim=-1, p=2)
        attnm = (km @ qm.transpose(-2, -1))  # A = K^T*Q
        attnm = attnm * self.rescalem
        attnm = attnm.softmax(dim=-1)

        q_inp2 = self.to_q2(y)
        k_inp2 = self.to_k2(y)
        v_inp2 = self.to_v2(y)
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                         (q_inp2, k_inp2, v_inp2))

        # q: b,heads,hw,c
        q2 = q2.transpose(-2, -1)
        k2 = k2.transpose(-2, -1)
        v2 = v2.transpose(-2, -1)
        q2 = F.normalize(q2, dim=-1, p=2)
        k2 = F.normalize(k2, dim=-1, p=2)
        attn_y = (k2 @ q2.transpose(-2, -1))  # base-self
        attn2 = (k2 @ qm.transpose(-2, -1))  # A = K^T*Q
        attn2 = attn2 * self.rescale2
        attn2 = attn2.softmax(dim=-1)

        x_base = x
        x_cross = attn2 @ vm
        x_cross_refined = attn_y @ x_cross
        x_self = attnm @ vm

        alpha, beta, gamma = F.softmax(self.weights, dim=0)
        x = alpha * (x_self) + beta * (x_cross) + gamma * (x_cross_refined)  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inpm.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class MLSIF(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MSA(dim=dim, dim_head=dim_head, heads=heads),
                MSA_Spectral(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, y, z):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        z = z.permute(0, 2, 3, 1)
        for (attn, attn_spe, ff) in self.blocks:
            x = attn(x, y) + x

            x = attn_spe(x, z) + x
            x = ff(x) + x
        x = x.permute(0, 3, 1, 2)
        return x


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1 / 16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac


class LN(nn.Module):
    def __init__(self, dim):
        super(LN, self).__init__()

        self.weights = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.ones(dim))

    def forward(self, X):
        h, w = X.shape[-2:]
        X = rearrange(X, 'b c h w -> b (h w) c')
        mu = X.mean(-1, keepdim=True)
        sigma = X.var(-1, keepdim=True, unbiased=False)
        X = (X - mu) / torch.sqrt(sigma + 1e-5) * self.weights + self.bias
        X = rearrange(X, 'b (h w) c -> b c h w', h=h, w=w)
        return X


class GN(nn.Module):
    def __init__(self, num_spectral, r=2):
        super(GN, self).__init__()

        self.num_spectral = num_spectral
        self.g = nn.Conv2d(num_spectral, int(num_spectral * r), 1, 1, 0, bias=False)
        self.q = nn.Conv2d(num_spectral, int(num_spectral * r), 1, 1, 0, bias=False)
        self.o = nn.Conv2d(int(num_spectral * r), num_spectral, 1, 1, 0, bias=False)
        self.gelu = nn.GELU()
        self.ln = LN(num_spectral)

    def forward(self, x):
        x0 = x
        x = self.ln(x)
        g = self.g(x)
        q = self.gelu(g)
        q = self.q(x)
        o = g * q
        o = self.o(o)
        x = x0 + o
        return x


class SELayer(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, norm=None):
        super(SELayer, self).__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv_du = torch.nn.Sequential(
            torch.nn.Conv2d(num_filter, num_filter // 2, 1, 1, 0, bias=bias),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(num_filter // 2, num_filter, 1, 1, 0, bias=bias),
            torch.nn.Sigmoid())

    def forward(self, x):
        mask = self.global_pool(x)
        mask = self.conv_du(mask)
        x = x * mask
        return x


class Res(nn.Module):
    def __init__(self, n_feat, groups=1):
        super(Res, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, bias=True, groups=groups),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, bias=True, groups=groups))
        self.se = SELayer(n_feat)

    def forward(self, x):
        res = self.conv1(x)
        res = self.se(res)
        return res + x


class CNet(torch.nn.Module):
    def __init__(self, scale_factor=8, num_channel=3, base_filter=64, num_spectral=31, num_block=1):
        super(CNet, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.in_conv = nn.Conv2d(num_channel + num_spectral, base_filter, 3, 1, 0, bias=True)
        self.tailh = nn.Conv2d(base_filter, 64, 3, 1, 0, bias=True)
        self.tail = nn.Conv2d(base_filter, 64, 3, 1, 0, bias=True)
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        self.down = nn.Upsample(scale_factor=1 / 2, mode='bilinear')
        body = [Res(base_filter) \
                for _ in range(num_block)]
        self.df = nn.Sequential(*body)

    def forward(self, Y, Z):
        Y = self.up(Y)  # uncommen this code for training
        res = torch.cat((Y, Z), dim=1)
        res = self.in_conv(self.pad(res))
        res = self.df(res)

        resh = self.tailh(self.pad(res))
        res1 = self.pad(self.tail(res))
        reslr = self.down(res1)

        return resh, reslr


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3


def conv_exp(input, kernel, terms=10, dynamic_truncation=0, verbose=False):
    B, C, H, W = input.size()

    assert kernel.size(0) == kernel.size(1)
    assert kernel.size(0) == C, '{} !={}'.format(kernel.size(0), C)

    padding = (kernel.size(2) - 1) // 2, (kernel.size(3) - 1) // 2

    result = input
    product = input

    for i in range(1, terms + 1):
        product = F.conv2d(product, kernel, padding=padding, stride=(1, 1)) / i
        result = result + product

        if dynamic_truncation != 0 and i > 5:
            if product.abs().max().item() < dynamic_truncation:
                break
    if verbose:
        print('Maximum element size in term:{}'.format(torch.max(torch.abs(product))))
    return result


def inv_conv_exp(input, kernel, terms=10, dynamic_truncation=0, verbose=False):
    return conv_exp(input, -kernel, terms, dynamic_truncation, verbose)


class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
        return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class MyBlock(nn.Module):
    def __init__(self, channel_in, channel_out, gc=48, mshape=31, bias=True):
        super(MyBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(channel_in + mshape, gc, 3, 1, 0, bias=bias))
        self.conv2 = Res(gc)
        self.conv3 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(gc, channel_out, 3, 1, 0, bias=bias))
        self.ca = GN(gc)
        initialize_weights_xavier([self.conv1, self.conv2], 0.1)

        initialize_weights(self.conv3, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ca(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x


class LowFreqExtractorLearnable(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel=5, use_spectral_conv=True):
        super().__init__()
        pad = kernel // 2
        # learnable anti-alias: depthwise conv across spatial dims
        self.lowpass = nn.Conv2d(in_channels, in_channels, kernel_size=kernel, padding=pad,
                                 groups=in_channels, bias=False)  # depthwise
        # downsample with stride 2 depthwise separable conv (preserve channels first)
        self.down_dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels,
                                 bias=False)
        self.down_pw = nn.Conv2d(in_channels, in_channels // 2 if in_channels // 2 > 0 else in_channels, kernel_size=1,
                                 bias=False)
        # expand channels
        self.expand = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.use_spectral_conv = use_spectral_conv
        if use_spectral_conv:
            # spectral mixing: 1D conv along channel dimension implemented as Conv1d on (B, C, N)
            self.spectral_conv = nn.Sequential(
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, groups=1),
                nn.GELU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=1)
            )
        self.se = SELayer(out_channels)

    def forward(self, x):
        # x: (B, C_in, H, W)
        x = self.lowpass(x)  # learnable low-pass\

        x = self.down_dw(x)  # (B, C_in, H/2, W/2)

        x = self.down_pw(x)

        x = self.expand(x)  # (B, out_channels, H/2, W/2)

        x = self.norm(x)
        x = self.act(x)
        x = self.se(x)
        return x


class CInvBlock(nn.Module):
    def __init__(self, channel_num, channel_split_num, gc=64, mshape=64):
        super(CInvBlock, self).__init__()

        self.channel_num = channel_num
        self.channel_split_num = channel_split_num
        self.gc = gc
        self.mshape = mshape
        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num
        self.clamp_s = nn.Parameter(torch.rand(1, self.split_len2, 1, 1))
        self.F = MyBlock(self.split_len2, self.split_len1, gc, mshape)
        self.G = MyBlock(self.split_len1, self.split_len2, gc, mshape)
        self.H = MyBlock(self.split_len1, self.split_len2, gc, mshape)

        self.norm = ActNorm(channel_num)
        self.kernel1 = nn.Parameter(torch.rand(channel_num, channel_num, 5, 5) * 1e-6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, m, rev=False):
        if not rev:

            x = self.norm.forward(x)
            x = conv_exp(x, self.kernel1)
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
            y1 = x1 + self.F(torch.cat((x2, m), dim=1))
            self.s = self.clamp_s * torch.sigmoid(self.H(torch.cat((y1, m), dim=1) * 2 - 1))
            y2 = x2.mul(torch.exp(self.s)) + self.G(torch.cat((y1, m), dim=1))
            out = torch.cat((y1, y2), 1)
        else:
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
            self.s = self.clamp_s * torch.sigmoid(self.H(torch.cat((x1, m), dim=1) * 2 - 1))
            y2 = (x2 - self.G(torch.cat((x1, m), dim=1))).div(torch.exp(self.s))
            y1 = x1 - self.F(torch.cat((y2, m), dim=1))
            x = torch.cat((y1, y2), 1)
            out = inv_conv_exp(x, self.kernel1)
            out = self.norm.reverse(out)

        return out


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, d=1, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out

class Net(nn.Module):
    def __init__(self, hschannels=31, mschannels=3, sf=8, hidden_num=64, block_num=2):
        super(Net, self).__init__()
        operations = []
        operations1 = []
        operations2 = []
        self.sf = sf
        self.conv_hs1 = nn.Conv2d(hschannels, hidden_num, kernel_size=3, padding=1, bias=False)
        self.conv_hs2 = nn.Conv2d(hidden_num, hschannels, kernel_size=3, padding=1, bias=False)
        self.down_hs = LowFreqExtractorLearnable(31)
        self.channel_in = hschannels
        self.inv_in = CInvBlock(64, 32, gc=32, mshape=64)
        for j in range(block_num):
            b = MLSIF(dim=64, num_blocks=2, dim_head=32, heads=2)
            b1 = MLSIF(dim=64, num_blocks=2, dim_head=32, heads=2)
            b2 = MLSIF(dim=64, num_blocks=2, dim_head=32, heads=2)
            operations2.append(b2)
            operations1.append(b1)
            operations.append(b)
        self.operations = nn.ModuleList(operations)
        self.operations1 = nn.ModuleList(operations1)
        self.operations2 = nn.ModuleList(operations2)
        self.inv_out = CInvBlock(64, 32, gc=32, mshape=64)
        self.har_in = HaarDownsampling(64)
        self.har_out = HaarDownsampling(64)
        self.c_net = CNet()

        
        self.Abundance = nn.Sequential(
            nn.Conv2d(hschannels, 8 * mschannels, kernel_size=3, padding=1),
            SELayer(8 * mschannels),
            nn.Conv2d(8 * mschannels, mschannels, kernel_size=3, padding=1),
            SELayer(mschannels),
            nn.ReLU(inplace=True),
            nn.Sigmoid()

        )
        self.endmember = nn.Sequential(
            # nn.Conv2d(mschannels, 32, kernel_size=3, padding=1),
            DenseBlock(3,32),
            torch.nn.Tanh(),
            # nn.Conv2d(32, 64, kernel_size=3, padding=1),
            DenseBlock(32,64),
            Res(64),
            torch.nn.Tanh(),
            # nn.Conv2d(64, 64, kernel_size=3, padding=1),
            DenseBlock(64,64),
            Res(64),
            torch.nn.Tanh(),
            nn.Conv2d(64, hschannels, kernel_size=1),
            nn.ReLU(inplace=True)
        )


    def forward(self, hsi, msi):

        hsi_up = F.interpolate(hsi, scale_factor=self.sf, mode='bicubic', align_corners=False)  # 上采样
        hsi_4 = F.interpolate(hsi, scale_factor=4, mode='bicubic', align_corners=False)
        hsi_abu = self.Abundance(hsi)  #
       
        pseudo_hsi = self.endmember(hsi_abu)  # 31 16 16
        
        pseudo_hr = self.endmember(msi)  # 3 64 64 --> 31 64 64
        p_base = F.interpolate(pseudo_hsi, scale_factor=self.sf, mode='bicubic', align_corners=False)  # 上采样
        x = pseudo_hr - p_base
        
        x = self.conv_hs1(x)
       
        hsi_4 = self.down_hs(hsi_up)
        m1, m = self.c_net(hsi, msi)
        
        x = self.inv_in(x, m1)
        x = self.har_in(x)
        c = 64
        # ll = x[:, :c, :, :]
        lh = x[:, c:2 * c, :, :]
        hl = x[:, 2 * c:3 * c, :, :]
        hh = x[:, 3 * c:4 * c, :, :]

        ll = hsi_4

        for op in self.operations:
            lh = op.forward(lh, m, hsi_4)
        for op in self.operations1:
            hl = op.forward(hl, m, hsi_4)
        for op in self.operations2:
            hh = op.forward(hh, m, hsi_4)

        x = torch.cat([ll, lh, hl, hh], dim=1)
        x = self.har_out(x, rev=True)
        x = self.inv_out(x, m1)
        x = self.conv_hs2(x)

        base = hsi_up

        return x + base, pseudo_hsi, hsi_abu

from thop import profile, clever_format
if __name__ == '__main__':

    a,b=64,64
    hsi = torch.randn(2, 31,a//8,b//8).cuda()
    msi = torch.randn(2, 3, a,b).cuda()
    HR = torch.randn(2, 31, a,b).cuda()
    model = Net(31,3,8).cuda()
    flops, params,DIC= profile(model, inputs=(hsi,msi),ret_layer_info=True)
    # flops, params,DIC= profile(model, inputs=(msi,hsi),ret_layer_info=True)
    flops, params= clever_format([flops, params], "%.3f")
    print(flops, params)
