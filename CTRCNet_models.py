import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from einops import rearrange
from torch.autograd import Function

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

def deconv(in_channels, out_channels, kernel_size=3, stride=1, groups=3, padding=1, output_padding=0):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)
    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None
class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """

    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=0.1,
                 reparam_offset=2 ** -18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset
        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = ((self.beta_min + self.reparam_offset ** 2) ** 0.5)
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class Attention(nn.Module):
    def __init__(self, dim=64, num_heads=2, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out
class MultiHeadedAttention(nn.Module):
    def __init__(self, in_model, out_model):
        super().__init__()

        self.output_linear = nn.Sequential(
            nn.Conv2d(out_model, out_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_model),
            nn.LeakyReLU(0.2, inplace=True),)

        self.conv1 = nn.Conv2d(
            in_model, out_model, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(
            out_model, out_model, kernel_size=3, stride=2, padding=1)

        self.SMAT = Attention(out_model, 8)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):

        x = self.conv1(x)
        x32 = self.conv2(x)
        x16 = self.conv2(x32)
        x8 = self.conv2(x16)


        x32 = self.SMAT(x32)
        x16 = self.SMAT(x16)
        x8 = self.SMAT(x8)

        x32 = self.upsample2(x32)
        x16 = self.upsample4(x16)
        x8 = self.upsample8(x8)

        output = x8 + x16 + x32

        self_attention = self.output_linear(output)

        return self_attention
class FeedForward2D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeedForward2D, self).__init__()

        self.conv1 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.lU1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.lU2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lU1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lU2(x)
        return x
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(
            'Network [%s] was created. Total number of parameters: %.1f million. '
            'To see the architecture, do print(network).'
            % (type(self).__name__, num_params / 1000000)
        )

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (
                classname.find('Conv') != -1 or classname.find('Linear') != -1
            ):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented'
                        % init_type
                    )
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)
class TransformerBlock(nn.Module):
    def __init__(self, in_channel=256, out_channel = 256, half=False):
        super().__init__()
        self.half = half

        self.attention = MultiHeadedAttention( in_model=in_channel, out_model=out_channel)
        self.feed_forward = FeedForward2D(in_channel, out_channel)
        self.conv1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=2, padding=1)

    def forward(self, x):    #torch.Size([128, 256, 8, 8])

        self_attention = self.attention(x)
        x = self.conv1(x)

        output = x + self_attention
        feed_forward = self.feed_forward(output)
        output = output + feed_forward

        if self.half:
            output = self.conv2(output)
        return output

class LMST(BaseNetwork):
    def __init__(self, in_channel, out_channle, half=False):     # 3,256; 256,256,  256,256, 256,48
        super(LMST, self).__init__()

        self.t = TransformerBlock( in_channel=in_channel, out_channel=out_channle, half=half )

    def forward(self, x):
        output = self.t(x)

        return output

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class FCF(nn.Module):

    def __init__(self, in_channels=96, r=4):
        super(FCF, self).__init__()

        inter_channels = int(in_channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels//2),
        )
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(in_channels)

    def forward(self, cnn, lmst):

        cnn_lmst = cnn * lmst

        g_f = cnn_lmst + lmst

        l_f = cnn_lmst + cnn

        lg_f = torch.cat((g_f, l_f), dim=1)

        lg_f_ca = self.channel_attention(lg_f) + lg_f
        lg_f_sa = self.spatial_attention(lg_f_ca) +lg_f_ca

        lg_out = self.local_att(lg_f_sa)

        return lg_out

class AF_block(nn.Module):
    def __init__(self, Nin, Nh, No):
        super(AF_block, self).__init__()
        self.fc1 = nn.Linear(Nin + 1, Nh)
        self.fc2 = nn.Linear(Nh, No)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, snr):

        if snr.shape[0] > 1:
            snr = snr.squeeze()
        snr = snr.unsqueeze(1)
        mu = torch.mean(x, (2, 3))
        out = torch.cat((mu, snr), 1)
        # print(out.size())
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2)
        out = out.unsqueeze(3)
        out = out * x
        return out

class conv_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv1x1=False, kernel_size=3, stride=1, padding=1):

        super(conv_ResBlock, self).__init__()
        self.conv1 = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = conv(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.gdn1 = GDN(out_channels)
        self.gdn2 = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.use_conv1x1 = use_conv1x1
        if use_conv1x1 == True:
            self.conv3 = conv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gdn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.gdn2(out)
        if self.use_conv1x1 == True:
            x = self.conv3(x)
        out = out + x
        out = self.prelu(out)
        return out
class deconv_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_deconv1x1=False, kernel_size=3, stride=1, padding=1, output_padding=0):
        super(deconv_ResBlock, self).__init__()
        self.deconv1 = deconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.deconv2 = deconv(out_channels, out_channels, kernel_size=1, stride=1, padding=0, output_padding=0)
        self.gdn1 = GDN(out_channels)
        self.gdn2 = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_deconv1x1 = use_deconv1x1
        if use_deconv1x1 == True:
            self.deconv3 = deconv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, output_padding=output_padding)

    def forward(self, x, activate_func='prelu'):
        out = self.deconv1(x)
        out = self.gdn1(out)
        out = self.prelu(out)
        out = self.deconv2(out)
        out = self.gdn2(out)
        if self.use_deconv1x1 == True:
            x = self.deconv3(x)
        out = out + x
        if activate_func == 'prelu':
            out = self.prelu(out)
        elif activate_func == 'sigmoid':
            out = self.sigmoid(out)
        return out

class Encoder(nn.Module):
    def __init__(self, enc_shape, kernel_sz, Nc_conv):
        super(Encoder, self).__init__()
        enc_N = enc_shape[0]
        Nh_AF = Nc_conv // 2
        padding_L = (kernel_sz - 1) // 2  # 2 = (5-1)/2

        self.lmst1 = LMST(3, Nc_conv, half=True)
        self.lmst2 = LMST(Nc_conv, Nc_conv, half=True)
        self.lmst3 = LMST(Nc_conv, Nc_conv)
        self.lmst4 = LMST(Nc_conv, enc_N)

        self.l_AF1 = AF_block(Nc_conv, Nh_AF, Nc_conv)
        self.l_AF2 = AF_block(Nc_conv, Nh_AF, Nc_conv)
        self.l_AF3 = AF_block(Nc_conv, Nh_AF, Nc_conv)

        self.cnn1 = conv_ResBlock(3, Nc_conv, use_conv1x1=True, kernel_size=kernel_sz, stride=2, padding=padding_L)
        self.cnn2 = conv_ResBlock(Nc_conv, Nc_conv, use_conv1x1=True, kernel_size=kernel_sz, stride=2, padding=padding_L)
        self.cnn3 = conv_ResBlock(Nc_conv, Nc_conv, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.cnn4 = conv_ResBlock(Nc_conv, enc_N, use_conv1x1=True, kernel_size=kernel_sz, stride=1, padding=padding_L)

        self.c_AF1 = AF_block(Nc_conv, Nh_AF, Nc_conv)
        self.c_AF2 = AF_block(Nc_conv, Nh_AF, Nc_conv)
        self.c_AF3 = AF_block(Nc_conv, Nh_AF, Nc_conv)

        self.fcf = FCF()

        self.lc_AF = AF_block(enc_N, enc_N*2, enc_N)

        self.flatten = nn.Flatten()

    def forward(self, x, snr=5):
        c_out = self.cnn1(x)
        c_out = self.c_AF1(c_out, snr)

        l_out = self.lmst1(x)
        l_out = self.l_AF1(l_out, snr)

        c_out = self.cnn2(c_out)
        c_out = self.c_AF2(c_out, snr)

        l_out = self.lmst2(l_out)
        l_out = self.l_AF2(l_out, snr)

        c_out = self.cnn3(c_out)
        c_out = self.c_AF3(c_out, snr)

        l_out = self.lmst3(l_out)
        l_out = self.l_AF3(l_out, snr)

        c_out = self.cnn4(c_out)
        l_out = self.lmst4(l_out)


        lc_out = self.fcf(l_out, c_out)

        out = self.lc_AF(lc_out, snr)

        out = self.flatten(out)
        return out

def mask_gen(N, cr, ch_max=48):
    MASK = torch.zeros(cr.shape[0], N).int()
    nc = N // ch_max
    for i in range(0, cr.shape[0]):
        L_i = nc * torch.round(ch_max * cr[i]).int()
        MASK[i, 0:L_i] = 1
    return MASK

class ASFM(nn.Module):
    def __init__(self, in_channel=3072, out_channel=3072):

        super(ASFM, self).__init__()

        self.fc1 = nn.Linear(in_channel + 1, in_channel//2)
        self.fc2 = nn.Linear(in_channel//2, out_channel)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, y, snr, cr):
        batch_size, length = y.shape
        mask = mask_gen(length, cr).cuda()
        y = y * mask
        y1 = torch.cat((y, snr), 1)
        y1 = self.fc1(y1)
        y1 = self.relu(y1)
        y1 = self.fc2(y1)
        y1 = self.sigmoid(y1)
        y = y * y1
        return y

class Decoder(nn.Module):
    def __init__(self, enc_shape, kernel_sz, Nc_deconv):
        super(Decoder, self).__init__()
        self.enc_shape = enc_shape
        Nh_AF1 = enc_shape[0] // 2
        Nh_AF = Nc_deconv // 2
        padding_L = (kernel_sz - 1) // 2

        self.deconv1 = deconv_ResBlock(self.enc_shape[0], Nc_deconv, use_deconv1x1=True, kernel_size=kernel_sz, stride=2, padding=padding_L, output_padding=1)
        self.deconv2 = deconv_ResBlock(Nc_deconv, Nc_deconv, use_deconv1x1=True, kernel_size=kernel_sz, stride=2, padding=padding_L, output_padding=1)
        self.deconv3 = deconv_ResBlock(Nc_deconv, Nc_deconv, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.deconv4 = deconv_ResBlock(Nc_deconv, 3, use_deconv1x1=True, kernel_size=kernel_sz, stride=1, padding=padding_L)

        self.AF1 = AF_block(self.enc_shape[0], Nh_AF1, self.enc_shape[0])
        self.AF2 = AF_block(Nc_deconv, Nh_AF, Nc_deconv)
        self.AF3 = AF_block(Nc_deconv, Nh_AF, Nc_deconv)
        self.AF4 = AF_block(Nc_deconv, Nh_AF, Nc_deconv)

    def forward(self, x, snr):
        out = x.view(-1, self.enc_shape[0], self.enc_shape[1], self.enc_shape[2])

        out = self.AF1(out, snr)
        out = self.deconv1(out)

        out = self.AF2(out, snr)
        out = self.deconv2(out)

        out = self.AF3(out, snr)
        out = self.deconv3(out)

        out = self.AF4(out, snr)
        out = self.deconv4(out, 'sigmoid')

        return out

def Channel(z, snr, channel_type='AWGN'):
    batch_size, length = z.shape
    gamma = 10 ** (snr / 10.0)

    p = 1
    z_power = torch.sqrt(torch.sum(z ** 2, 1))
    z_M = z_power.repeat(length, 1)
    z = np.sqrt(p * length) * z / z_M.t()

    if channel_type == 'AWGN':
        P=2
        noise = torch.sqrt(P / gamma) * torch.randn(batch_size, length).cuda()
        y = z + noise
        return y
    elif channel_type == 'Fading':
        P = 2
        K = length // 2

        h_I = torch.randn(batch_size, K).cuda()
        h_R = torch.randn(batch_size, K).cuda()
        h_com = torch.complex(h_I, h_R)
        x_com = torch.complex(z[:, 0:length:2], z[:, 1:length:2])
        y_com = h_com * x_com

        n_I = torch.sqrt(P / gamma) * torch.randn(batch_size, K).cuda()
        n_R = torch.sqrt(P / gamma) * torch.randn(batch_size, K).cuda()
        noise = torch.complex(n_I, n_R)

        y_add = y_com + noise
        y = y_add / h_com

        y_out = torch.zeros(batch_size, length).cuda()
        y_out[:, 0:length:2] = y.real
        y_out[:, 1:length:2] = y.imag
        return y_out


class CTRCNet(nn.Module):
    def __init__(self, enc_shape, Kernel_sz, Nc):
        super(CTRCNet, self).__init__()
        self.encoder = Encoder(enc_shape, Kernel_sz, Nc)
        self.asfm = ASFM(in_channel=3072, out_channel=3072)
        self.decoder = Decoder(enc_shape, Kernel_sz, Nc)

    def forward(self, x, snr, cr, channel_type='AWGN'):
        z = self.encoder(x, snr)
        z = self.asfm(z, snr, cr)
        z = Channel(z, snr, channel_type)
        out = self.decoder(z, snr)
        return out


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_input = torch.randn((128, 3, 32, 32)).to(device)
    enc_shape = [48, 8, 8]
    enc_N = enc_shape[0]
    SNR_TRAIN = torch.randint(0, 28, (x_input.shape[0], 1)).cuda()
    CR = 0.1 + 0.9 * torch.rand(x_input.shape[0], 1).cuda()
    model = CTRCNet(enc_shape, 5, 256).to(device)
    o = model(x_input, SNR_TRAIN, CR, "AWGN")
