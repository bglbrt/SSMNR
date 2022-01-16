#!/usr/bin/env python

# numerical and computer vision libraries
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class N2V(nn.Module):
    '''
    N2V model.
    '''

    def __init__(self, n_channels, n_feat):
        '''
        Initialization function.
        '''

        super(N2V, self).__init__()

        self.n_channels = n_channels
        self.n_feat = n_feat

        #########
        # ENCODER

        # N2V U-NET encoding in block
        self.encode_block_in = self._encode_block_in(self.n_feat)

        # N2V U-NET encoding blocks
        self.encode_block = self._encode_block(2 * self.n_feat)

        # N2V U-NET encoding out block
        self.encode_block_out = self._encode_block_out(4 * self.n_feat)

        #########
        # DECODER

        # N2V U-NET decoding up-sampling block
        self.decode_block_up = self._decode_block_up()

        # N2V U-NET decoding in block
        self.decode_block_in = self._decode_block_in(2 * self.n_feat, 2 * self.n_feat)

        # N2V U-NET decoding blocks
        self.decode_block = self._decode_block(self.n_feat, self.n_feat)

        ########
        # OUTPUT

        # N2V U-NET output block
        self.output_block = self._output_block(3, 3)

    def forward(self, x):
        '''
        Forward function.

        Arguments:
            x: torch.Tensor
                - N2V input Tensor

        Returns:
            x: torch.Tensor
                - N2V output Tensor
        '''

        #########
        # ENCODER

        # N2V U-NET encoding in block
        pool_in = self.encode_block_in(x)

        # N2V U-NET encoding blocks
        pool = self.encode_block(pool_in)

        # N2V U-NET encoding out block
        pool_out = self.encode_block_out(pool)

        #########
        # DECODER

        # N2V U-NET decoding up-sampling block
        upsample = self.decode_block_up(pool_out)
        concat = torch.cat((upsample, pool), dim=1)

        # N2V U-NET decoding in block
        upsample_in = self.decode_block_in(concat)
        concat_in = torch.cat((upsample_in, pool_in), dim=1)

        # N2V U-NET decoding block
        decode_out = self.decode_block(concat_in)
        concat_out = torch.concat((decode_out, x), dim=1)

        ########
        # OUTPUT

        # N2V U-NET output block
        x = self.output_block(concat_out)

        # return x
        return x

    def _encode_block_in(self, n_feat):
        '''
        N2V U-NET encoding in block.
        '''

        # initialise block
        block = nn.Sequential(
            nn.Conv2d(self.n_channels, n_feat, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # return block
        return block

    def _encode_block(self, n_feat):
        '''
        N2V U-NET encoding block.
        '''

        # initialise block
        block = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(int(n_feat / 2), n_feat, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # return block
        return block

    def _encode_block_out(self, n_feat):
        '''
        N2V U-NET encoding out block.
        '''

        # initialise block
        block = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(int(n_feat / 2), n_feat, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(n_feat, int(n_feat / 2), 3, stride=1, padding=1),
            nn.BatchNorm2d(int(n_feat / 2)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # return block
        return block

    def _decode_block_up(self):
        '''
        N2V U-NET decoding up-sampling block.
        '''

        # initialise block
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest")
        )

        # return block
        return block

    def _decode_block_in(self, n_feat, n_feat_concat):
        '''
        N2V U-NET decoding in block.
        '''

        # initialise block
        block = nn.Sequential(
            nn.Conv2d(n_feat + n_feat_concat, n_feat, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(n_feat, int(n_feat / 2), 3, stride=1, padding=1),
            nn.BatchNorm2d(int(n_feat / 2)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest")
        )

        # return block
        return block

    def _decode_block(self, n_feat, n_feat_concat):
        '''
        N2V U-NET decoding block.
        '''

        # initialise block
        block = nn.Sequential(
            nn.Conv2d(n_feat + n_feat_concat, n_feat, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(n_feat, 3, 3, stride=1, padding=1)
        )

        # return block
        return block

    def _output_block(self, n_feat, n_feat_concat):
        '''
        N2V U-NET output block.
        '''

        # initialise block
        block = nn.Sequential(
            nn.Conv2d(n_feat + n_feat_concat, n_feat, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(n_feat, self.n_channels, 3, stride=1, padding=1)
        )

        # return block
        return block

########################################################
########################################################
########################################################
######################### N2VT #########################
########################################################
########################################################
########################################################

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MultiHeadDense(nn.Module):
    def __init__(self, d, bias=False):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))
        if bias:
            raise NotImplementedError()
            self.bias = Parameter(torch.Tensor(d, d))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x:[b, h*w, d]
        b, wh, d = x.size()
        x = torch.bmm(x, self.weight.repeat(b, 1, 1))
        # x = F.linear(x, self.weight, self.bias)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def positional_encoding_2d(self, d_model, height, width):
        """
        reference: wzlxjtu/PositionalEncoding2D
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        try:
            pe = pe.to(torch.device("cuda:0"))
        except RuntimeError:
            pass
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe

    def forward(self, x):
        raise NotImplementedError()


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000
                         **(torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x,
                             device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y,
                             device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()),
                          dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2),
                          device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)


class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, channel):
        super(MultiHeadSelfAttention, self).__init__()
        self.query = MultiHeadDense(channel, bias=False)
        self.key = MultiHeadDense(channel, bias=False)
        self.value = MultiHeadDense(channel, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.pe = PositionalEncodingPermute2D(channel)

    def forward(self, x):
        b, c, h, w = x.size()
        # pe = self.positional_encoding_2d(c, h, w)
        pe = self.pe(x)
        x = x + pe
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  #[b, h*w, d]
        Q = self.query(x)
        K = self.key(x)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) /
                         math.sqrt(c))  #[b, h*w, h*w]
        V = self.value(x)
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(b, c, h, w)
        return x


class MultiHeadCrossAttention(MultiHeadAttention):
    def __init__(self, channelY, channelS):
        super(MultiHeadCrossAttention, self).__init__()
        self.Sconv = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.Yconv = nn.Sequential(
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.query = MultiHeadDense(channelS, bias=False)
        self.key = MultiHeadDense(channelS, bias=False)
        self.value = MultiHeadDense(channelS, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.Yconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channelY, channelY, kernel_size=3, padding=1),
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.softmax = nn.Softmax(dim=1)
        self.Spe = PositionalEncodingPermute2D(channelS)
        self.Ype = PositionalEncodingPermute2D(channelY)

    def forward(self, Y, S):
        Sb, Sc, Sh, Sw = S.size()
        Yb, Yc, Yh, Yw = Y.size()
        # Spe = self.positional_encoding_2d(Sc, Sh, Sw)
        Spe = self.Spe(S)
        S = S + Spe
        S1 = self.Sconv(S).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)
        V = self.value(S1)
        # Ype = self.positional_encoding_2d(Yc, Yh, Yw)
        Ype = self.Ype(Y)
        Y = Y + Ype
        Y1 = self.Yconv(Y).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)
        Y2 = self.Yconv2(Y)
        Q = self.query(Y1)
        K = self.key(Y1)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(Sc))
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(Yb, Sc, Yh, Yw)
        Z = self.conv(x)
        Z = Z * S
        Z = torch.cat([Z, Y2], dim=1)
        return Z


class TransformerUp(nn.Module):
    def __init__(self, Ychannels, Schannels):
        super(TransformerUp, self).__init__()
        self.MHCA = MultiHeadCrossAttention(Ychannels, Schannels)
        self.conv = nn.Sequential(
            nn.Conv2d(Ychannels,
                      Schannels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(Schannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(Schannels,
                      Schannels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(Schannels),
            nn.ReLU(inplace=True))

    def forward(self, Y, S):
        x = self.MHCA(Y, S)
        x = self.conv(x)
        return x


class N2VT(nn.Module):
    def __init__(self, n_channels, n_feat):
        super(N2VT, self).__init__()

        self.n_channels = n_channels
        self.n_feat = n_feat

        self.inc = DoubleConv(n_channels, self.n_feat)
        self.down1 = Down(self.n_feat, 2 * self.n_feat)
        self.down2 = Down(2 * self.n_feat, 4 * self.n_feat)
        self.down3 = Down(4 * self.n_feat, 8 * self.n_feat)
        self.MHSA = MultiHeadSelfAttention(8 * self.n_feat)
        self.up1 = TransformerUp(8 * self.n_feat, 4 * self.n_feat)
        self.up2 = TransformerUp(4 * self.n_feat, 2 * self.n_feat)
        self.up3 = TransformerUp(2 * self.n_feat, self.n_feat)
        self.outc = OutConv(self.n_feat, n_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.MHSA(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x

# function to initialise transformer model
def initialize_model(model_name, n_channels, n_feat):
    '''
    Model initialization.

    Arguments:
        model_name: str
            - name of model

    Returns:
        model: torchvision.model or str
            - model
    '''

    # N2V model
    if model_name == 'N2V':

        # initialise N2V
        model = N2V(n_channels, n_feat)

    # N2VT model
    elif model_name == 'N2VT':

        # initialise N2VT
        model = N2VT(n_channels, n_feat)

    # evaluation models: MEAN, MEDIAN and BM3D
    elif model_name in ['MEAN', 'MEDIAN', 'BM3D']:

        # print warning about training
        print('Warning! ' + model_name + ' can only be used for evaluation!')

        model = model_name

    # raise exception otherwise
    else:
        raise Exception("Error! Model name not recognised. Please use either: N2VS, N2V.")

    # return model
    return model
