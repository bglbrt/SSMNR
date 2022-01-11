#!/usr/bin/env python

# numerical and computer vision libraries
import torch
import torch.nn as nn

class N2VS(nn.Module):
    '''
    N2VS model.
    '''

    def __init__(self):
        '''
        Initialization function.
        '''

        super(N2VS, self).__init__()

        # ENCODER
        self.encode = nn.Sequential(
            nn.Conv2d(3, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2)
        )

        # DECODER
        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # OUTPUT
        self.output = nn.Conv2d(48, 3, 1)

    def forward(self, x):
        '''
        Forward function.

        Arguments:
            x: torch.Tensor
                - N2VS input

        Returns:
            x: torch.Tensor
                - NV2S output
        '''

        # ENCODER
        encoder = self.encode(x)

        # DECODER
        decoder = self.decode(encoder)

        # OUTPUT
        x = self.output(decoder)

        # return x
        return x.double()

class N2V(nn.Module):
    '''
    N2V model.
    '''

    def __init__(self, n_feat):
        '''
        Initialization function.
        '''

        super(N2V, self).__init__()

        self.n_feat = n_feat

        #########
        # ENCODER

        # N2V U-NET encoding in block
        self.encode_block_in = self._encode_block_in(1 * self.n_feat)

        # N2V U-NET encoding blocks
        self.encode_block_1 = self._encode_block(1 * self.n_feat)
        self.encode_block_2 = self._encode_block(2 * self.n_feat)

        # N2V U-NET encoding out block
        self.encode_block_out = self._encode_block_out(4 * self.n_feat)

        #########
        # DECODER

        # N2V U-NET decoding up-sampling block
        self.decode_block_up = self._decode_block_up()

        # N2V U-NET decoding in block
        self.decode_block_in = self._decode_block_in(8 * self.n_feat, 2 * self.n_feat)

        # N2V U-NET decoding blocks
        self.decode_block_1 = self._decode_block(4 * self.n_feat, 1 * self.n_feat)

        # N2V U-NET decoding out block
        self.decode_block_out = self._decode_block_out(2 * self.n_feat)

        ########
        # OUTPUT

        # N2V U-NET output block
        self.output_block = self._output_block(self.n_feat)

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
        pool1 = self.encode_block_1(pool_in)
        pool2 = self.encode_block_2(pool1)

        # N2V U-NET encoding out block
        pool_out = self.encode_block_out(pool2)

        #########
        # DECODER

        # N2V U-NET decoding up-sampling block
        upsample = self.decode_block_up(pool_out)
        concat_up = torch.cat((upsample, pool1), dim=1)

        # N2V U-NET decoding in block
        upsample_in = self.decode_block_in(concat_up)
        concat_in = torch.cat((upsample_in, pool_in), dim=1)

        # N2V U-NET decoding blocks
        upsample_1 = self.decode_block_1(concat_in)
        concat_1 = torch.cat((upsample_1, x), dim=1)

        # N2V U-NET decoding out block
        x = self.decode_block_out(concat_1)

        ########
        # OUTPUT

        # N2V U-NET output block
        x = self.output_block(x)

        # return x
        return x.double()

    def _encode_block_in(self, n_feat):
        '''
        N2V U-NET encoding in block.
        '''

        # initialise block
        block = nn.Sequential(
            nn.Conv2d(3, n_feat, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(n_feat, n_feat, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2)
        )

        # return block
        return block

    def _encode_block(self, n_feat):
        '''
        N2V U-NET encoding block.
        '''

        # initialise block
        block = nn.Sequential(
            nn.Conv2d(n_feat, 2 * n_feat, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(2 * n_feat, 2 * n_feat, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2)
        )

        # return block
        return block

    def _encode_block_out(self, n_feat):
        '''
        N2V U-NET encoding out block.
        '''

        # initialise block
        block = nn.Sequential(
            nn.Conv2d(n_feat, 2 * n_feat, 3, stride=1, padding=1),
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
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(n_feat, int(n_feat / 2), 3, stride=1, padding=1),
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
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(n_feat, int(n_feat / 2), 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest")
        )

        # return block
        return block

    def _decode_block_out(self, n_feat):
        '''
        N2V U-NET decoding out block.
        '''

        # initialise block
        block = nn.Sequential(
            nn.Conv2d(n_feat + 3, n_feat, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(n_feat, int(n_feat / 2), 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # return block
        return block

    def _output_block(self, n_feat):
        '''
        N2V U-NET output block.
        '''

        # initialise block
        block = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(n_feat, n_feat, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(n_feat, 3, 1)
        )

        # return block
        return block

class SSDN(nn.Module):
    '''
    SSDN model.
    '''

    def __init__(self):
        '''
        Initialization function.
        '''

        super(SSDN, self).__init__()

        #########
        # ENCODER

        # SSDN U-NET encoding in block
        self.encode_block_in = self._encode_block_in()

        # SSDN U-NET encoding blocks
        self.encode_block_1 = self._encode_block()
        self.encode_block_2 = self._encode_block()
        self.encode_block_3 = self._encode_block()
        self.encode_block_4 = self._encode_block()

        # SSDN U-NET encoding out block
        self.encode_block_out = self._encode_block_out()

        #########
        # DECODER

        # SSDN U-NET decoding up-sampling block
        self.decode_block_up = self._decode_block_up()

        # SSDN U-NET decoding in block
        self.decode_block_in = self._decode_block_in()

        # SSDN U-NET decoding blocks
        self.decode_block_1 = self._decode_block()
        self.decode_block_2 = self._decode_block()
        self.decode_block_3 = self._decode_block()
        self.decode_block_4 = self._decode_block()

        # SSDN U-NET decoding out block
        self.decode_block_out = self._decode_block_out()

        ########
        # OUTPUT

        # SSDN U-NET output block
        self.output_block = self._output_block()

    def forward(self, x):
        '''
        Forward function.

        Arguments:
            x: torch.Tensor
                - SSDN input Tensor

        Returns:
            x: torch.Tensor
                - SSDN output Tensor
        '''

        #########
        # ENCODER

        # SSDN U-NET encoding in block
        pool_in = self.encode_block_in(x)

        # SSDN U-NET encoding blocks
        pool1 = self.encode_block_1(pool_in)
        pool2 = self.encode_block_2(pool1)
        pool3 = self.encode_block_3(pool2)
        pool4 = self.encode_block_4(pool3)

        # SSDN U-NET encoding out block
        pool_out = self.encode_block_out(pool4)

        #########
        # DECODER

        # SSDN U-NET decoding up-sampling block
        upsample = self.decode_block_up(pool_out)
        concat_up = torch.cat((upsample, pool4), dim=1)

        # SSDN U-NET decoding in block
        upsample_in = self.decode_block_in(concat_up)
        concat_in = torch.cat((upsample_in, pool3), dim=1)

        # SSDN U-NET decoding blocks
        upsample_1 = self.decode_block_1(concat_in)
        concat_1 = torch.cat((upsample_1, pool2), dim=1)
        upsample_2 = self.decode_block_2(concat_1)
        concat_2 = torch.cat((upsample_2, pool1), dim=1)
        upsample_3 = self.decode_block_3(concat_2)
        concat_3 = torch.cat((upsample_3, pool_in), dim=1)
        upsample_4 = self.decode_block_4(concat_3)
        concat_4 = torch.cat((upsample_4, x), dim=1)

        # SSDN U-NET decoding out block
        x = self.decode_block_out(concat_4)

        ########
        # OUTPUT

        # SSDN U-NET output block
        x = self.output_block(x)

        # return x
        return x.double()

    def _encode_block_in(self):
        '''
        SSDN U-NET encoding in block.
        '''

        # initialise block
        block = nn.Sequential(
            nn.Conv2d(3, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2)
        )

        # return block
        return block

    def _encode_block(self):
        '''
        SSDN U-NET encoding block.
        '''

        # initialise block
        block = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2)
        )

        # return block
        return block

    def _encode_block_out(self):
        '''
        SSDN U-NET encoding out block.
        '''

        # initialise block
        block = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # return block
        return block

    def _decode_block_up(self):
        '''
        SSDN U-NET decoding up-sampling block.
        '''

        # initialise block
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest")
        )

        # return block
        return block

    def _decode_block_in(self):
        '''
        SSDN U-NET decoding in block.
        '''

        # initialise block
        block = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest")
        )

        # return block
        return block

    def _decode_block(self):
        '''
        SSDN U-NET decoding block.
        '''

        # initialise block
        block = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest")
        )

        # return block
        return block

    def _decode_block_out(self):
        '''
        SSDN U-NET decoding out block.
        '''

        # initialise block
        block = nn.Sequential(
            nn.Conv2d(96 + 3, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # return block
        return block

    def _output_block(self):
        '''
        SSDN U-NET output block.
        '''

        # initialise block
        block = nn.Sequential(
            nn.Conv2d(96, 96, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(96, 96, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(96, 3, 1)
        )

        # return block
        return block

# function to initialise transformer model
def initialize_model(model_name, n_feat):
    '''
    Model initialization.

    Arguments:
        model_name: str
            - name of model

    Returns:
        model: torchvision.model
            - model
    '''

    # N2VS model
    if model_name == 'N2VS':

        # initialise N2VS
        model = N2VS()

    # N2V model
    elif model_name == 'N2V':

        # initialise N2V
        model = N2V(n_feat)

    # SSDN model
    elif model_name == 'SSDN':

        # initialise N2V
        model = SSDN()

    else:
        raise Exception("Error! Model name not recognised. Please use either: N2VS, N2V.")

    # return model
    return model
