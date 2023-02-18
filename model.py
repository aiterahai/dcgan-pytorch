"""
file name: model.py

create time: 2023-02-18 05:41
author: Tera Ha
e-mail: terra2007@naver.com
github: https://github.com/terra2007
"""
import torch.nn as nn


# weights initialization called on netG and netD
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    """
    Generator model proposed in DCGAN paper.
    """

    def __init__(self, input_dim, output_dim, ngf):
        """
        Args:
            input_dim (int) : Number of dimensions in model input.
            output_dim (int) : Number of dimensions in model output.
            ngf (int) : Size of feature maps in generator
        """
        super(Generator, self).__init__()
        # Input is Z, going into a convolution
        self.net = nn.Sequential(
            self.block(self.block(input_dim, ngf * 16, stride=1, padding=0)),  # ((ngf * 16) x 4 x 4)
            self.block(self.block(ngf * 16, ngf * 8)),  # ((ngf * 8) x 8 x 8)
            self.block(self.block(ngf * 8, ngf * 4)),  # ((ngf * 4) x 16 x 16)
            self.block(self.block(ngf * 4, ngf * 2)),  # ((ngf * 2) x 32 x 32)
            self.block(self.block(ngf * 2, output_dim))  # (output_dim x 64 x 64)
        )

    def block(self,
              in_channel,
              out_channel,
              kernel_size=4,
              stride=2,
              padding=1,
              final_layer=False):

        layers = nn.Sequential()

        layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=False))

        if final_layer:
            layers.append(nn.Tanh())
        else:
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))

        return layers

    def forward(self, noise):
        return self.net(noise)


class Discriminator(nn.Module):
    """
    Discriminator model proposed in DCGAN paper.
    """

    def __init__(self, input_dim, ndf):
        """
        Args:
            input_dim (int) : Number of dimensions in model input.
            ndf (int) : Size of feature maps in discriminator
        """
        super(Discriminator, self).__init__()
        # Input : (nc x 64 x 64)
        self.net = nn.Sequential(
            self.block(input_dim, ndf),  # ((ndf) x 32 x 32)
            self.block(ndf, ndf * 2),  # (ndf * 2) x 16 x 16
            self.block(ndf * 2, ndf * 4),  # (ndf * 4) x 8 x 8
            self.block(ndf * 4, ndf * 8),  # (ndf * 8) x 4 x 4
            self.block(ndf * 8, 1, padding=0, final_layer=True),
            nn.Sigmoid()
        )

    def block(self,
              in_channel,
              out_channel,
              kernel_size=4,
              stride=2,
              padding=1,
              final_layer=False):

        layers = nn.Sequential()

        layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False))

        if not final_layer:
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        return layers

    def forward(self, image):
        return self.net(image)
