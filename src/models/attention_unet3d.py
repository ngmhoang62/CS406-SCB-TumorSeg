"""M2 convolutional U-Net with attention gates at skip connections."""

from monai.networks.nets import AttentionUnet


class SCBAttentionUNet3D(AttentionUnet):
    def __init__(self, in_channels=2, channels=(32, 64, 128, 256, 320)):
        super().__init__(spatial_dims=3, in_channels=in_channels, out_channels=1,
                         channels=tuple(channels), strides=(2,) * (len(channels) - 1))

    def forward(self, x, subtype=None):
        return super().forward(x)
