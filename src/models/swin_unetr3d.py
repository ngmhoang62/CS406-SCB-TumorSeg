"""M3 MONAI Swin UNETR, with its main segmentation output."""

import inspect

from monai.networks.nets import SwinUNETR


class SCBSwinUNETR3D(SwinUNETR):
    def __init__(self, in_channels=2, feature_size=48, use_checkpoint=True,
                 patch_size=(96, 96, 96)):
        kwargs = dict(in_channels=in_channels, out_channels=1, spatial_dims=3,
                      feature_size=feature_size, use_checkpoint=use_checkpoint)
        # MONAI 1.4 requires img_size. MONAI 1.5 removed it.
        if "img_size" in inspect.signature(SwinUNETR).parameters:
            kwargs["img_size"] = tuple(patch_size)
        super().__init__(**kwargs)

    def forward(self, x, subtype=None):
        return super().forward(x)
