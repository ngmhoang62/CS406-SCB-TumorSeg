"""M1 PlainConvUNet with optional dual encoders and a training boundary head."""

from copy import deepcopy

import torch
import torch.nn as nn


class SpatialFusion3D(nn.Module):
    """Learn two modality weights at each voxel, shared across feature channels."""

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Conv3d(2 * channels, 2, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, post1, subtraction):
        weights = self.gate(torch.cat([post1, subtraction], dim=1)).softmax(dim=1)
        return weights[:, 0:1] * post1 + weights[:, 1:2] * subtraction


class NNUNet3D(nn.Module):
    def __init__(self, in_channels=2, features=(32, 64, 128, 256, 320, 320),
                 encoder_fusion="concat", boundary_aux=False):
        super().__init__()
        from dynamic_network_architectures.architectures.unet import PlainConvUNet

        if encoder_fusion not in {"concat", "add", "gated"}:
            raise ValueError("encoder_fusion must be concat, add or gated")
        if encoder_fusion != "concat" and in_channels != 2:
            raise ValueError("Dual encoders require IP3 with Post1 and subtraction channels")
        self.encoder_fusion = encoder_fusion
        self.boundary_aux = boundary_aux
        stages = len(features)
        self.network = PlainConvUNet(
            input_channels=in_channels if encoder_fusion == "concat" else 1,
            n_stages=stages, features_per_stage=list(features),
            conv_op=nn.Conv3d, kernel_sizes=[(3, 3, 3)] * stages,
            strides=[(1, 1, 1)] + [(2, 2, 2)] * (stages - 1),
            n_conv_per_stage=2, num_classes=1, n_conv_per_stage_decoder=2,
            conv_bias=True, norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True}, dropout_op=None,
            nonlin=nn.LeakyReLU, nonlin_kwargs={"negative_slope": 0.01, "inplace": True},
            deep_supervision=False,
        )
        self.network.apply(PlainConvUNet.initialize)
        self.subtraction_encoder = None
        self.fusions = nn.ModuleList()
        if encoder_fusion != "concat":
            self.subtraction_encoder = deepcopy(self.network.encoder)
            self.subtraction_encoder.apply(PlainConvUNet.initialize)
            if encoder_fusion == "gated":
                self.fusions.extend(SpatialFusion3D(channels) for channels in features)
        self.boundary_head = None
        if boundary_aux:
            self.boundary_head = nn.Conv3d(features[0], 1, kernel_size=1)
            self.boundary_head.apply(PlainConvUNet.initialize)

    def forward(self, x, subtype=None):
        if self.encoder_fusion == "concat" and not self.boundary_aux:
            return self.network(x)
        if self.subtraction_encoder is None:
            skips = self.network.encoder(x)
        else:
            if x.shape[1] != 2:
                raise ValueError("Dual encoders expect (B,2,D,H,W) in [Post1, subtraction] order")
            post1 = self.network.encoder(x[:, 0:1])
            subtraction = self.subtraction_encoder(x[:, 1:2])
            if self.encoder_fusion == "add":
                skips = [first + second for first, second in zip(post1, subtraction)]
            else:
                skips = [fusion(first, second) for fusion, first, second
                         in zip(self.fusions, post1, subtraction)]
        decoder = self.network.decoder
        if not self.boundary_aux:
            return decoder(skips)
        # Reuse the official decoder blocks and expose the final feature map
        # to the auxiliary head, without storing per-forward state or hooks.
        features = skips[-1]
        for upsample, stage, skip in zip(decoder.transpconvs, decoder.stages,
                                         reversed(skips[:-1])):
            features = stage(torch.cat([upsample(features), skip], dim=1))
        segmentation = decoder.seg_layers[-1](features)
        if self.training:
            return {"seg": segmentation, "boundary": self.boundary_head(features)}
        return segmentation
