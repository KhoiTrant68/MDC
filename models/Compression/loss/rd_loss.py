import math

import torch
import torch.nn as nn


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in
                              output["likelihoods"].values())
        out["ssim_loss"] = output["loss"][0]
        out["L1_loss"] = output["loss"][1]
        out["vgg_loss"] = output["loss"][2]

        # gather
        out["loss"] = self.lmbda * (0.25 * out["ssim_loss"] + 10 *
                                    out["L1_loss"] + 0.1 * out["vgg_loss"]) + out["bpp_loss"]
        return out
