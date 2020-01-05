import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class _BatchNormEmulateMultiGPU(_BatchNorm):
    def __init__(
        self,
        num_features,
        num_groups,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        shuffle=True,
    ):
        self.num_groups = num_groups
        self.shuffle = shuffle
        super(_BatchNormEmulateMultiGPU, self).__init__(
            num_features * self.num_groups,
            eps,
            momentum,
            affine,
            track_running_stats,
        )

    def forward(self, x):
        N, C = x.shape[:2]
        orig_shape = x.shape
        assert N % self.num_groups == 0

        # Shuffle inputs
        if self.shuffle:
            perm = torch.randperm(N)
            x = x[perm]

        # Reshape for grouped batch norm
        x = x.view(N // self.num_groups, C * self.num_groups, *x.shape[2:])

        # Apply batch norm
        x = super(_BatchNormEmulateMultiGPU, self).forward(x)

        # Reshape to input shape
        x = x.view(orig_shape)

        # Undo shuffling if necessary
        if self.shuffle:
            x = x[torch.arange(len(perm))[torch.argsort(perm)]]

        # Running mean and var is mean across groups
        if self.train:
            self.running_mean = (
                self.running_mean.view(self.num_groups, -1)
                .mean(0)
                .repeat(self.num_groups)
            )
            self.running_var = (
                self.running_var.view(self.num_groups, -1)
                .mean(0)
                .repeat(self.num_groups)
            )

        return x

    def extra_repr(self):
        return (
            "{num_features}, num_groups={num_groups}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )


class BatchNorm1dEmulateMultiGPU(_BatchNormEmulateMultiGPU):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )


class BatchNorm2dEmulateMultiGPU(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(
                "expected 4D input (got {}D input)".format(input.dim())
            )


class BatchNorm3dEmulateMultiGPU(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError(
                "expected 5D input (got {}D input)".format(input.dim())
            )
