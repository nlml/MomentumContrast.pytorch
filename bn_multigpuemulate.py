import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class _BatchNormEmulateMultiGPU(_BatchNorm):

    def __init__(self, num_features, num_groups, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, shuffle=True):
        self.num_groups = num_groups
        self.shuffle = shuffle
        super(_BatchNormEmulateMultiGPU, self).__init__(
            num_features * self.num_groups, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        orig_shape = input.shape
        N, C = orig_shape[:2]
        perm = torch.randperm(N)
        assert N % self.num_groups == 0
        input_reshaped = input
        if self.shuffle:
            input_reshaped = input_reshaped[perm]
        input_reshaped = input.view(N // self.num_groups, C * self.num_groups, *input.shape[2:])
        out = super(_BatchNormEmulateMultiGPU, self).forward(input_reshaped)
        out = out.view(orig_shape)
        if self.shuffle:
            out = out[perm]
        if self.train:
            self.running_mean = self.running_mean.view(self.num_groups, -1).mean(0).repeat(self.num_groups)
            self.running_var = self.running_var.view(self.num_groups, -1).mean(0).repeat(self.num_groups)
        return out

    def extra_repr(self):
        return '{num_features}, num_groups={num_groups}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

class BatchNorm1dEmulateMultiGPU(_BatchNormEmulateMultiGPU):

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

class BatchNorm2dEmulateMultiGPU(_BatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class BatchNorm3dEmulateMultiGPU(_BatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
