import random
from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import COOPFUSERS

__all__ = ["MaxFuser"]


@COOPFUSERS.register_module()
class MaxFuser(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        assert self.in_channels[0] == self.in_channels[1], "The inputs have different number of channels!"
        assert self.in_channels[0] == self.out_channels, "The first input has different number of channels to output!"
        assert self.in_channels[1] == self.out_channels, "The second input has different number of channels to output!"

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        output = torch.fmax(inputs[0], inputs[1])

        if self.training and random.random() < self.dropout:
            output = torch.nn.functional.dropout(output, p=self.dropout)

        return output
