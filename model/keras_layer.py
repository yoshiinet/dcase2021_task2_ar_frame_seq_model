# Copyright (c) 2021 ralabo.jp
# This software is released under the MIT License.
# see https://opensource.org/licenses/mit-license.php
# ====================================================
import torch
from torch import nn
from torch.nn import functional as F

def Dense(dim_in, dim_out):
    """
    Linear layer with initialization as same as keras
    """
    m = nn.Linear(dim_in, dim_out)
    nn.init.xavier_uniform_(m.weight,gain=1)
    nn.init.zeros_(m.bias)
    return m

def BatchNormalization(dim_out):
    m = nn.BatchNorm1d(dim_out, eps=0.001, momentum=0.99)
    return m
