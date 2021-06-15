# Copyright (c) 2021 ralabo.jp
# This software is released under the MIT License.
# see https://opensource.org/licenses/mit-license.php
# ====================================================
import os
import numpy as np
import torch

def provide_modeler(com, fit_key=None, eval_key=None):
    param = com.param


    input_dim = param["feature"]["n_mels"] * param["feature"]["n_frames"]
    model=param['net']['model']

    if model == 'pptunetf': # preprocessing tuning transformers system
        from model.pptunetf import PrepTuneTransformerModel # on demand import
        class Model(PrepTuneTransformerModel):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            def __init__(self,input_dim):
                super().__init__(param, fit_key, eval_key)

    else:
        assert model in ['pptunetf']

    modeler = Model(input_dim)
    modeler.to(modeler.device)
    return modeler
