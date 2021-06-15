# Copyright (c) 2021 ralabo.jp
# This software is released under the MIT License.
# see https://opensource.org/licenses/mit-license.php
# ====================================================
from torch.utils.data import DataLoader

class Task2DataLoader(DataLoader):
    def __init__(self,dataset,batch_size,shuffle,drop_last,unit):
        dataset.set_unit(unit)
        super().__init__(dataset, batch_size=batch_size,
                         shuffle=shuffle, drop_last=bool(drop_last))
