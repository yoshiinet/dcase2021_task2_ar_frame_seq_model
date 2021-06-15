# Copyright (c) 2021 ralabo.jp
# This software is released under the MIT License.
# see https://opensource.org/licenses/mit-license.php
# ====================================================
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset

class Task2Dataset(Dataset):
    """
    basedata is a Dataset comprised of ( block, y_true, file_path )
    """
    def __init__(self, basedata, unit='frame'):
        self.set_unit(unit)

        if isinstance(basedata, Subset):# splitの結果の時
            basedataset = basedata.dataset
        else:
            basedataset = basedata

        self.basedata = basedataset
        self.basedata_memory = basedataset.basedata_memory

        if self.basedata_memory == 'compact':
            # 一つのファイル当たりの特徴ベクトル数を計算する
            self.n_vectors = basedataset.n_vectors
            self.n_frames = basedataset.n_frames
            self.dims = basedataset.dims
            self.n_mels = basedataset.n_mels
        elif self.basedata_memory == 'extend':
            # 一つのファイル当たりの特徴ベクトル数
            self.n_vectors = len(basedataset.data[0])
        else:
            assert self.basedata_memory in ['compact', 'extend']


    def set_unit(self,unit):
        assert unit in ['frame', 'block']
        self.unit = unit

    def __len__(self):
        if self.unit=='frame':
            return len(self.basedata) * self.n_vectors
        elif self.unit=='block':
            return len(self.basedata)
        else:
            assert self.unit in ['frame', 'block']

    def __getitem__(self, index):
        if self.basedata_memory == 'compact': # フレーム系列から特徴ベクトル列を生成する場合
            if self.unit=='frame':
                file_idx = index//self.n_vectors # ファイル位置を取得する
                frames, y_true, file_path = self.basedata[file_idx]
                frame_offset = index - file_idx * self.n_vectors # フレーム位置
                vector = frames[frame_offset:frame_offset+self.n_frames, :].flatten()
                return vector, y_true
            elif self.unit=='block':
                # フレームを連結して特徴ベクトル列に変換する
                frames, y_true, file_path = self.basedata[index] # 1ファイルのフレーム列
                block = np.empty((self.n_vectors, self.dims), np.float32) # alloc memory
                for t in range(self.n_frames): # for cocatenating frames
                    # 't'ずらしたフレーム列([n_vectors,n_mels])を
                    # blockのt*n_mels次元から始まる位置にコピーする
                    block[:, self.n_mels*t:self.n_mels*(t+1)] = frames[t:t+self.n_vectors, :]
                return block, y_true, file_path
            else:
                assert self.unit in ['frame', 'block']
        elif self.basedata_memory == 'extend': # 展開済みの特徴ベクトル列をそのまま使う場合
            if self.unit=='frame':
                file_idx = index//self.n_vectors
                frame_offset = index - file_idx * self.n_vectors
                block, y_true, file_path = self.basedata[file_idx]
                frame = block[frame_offset]
                return frame, y_true
            elif self.unit=='block':
                return self.basedata[index]
            else:
                assert self.unit in ['frame', 'block']
        else:
            assert self.basedata_memory in ['compact', 'extend']

"""
実行時間の比較(compact vs. extend)

basedata_memroy=compact
datasize: 6018 --> 100
generate dataset * / train section_* *: 100%|##########| 100/100 [00:02<00:00, 46.59it/s]
starting from epoch 0
epoch:0 loss:12653.2 val_loss:12503.3 frames:11 lr:1e-05 elapsed:29.3s
epoch:1 loss:12393.6 val_loss:12252.7 frames:11 lr:1e-05 elapsed:28.7s
epoch:2 loss:12111.9 val_loss:11949.2 frames:11 lr:1e-05 elapsed:28.6s
epoch:3 loss:11796.7 val_loss:11625.3 frames:11 lr:1e-05 elapsed:28.9s
epoch:4 loss:11465.5 val_loss:11287.8 frames:11 lr:1e-05 elapsed:28.5s
epoch:5 loss:11121.3 val_loss:10938.4 frames:11 lr:1e-05 elapsed:28.8s
epoch:6 loss:10765.7 val_loss:10578.2 frames:11 lr:1e-05 elapsed:28.7s
epoch:7 loss:10400 val_loss:10208.5 frames:11 lr:1e-05 elapsed:29.8s
epoch:8 loss:10025.3 val_loss:9830.32 frames:11 lr:1e-05 elapsed:29.7s
epoch:9 loss:9643.09 val_loss:9445.28 frames:11 lr:1e-05 elapsed:28.8s
total elapsed: 338.429s

basedata_memroy=extend
datasize: 6018 --> 100
generate dataset * / train section_* *: 100%|##########| 100/100 [00:01<00:00, 63.31it/s]
epoch:0 loss:12653.2 val_loss:12503.3 frames:11 lr:1e-05 elapsed:28.5s
epoch:1 loss:12393.6 val_loss:12252.7 frames:11 lr:1e-05 elapsed:28.0s
epoch:2 loss:12111.9 val_loss:11949.2 frames:11 lr:1e-05 elapsed:27.9s
epoch:3 loss:11796.7 val_loss:11625.3 frames:11 lr:1e-05 elapsed:27.9s
epoch:4 loss:11465.5 val_loss:11287.8 frames:11 lr:1e-05 elapsed:27.9s
epoch:5 loss:11121.3 val_loss:10938.4 frames:11 lr:1e-05 elapsed:28.1s
epoch:6 loss:10765.7 val_loss:10578.2 frames:11 lr:1e-05 elapsed:28.1s
epoch:7 loss:10400 val_loss:10208.5 frames:11 lr:1e-05 elapsed:28.1s
epoch:8 loss:10025.3 val_loss:9830.32 frames:11 lr:1e-05 elapsed:28.2s
epoch:9 loss:9643.09 val_loss:9445.28 frames:11 lr:1e-05 elapsed:28.5s
total elapsed: 320.872s

"""