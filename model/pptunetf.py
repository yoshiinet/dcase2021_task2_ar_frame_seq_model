# Copyright (c) 2021 ralabo.jp
# This software is released under the MIT License.
# see https://opensource.org/licenses/mit-license.php
# ====================================================
import os
import math
import numpy as np
import random
import time
import json
import glob
import shutil
from collections import defaultdict
import torch
from fast_transformers.builders import TransformerEncoderBuilder
#from torch import optim
import torch_optimizer as optim
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from model.keras_layer import Dense, BatchNormalization
from fast_transformers.masking import TriangularCausalMask, LengthMask, FullMask
from common import com
from utils.safely_saver import SafelySaver


class InputLayer(nn.Module):
    """
    入力層:
    入力フレームの次元を内部ベクトル次元に変換する
    正規化と非線形性を持たせるため、batch normalization + ReLU を使う
    Args:
        type  :
            type == 'linear':
                線形変換のみ
            type == 'non-linear':
                線形変換+レイヤー正規化+ドロップアウト+ReLU
                入力フレーム系列のパターンを崩さないため、
                入力フレーム系列全体の normalization(Layer Normalization) を行う
            type == 'none':
                入力フレームを変換無でそのまま渡す
                TransFormerLayerのd_modelはn_melsと同じでなければならない

        X:  入力フレーム系列

        x  : [ N, S, D ] -> [ N, S, E ]
    """
    def __init__(self, max_seq_len, n_mels, d_model, type='linear', dropout=0.1):
        super().__init__()

        self.type = type
        if self.type=='linear':
            self.fc = Dense(n_mels, d_model)
        elif self.type == 'non-linear':
            self.non_linear_input = nn.Sequential(
                Dense(n_mels, d_model), # Transformerの内部次元数に合わせる
                nn.LayerNorm(d_model), # フレーム列の正規化
                nn.Dropout(dropout), # フレーム列のドロップアウト
                nn.ReLU(), # 非線形化
                )
        elif self.type == 'none':
            assert com.param['feature']['n_mels'] == com.param['net']['d_model'], \
                "inp_layer == 'none'のとき n_mels({}) == d_model({})でなければならない" \
                .format(com.param['feature']['n_mels'], com.param['net']['d_model'])
            pass # no conversion
        else:
            assert type in ['linear', 'non-linear', 'raw-tf6']

    def forward(self,X):
        """
        X : [ N, S, D ]
        Y : [ N, S, E ]
        where, S : src length, N : batch size, D : input dim, E : embed dim
        """
        if self.type=='linear':
            Y = self.fc(X)
        elif self.type=='non-linear':
            Y = self.non_linear_input(X)
        elif self.type == 'none':
            Y = X # no conversion
        else:
            assert type in ['linear', 'non-linear', 'raw-tf6']
        return Y # [N, S, E]

class OutputLayer(nn.Module):
    """
    出力層：
    out_layer type=='linear'

    Transformer_Blockの出力を用い、最後フレームを予測する
    """

    def __init__(self, max_context, d_model, output_dim, type='linear'):
        super().__init__()

        self.max_context =max_context
        self.type = type

        if self.type=='linear':
            # 全結合層
            self.fc = Dense(d_model, output_dim)
        else:
            assert type in ['linear']

    def forward(self, X):
        """
        X : [ N, S, E ]
        Y : [ N, D ]
        where, S : src length, N : batch size, D : input dim, E : embed dim
        """
        out_layer = self.type
        assert out_layer in ['linear']

        if self.type=='linear':
            x_last = X[:, self.max_context, :]  # 系列の最後の要素, -> [N, E]
            Y = self.fc(x_last) # -> [N, D]

        return Y

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_seq_len):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # バッチ次元に1を追加
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x : [N, S, E]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)

class PpTransformerLayersModel(nn.Module):
    """
    前処理(ppNtf_model)： Nは層数
        Transformer層3層により、入力フレーム系列を出力フレーム系列に写像する
    """
    def __init__(self, parent, n_layers=3):
        super().__init__()

        # 前処理
        # https://fast-transformers.github.io/transformers/#transformerencoderlayer
        d_model = parent.n_mels # フレームの次元数
        nhead = 8
        attn_type = parent.attn_type
        d_ff = d_model * 4 # d_modelの4倍が適切らしい
        dropout_factor = parent.dropout_factor

        assert d_model % nhead == 0,\
            'd_model({})はnhead{})で割り切れる必要がある'.format(d_model, nhead)

        self.pp3tf_model = TransformerEncoderBuilder.from_kwargs(
            n_layers=n_layers,
            n_heads=nhead,
            query_dimensions=d_model//nhead,
            value_dimensions=d_model//nhead,
            feed_forward_dimensions=d_ff,
            attention_type = attn_type,
            activation="relu", # activation: Choose which activation to use
                                # for the feed forward part of the layer from
                                # the set {'relu', 'gelu'} (default: relu)
            dropout=dropout_factor,
            attention_dropout=dropout_factor,
            ).get()
    
    def forward(self, x):
        # 前処理(pp_model)を適用する
        x_pp = self.pp3tf_model(x)
        # pp3tf_modelの出力x_ppを返す
        return x_pp

class PpTransformer3Model(PpTransformerLayersModel):
    """
    前処理(pp3tf_model)：
        Transformer層3層により、入力フレーム系列を出力フレーム系列に写像する
    """
    def __init__(self, parent):
        super().__init__(parent, n_layers=3)

class PpTransformer6Model(PpTransformerLayersModel):
    """
    前処理(pp6tf_model)：
        Transformer層6層により、入力フレーム系列を出力フレーム系列に写像する
    """
    def __init__(self, parent):
        super().__init__(parent, n_layers=6)

class PpTransformer9Model(PpTransformerLayersModel):
    """
    前処理(pp9tf_model)：
        Transformer層6層により、入力フレーム系列を出力フレーム系列に写像する
    """
    def __init__(self, parent):
        super().__init__(parent, n_layers=9)

class PpTransformer12Model(PpTransformerLayersModel):
    """
    前処理(pp12tf_model)：
        Transformer層6層により、入力フレーム系列を出力フレーム系列に写像する
    """
    def __init__(self, parent):
        super().__init__(parent, n_layers=12)


class PreProcessModel(nn.Module):
    """
    前処理(pp_mdeol) + 親(呼出し元)の学習済みモデル(base_model)


    前処理(pp_model)：
        Transformer層3層により、入力フレーム系列を出力フレーム系列に写像する


    base_model_class: base_modelを構築するため
    """
    def __init__(self, parent, pp_model_builder, base_model_builder):
        super().__init__()

        self.device = parent.device
        self.n_frames = parent.n_frames
        self.n_mels = parent.n_mels
        self.parent = parent

        # pp_model
        self.pp_model = pp_model_builder(parent)

        # ベースモデル(親)
        self.base_model = base_model_builder(parent)

        # optimizerをセットする
        lr = parent.lr
        weight_decay = parent.weight_decay
        self.optimizer = optim.RAdam(self.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)


    def forward(self, x):
        """Return: x_pred, x_pp
            x_pred: 予測最終フレーム
            x_pp  : 前処理後のフレーム系列
        """
        # 前処理(pp_model)を適用する
        x_pp = self.pp_model(x)

        # base_model(親の学習済みモデル)を適用する
        x_pp_last_pred, normalized_weights_ = self.base_model(x_pp)

        # 本モデル(pp+base_model)の出力(x_ppの最終フレームの予測x_pp_last_pred)と
        # pp_modelの出力x_ppを返す
        return x_pp_last_pred, x_pp

    def calc_anomaly_scores(self, x):
        """
        x(xはframeの連続)に対して異常スコアを計算する

        異常スコアは、data 内の frame 当たりの損失

        x              : tensor, [nbatch, nblock, n_frames*n_mels]
        anomaly_scores : tensor, [nbatch]
        """
        nbatch = len(x) # データ数(=wavファイル数)
        
        # 前処理を充てるためフレーム系列に変換
        # 入力データは特徴フレーム系列(次元数n_frames x n_mels)である
        # eg. torch.Size([4, 303, 1408]) -> torch.Size([4*303, 11, 128])
        x = x.view(-1, self.n_frames, self.n_mels) # flatten blocks to frames

        # このモデルを適用する
        outputs = self(x)

        # 損失を計算する
        loss = self.calc_loss(outputs, None, reduction='each') # フレーム別(each)の損失を計算

        # data 内の frame 当たりの損失を異常スコアとする
        anomaly_scores = loss.view(nbatch,-1).mean(1)

        return anomaly_scores

    def calc_loss(self, outputs, labels=None, reduction='mean'):
        """
        このモデルの出力に対して損失を計算する

        outputs: a tuple, (x_pred, x_pp), outputs of this model
        labels : not used
        """
        x_pred, x_pp = outputs
        labels = x_pp[:, -1] # 前処理後のフレーム系列x_ppの最後のフレーム

        if reduction=='mean': # 学習または評価時
            # average over all in batch
            loss = F.mse_loss( x_pred, labels, reduction='mean')
        elif reduction=='each': # フレーム毎の損失を求めるとき
            # mean of each sample in batch
            loss = F.mse_loss( x_pred, labels, reduction = 'none').mean(1)
        elif reduction=='none': # フレーム, ビン毎の損失を求めるとき
            loss = F.mse_loss( x_pred, labels, reduction = 'none')
        else:
            assert reduction in ['mean', 'each', 'none']

        return loss

class BaseModel(nn.Module):

    """
    このモデルでは、Transformerで前のフレームから最後のフレームを予測する

    [x(1), x(2), ..., x(n-1)] --> x~(n)
    
    # Fast TransFomer
    #  (Transformers are RNNs:Fast Autoregressive Transformers with Linear Attention)
    # https://arxiv.org/pdf/2006.16236.pdf
    #  Document
    # https://fast-transformers.github.io/
    #
    """
    def __init__(self, parent):
        """
        param    :
        fit_key  :  'fit' or 'tl_fit'
        eval_key :  'eval' or 'tl_eval'
        """
        super().__init__()

        # Hyper parameters
        n_enc_l         = parent.n_enc_l   
        nhead           = parent.nhead     
        d_ff            = parent.d_ff      
        d_model         = parent.d_model    # 埋め込み次元
        inp_layer       = parent.inp_layer  # 入力層
        out_layer       = parent.out_layer  # 出力層
        attn_type       = parent.attn_type  # attention_type
        pos_enc         = parent.pos_enc    # position encoding

        n_mels          = parent.n_mels
        n_frames        = parent.n_frames
        device          = parent.device
        lr              = parent.lr
        dropout_factor  = parent.dropout_factor
        weight_decay    = parent.weight_decay

        max_context       = n_frames - 1 # 予測に使う入力フレーム数(末尾は予測に使わない)
        max_seq_len       = n_frames # 予測に使う入力フレーム数+1
        output_dim        = n_mels


        self.parent       = parent
        self.device       = device

        #self.n_frames     = n_frames    # get_mini_batch で使う
        self.n_mels       = n_mels      # PreprocessRawTransformer3 で使う
        self.max_context  = max_context # create_maskで使う
        self.pos_enc      = pos_enc     # forward で使う
        self.d_model      = d_model
        self.output_dim   = output_dim
        self.out_layer    = out_layer
        self.lr           = lr
        self.weight_decay = weight_decay
        self.dropout_factor  = dropout_factor # 前処理モデルで使う(注意:self.dropoutはlayerとして使用済み)

        # 定義
        #S = max_seq_len # sequence length
        #N = None # batch size
        #D = n_mels # input feature vector size
        #E = d_model # embed vector size

        # モデル構築
        if pos_enc == 'sin_cos':
            self.pos_encoder = PositionalEncoding(d_model, dropout_factor, max_seq_len)
        elif pos_enc == 'embed':
            self.pos_embeddings = nn.Embedding(max_seq_len, d_model)
            self.layernorm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout_factor)
            self.position_ids = torch.arange(max_seq_len, dtype=torch.long,
                                             device=self.device)
        elif pos_enc == 'none':
            self.layernorm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout_factor)
        else:
            assert pos_enc in ['sin_cos', 'embed', 'none']

        # create transformer
        # https://fast-transformers.github.io/transformers/#transformerencoderlayer
        self.transformer = TransformerEncoderBuilder.from_kwargs(
            n_layers=n_enc_l,
            n_heads=nhead,
            query_dimensions=d_model//nhead,
            value_dimensions=d_model//nhead,
            feed_forward_dimensions=d_ff,
            attention_type = attn_type,
            activation="relu", # activation: Choose which activation to use
                               # for the feed forward part of the layer from
                               # the set {'relu', 'gelu'} (default: relu)
            dropout=dropout_factor,
            attention_dropout=dropout_factor,
            ).get()

        # 入力層
        self.inp_layer = InputLayer(max_seq_len, n_mels, d_model, type=inp_layer,
                                    dropout=dropout_factor)
        # 出力層
        self._set_out_layer()

        # マスクの初期値
        if attn_type == 'causal-linear':
            self.src_mask = TriangularCausalMask(max_seq_len, device=self.device)
        elif attn_type == 'linear':
            self.src_mask = FullMask(max_seq_len, device=self.device)
        else:
            assert attn_type in ['linear', 'causal-linear']

        # optimizers
        self._set_optimizer()

    def forward(self, x):
        """
        return           : output, normlized_weights

        x                : [N, S, D] (注意)バッチ次元と系列次元を反転
        src_mask         : [S, S]
        src_padding_mask : [N, S]
        emb_output       : [N, E]
        output           : [N, D]

        where, E : embed size, N : batch size, S : seq length
        """
        if self.pos_enc == 'sin_cos':
            x = self.inp_layer(x) # encode [N, S, D] -> [N, S, E]
            x *= math.sqrt(x.size(-1)) # weight against positional encoding
            x = self.pos_encoder(x) # add positional encoding
        elif self.pos_enc == 'embed':
            emb_inp = self.inp_layer(x) # encode [N, S, D] -> [N, S, E]
            emb_pos = self.pos_embeddings(self.position_ids)
            x = emb_inp + emb_pos
            x = self.layernorm(x)
            x = self.dropout(x)
        elif self.pos_enc == 'none':
            x = self.inp_layer(x) # encode [N, S, D] -> [N, S, E]
            x = self.layernorm(x)
            x = self.dropout(x)
        else:
            assert self.pos_enc in ['sin_cos', 'embed']

        src_mask, length_mask = self.create_mask(x) # get masks
        x = self.transformer(x, attn_mask=src_mask, length_mask=length_mask)# -> [N, S, E]
        # 警告：'output_layer'の改名時はswitch_to_transfer_learning_model()内の
        #       文字列も同様に変更すること
        x = self.output_layer(x)    # decode [N, S, E] -> [N, D]

        return x, 'TODO: normalized_weights to visualize attention'

    def calc_loss(self, outputs, labels, reduction='mean'):
        """
        outputs: a tuple, (x_pred, normalized_weights)
        labels : last frame to be predicted
        """
        x_pred, normalized_weights = outputs
        if reduction=='mean': # 学習または評価時
            # average over all in batch
            loss = F.mse_loss( x_pred, labels, reduction='mean')
        elif reduction=='each': # フレーム毎の損失を求めるとき
            # mean of each sample in batch
            loss = F.mse_loss( x_pred, labels, reduction = 'none').mean(1)
        elif reduction=='none': # フレーム, ビン毎の損失を求めるとき
            loss = F.mse_loss( x_pred, labels, reduction = 'none')
        else:
            assert reduction in ['mean', 'each', 'none']

        return loss

    def calc_anomaly_scores(self, x):
        """
        x(xはframeの連続)に対して異常スコアを計算する

        異常スコアは、data 内の frame 当たりの損失

        x              : tensor, [nbatch, nblock, n_frames*n_mels]
        anomaly_scores : tensor, [nbatch]
        """
        nbatch = len(x) # データ数(=wavファイル数)
        
        # 最終次元を保存して平坦化+モデル適用
        x = x.view( (-1, x.shape[-1]) ) # flatten blocks to frames
        x, y = self.parent.get_mini_batch(x) # xを最終フレームyと残り入力xに分離
        x = self(x) # Transformerを適用する
        loss = self.calc_loss(x, y, reduction='each') # フレーム別(each)の損失を計算

        # data 内の frame 当たりの損失を異常スコアとする
        anomaly_scores = loss.view(nbatch,-1).mean(1)

        return anomaly_scores

    def _set_optimizer(self):
        """共通関数
        """
        # https://github.com/jettify/pytorch-optimizer
        self.optimizer = optim.RAdam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

    def _set_out_layer(self):
        """
        出力層を新品に交換する
        警告：'output_layer'の改名時はswitch_to_transfer_learning_model()内の
              文字列も同様に変更すること
        """
        self.output_layer = OutputLayer(self.max_context,
                                        self.d_model,
                                        self.output_dim,
                                        type=self.out_layer)
        # 必要ならばgpuに移動
        self.output_layer.to(self.device)


    def generate_square_subsequent_mask(self, sz):
        """
        -infを成分とする上三角行列を生成する
        for example, sz with 5, mask will be
            tensor([[0., -inf, -inf, -inf, -inf],
                    [0.,   0., -inf, -inf, -inf],
                    [0.,   0.,   0., -inf, -inf],
                    [0.,   0.,   0.,   0., -inf],
                    [0.,   0.,   0.,   0.,   0.]])
        """
        #mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        #mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        # https://github.com/pytorch/pytorch/issues/48360
        mask = torch.tril(torch.ones(sz, sz))
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, emb):
        """
        emb              : [N, S, E]
        src_mask         : [S, S], TriangularCausalMask
        length_mask      : [N, S], LengthMask
        
        where, E : embed size, N : batch size, S : seq length
        """
        N, S, E = emb.shape

        # 下三角マスク
        src_mask = self.src_mask # 初期化したマスク(固定長S)を使う

        lengths = torch.empty(N)
        lengths[:] = self.max_context
        length_mask = LengthMask(lengths, max_len=S, device=emb.device)

        return src_mask, length_mask

class PrepTuneTransformerModel(object):
    def __init__(self, param, fit_key, eval_key):

        self.n_mels         = param['feature']['n_mels']    # get_mini_batch, base_model で使う
        self.n_frames       = param['feature']['n_frames']  # get_mini_batch, base_model で使う
        self.lr             = param[fit_key]['lr']          # fit verboseで使う
        self.dropout_factor = param['fit']['dropout']
        self.weight_decay   = param[fit_key].get('weight_decay',0.0)

        self.n_enc_l        = param['net']['n_enc_l']
        self.nhead          = param['net']['nhead']
        self.d_ff           = param['net']['d_ff']
        self.d_model        = param['net']['d_model'] # 埋め込み次元
        self.inp_layer      = param['net']['inp_layer'] # 入力層
        self.out_layer      = param['net']['out_layer'] # 出力層
        self.attn_type      = param['net']['attn_type'] # attention_type
        self.pos_enc        = param['net']['pos_enc'] # position encoding

        self.base_model     = BaseModel(parent=self)

        self._target_depend_model = self.base_model # 現在の学習・評価の対象モデル

    def to(self, device):
        self.base_model.to(device)

    def _load(self, model, model_file_path, restore_state=None):
        states = torch.load(model_file_path, map_location={'cuda:0':'cpu'})

        model.epochs = states['epochs']
        if restore_state: # state_dictの以外の情報を復元する
            # 乱数状態の復元
            torch_rng_state = states.get('torch_rng_state')
            if torch_rng_state is not None:
                torch.random.set_rng_state(torch_rng_state)
            else:
                print("can't recover torch_rng_state",
                        os.path.basename(model_file_path))
            numpy_rng_state = states.get('numpy_rng_state')
            if numpy_rng_state is not None:
                np.random.RandomState = numpy_rng_state
            else:
                print("can't recover numpy_rng_state",
                        os.path.basename(model_file_path))
            # 学習履歴の復元 --> model.history
            history = states.get('history')
            if history is not None:
                model.history = history
            else:
                model.history = None
                print("can't recover history",
                      os.path.basename(model_file_path))

        # モデルのパラメータの復元
        state_dict = states['state_dict']
        model.load_state_dict(state_dict)
        com.print(3,"load_model -> {}".format(model_file_path))

        return model

    def _save(self, model, model_file_path, epochs):
        with SafelySaver(model_file_path) as tmp_file:
            torch.save(dict(epochs=epochs,
                            torch_rng_state=torch.random.get_rng_state(),
                            numpy_rng_state=np.random.RandomState,
                            history=model.history,
                            state_dict=model.state_dict(),
                            ),
                       tmp_file)

    def summary(self):
        model = self.model() # 現在のモデルを取得
        com.print(1, '-'*60)
        total = 0
        for name_module, module in model.named_modules():
            sub_total = 0
            for name_param, param in module.named_parameters():
                size = param.size()
                sub_total += np.prod(size)

            com.print(2, "'{}' \t: {}".format(name_module, sub_total))
            total += sub_total

        com.print(1, 'total parameters :', total)
        com.print(1, '-'*60)

    def load_model(self, model_file_path, model=None, restore_state=None):
        """指定されたモデルのパラメータをロードする"""
        model = model or self.model()
        return self._load(model, model_file_path, restore_state)

    def save_model(self, model_file_path, model, epochs):
        """指定されたモデルのパラメータをセーブする"""
        return self._save(model, model_file_path, epochs)

    def model(self):
        """
        現在学習・評価の対象としているモデルを取得する
        """
        return self._target_depend_model

    def set_current_model(self, model):
        """
        modelを現在習・評価の対象としているモデルとする
        """
        self._target_depend_model = model

    def swtich_model(self, item, f_target):#, phase='train'):
        """
        必要ならtarget依存モデルに切り替える
        """
        self.set_current_model(self.base_model) # 初期設定(base_model)

        target = item.domain.target
        if target == 'target': # tagert依存モデル(転移学習、前処理追加など)のとき
            if f_target == 1:
                # 転移学習モデルへの切り替え
                self.switch_to_transfer_learning_model()

            else:
                # 前処理付加モデルへの切り替え
                self.switch_to_preprocess_model(f_target)

    def switch_to_transfer_learning_model(self):#, source_model_file_path, phase):
        """
        target依存モデルとして、最終層を付け替えたモデル(転移学習)を準備する
        """
        pass # このモデルへの切り替えは不要(切り替え先は元のモデル構造と同じため)

    def switch_to_preprocess_model(self, f_target):#, source_model_file_path, phase):
        """
        target依存モデルとして、前処理層追加モデルを準備する
        追加する前処理モデルはf_targetに依存する
            f_target:
                'pp-raw-tf3'のとき,前処理モデルとして、
                生のメルスペクトログラムを入力とする3層のTransformerを用いる
        """

        # f_targetにより前処理用モデルを選択する
        if f_target == 'pp-raw-tf3':
            # 前処理層を追加したモデル(pp_model)を作成
            pp_model = PreProcessModel(parent=self,
                                       pp_model_builder=PpTransformer3Model,
                                       base_model_builder=BaseModel)
        elif f_target == 'pp-raw-tf6':
            # 前処理層を追加したモデル(pp_model)を作成
            pp_model = PreProcessModel(parent=self,
                                       pp_model_builder=PpTransformer6Model,
                                       base_model_builder=BaseModel)
        elif f_target == 'pp-raw-tf9':
            # 前処理層を追加したモデル(pp_model)を作成
            pp_model = PreProcessModel(parent=self,
                                       pp_model_builder=PpTransformer9Model,
                                       base_model_builder=BaseModel)
        elif f_target == 'pp-raw-tf12':
            # 前処理層を追加したモデル(pp_model)を作成
            pp_model = PreProcessModel(parent=self,
                                       pp_model_builder=PpTransformer12Model,
                                       base_model_builder=BaseModel)
        else:
            assert f_target in ['pp-raw-tf3', 'pp-raw-tf6', 'pp-raw-tf9', 'pp-raw-tf12']

        # device を base_modelのものとする
        pp_model.to(self.base_model.device)

        # 学習・評価対象のモデルをこのモデルに切り替える
        self.set_current_model(pp_model)

    def prepare_fit(self, item, f_target):
        """
        self.epoch_start==0のときモデルを初期化する
        """

        target = item.domain.target # 転移学習のとき'source' or 'target'
        if target == 'target': # tagert依存モデル(転移学習、前処理追加など)のとき

            # sourceモデルのパスを得る
            source_model_file_path = item.model_file_path(purpose='eval',
                                                          target='source',
                                                          eval_key='eval')
            if f_target == 1:
                # 転移学習モデルの準備
                self.prepare_transfer_learning_model(source_model_file_path)

            else:
                # 前処理付加モデルの準備
                self.prepare_preprocess_model(source_model_file_path)

    def prepare_transfer_learning_model(self, source_model_file_path):
        """
        target依存モデルとして、最終層を付け替えたモデル(転移学習)を準備する
        """
        if self.epoch_start==0: # restartがepoch=0からのとき

            # 対象モデルを取得
            model_ = self.model()

            # 学習済みsourceモデルをロードする
            self.load_model(source_model_file_path, model_)

            # 最終層(model_.out_layer)を新品に付け替える
            model_._set_out_layer()
        
            # 最終層以外の荷重を固定する
            for name_, p_ in model_.named_parameters():
                # 警告:文字列'output_layer'は_
                # set_out_layer()中の変数名と一致させること
                if name_.startswith('output_layer'):
                    p_.requires_grad = True
                else:
                    p_.requires_grad = False

            # パラメータが変更されたのでoptimizerを再設定
            model_._set_optimizer()

    def prepare_preprocess_model(self, source_model_file_path):
        """
        target依存モデルとして、前処理層追加モデルを準備する
        追加する前処理モデルはf_targetに依存する
            f_target:
                'pp-raw-tf3'のとき,前処理モデルとして、
                生のメルスペクトログラムを入力とする3層のTransformerを用いる
        """
        # 対象モデルを取得
        pp_model = self.model()

        if self.epoch_start==0: # restartがepoch=0からのとき
            # 学習のためモデルのパラメータを初期化する

            # pp_modelのbase_modelにアクセス
            pp_base_model = pp_model.base_model

            # pp_base_model部分に学習済みsourceモデルをロードする
            self.load_model(source_model_file_path, pp_base_model)

            # pp_base_model部分のパラメータを固定する
            for name_, p_ in pp_base_model.named_parameters():
                p_.requires_grad = False
    
    def get_mini_batch(self, X):
        """
        Helper function to get a tuple of inputs, mask and labels
        return: src, tgt

        src   : [N, S, D] 入力フレーム列(ターゲットの最後のフレームを含む)
        tgt   : [N, D] 損失最小化のターゲット(src の最後のフレーム)
        """
        X = X.to(self.device)
        X = X.view(-1, self.n_frames, self.n_mels) # フーム系列として見る
        src = X # [N, S, D] コンテキスト+目的フレーム全体
        tgt = src[:, self.n_frames-1, :] # 最後の目的フレームである

        return src, tgt

    def prepare_restart(self, restart, num_epochs, model_file_path,
                        history_json_path, history_img_path):
        """保存ファイルを探す
        保存ファイルがないとき、同じ親ディレクト西谷同名のファイルを探す
        """
        epoch_start = 0 # 初期化
        history = defaultdict(list) # 初期化
        if restart:
            # 保存ファイルがあり、epochsに達しているか調べる
            if os.path.exists(model_file_path): # 保存したファイルがあるとき
                state_dict = torch.load(model_file_path, map_location=torch.device('cpu'))
                epochs = state_dict['epochs']
                if epochs == num_epochs: # epochsに達しているとき
                    epoch_start = num_epochs # epochs_startをセット

            # 保存ファイルがないときの処理
            if epoch_start == 0: # epoch_startが未セット
                # 別の場所で最大epochsモデルを探す

                # 親の下のディレクトリで同名のファイルを検索
                dn, bn = os.path.split(model_file_path)
                globs = glob.glob(os.path.join(dn,'../*',bn))

                # 最大のepochs数を探す
                max_epochs = 0 # 初期化
                for file in globs:
                    state_dict = torch.load(file, map_location=torch.device('cpu'))
                    epochs = state_dict['epochs']
                    if epochs > max_epochs:
                        max_epochs = epochs
                        max_file = file

                # 最大のepochsが見つかった時の処理
                if max_epochs: # 最大epochsのファイルがあるとき
                    # 最大epochが必要epochを超えていたらエラー
                    assert max_epochs <= num_epochs,\
                        "can't decrease epochs to {}, because larger epochs({}) exists,"\
                        .format(num_epochs, max_epochs)

                    model_ = self.model() # 現在のモデルを取得
                    if os.path.abspath(max_file) != os.path.abspath(model_file_path):
                        # 最大epochファイルが別のディレクトリにある場合

                        # max_fileをロードする
                        self.load_model(max_file, model_, restore_state=True)

                        # self.callback(保存処理)を呼ぶための準備
                        history = model_.history # hisotry をコピー
                        iepoch = model_.epochs - 1 # epochs をコピー

                        # self.callback(epoch, history)とほぼ共通の処理
                        self.history = history # for self.save()
                        self.model_file_path = model_file_path # for callback
                        self.history_json_path = history_json_path # for callback
                        self.history_img_path = history_img_path # for callback

                        self.callback(model_, iepoch, history, save_best=False, save_freq=False)

                        # callbackで保存できない、bestファイルをmax_fileの場所からコピー
                        body_, ext_ = os.path.splitext(max_file)
                        max_best_file_ = body_ + '_best' + ext_ # max_fileの場所のbestファイル
                        body_, ext_ = os.path.splitext(model_file_path)
                        best_file_ = body_ + '_best' + ext_ # 作成場所のbestファイル
                        shutil.copy2(max_best_file_, best_file_) # ファイルコピー
                    else:
                        # 作成場所のモデルをロードする
                        self.load_model(model_file_path, model_, restore_state=True)

                    epoch_start = model_.epochs
                    history = model_.history
                    if history is None: # model does not have 'history'
                        if os.path.exists(history_json_path):#保存ファイルがあるとき
                            with open(history_json_path) as f: # read from existing file
                                history_dict=json.load(f)
                            history = history_dict['history'] # 上書き
                        else: # 保存ファイルがないとき
                            # max_fileと同じディレクトのhisotry.jsonを探す
                            bn = os.path.basename(history_json_path)
                            dn = os.path.dirname(max_file)
                            max_history_pth = os.path.join(dn, bn)
                            if os.path.exists(max_history_pth):
                                with open(max_history_pth) as f:
                                    history_dict=json.load(f)
                                history = history_dict['history'] # 上書き
                            else:
                                # historyのfakeを作る
                                history = defaultdict(list)
                                history['loss'] = [np.nan]*epoch_start
                                history['val_loss'] = [np.nan]*epoch_start

        # 結果を残す
        self.history = history
        self.epoch_start = epoch_start
        need_train_model = epoch_start < num_epochs
        return need_train_model

    def fit(self, dl_tra, dl_val, num_epochs,
            save_freq=None,
            restart=None,
            model_file_path=None,
            history_json_path=None,
            history_img_path=None,
            ):
        """provide the training loop
        """
        torch.backends.cudnn.benchmark = True

        # restart で取得した情報
        epoch_start = self.epoch_start  # 再開するエポック
        history = self.history          # これまでの履歴

        model = self.model() # 現在のモデルを取得

        model_updated = None # 下のloopで内で更新されたかどうか
        for epoch in range(epoch_start,num_epochs):
            t0 = time.time()
    
            # 学習
            model.train() # to training mode

            data_size = 0 # data_size
            loss_tra = 0
            for X, _ in dl_tra:
                src, tgt = self.get_mini_batch(X)     # Xを入力/目的データに変換
                model.optimizer.zero_grad()           # 勾配リセット
                outputs = model(src)                  # Transformerに入力
                loss = model.calc_loss(outputs, tgt)  # 損失を計算
                loss.backward()                       # 逆伝搬
                model.optimizer.step()                # パラメータの更新
                loss_tra += loss.item() * src.size(0)
                data_size += src.size(0)
                model_updated = True                  # 更新された


            history['loss'].append(loss_tra/len(dl_tra))

            # 検証
            model.eval()

            loss_val = 0
            with torch.no_grad():
                for X,_ in dl_val:
                    src, tgt = self.get_mini_batch(X)    # Xを入力/目的データに変換
                    outputs = model(src)                 # Transformerに入力
                    loss = model.calc_loss(outputs, tgt) # 損失を計算
                    loss_val += loss.item() * src.size(0)

            history['val_loss'].append(loss_val/len(dl_val))

            com.print(1,
                      'epoch:{}'.format(epoch),
                      ' '.join(["{}:{:8g}".format(k,v[-1]) for k,v in history.items()]),
                      'frames:{}'.format(self.n_frames),
                      'lr:{}'.format(self.lr),
                      'ds:{}'.format(data_size),
                      'elapsed:{:.1f}s'.format(time.time()-t0),
                      '({})'.format(os.path.basename(model_file_path)),
                      )


            self.model_file_path = model_file_path # for callback
            self.save_freq = save_freq # for callback
            self.history_json_path = history_json_path # for callback
            self.history_img_path = history_img_path # for callback

            self.callback(model, epoch, history, save_best=True, save_freq=True)

        com.print(2, 'save'if model_updated else 'skip', model_file_path)

        return

    def callback(self, model, epoch, history, save_best=None, save_freq=None):
        """
        epoch 毎によばれる
        """
        # モデルのパラメータを保存する(毎epoch)
        model.history = history
        self.save_model(self.model_file_path, model, epochs=epoch+1)
        com.print(2, 'save', self.model_file_path)

        # historyを保存する(毎epoch)
        with open(self.history_json_path, 'w') as f:
            json.dump(dict(epochs=epoch+1,history=history), f, indent=2)
        com.print(2, 'save', self.history_json_path)

        # 学習曲線を描画して保存(毎epoch)
        com.loss_plot(history, self.history_img_path)
        com.print(2, 'save', self.history_img_path)

        # bestのパラメータを保存
        if save_best:
            val_loss = history['val_loss']
            if val_loss[-1] == min(val_loss): # best loss
                body, ext = os.path.splitext(self.model_file_path)
                best_model_file_path = body + '_best' + ext
                self.save_model(best_model_file_path, model, epochs=epoch+1)
                com.print(2, 'save', best_model_file_path)

        if save_freq:
            # save_freq毎にパラメータを保存
            if len(val_loss) % self.save_freq == 0:
                body, ext = os.path.splitext(self.model_file_path)
                model_file_epoch_path = body + '_epochs={}'.format(epoch+1) + ext
                self.save_model(model_file_epoch_path, model, epochs=epoch+1)
                com.print(2, 'save', model_file_epoch_path)

    def reconstruct_data(self, x):
        """
        入力スペクトログラムを予測データを用いて再現する
        x              : [nbatch, nblock, n_frames*n_mels]
        y              : ndarray, [nbatch, nblock, n_mels]
        y_recon        : ndarray, [nbatch, nblock, n_mels]
        a_scores       : ndarray, [nbatch, nblock, n_mels]
        """
        nbatch = len(x) # ndata:データ数(=wavファイル数)
        x = x.view(-1, x.shape[-1]) # 最終次元(n_frames*n_mels)で平坦化
        x, y = self.get_mini_batch(x) # コンテキストxと最終フレームyに分離
        x = self(x) # Transformerを適用する
        loss = self.calc_loss(x, y, reduction='none') # フレーム, ビン別(none)の損失を計算

        x, weights_ = x # 予測データとattention 荷重に分離

        # 入力スペクトログラム(最終フレーム)
        y = y.view(nbatch, -1, self.n_mels)

        # 再現スペクトログラム
        y_recon = x.view(nbatch, -1, self.n_mels)

        # frame, bin 毎の損失
        a_scores = loss.view(nbatch, -1, self.n_mels)

        # return オリジナル、再現、異常スコア
        return y.numpy(), y_recon.numpy(), a_scores.numpy()
