# Copyright (c) 2021 ralabo.jp
# This software is released under the MIT License.
# see https://opensource.org/licenses/mit-license.php
# ====================================================
from copy import deepcopy
from common import com

def switch_training_param(switch):
    """
    true時、転移学習用paramに切り替える
    切替え直前の、paramはparam['_base_param']に保存する
    args:
        switch : true時、転移学習用paramに切り替える
    return:
         fit_key  : 'fit' or 'tl_fit'
         eval_key : 'eval' or 'tl_eval'
    """
    # 切り替え有無にかかわらず、
    # 切替え直前の、paramをparam['_base_param']に保存する
    if '_base_param' in com.param:
        del com.param['_base_param']
    com.param['_base_param'] = deepcopy(com.param)

    if switch: # 転移学習時(trainのとき)

        # com.param['transfer_learning']['tl_fit']及び
        # com.param['transfer_learning']['tl_eval']のdefault値をセットする
        com.set_default_tl_param()

        # 最後に'tl_fit', 'tl_eval'を更新する
        com.update_param(com.param['transfer_learning'])

        fit_key = 'tl_fit'
        eval_key = 'tl_eval'
    else:
        fit_key = 'fit'
        eval_key = 'eval'

    return fit_key, eval_key
