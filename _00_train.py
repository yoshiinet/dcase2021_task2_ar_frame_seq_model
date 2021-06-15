# Copyright (c) 2021 ralabo.jp
# This software is released under the MIT License.
# see https://opensource.org/licenses/mit-license.php
# ====================================================
import os
import sys
import json
import numpy as np
import torch
import shutil

from model.model_provider import provide_modeler
from data.task2_basedata import BaseData
from data.task2_dataset import Task2Dataset
from data.task2_dataloader import Task2DataLoader
from utils.score import calc_data_anomaly_scores
from utils.score import plot_score_distribution
from utils.score import estimate_score_distribution

from flow.flow_provider import get_flow_instance
from flow.transfer_learn import switch_training_param
from common import com

      
def main():
    com.parse_command_line()

    # init random generators for pytorch
    com.init_random_generators()

    restart = com.param['misc'].get('restart_train')
    if com.overwrite:
        if restart:
            restart = None
            print('"restart" is canceled, because "overwrite" is ture')

    com.save_param() # loop前にcom.paramを保存する

    flow = get_flow_instance(com.param)
    for item in flow.items(): # 1モデルの学習処理ループ
        com.restore_param() # 保存していたparamを使う(loop状態の最初から)

        target = item.domain.target # 転移学習のとき'source' or 'target'

        # true時、転移学習用paramに切り替える
        fit_key, eval_key = switch_training_param(target == 'target')

        # 変更後のcom.paramでモデルの作成先を得る
        # (eval_keyの指すeval_epoch['best'or'last']を使用)
        model_file_path = item.model_file_path(purpose='train',
                                               target=target,
                                               eval_key=eval_key)
        history_json_path = item.history_json_path()

        need_train_model = com.overwrite \
            or not os.path.exists(model_file_path) or restart

        # ガンマ分布パラメータの保存先(モデルの作成先と同じ場所に作成)
        score_distr_param_path = item.score_distr_param_path()
        score_distr_csv_path = item.score_distr_csv_path()
        score_distr_fig_path = item.score_distr_fig_path()
        history_img_path = item.history_img_path()

        need_calc_score_distribution = com.overwrite or \
            com.args.re_calc_score_distribution or \
            not os.path.exists(score_distr_param_path)

        # modeler のplace holderを準備する(fit_keyの指すlrを使用)
        modeler = provide_modeler(com, fit_key, eval_key)

        # 必要ならtarget依存モデルに切り替える
        modeler.swtich_model(item, flow.f_target)

        if need_train_model:
            # make output directory
            os.makedirs(com.model_directory(),exist_ok=True)

            # restartの準備
            num_epochs = com.param[eval_key]['epochs'] # (eval_keyの指すepochsを使用)
            need_train_model = modeler.prepare_restart(restart,
                                                       num_epochs,
                                                       model_file_path,
                                                       history_json_path,
                                                       history_img_path)

        if not ( need_train_model or need_calc_score_distribution ):
            com.print(2,'skip',model_file_path)
            continue # skip this machine_type

        data_dict = None # 遅延ロードのため

        if need_train_model:
            modeler.summary()

            # modeler.epoch_start==0のときmodelを初期化する
            modeler.prepare_fit(item, flow.f_target)

            # 必要ならデータを準備する
            data_dict = data_dict or prepare_data(item, fit_key, eval_key)

            # モデルパラメータを学習、ファイルに保存する
            modeler.fit(data_dict['dl_tra'], data_dict['dl_val'],
                        num_epochs, restart = restart,
                        save_freq = com.param['misc']['save_freq'],
                        model_file_path = model_file_path,
                        history_json_path = history_json_path,
                        history_img_path = history_img_path,
                        )
        else:
            # 学習済みモデルをロードする
            modeler.load_model(model_file_path)

        if need_calc_score_distribution:
            # 必要ならデータを準備する
            data_dict = data_dict or prepare_data(item, fit_key, eval_key)

            # 異常スコア分布パラメータを推定する
            train_score_distribution_param(flow, item, target,
                                           modeler.model(),
                                           data_dict['ds_tra'],
                                           score_distr_param_path,
                                           score_distr_csv_path,
                                           score_distr_fig_path)

    
    torch.cuda.empty_cache() # GPU メモリをすべて開放する
    return

def prepare_data(item, fit_key, eval_key):
    """データを準備する
    """
    base_data =  item.base_data(target_dir='train', augment=True,
                                fit_key=fit_key, eval_key=eval_key)
    data_tra, data_val = base_data.split(com.param[fit_key]["val_split"])
    ds_tra = Task2Dataset(data_tra)
    ds_val = Task2Dataset(data_val)

    data_dict = dict(
        base_data = base_data,
        data_tra = data_tra,
        data_val = data_val,
        ds_tra = ds_tra,
        ds_val = ds_val,
        dl_tra = Task2DataLoader(ds_tra,
                                 batch_size=com.param[fit_key]['batch_size'],
                                 shuffle=com.param[fit_key]["shuffle"],
                                 drop_last=com.param[fit_key].get('drop_last'),
                                 unit='frame'),
        dl_val = Task2DataLoader(ds_val,
                                 batch_size=com.param[fit_key]['batch_size'],
                                 shuffle=False,
                                 drop_last=com.param[fit_key].get('drop_last'),
                                 unit='frame'),
        )
    return data_dict

def train_score_distribution_param(flow, item, target,
                                   model, ds_tra,
                                   score_distr_param_path,
                                   score_distr_csv_path,
                                   score_distr_fig_path):
    """
    異常スコアの分布のパラメータを学習する
    異常スコアの分布はガンマ分布とする
    """
    if target == 'target': # tagert依存モデル(転移学習、前処理追加など)のとき
        if flow.f_target in ['pp-raw-tf3', 'pp-raw-tf6', 'pp-raw-tf9', 'pp-raw-tf12']:
            # ソースモデルによる結果をコピーする
            source_score_distr_param_path = item.score_distr_param_path(target='source')
            source_score_distr_csv_path = item.score_distr_csv_path(target='source')
            source_score_distr_fig_path = item.score_distr_fig_path(target='source')
            shutil.copy2(source_score_distr_param_path, score_distr_param_path)
            shutil.copy2(source_score_distr_csv_path, score_distr_csv_path)
            shutil.copy2(source_score_distr_fig_path, score_distr_fig_path)
            
            # 戻る
            return
        elif flow.f_target == 1: # 転移学習のとき
            pass # データ数少ないが下記で異常スコア分布パラメータを推定する

    # 上記以外の時の処理

    # 異常スコア分布パラメータを推定する
    dl_tra = Task2DataLoader(ds_tra,
                             batch_size=com.param['misc']['batch_size'],
                             shuffle=False,
                             drop_last=False,
                             unit='block')
    estimate_score_distribution(model, dl_tra, com.param,
                                score_distr_param_path,
                                score_distr_csv_path,
                                score_distr_fig_path)
    

if __name__=='__main__':

    main()
