# Copyright (c) 2021 ralabo.jp
# This software is released under the MIT License.
# see https://opensource.org/licenses/mit-license.php
# ====================================================
import os
import sys
import numpy as np
import json
import torch

from sklearn import metrics

from model.model_provider import provide_modeler
from data.task2_basedata import BaseData
from data.task2_dataset import Task2Dataset
from data.task2_dataloader import Task2DataLoader
from utils.score import calc_batch_anomaly_scores
from utils.score import gamma_ppf

from common import com
from flow.flow_provider import get_flow_instance
from flow.transfer_learn import switch_training_param

def main():

    com.parse_command_line()
    if com.mode is None:
        sys.exit(-1)

    com.save_param() # loop前にcom.paramを保存する

    flow = get_flow_instance(com.param)
    for item in flow.items(): # 1モデルの評価処理ループ
        com.restore_param() # 保存していたparamを使う(loop状態の最初から)

        target = item.domain.target # 転移学習のとき'source' or 'target'

        # 転移学習時、ハイパーパラメータ(param)を切り替える
        # (target=='source' or target=='target'なら転移学習モードである)
        fit_key, eval_key = switch_training_param(target in ['source', 'target'])

        # 学習済みの評価対象モデル,及び,異常スコア分布パラメータの場所を得る
        model_file_path = item.model_file_path(purpose='eval',
                                               target=target,
                                               eval_key=eval_key)
        score_distr_param_path = item.score_distr_param_path()

        modeler = None # 遅延ロードのため
        for machine_type, section_name, target_dir in item.test_items():#評価項目(itemに依存)
            # 結果が作成済みであれば次のセクションへ
            anomaly_score_path = com.anomaly_score_path(machine_type, section_name, target_dir)
            decision_result_path = com.decision_result_path(machine_type, section_name, target_dir)
            com.print(2, '@test@:',machine_type, section_name, target_dir,
                  'domain:',
                  item.domain.machine_type, item.domain.section_index,
                  item.domain.target) #for debug
            if com.args.re_make_result_dir: # 評価結果の再作成時
                # 既存の結果が存在するときに再作成する
                if not os.path.exists(anomaly_score_path): # 既存の結果がないとき
                    continue # skip(再作成しない)
                if os.path.getmtime(anomaly_score_path) > os.path.getmtime(com.args.re_make_result_dir):
                    continue # skip(より新しい)
                save_message = 'update'
            else:
                # 通常の処理の時
                if os.path.exists(anomaly_score_path) \
                    and os.path.exists(decision_result_path):
                    if not com.args.re_calc_score_distribution:
                        if not (com.overwrite_test or com.overwrite):
                            com.print(2,'skip',anomaly_score_path)
                            com.print(2,'skip',decision_result_path)
                            continue # skip this section
                save_message = 'save'

            # make output result directory
            os.makedirs(com.result_directory(), exist_ok=True)
            if modeler is None: # 遅延ロード
                if not os.path.exists(model_file_path): # モデルファイルが無い
                    self.print(0,"{} modeler not found ".format(machine_type))
                    sys.exit(-1)
                # モデルを準備
                modeler = provide_modeler(com, fit_key, eval_key)

                # 必要ならtarget依存モデルに切り替える
                modeler.swtich_model(item, flow.f_target)

                # 学習済みモデルをロードする
                modeler.load_model(model_file_path)

                # 異常スコアの学習時ガンマ分布パラメータを取得
                with open(score_distr_param_path) as f:
                    gamma_params = json.load(f)

                # 判定閾値を計算
                thresh = gamma_ppf(com.param['decision']["decision_thr"], gamma_params)

            # セクションのテストデータを取得
            base_data = BaseData(machine_type, section_name=section_name,
                                 domain='*', target_dir=target_dir, augment=False)
            ds_test = Task2Dataset(base_data)
            dl_test = Task2DataLoader(ds_test,
                                      batch_size=com.param['misc']['batch_size'],
                                      shuffle=False, drop_last=False, unit='block')
            # modeler を評価する
            anomaly_score_list, decision_result_list = test_model(modeler.model(),
                                                                  dl_test,
                                                                  thresh)

            # output anomaly scores
            com.to_csv(anomaly_score_list, anomaly_score_path)
            com.print(2, save_message, anomaly_score_path)

            # output decision results
            com.to_csv(decision_result_list, decision_result_path)
            com.print(2, save_message, decision_result_path)

    torch.cuda.empty_cache() # GPU メモリをすべて開放する
    return

def test_model(model, dl_test, thresh):
    model.eval()

    anomaly_score_list = []
    decision_result_list = []

    with torch.no_grad():
        for blocks, labels, file_paths in dl_test:
            blocks = blocks.to(model.device)

            anomaly_scores = calc_batch_anomaly_scores(model, blocks) # a list
            decisions = np.array(anomaly_scores) > thresh # ndarray
            decisions = decisions.astype(int) # convert to int

            # ファイル名と異常スコアを保存
            basenames = [os.path.basename(file_path) for file_path in file_paths]
            anomaly_score_list.extend([x for x in zip(basenames,anomaly_scores)])
            # ファイル名と異常識別結果を保存
            decision_result_list.extend([x for x in zip(basenames,decisions)])

    return anomaly_score_list, decision_result_list

if __name__=='__main__':

    main()
