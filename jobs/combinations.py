import os
import re
import glob
from common import com
from copy import deepcopy

def search_combinations(com, transfer_learning=False,
                        use_existing_combinations=False):
    """
    既存のモデルファイルを探索して、ハイパーパラメータの組合せを返す

    use_existing_combinations: True なら現存するハイパーパラメータの組合せを使う
                               False なら評価結果を再作成するための組合せを探す

    transfer_learning : True なら転移学習の条件を探索する
                        False なら通常学習の条件を探索する
    """
    if use_existing_combinations:
        return search_combinations_use_existing_combinations(com,transfer_learning)

    # 評価結果を再作成するための組合せを探す
    # param['result_directory']/anomaly_score_*.csv を再作成する
    param = deepcopy(com.param) # コピーを作成する

    if transfer_learning:
        # 転移学習の条件探索のため param を編集する
        # param['transfer_learning']の下のハイパーパラメータに
        # プレフィックス'tl_'を付け param の下に追加
        def prefix_(x):
           if x.startswith('tl_'):
               return x
           return 'tl_' + x

        # param['transfer_learning']['tl_fit']及び
        # param['transfer_learning']['tl_eval']のdefault値をセットする
        com.set_default_tl_param(param)

        for k1, v1 in param['transfer_learning'].items():
            if isinstance(v1, dict): # v1が辞書のとき
                for k2,v2 in v1.items():
                    param[prefix_(k1)][prefix_(k2)] = v2
            else: # v1が値の時
                param[prefix_(k1)] = v1

        # 転移学習の条件キー
        model_directory_key = 'tl_model_directory'
        result_directory_key = 'tl_result_directory'
    else:
        # 通常学習の条件キー
        model_directory_key = 'model_directory'
        result_directory_key = 'result_directory'

    # 次のanomaly_score_pathを探索するためのワイルドカードを作成する
    # <result_directory>/anomaly_score_{machine_type}_{section_name}_{target_dir}.csv
    result_directory_spec = com.subst_model_config(param[result_directory_key])
    anomaly_score_path = com.anomaly_score_path(machine_type='*',
                                                section_name='*',
                                                target_dir='*',
                                                result_directory=result_directory_spec)
    anomaly_score_path_query = re.sub(r'{\w+}', '*', anomaly_score_path)
    anomaly_score_paths = glob.glob(anomaly_score_path_query)

    # model_dirs からハイパーパラメータを復元する
    combinations = []
    for anomaly_score_path in anomaly_score_paths:
        dirname, basename = os.path.split(anomaly_score_path)
        # model_pathからハイパーパラメータを取得する
        m = re.match(r"anomaly_score_([^_]+)_(section_\d+)_(.+).csv",basename)
        domain_dict = dict(machine_types=m.group(1),
                           section_name=m.group(2),
                           target_dir=m.group(3))
        cond_tuples = com.get_cond_tuples_from_dirname(dirname,
                                                       result_directory_spec)
        combination = dict(**domain_dict) # a new dict
        for k, v in cond_tuples.items():
            sub_param = com.get_cond_sub_params(cond_tuples, k, param)
            combination.update(sub_param)

        combinations.append(combination)

    return combinations

def search_combinations_use_existing_combinations(com, transfer_learning=False):
    """
    現存するハイパーパラメータの組合せを使う
    既存のモデルファイルを探索して、ハイパーパラメータの組合せを返す

    transfer_learning : True なら転移学習の条件を探索する
                        False なら通常学習の条件を探索する
    """
    assert False,'use_existing_combinations is not supported'
    param = deepcopy(com.param) # コピーを作成する

    if transfer_learning:
        # 転移学習の条件探索のため param を編集する
        # param['transfer_learning']の下のハイパーパラメータに
        # プレフィックス'tl_'を付け param の下に追加
        def prefix_(x):
           if x.startswith('tl_'):
               return x
           return 'tl_' + x

        for k1, v1 in param['transfer_learning'].items():
            if isinstance(v1, dict): # v1が辞書のとき
                for k2,v2 in v1.items():
                    param[prefix_(k1)][prefix_(k2)] = v2
            else: # v1が値の時
                param[prefix_(k1)] = v1

        # 転移学習の条件キー
        model_directory_key = 'tl_model_directory'
        result_directory_key = 'tl_result_directory'
    else:
        # 通常学習の条件キー
        model_directory_key = 'model_directory'
        result_directory_key = 'result_directory'

    # 探索するファイルのワイルドカードを作成する
    model_directory_spec = com.subst_model_config(param[model_directory_key])
    model_dirs = re.sub(r'{\w+}','*',model_directory_spec)
    model_dirs = glob.glob(model_dirs+'\\model_*.pt')

    # model_dirs からハイパーパラメータを復元する
    combinations = []
    for model_path in model_dirs:
        basename = os.path.basename(model_path)
        if not re.match(r"model_[^_]+_section_[^_]+_[^_]+\.pt", basename):
            # 基本ファイル名以外のとき
            continue # skip

        # model_pathからハイパーパラメータを取得する
        m = re.match(r"model_(.+).pt",basename)
        machine_types = m.group(1)
        model_dir = os.path.dirname(model_path)
        cond_tuples = com.get_cond_tuples_from_dirname(model_dir,
                                                       model_directory_spec)
        combination=dict(machine_types=machine_types)
        for k, v in cond_tuples.items():
            sub_param = com.get_cond_sub_params(cond_tuples, k, param)
            combination.update(sub_param)

        # 'decision'パラメータを追加する
        result_directory = com.subst_model_config(param[result_directory_key])
        # {decision}に'*'を代入したいが、glob.escapeされるので'__decision__'を代入
        result_dirs = result_directory.format(**cond_tuples, decision='__decision__')
        # '__decision__'を'*'に置換, 末尾に'/'を付加
        result_dirs = glob.escape( result_dirs ).replace('__decision__','*')+'/'
        # ファイル(ここではディレクトリのみ)を検索
        result_dirs = glob.glob(result_dirs)
        for result_dir in result_dirs:
            # {decision} tuple 中のハイパーパラメータをcombinationのコピーに追加
            cond_tuples = com.get_cond_tuples_from_dirname(result_dir, result_directory)
            decision_param=com.get_cond_sub_params(cond_tuples,'decision',param)
            # コピーを更新して追加
            combination_copy = combination.copy()
            combination_copy.update(decision_param)
            combinations.append(combination_copy)

    return combinations
