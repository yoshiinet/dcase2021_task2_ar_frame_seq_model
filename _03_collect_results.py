# Copyright (c) 2021 ralabo.jp
# This software is released under the MIT License.
# see https://opensource.org/licenses/mit-license.php
# ====================================================
import os
import sys
import re
import json
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import defaultdict

from scipy.stats import hmean
from natsort import natsorted, natsort_keygen
from sklearn import metrics
from glob import glob

from utils.score import gamma_ppf
from common import com, TargetDomain
from flow.flow_provider import get_flow_instance
from flow.transfer_learn import switch_training_param

class IterFiles(object):
    def __init__(self):
        pass

    def result_dirs(self,csv_files):
        """
        result_dir が同じファイルを集める
    
        return result_dir, selected_files
        """
        selected_files = []
        prev_result_dir = None
        for x in csv_files:
            result_dir = x[0] # get dir
            if result_dir != prev_result_dir:
                if selected_files:
                    yield prev_result_dir, selected_files
                    selected_files = []

            selected_files += [x]
            prev_result_dir = result_dir

        if selected_files:
            yield prev_result_dir, selected_files

    def matchine_types(self,csv_files):
        """
        return machine_type, selected_files
        """
        selected_files = []
        prev_machine_type = None
        for x in csv_files:
            basename = x[1] # get base
            machine_type = basename.split('_')[2] # machine_type
            if machine_type != prev_machine_type:
                if selected_files:
                    yield prev_machine_type, selected_files
                    selected_files = []

            selected_files += [x]
            prev_machine_type = machine_type

        if selected_files:
            yield prev_machine_type, selected_files

    def target_dirs(self,csv_files):
        """
        return target_dir, selected_files
        target_dir: eg. 'source_test', 'target_test'
        """
        selected_files = []
        prev_dir_name = None
        for x in csv_files:
            basename = x[1] # get base
            target_dir = '_'.join(basename.split('_')[5:7]) # target_dir
            if target_dir != prev_dir_name:
                if selected_files:
                    yield prev_dir_name, selected_files
                    selected_files = []
            selected_files += [x]
            prev_dir_name = target_dir

        if selected_files:
            yield prev_dir_name, selected_files

    def section_names(self,csv_files):
        """
        return section_name, selected_files
        section_name: eg. 'section_00', 'section_01', ...
        """
        selected_files = []
        prev_section_name = None
        for x in csv_files:
            basename = x[1] # get base
            section_name = '_'.join(basename.split('_')[3:5]) # section_name
            if section_name != prev_section_name:
                if selected_files:
                    yield prev_section_name, selected_files
                    selected_files = []
            selected_files += [x]
            prev_section_name = section_name

        if selected_files:
            yield prev_section_name, selected_files

def get_thresh(machine_type, section_index, target, decision_thr, flow):
    domain = flow.domain(machine_type, section_index, target)
    score_distr_param_path = com.score_distr_param_path(domain, target=target)
    with open(score_distr_param_path) as f:
        gamma_params = json.load(f)

    thresh = gamma_ppf(decision_thr, gamma_params)
    assert not np.isnan(thresh), 'thresh is nan.'\
        ' gamma param=(q={q} a={a} loc={loc} scale={scale} thresh={thresh} file={file})'\
        .format(q=decision_thr, **gamma_params, file=score_distr_param_path)
    return thresh, gamma_params

def make_result_files(combinations):
    """
    _result_*.csv を収集し result_files を作成する

    yamlで指定された result_directory:
        result_directory: <somepath>/{config}/{feature}/{net}/{fit}/{decision}
    にある
    anomaly_score_path()関数で定義されるファイル:
        anormaly_score_{machine_type}_section_{section_name}_{target_dir}.csv
    を収集し、スコアを計算する
    スコアは dataframe として返す

    return: result_path_params
        a list of tuples, (result_path, param, transfer_learning)
    """
    com.save_param() # 作業前にcom.paramを保存する

    # target独立(転移学習ではない)の結果を得る
    csv_files = make_result_file_common(combinations, transfer_learning=False)

    # target依存(転移学習)の結果を得る
    csv_files += make_result_file_common(combinations, transfer_learning=True)

    com.restore_param() # 作業後に保存したcom.paramを復帰
    
    assert csv_files,'no _result_*.csv files.'
    return csv_files

def make_result_file_common(combinations, transfer_learning):
    """
    transfer_learning:
        trueなら転移学習の結果ファイルをリストアップする
        falseなら通常学習の結果ファイルをリストアップする
    return:
        a list of tuples, [(result_path, param.copy(), transfer_learning),...]
    """
    if transfer_learning:
        # 転移学習結果を探索するためのワイルドカードを作成する
        tl_result_directory = com.param.get('transfer_learning') and \
                              com.param['transfer_learning'].get('result_directory')
        if tl_result_directory:
            # true時、転移学習用paramに切り替える
            fit_key, eval_key = switch_training_param(transfer_learning)
        else:
            return []


    # 探索するファイルのワイルドカードを作成する
    result_dir = com.subst_model_config(com.param['result_directory'])
    result_dir = re.sub(r'{\w+}','*',result_dir)
    anomaly_score_csv = com.anomaly_score_path(machine_type='*',
                                               #section_name='*', # exclude 03, 04, 05
                                               section_name='section_0[012]',
                                               target_dir='*',
                                               result_directory=result_dir)
    # glob でファイルを探索する
    csv_files = glob(anomaly_score_csv)
    csv_files = [x.replace('\\','/') for x in csv_files] # 区切りを'/'にそろえる
    csv_files = sorted(csv_files) # 順番を並べ替える
    # ファイル名を (dir, base, ext) に分解
    csv_files = [(os.path.dirname(x),)+tuple(os.path.splitext(os.path.basename(x)))
                 for x in csv_files]
    if combinations:
        # フィルターする
        def _filter_fun(x):
            for c in combinations: # 組み合わせの中を探索する
                if com.is_match_combination(x,c): # 条件がマッチするとき
                    return True

        csv_files = list(filter(_filter_fun, csv_files ))

    # ディレクトリ名ごとにスコアを計算する
    iter = IterFiles()
    result_path_params = []
    for result_dir, files_1 in iter.result_dirs(csv_files):
        com.restore_param() # 上位で保存していたcom.paramを復帰
        # transfer_learning==true時、転移学習用paramに切り替える
        fit_key, eval_key = switch_training_param(transfer_learning)
        # result_dirからハイパーパラメータを取得する
        cond_tuples = com.get_cond_tuples_from_dirname(result_dir)
        com.update_param_from_cond_tuples(cond_tuples) # update com.param
        if '_base_param' in com.param:
            # cond_tuplesのkeyとcom.param[key]の組を辞書base_param_とする
            base_param_ = {k:com.param[k] for k,v in cond_tuples.items()}
            # 上記辞書base_param_で com.param['_base_param']を更新する
            com.param['_base_param'].update(base_param_)
            # com.param['_base_param']['cond']を更新する
            com.update_cond_params(com.param['_base_param'])

        param = com.param # update local param
        # get decision parameters
        decision_thr = param['decision']['decision_thr']
        max_fpr = param['decision']['max_fpr']
        # get flow for model_for
        flow = get_flow_instance(param)
        # machine_types > target_dirs > section_names  ごとに集計する
        # initialize lines in csv for AUC and pAUC
        csv_lines = []
        performance_over_all = [] if com.mode else None
        for machine_type, files_2 in iter.matchine_types(files_1):
            # determine decision threshold
            # load anomaly score distribution for training
            if com.mode:
                # results for each machine type
                csv_lines.append([machine_type])
                csv_lines.append(["section", "target", "AUC", "pAUC", "precision",
                                  "recall", "F1 score"])
                performance = []
            for target_dir, files_3 in iter.target_dirs(files_2):
                for section_name,files_4 in iter.section_names(files_3):
                    assert len(files_4) == 1 , 'not unique,  check source code'

                    section_index = section_name.split('_')[1] # eg. '00', '01', ...
                    target = target_dir.split('_')[0] # eg. 'source' or 'target'

                    # get threshold
                    thresh, gamma_params = get_thresh(machine_type, section_index, target,
                                                      decision_thr, flow)

                    # read anomaly_score_csv
                    dir, base, ext = files_4[0]
                    anomaly_score_csv = dir + '/' + base + ext # compose a full path
                    df = pd.read_csv(anomaly_score_csv,header=None)
                    df.columns = ['wav','a_score']
                    if com.mode:
                        # get y_true from wav_filename
                        # section_00_target_test_normal_0000.wav, anormaly_score
                        # 正解はファイル名('wav'欄)中の'anomaly'か'normal'かで取得する
                        y_true = [1 if x.split('_')[4]=='anomaly' else 0 for x in df.loc[:,'wav']]
                        # 異常スコア('a_score'欄)を取得する
                        y_pred = df.loc[:,'a_score'] # anomaly score

                        # append AUC and pAUC to lists
                        auc = metrics.roc_auc_score(y_true, y_pred)
                        p_auc = metrics.roc_auc_score(y_true, y_pred,
                                                      max_fpr=max_fpr)
                        tn, fp, fn, tp = metrics.confusion_matrix(
                            y_true, [1 if x > thresh
                                     else 0 for x in y_pred]).ravel()
                        prec = tp / np.maximum(tp + fp, sys.float_info.epsilon)
                        recall = tp / np.maximum(tp + fn, sys.float_info.epsilon)
                        f1 = 2.0 * prec * recall / np.maximum(prec + recall,
                                                              sys.float_info.epsilon)
                        section_index_ = section_name.split("_", 1)[1]
                        target_ = target_dir.split("_", 1)[0]
                        csv_lines.append([section_index_, # 'section'
                                          target_ + ' ' + section_index_, # 'target'
                                          auc, p_auc, prec, recall, f1])
                        performance.append([auc, p_auc, prec, recall, f1])
                        performance_over_all.append([auc, p_auc, prec, recall, f1])
            # end for dirname
            if com.mode:
                # calculate averages for AUCs and pAUCs
                amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
                csv_lines.append(["arithmetic mean", ""] + list(amean_performance))
                hmean_performance = hmean(np.maximum(np.array(performance, dtype=float),
                                                     sys.float_info.epsilon), axis=0)
                csv_lines.append(["harmonic mean", ""] + list(hmean_performance))
                csv_lines.append([])
        # end for matchie_type
        if com.mode:
            csv_lines.append(["", "", "AUC", "pAUC", "precision", "recall", "F1 score"])
            # calculate averages for AUCs and pAUCs
            amean_performance = np.mean(np.array(performance_over_all, dtype=float), axis=0)
            csv_lines.append(["arithmetic mean over all machine types, sections, and domains",
                                ""] + list(amean_performance))
            hmean_performance = hmean(np.maximum(np.array(performance_over_all, dtype=float),
                                                 sys.float_info.epsilon), axis=0)
            csv_lines.append(["harmonic mean over all machine types, sections, and domains",
                                ""] + list(hmean_performance))
            csv_lines.append([])
        
            # output results
            result_basename = param['result_file'].format(**cond_tuples)
            result_path = result_dir + os.sep + result_basename
            com.to_csv(csv_lines, result_path)
            com.print(2,'save',os.path.basename(result_path))
            param_ = deepcopy(param)
            # check param_ consisttency
            if param_.get('_base_param'):
                assert param_['model_for'] == param_['_base_param']['model_for']
                assert str(tuple(v for k,v in param_['model_for'].items())) \
                       == param_['cond']['model_for']
                assert str(tuple(v for k,v in param_['model_for'].items())) \
                       == param_['_base_param']['cond']['model_for']
            # append new row
            result_path_params += [(result_path, param_, transfer_learning)]

    return result_path_params

def collect_results(combinations):
    result_path_params = make_result_files(combinations)

    # ファイル内容をデータフレームに読み込む
    col_names = ['section','target','AUC','pAUC','precision','recall','F1 score']
    rows=[]
    for result_path_, param_, transfer_learning_ in result_path_params:

        if param_.get('_base_param'):
            assert param_['model_for'] == param_['_base_param']['model_for']

        # 拡張子なしのベースネーム
        basename = os.path.splitext(os.path.basename(result_path_))[0]
        #print('reading',basename)

        # csv を読む
        df = pd.read_csv(result_path_,names=col_names,error_bad_lines=False)

        # machin_type と　basename 欄を加える
        df.insert(0,'machine_type','') # insert column
        df.insert(0,'basename','') # insert column
        df.insert(0,'result_path_','') # insert column
        df.insert(0,'param_','') # insert column

        # 行を加工する
        for idx,row in df.iterrows():
            section = row['section']
            if section=='section' or pd.isna(section):
                pass
            elif 'mean' in section or section.isdigit():
                if 'over all' in section:
                    machine_type = '00 over all'

                row['basename']=basename
                row['machine_type']=machine_type
                row['section']=section
                row['result_path_']=result_path_
                row['param_']=param_
                rows.append(row)
            elif section:
                machine_type = section

    df = pd.DataFrame(rows)

    return df

def normalize_results(df):
    """
    # #####################################
    # 扱いやすくするための後処理
    # #####################################
    """
    # 行をソートする
    df = df.sort_values(['machine_type','target','section','basename'],key=natsort_keygen())

    # sectionの変化する点に空行を入れる(グラフを見やすくするため)
    out_rows = []
    prev_section = None
    prev_machine_type = None
    col_names = df.columns
    for idx,row in df.iterrows():

        # section間に空行を挿入
        section = row['section']
        if pd.isna(section):
           pass
        elif section != prev_section:
            out_rows.append(pd.Series(dtype=object))
            prev_section = section

        # machine_type 間にヘッダー行を挿入
        machine_type = row['machine_type']
        if pd.isna(machine_type):
           pass
        elif machine_type != prev_machine_type:
            header = row.copy()
            header.values[:] = col_names
            out_rows.append(header)
            prev_machine_type = machine_type

        # 空白の'target'(target or souce) に'section'を簡略し転記する
        if pd.isna(row['target']):
            if pd.isna(section):
                pass
            else:
                if section.startswith('arithmetic mean over all'):
                    row['target'] = 'a. m. all '
                elif section.startswith('harmonic mean over all'):
                    row['target'] = 'h. m. all '
                elif section == 'arithmetic mean':
                    row['target'] = 'a. m. '+row['machine_type']
                elif section == 'harmonic mean':
                    row['target'] = 'h. m. '+row['machine_type']

        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)

    return out_df

def insert_mean_score_each_machine_type(df, score_key='AUC'):
    """
    マシンタイプ別の平均スコアAUC列を追加する(pd.DataFrame.assign()を使用)
   
    """
    # dfを舐めて、追加するデータを集める

    # arithmetic mean用の辞書を用意する
    # harmonic mean 用の辞書も用意する
    mean_data = dict()
    mean_data['a_mean'] = defaultdict(list)
    mean_data['h_mean'] = defaultdict(list)

    # 条件記入用に5列分のkeyを作成する
    note_keys = ['{}{}'.format(score_key, j) for j in range(5,0,-1)]

    # 最初にnote_keysを辞書に登録する
    for key in note_keys:
        mean_data['a_mean'][key] = None
        mean_data['h_mean'][key] = None

    # dataframeを調べて各machineのarithmetic meanとharmonic meanを追加する
    for idx, row in df.iterrows():
        section = row['section']
        if not pd.isna(section) and section.startswith('arithmetic mean'):
            key = row['machine_type'] + ' ' + score_key
            mean_data['a_mean'][key].append(row[score_key]) # arithmetic mean of score
        if not pd.isna(section) and section.startswith('harmonic mean'):
            key = row['machine_type'] + ' ' + score_key
            mean_data['h_mean'] [key].append(row[score_key]) # harmonic mean of score

    # 実験条件数(行数)を求める
    num_conds = len(mean_data['a_mean'][key]) # keyは最後のkeyを使う

    # 条件コラムに空白行をセットする
    for key in note_keys:
        mean_data['a_mean'][key] = [np.nan] * num_conds # 空白行
        mean_data['h_mean'][key] = [np.nan] * num_conds # 空白行

    # 集めたデータが4行目からに来るように空行とヘッダーを挿入する
    assign_data = defaultdict(list)
    for key in mean_data['a_mean'].keys():
        data = [np.nan] # 本当のヘッダーの次の空行
        data += [key] # ヘッダー(kがヘッダー文字である)を追加
        data += mean_data['a_mean'][key] # arithmetic meanを追加
        data += [key] # 個別ヘッダー行を入れる
        data += mean_data['h_mean'][key] # harmonic mean を追加
        data += [np.nan]*(len(df)-len(data)) # 残りの行を空行で埋める
        assign_data[key] = data # 列 k を更新する

    # 列を追加する
    df = df.assign(**assign_data)

    return df

def add_anomaly_distr_param(df):
    """
    異常スコア分布パラメータを追記する
    """

    # index をユニークに更新する
    df = df.reset_index(drop=True)
    for idx,row in df.iterrows():
        machine_type = row['machine_type']
        basename = row['basename']
        basename = basename if isinstance(basename,str) and basename.startswith('_result_') else None
        section = row['section']
        section = section if isinstance(section,str) and section[0].isdigit() else None

        if df.loc[idx,'basename'] == 'basename':
            prev_head_idx = idx # 最後の見つかったヘッダー行の位置

        # 個別 machine_type, section, domain の行であるとき
        if basename and section and machine_type != '00 over all':
            param_ = row['param_']
            
            if param_.get('_base_param'):
                assert param_['model_for'] == param_['_base_param']['model_for']
                assert str(tuple(v for k,v in param_['model_for'].items())) \
                       == param_['cond']['model_for']
                assert str(tuple(v for k,v in param_['model_for'].items())) \
                       == param_['_base_param']['cond']['model_for']

            com.update_param(param_) # update com.param
            # 分布パラメータファイルパスを取得する(flow->domainを求め取得)
            target = row['target'].split()[0] # eg. row['target'] looks like 'source 00'
            section_index = section
            flow = get_flow_instance(com.param)
            domain = flow.domain(machine_type, section_index, target)
            score_distr_param_path = com.score_distr_param_path(domain)

            # 分布パラメータを取得する
            with open(score_distr_param_path) as f:
                gamma_param = json.load(f)

            # 分布パラメータを追記する
            for k in gamma_param.keys():
                df.loc[idx,k] = gamma_param[k]

            if pd.isna(df.loc[prev_idx,'basename']):# 前行が空行の時
                for k in gamma_param.keys():
                    if pd.isna(df.loc[prev_head_idx, k]): # 未記入
                        df.loc[prev_head_idx, k] = k
                    else: # すでに記入済み
                        break # 他も記入済みのはずなのでループを終了

        prev_idx = idx

    return df

def add_cond_tuples(df):
    """
    列の最後に新しい列を追加する
    'basename'の条件をtupleに分離して
     可能ならばサブパラメータに分解して列の最後に追加する
    """
    # index をユニークなものにする
    df = df.reset_index(drop=True)
    new_columns = defaultdict(dict)
    first = True
    for idx, row in df.copy().iterrows():# df はループ内で更新されることに注意
        machine_type = row['machine_type']

        # 個別 machine_type の平均部分であるとき
        basename = row['basename']
        if not pd.isna(basename) and basename != 'basename':
            com.update_param(row['param_']) # update com.param
            # 'result_path_'から条件を cond_tuples に抽出する
            result_dir = os.path.dirname(row['result_path_'])
            cond_tuples = com.get_cond_tuples_from_dirname(result_dir)

            # 条件名をキーとする階層辞書を作成する
            # cond_params::= dict(cond, cond_value or dict(subparam, subparam_value) )
            cond_params =  {}
            for k,v in cond_tuples.items():
                if v.startswith('('): # is v tuple ?
                    # make a dict(subparam, subparam_value)
                    sub_param = com.get_cond_sub_params(cond_tuples, k)
                    if k.startswith('tl_'):
                        # sub_paramのkeyにプレフィックス'tl_'を付ける
                        sub_param = {'tl_'+k : v for k,v in sub_param.items()}
                    cond_params[k] = sub_param
                else: # v is a value
                    cond_params[k] = v

            # 新しい列を追加する(全体のヘッダーを自動作成)
            if first: # 最初のみ
                kwds = {} # 追加する列を{kwd:value}形式で作成
                rows = [np.nan]*len(df) # 追加する行(空白)
                for k1, v1 in cond_params.items():
                    if isinstance(v1, dict):
                        for k2 in v1.keys():
                            kwds[k2] = rows
                    else:
                        kwds[k1] = rows
                df = df.assign(**kwds) # 新しい列を追加
                first = False

            # 現在行idxに値を記入する
            for k1, v1 in cond_params.items():
                if isinstance(v1, dict):
                    for k2, v2 in v1.items():
                        df.loc[idx, k2] = str(v2)
                else:
                    df.loc[idx, k1] = str(v1)

            # 前行がヘッダー行であれば前行にヘッダーを追記する
            if df.loc[prev_idx,'basename']=='basename': # prev is a header
                for k1, v1 in cond_params.items():
                    if isinstance(v1, dict):
                        for k2, v2 in v1.items():
                            df.loc[prev_idx, k2] = k2
                    else:
                        df.loc[prev_idx, k1] = k1

        prev_idx = idx

    return df


def main(combinations, save_to='csv', **options):
    """
    すべての処理をする
    """
    com.parse_command_line(**options)

    # #####################
    # result_csv を収集する
    # #####################
    df = collect_results(combinations) # result_csv を収集、dataframeを取得
    df = normalize_results(df) # 扱いやすくするため、情報を追加
    df = insert_mean_score_each_machine_type(df, score_key='AUC') # マシンタイプ別の平均スコア列を追加
    df = insert_mean_score_each_machine_type(df, score_key='pAUC') # マシンタイプ別の平均スコア列を追加
    df = add_anomaly_distr_param(df) # 異常スコア分布パラメータを追記する(machine_type=='00 over all')
    df = add_cond_tuples(df) # 条件tupleを列に追加

    del df['result_path_']
    del df['param_']

    if save_to == 'ods':
        summary_ods = com.summary_ods_path() # 保存先
        with pd.ExcelWriter(summary_ods, engine='odf') as writer:
            df.to_excel(writer, index=False)
        com.print(2,'save',summary_ods)
    elif save_to == 'xlsx':
        summary_xlsx = com.summary_xlsx_path() # 保存先
        with pd.ExcelWriter(summary_xlsx) as writer:
            df.to_excel(writer, index=False)
        com.print(2,'save',summary_xlsx)
    elif save_to == 'csv':
        summary_csv = com.summary_csv_path() # 保存先
        df.to_csv(summary_csv, index=False)
        com.print(2,'save',summary_csv)

if __name__=='__main__':
    sys.argv += re.sub("\s+"," ","""
    --dev
    --defaults ./run/ave/ave_config.yaml
    --machine_types ToyCar
    --nunits 4096 2048 1024 512
    --n_frames 5
    --epochs 20
    --kld_weight 1.0
    --lr 0.0005
    --zdim 256
    --edim 32
    """).split() # split()は先頭と末尾の空白は無視するようだ

    main(combinations=None)

    if 0: # debug only
        import os
        os.system(r"../../../../Temp/dcase2021_task2_pytorch/ave/debug/_summary_debug.csv".replace('/','\\'))
