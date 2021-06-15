# Copyright (c) 2021 ralabo.jp
# This software is released under the MIT License.
# see https://opensource.org/licenses/mit-license.php
# ====================================================
import sys
import os
import re
import glob
import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
import yaml
import torch
import csv
from copy import deepcopy

class TargetDomain(object):
    def __init__(self, machine_type, section_index, target):
        self.machine_type = machine_type if machine_type != 'any' else '*'
        self.section_index = section_index if section_index != 'any' else '*'
        self.target = target if target != 'any' else '*'
        pass

    def basename_parts(self, target=None):
        """
        return a part of basename of a file
        a string of "<machine_type>_section_<section_index>_<domain>"
        if '*' is used, it is replaced with 'any'
        target: 転移学習時 target='source'でsourceモデルのファイ表題を得る
        """
        target = target or self.target
        return '_'.join([self.machine_type if self.machine_type!='*' else 'any',
                         'section',
                         self.section_index if self.section_index!='*' else 'any',
                         target if target!='*' else 'any',
                         ])

def as_path(path):
    """
    path の中の空白を除く
    """
    path = path.replace(', ', ',') # ','の後ろの' 'を除去する
    return path

class Common(object):
    debug = False
    param = None
    mode = None
    overwrite = None
    verbose = None
    fig = None

    def print(self, level, *args, **kwds):
        if self.verbose and self.verbose>=level:
            print(*args,**kwds)

    def to_csv(self, save_data, save_file_path):
        with open(save_file_path, "w", newline="") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(save_data)

    def get_cond_tuples_from_dirname(self, result_dir, result_directory=None):
        """
        result_dir と result_directory を照合し、
        {}で囲まれたキーワードに対応する文字列を抽出する

        return a dict( param = str(tuple), ... )
        """
        result_directory = result_directory or self.param['result_directory']
        # 照合する両者を正規化する
        result_dir = result_dir.replace('\\','/')
        if result_dir[-1]=='/':
            result_dir = result_dir[:-1] # drop trailing '/'
        result_directory = result_directory.replace('\\','/')
        if result_directory[-1]=='/':
            result_directory = result_directory[:-1] # drop trailing '/'

        # パス要素の後ろから照合をとる
        cond_tuples = {}
        for val,key in zip(reversed(result_dir.split('/')),
                            reversed(result_directory.split('/'))):
            m=re.match(r"{(\w+)}",key)
            if m:
                key = m.group(1)
                cond_tuples[key] = val

        # 逆になった順番を正順に戻しておく
        cond_tuples = {k:cond_tuples[k] for k in reversed(list(cond_tuples.keys()))}

        return cond_tuples

    def get_saved_param(self):
        """
        self.save_param()で保存したparamへの参照を得る
        """
        return self.param_save

    def save_param(self):
        """
        self.paramのdeepcopyを保存する
        """
        self.param_save = deepcopy(self.param)

    def restore_param(self):
        """
        保存したparam_saveをparamに復帰する
        """
        self.param = deepcopy(self.param_save)

    def update_param(self, new_param_dict):
        """
        self.param を更新する
        param_dict: 辞書
        {k:v} のvがdictであればparam[k]のdeepcopyを更新する
        """
        for k, v in new_param_dict.items():
            if isinstance(v, dict):
                if k in self.param:
                    self.param[k] = deepcopy(self.param[k]) # deepcopyを更新する
                    self.param[k].update(v)
                else:
                    self.param[k] = deepcopy(v)
            else:
                self.param[k] = v
        self.update_cond_params() # param['cond']も更新する

    def update_cond_params(self, param=None):
        """
        com.param['cond']を更新する

            param['cond'][key] に、param[key]が与えるハイパーパラメーターtupleの
            文字列表現、または、一部のハイパーパラメータを抜き出しセットする
        
            ハイパーパラメーター tupleの文字列表現は、
            名前の昇順にソートされたパラメーターのtupleの文字列表現である
        
        param[key]がNoneであれば、param[key]に空の辞書をセットする
        """
        param = param or self.param

        # 'model'については独立した条件'cond'とする
        param['cond']['model'] = param['net']['model']

        # keyに対するヘルパー関数
        def _set_cond_str(key):
            """
            param['cond'][key] に ハイパーパラメーター tupleの文字列表現をセットする
            param[key]がNoneであれば、param[key]に空の辞書をセットする
            """
            if param.get(key) is None:
                param[key] = dict() # assign new dict

            param['cond'][key]= str(
                tuple([param[key][k] for k in sorted(param[key].keys())]))

        # keyのリストに対して ヘルパー関数を適用する
        for key in ['feature', 'net', 'model_for', 'fit',
                    'eval', 'misc', 'decision', 'tl_fit', 'tl_eval']:
            _set_cond_str(key)

    def update_param_from_cond_tuples(self, cond_tuples):
        """
        param を更新する
        param['cond']の内容をcond_tuplesの内容で更新するのも忘れないこと
        更新するparam[key]がdictであればそれのdeepcopyを更新する
        """
        # key の内容を更新する
        for key in cond_tuples.keys():
            subparam = self.get_cond_sub_params(cond_tuples, key)
            if isinstance(subparam, dict):
                self.param[key] = deepcopy(self.param[key]) # deepcopyを更新する
                self.param[key].update(subparam)
            else:
                self.param[key] = subparam

        self.update_cond_params() # param['cond']も更新する

    def set_default_tl_param(self, param=None):
        """
        param['transfer_learning']['tl_fit']及び
        param['transfer_learning']['tl_eval']のdefault値を
        それぞれ
        param['fit']及び
        param['eval']から
        コピーする
        """
        param = param or self.param
        # 'tl_fit'のdefaultを'fit'からコピーする
        fit_param = param['fit'].copy()
        fit_param.update(param['transfer_learning']['tl_fit'])
        param['transfer_learning']['tl_fit'].update(fit_param)

        # 'tl_eval'のdefaultを'eval'からコピーする
        eval_param = param['eval'].copy()
        eval_param.update(param['transfer_learning']['tl_eval'])
        param['transfer_learning']['tl_eval'].update(eval_param)

    def get_cond_sub_params(self, cond_tuples, subparam, param=None):
        """
        tuple表現からsubparamに対応する部分を辞書 or atomとして取得する
        cond_tuples: a dict( subparam = tuple_string, ... )

        return: a dict( param_name = param_value, ... ) for subparam
        """
        return self.get_subparam_from_tuple_string(cond_tuples[subparam],subparam,param)

    def get_subparam_from_tuple_string(self,tuple_string,subparam, param=None):
        """
        tuple_string が辞書であれば subparamに対応する部分を辞書として取得する
        tuple_string がatomであればそれを返す
        """
        param = param or self.param
        if tuple_string.startswith('('):# if it looks like a tuple
            cond_tuples = eval(tuple_string)
            subparam = sorted(param[subparam].keys())
            cond_sub_param = {}
            for x,y in zip(cond_tuples,subparam):
                cond_sub_param[y] = x
            return cond_sub_param
        else: # it may be a simple string
            return tuple_string

    def get_subparam_from_dirname(self, result_dir, major_key ):
        """
        result_dirからハイパーパラメータを取得する
        args:
            major_key:  辞書 param のと同じ key
        
        return: a dict( subparam = value, ... ), a dict( param = str(tuple), ... )
        """
        cond_tuples = self.get_cond_tuples_from_dirname(result_dir,
                                                        self.param['result_directory'])
        subparam = self.get_cond_sub_params(cond_tuples, major_key )
        return subparam, cond_tuples

    def find_param(self,name,param):
        """
        name という ハイパーパラメータを param の中に探し値を返す
        見つからなければ None を返す
        """
        val = param.get(name)
        if val:
            return val

        for sub_key, sub_dict in param.items():
            if isinstance(sub_dict,dict):
                val = sub_dict.get(name)
                if val:
                    return val

        return None

    def subst_model_config(self, result_or_model_directory):
        """
        result_or_model_directory の中の
        {model}, {config} にハイパーパラメータを代入する
        """
        # substituite '{model}'
        result_or_model_directory = \
            result_or_model_directory.replace('{model}', self.param['net']['model'])

        # substituite '{config}'
        config = self.config
        result_or_model_directory = \
            result_or_model_directory.replace('{config}', config)

        return result_or_model_directory


    def is_match_combination(self,csv_file,combination):
        """
        combinationで指定されるハイパーパラメータがcsv_fileにマッチするとき真を返す

        csv_file := tuple( result_dir, base, ext )
        """
        result_dir, basename, ext = csv_file
        cond_tuples = self.get_cond_tuples_from_dirname(result_dir, self.param['result_directory'])
        for k,v in combination.items():
            # find major_key
            for sub_key,sub_dict in self.param.items():
                if isinstance(sub_dict,dict):
                    if sub_dict.get(k) is not None:
                        # sub_dict中に k が見つかった
                        # cond_tuplesから sub_key 部分のハイパーパラメータを取得
                        subparam = self.get_cond_sub_params(cond_tuples, sub_key )
                        if subparam[k] != v: # 値が一致しないとき
                            return False
                        break # 値が一致したとき内側のループから抜ける
            else:
                # self.param に {k:v} が含まれないとき
                # basename 中に v を探索する
                if v not in basename: # v が見つからないとき
                    return False

        return True

    def model_directory(self, param=None, **cond_tuples):
        param = param or self.param
        cond_tuples = cond_tuples or param['cond']
        return as_path(param['model_directory'].format(**cond_tuples))

    def result_directory(self):
        return as_path(self.param['result_directory'].format(**self.param['cond']))

    def summary_csv_path(self):
        return as_path(self.param['summary_csv'].format(**self.param['cond']))

    def summary_ods_path(self):
        return as_path(self.param['summary_ods'].format(**self.param['cond']))

    def summary_xlsx_path(self):
        return as_path(self.param['summary_xlsx'].format(**self.param['cond']))

    def result_csv_file_path(self):
        result_basename = self.param['result_file'].format(**self.param['cond'])
        return as_path(self.result_directory() +'/' + result_basename)

    def machine_dir(self, machine_type):
        if self.mode:
            return as_path(self.param['dev_directory'] + '/'+machine_type)
        else:
            return as_path(self.param['eval_directory'] + '/'+machine_type)

    def model_file_path(self, domain, purpose, target, eval_key):
        """
        purpose  : 'train' or 'eval'

        target   : 'source' or 'target' or '*'(domainの属性と異なることもある)
            target=='source'ならばsourceモデルのパスを返す
            sourceモデルとしてbestモデルを参照するため、phase='eval'とする必要あり
            self.param はユーザの責任で適切にセットする必要あり
        
        eval_key : one of 'eval' or 'tl_eval'
        """
        if purpose == 'train':
            return as_path("{}/model_{}.pt".format(self._model_dir(target),
                                           domain.basename_parts(target)))
        elif purpose == 'eval':
            # eval_keyが指す'eval_epoch'を得る
            eval_epoch = self.param[eval_key]['eval_epoch']
            # eval_epochに応じたtemplateを得る
            if eval_epoch == 'best':
                template = "{}/model_{}_best.pt" # use 'best' model
            elif eval_epoch == 'last':
                template = "{}/model_{}.pt" # use 'last' model
            else:
                assert eval_epoch in ['best', 'last']
            # templateに代入し、パスを返す
            return as_path(template.format(self._model_dir(target),
                                   domain.basename_parts(target)))
        else:
            assert purpose in ['train', 'eval']

    def history_img_path(self, domain):
        return as_path("{}/history_{}.png".format(self._model_dir(domain.target),
                                          domain.basename_parts()))

    def history_json_path(self, domain):
        return as_path("{}/history_{}.json".format(self._model_dir(domain.target),
                                           domain.basename_parts()))

    def _model_dir(self, target, model_dir=None):
        """
        model_directoryを得る

        target:
            target == 'source' ならば　self.param['_base_param']を使う
        """
        if model_dir is None:
            if target == 'source':
                # self.param['_base_param'](tl_model_directoryでない方)を使う
                param_ = self.param['_base_param']
                if param_.get('model_for'):
                    # f_targetが0以外であればf_targetを'_source'に読替える
                    if param_['model_for'].get('f_target'): # 0以外の時
                        param_ = deepcopy(param_)
                        param_['model_for']['f_target'] = 'source'
                        self.update_cond_params(param_)

                model_dir = self.model_directory(param=param_)
            else:
                # self.paramを使う
                model_dir = self.model_directory()

        return as_path(model_dir)

    def score_distr_param_path(self, domain, model_dir=None, target=None):
        """
        domain:
            domain.target == 'source' ならば　param['_base_param']を使う
        """
        target = target or domain.target
        model_dir = self._model_dir(target, model_dir)
        if domain.target == '*':
            target = domain.target
        return as_path("{}/score_distr_{}.json".format(
            model_dir, domain.basename_parts(target)))

    def score_distr_csv_path(self, domain, model_dir=None, target=None):
        """
        domain:
            domain.target == 'source' ならば　param['_base_param']を使う
        """
        target = target or domain.target
        model_dir = self._model_dir(target, model_dir)
        return as_path("{}/score_distr_{}.csv".format(
            model_dir, domain.basename_parts(target)))

    def score_distr_fig_path(self, domain, model_dir=None, target=None):
        """
        domain:
            domain.target == 'source' ならば　param['_base_param']を使う
        """
        target = target or domain.target
        model_dir = self._model_dir(target, model_dir)
        return as_path("{}/score_distr_{}.png".format(
            model_dir, domain.basename_parts(target)))

    def anomaly_score_path(self,machine_type, section_name, target_dir,
                           result_directory=None):
        result_directory = result_directory or self.result_directory()
        return as_path("{result}/anomaly_score_{machine_type}_{section_name}_{target_dir}.csv"\
            .format(result=result_directory,  machine_type=machine_type,
                    section_name=section_name, target_dir=target_dir))

    def decision_result_path(self, machine_type, section_name, target_dir):
        return as_path("{result}/decision_result_{machine_type}_{section_name}_{target_dir}.csv"\
            .format(result=self.result_directory(), machine_type=machine_type,
                    section_name=section_name,  target_dir=target_dir))

    def parse_command_line(self, **options):
        ap = argparse.ArgumentParser()
        ap.add_argument('--mode',help="mode, one of 'dev', 'eval'")
        ap.add_argument('--debug', action='store_true', help="for debug only")
        ap.add_argument('--show', action='store_true', help="show plot")
        ap.add_argument('--remote', action='store_true', help="indicate remote machine")
        ap.add_argument('--local', action='store_true', help="indicate local machine")
        ap.add_argument('--defaults', default='config_ae_pytorch.yaml', help="path to yaml file")
        ap.add_argument('--config', help="param['misc']['config']")
        ap.add_argument('--seed', type=int, default=1234, help="random generator seeds")
        ap.add_argument('--usable_machine_types', nargs='+', help="param['limit']['usable_machine_types']")
        ap.add_argument('--n_frames', type=int, help="param['feature']['n_frames']")
        ap.add_argument('--n_mels', type=int, help="param['feature']['n_mels']")
        ap.add_argument('--n_hop_frames', type=int, help="param['feature']['n_hop_frames']")
        ap.add_argument('--nunits', type=int, nargs='+', help="各階層のユニット数 eg.[128,64,32,16]")
        ap.add_argument('--w_loss', type=float, nargs=3, help="損失の加重[w_adv, w_con, w_enc]")
        ap.add_argument('--epochs', type=int, help="param['eval']['epochs']")
        ap.add_argument('--kld_weight', type=float, help="KLD 損失の係数")
        ap.add_argument('--lr', type=float, help="learning rate")
        ap.add_argument('--zdim', type=int, help="z-space次元数")
        ap.add_argument('--edim', type=int, help="epsilon次元数")
        ap.add_argument('--n_enc_l', type=int, help="param['net']['n_enc_l']")
        ap.add_argument('--nhead', type=int, help="param['net']['nhead']")
        ap.add_argument('--inp_layer', type=str, help="param['net']['inp_layer']")
        ap.add_argument('--out_layer', type=str, help="param['net']['out_layer']")
        ap.add_argument('--pos_enc', type=str, help="param['net']['pos_enc']")
        ap.add_argument('--attn_type', type=str, help="param['net']['attn_type']")
        ap.add_argument('--batch_size', type=int, help="param['fit']['batch_size']")
        ap.add_argument('--aug_count', type=int, help="param['fit']['aug_count']")
        ap.add_argument('--aug_gadd', type=float, help="param['fit']['aug_gadd']")
        ap.add_argument('--aug_wcut', type=int, help="param['fit']['aug_wcut']")
        ap.add_argument('--data_size', type=int, help="param['fit']['data_size']")
        ap.add_argument('--d_model', type=int, help="param['net']['d_model']")
        ap.add_argument('--d_ff', type=int, help="param['net']['d_ff']")
        ap.add_argument('--dropout', type=float, help="dropout")
        ap.add_argument('--f_machine', type=int, help="param['model_for']['f_machine']")
        ap.add_argument('--f_section', type=int, help="param['model_for']['f_section']")
        ap.add_argument('--f_target', type=str, help="param['model_for']['f_target']")

        # transfer learning, fine tuning, other tuning
        ap.add_argument('--tl_lr', type=float, help="param['tl_fit']['tl_lr']")
        ap.add_argument('--tl_dropout', type=float, help="param['tl_fit']['tl_dropout']")
        ap.add_argument('--tl_batch_size', type=int, help="param['tl_fit']['batch_size']")
        ap.add_argument('--tl_data_size', type=int, help="param['tl_fit']['tl_data_size']")
        ap.add_argument('--tl_shuffle', type=int, help="param['tl_fit']['tl_shuffle']")
        ap.add_argument('--tl_val_split', type=float, help="param['tl_fit']['tl_val_split']")
        ap.add_argument('--tl_aug_count', type=int, help="param['tl_fit']['tl_aug_count']")
        ap.add_argument('--tl_aug_gadd', type=float, help="param['tl_fit']['tl_aug_gadd']")
        ap.add_argument('--tl_aug_wcut', type=int, help="param['tl_fit']['tl_aug_wcut']")
        ap.add_argument('--tl_epochs', type=int, help="param['tl_eval']['tl_epochs']")
        ap.add_argument('--tl_eval_epoch', help="param['tl_eval']['eval_epoch']")


        ap.add_argument('-v', '--verbose', action='count', default=0,
                            help="increase verbose level")

        ap.add_argument('--save_freq', type=int, help="param['misc']['save_freq']")
        ap.add_argument('--restart_train', type=int, help="param['misc']['restart_train']")
        ap.add_argument('--basedata_memory', type=str, help="param['misc']['basedata_memory']")
        ap.add_argument('--eval_epoch', help="param['eval']['eval_epoch']")
        ap.add_argument('--re_calc_score_distribution', action='store_true', help="異常スコア分布を再計算")
        ap.add_argument('--re_make_result_dir', help="このファイルより古い評価結果を再作成する")
        ap.add_argument('--overwrite', action='store_true', help="param['misc']['overwrite']")
        ap.add_argument('--overwrite_test', action='store_true', help="param['misc']['overwrite_test']")
        # チャートの表示
        ap.add_argument('--chart_columns', type=str,nargs='+',default=['target','AUC','pAUC','F1 score'],
                            help="x軸ラベル(default:'target')と棒ラベル(default:'pAUC')のリスト")
        ap.add_argument('--chart_layout',default='grid', help="チャートのレイアウト(vertical,grid")
        # (注意) 追加したときは、 下方の「パラメータをコマンドラインの指定値で上書きする」にも追加
        # 同様に、 _04_run_job.py の run_job(), f_run() にも追加

        self.args = ap.parse_args()
        # 関数引数で上書きする
        for k,v in options.items():
            if v is not None:
                setattr(self.args,k,v)

        self.debug=self.args.debug
        self.show = self.args.show
        self.overwrite = self.args.overwrite
        self.overwrite_test = self.args.overwrite_test

        if self.args.mode == 'dev':
            self.mode = True
        elif self.args.mode == 'eval':
            self.mode = False
        else:
            assert self.args.mode in ['dev', 'eval'], "--mode dev or --mode eval"

        # load parameter.yaml
        com.load_defaults()

        return

    def load_defaults(self):
        """
        load defaults parameters from path
        path is given by the '--defaults' option
        """
        path = self.args.defaults
        with open(path) as stream:
            self.param = yaml.safe_load(stream)


        # パラメータをコマンドラインの指定値で上書きする
        if self.args.n_frames is not None:
            self.param['feature']['n_frames'] = self.args.n_frames
        if self.args.n_mels is not None:
            self.param['feature']['n_mels'] = self.args.n_mels
        if self.args.n_hop_frames is not None:
            self.param['feature']['n_hop_frames'] = self.args.n_hop_frames
        if self.args.chart_columns is not None:
            self.param['chart_columns'] = self.args.chart_columns
        if self.args.chart_layout is not None:
            self.param['chart_layout'] = self.args.chart_layout
        if self.args.nunits is not None:
            self.param['net']['nunits'] = self.args.nunits
        if self.args.dropout is not None:
            self.param['fit']['dropout'] = self.args.dropout
        if self.args.f_machine is not None:
            self.param['model_for']['f_machine'] = self.args.f_machine
        if self.args.f_section is not None:
            self.param['model_for']['f_section'] = self.args.f_section
        if self.args.f_target is not None:
            # f_targetは文字列なので数字をintに変換
            if self.args.f_target.isdigit():
                self.args.f_target = int(self.args.f_target)
            self.param['model_for']['f_target'] = self.args.f_target
        if self.args.w_loss is not None:
            self.param['fit']['w_loss'] = self.args.w_loss
        if self.args.epochs is not None:
            self.param['eval']['epochs'] = self.args.epochs
        if self.args.kld_weight is not None:
            self.param['fit']['kld_weight'] = self.args.kld_weight
        if self.args.lr is not None:
            self.param['fit']['lr'] = self.args.lr
        if self.args.zdim is not None:
            self.param['net']['zdim'] = self.args.zdim
        if self.args.edim is not None:
            self.param['net']['edim'] = self.args.edim
        if self.args.n_enc_l is not None:
            self.param['net']['n_enc_l'] = self.args.n_enc_l
        if self.args.nhead is not None:
            self.param['net']['nhead'] = self.args.nhead
        if self.args.inp_layer is not None:
            self.param['net']['inp_layer'] = self.args.inp_layer
        if self.args.out_layer is not None:
            self.param['net']['out_layer'] = self.args.out_layer
        if self.args.pos_enc is not None:
            self.param['net']['pos_enc'] = self.args.pos_enc
        if self.args.attn_type is not None:
            self.param['net']['attn_type'] = self.args.attn_type
        if self.args.batch_size is not None:
            self.param['fit']['batch_size'] = self.args.batch_size
         # -------------- データ拡張(通常学習) -------------------------
        if self.args.aug_count is not None:
            self.param['fit']['aug_count'] = self.args.aug_count
        if self.args.aug_gadd is not None:
            self.param['fit']['aug_gadd'] = self.args.aug_gadd
        if self.args.aug_wcut is not None:
            self.param['fit']['aug_wcut'] = self.args.aug_wcut
        # --------------------------------------------------------------------------
        if self.args.data_size is not None:
            self.param['fit']['data_size'] = self.args.data_size
        if self.args.d_model is not None:
            self.param['net']['d_model'] = self.args.d_model
        if self.args.d_ff is not None:
            self.param['net']['d_ff'] = self.args.d_ff
        if self.args.overwrite is not None:
            self.param['misc']['overwrite'] = self.args.overwrite
        if self.args.overwrite_test is not None:
            self.param['misc']['overwrite_test'] = self.args.overwrite_test
        if self.args.save_freq is not None:
            self.param['misc']['save_freq'] = self.args.save_freq
        if self.args.restart_train is not None:
            self.param['misc']['restart_train'] = self.args.restart_train
        if self.args.basedata_memory is not None:
            self.param['misc']['basedata_memory'] = self.args.basedata_memory
        if self.args.eval_epoch is not None:
            self.param['eval']['eval_epoch'] = self.args.eval_epoch

        # 'limit'
        if 'limit' not in self.param:
            self.param['limit'] = {}
        if self.args.usable_machine_types is not None:
            self.param['limit']['usable_machine_types'] = self.args.usable_machine_types
        elif not self.param['limit'].get('usable_machine_types'):
            self.param['limit']['usable_machine_types'] = None # default is None

        # 'inp_layer' =='none' のとき、'd_model' ==> 'n_mels'
        if self.param['net']['inp_layer'] == 'none':# 直結
            if self.param['net']['d_model'] != self.param['feature']['n_mels']:
                print("set d_model({}) to n_mels({})"\
                    .format(self.param['net']['d_model'],
                            self.param['feature']['n_mels']))
                self.param['net']['d_model'] = self.param['feature']['n_mels']


        # ---------------- 転移学習 -------------------------
        #  yamlの構成
        #    transfer_learning:
        #      tl_fit:
        #        lr: <learning_rate_for_transfer_learning>
        #      tl_eval:
        #         epochs: <num_epochs_for_transfer_leraning>
        #
        if 'transfer_learning' not in self.param:
            self.param['transfer_learning'] = {}
        if 'tl_fit' not in self.param['transfer_learning']:
            self.param['transfer_learning']['tl_fit'] = {}
        if 'tl_eval' not in self.param['transfer_learning']:
            self.param['transfer_learning']['tl_eval'] = {}
        if self.args.tl_lr is not None:
            self.param['transfer_learning']['tl_fit']['lr'] = self.args.tl_lr
        if self.args.tl_dropout is not None:
            self.param['transfer_learning']['tl_fit']['dropout'] = self.args.tl_dropout
        if self.args.tl_batch_size is not None:
            self.param['transfer_learning']['tl_fit']['batch_size'] = self.args.tl_batch_size
        if self.args.tl_data_size is not None:
            self.param['transfer_learning']['tl_fit']['data_size'] = self.args.tl_data_size
        if self.args.tl_shuffle is not None:
            self.param['transfer_learning']['tl_fit']['shuffle'] = self.args.tl_shuffle
        if self.args.tl_val_split is not None:
            self.param['transfer_learning']['tl_fit']['val_split'] = self.args.tl_val_split
         # -------------- データ拡張(転移学習) -------------------------
        if self.args.tl_aug_count is not None:
            self.param['transfer_learning']['tl_fit']['aug_count'] = self.args.tl_aug_count
        if self.args.tl_aug_gadd is not None:
            self.param['transfer_learning']['tl_fit']['aug_gadd'] = self.args.tl_aug_gadd
        if self.args.tl_aug_wcut is not None:
            self.param['transfer_learning']['tl_fit']['aug_wcut'] = self.args.tl_aug_wcut
        # --------------------------------------------------------------------------
        if self.args.tl_epochs is not None:
            self.param['transfer_learning']['tl_eval']['epochs'] = self.args.tl_epochs
        if self.args.tl_eval_epoch is not None:
            self.param['transfer_learning']['tl_eval']['eval_epoch'] = self.args.tl_eval_epoch
        # ---------------------------------------------------
        if self.args.config is not None:
            self.param['misc']['config'] = self.args.config

        # 辞書param['cond']を初期化する
        self.param['cond']={}
        # param['cond']にparam['misc']['config']を追加する
        self.param['cond']['config'] = self.param['misc']['config']

        # param['cond'を更新する
        self.update_cond_params()

        # 上書きをセットする
        self.overwrite = self.param['misc']['overwrite']
        self.overwrite_test = self.param['misc']['overwrite_test']

        # cofig をセットする
        self.config = self.param['misc']['config']

        # verbose をセットする
        self.verbose = self.args.verbose

        return

    def init_random_generators(self):
        seed = self.args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    def select_dirs(self):
        if self.mode:
            self.print(3,"load_directory <- development")
            query = os.path.abspath("{base}/*".format(base=self.param["dev_directory"]))
        else:
            self.print(3,"load_directory <- evaluation")
            query = os.path.abspath("{base}/*".format(base=self.param["eval_directory"]))
        dirs = sorted(glob.glob(query))
        dirs = [f for f in dirs if os.path.isdir(f)]
        return dirs


    def get_section_names(self, machine_dir,  target_dir, ext="wav"):
        # create test files
        query = os.path.abspath("{machine_dir}/{target_dir}/*.{ext}".format(
            machine_dir=machine_dir, target_dir=target_dir, ext=ext))
        file_paths = sorted(glob.glob(query))
        # extract section names
        section_names = sorted(list(set(itertools.chain.from_iterable(
            [re.findall('section_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
        return section_names

    def file_list_generator(self,
                            machine_dir,    # eg. .../dev_data/ToyCar, .../eval_data/ToyCar,
                            section_name,   # 'section_00', ..., 'section_05'
                            target_dir,     # 'train', 'source_test', 'target_test'
                            prefix_normal="normal",
                            prefix_anomaly="anomaly",
                            domain="*", # {'source'|'target'}
                            ext="wav",
                            include_evaluation_data=None # 評価用学習データも含める
                            ):
        self.print(3,"machine_dir : {}".format(machine_dir + "_" + section_name))

        # usable_machine_types に限定するためのヘルパー関数
        def glob_filter(query):
            usable_machine_types = self.param['limit']['usable_machine_types']
            if usable_machine_types:
                pattern = '({})'.format('|'.join(usable_machine_types))
                return sorted([x for x in glob.glob(query) if re.search(pattern, x)])
            else:
                return sorted(glob.glob(query))

        # development
        if self.mode or target_dir == 'train': # dev mode 又は target=='train'
            query = os.path.abspath(
                "{machine_dir}/{target_dir}/{section_name}_{domain}_*_normal_*.{ext}"\
                    .format(machine_dir=machine_dir, target_dir=target_dir,
                            section_name=section_name, domain=domain, ext=ext))
            normal_files = glob_filter(query)
            normal_labels = np.zeros(len(normal_files))

            query = os.path.abspath(
                "{machine_dir}/{target_dir}/{section_name}_{domain}_*_anomaly_*.{ext}"\
                    .format(machine_dir=machine_dir, target_dir=target_dir,
                            section_name=section_name, domain=domain, ext=ext))
            anomaly_files = glob_filter(query)
            anomaly_labels = np.ones(len(anomaly_files))

            if include_evaluation_data:
                # 評価用学習データも含める
                machine_dir_eval = machine_dir.replace('dev_data','eval_data')
                com.print(1, 'includes eval data', machine_dir, machine_dir_eval)
                query = os.path.abspath(
                    "{machine_dir}/{target_dir}/{section_name}_{domain}_*_normal_*.{ext}"\
                        .format(machine_dir=machine_dir_eval, target_dir=target_dir,
                                section_name=section_name, domain=domain, ext=ext))
                normal_files_eval = glob_filter(query)
                normal_labels_eval = np.zeros(len(normal_files_eval))

                files = np.concatenate((normal_files, normal_files_eval, anomaly_files), axis=0)
                labels = np.concatenate((normal_labels, normal_labels_eval, anomaly_labels), axis=0)
                
            else:

                files = np.concatenate((normal_files, anomaly_files), axis=0)
                labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
        
            self.print(3,"#files : {num}".format(num=len(files)))
            if len(files) == 0:
                self.print(0,"no_wav_file!!")

        else: # not dev mode and target_dir is 'source_test' or 'target_test'
            assert not self.mode
            assert target_dir in ['source_test', 'target_test']
            query = os.path.abspath(
                "{machine_dir}/{target_dir}/{section_name}_{domain}_*_*.{ext}"\
                    .format(machine_dir=machine_dir, target_dir=target_dir,
                            section_name=section_name, domain=domain, ext=ext))
            files = glob_filter(query)
            labels = -np.ones(len(files)) # set to -1
        
            self.print(3,"#files : {num}".format(num=len(files)))
            if len(files) == 0:
                self.print(0,"no_wav_file!!")

        return files, labels

    def loss_plot(self, history, save_path, yscale='log'):
        """
        収束の様子を可視化する
        """
        if self.fig is None:
            self.fig = plt.figure(figsize=(7, 5))
            plt.subplots_adjust(wspace=0.3, hspace=0.3)

        for k, v in history.items():
            x = (np.arange(len(v)) + 1) # epoch
            plt.plot(x, v, label=k)

        plt.title(os.path.basename(save_path))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale(yscale)
        plt.legend(loc="upper right")

        plt.savefig(save_path)
        plt.clf() # clear figure

com = Common()
