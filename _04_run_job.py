# Copyright (c) 2021 ralabo.jp
# This software is released under the MIT License.
# see https://opensource.org/licenses/mit-license.php
# ====================================================
import os
import sys
import subprocess
import traceback
from itertools import product
from joblib import Parallel, delayed
from pprint import pprint
from natsort import natsorted

from _00_train import main as train_main
from _01_test import main as test_main
from _03_collect_results import main as collect_main
from jobs.combinations import search_combinations
from flow.flow_provider import get_flow_instance

from common import com


def f_run(nunits=None,
          usable_machine_types=None,
          n_frames=None, n_mels=None, n_hop_frames=None,
          epochs=None, kld_weight=None,
          lr=None, zdim=None, edim=None, n_enc_l=None,
          nhead=None, inp_layer=None, out_layer=None,
          pos_enc=None, attn_type=None,
          batch_size=None, data_aug=None,
          data_size=None, d_model=None, d_ff=None,
          w_loss=None, dropout=None, model_for=None,
          save_freq=None, restart_train=None, basedata_memory=None,
          eval_epoch=None,
          # ---------------- 転移学習 -------------------------
          tl_lr=None,
          tl_dropout=None,
          tl_batch_size=None,
          tl_data_size=None,
          tl_shuffle=None,
          tl_val_split=None,
          # データ拡張(転移学習)
          tl_aug=None, # a dict(tl_aug_count, tl_aug_gadd, tl_aug_wcut)

          tl_epochs=None,
          tl_eval_epoch=None,
          # ---------------------------------------------------
          overwrite=None,
          overwrite_test=None,
          mode=None,
          defaults=None, config=None, verbose=None, debug=None,
          re_calc_score_distribution=None,
          re_make_result_dir=None,

          # -------- 以下は参照していないキーワードである -----
          section_name=None,
          target_dir=None,
          # ------- 参照していないキーワードは、ここまで ------

          param=None, **kwds):
    """
    f_run()は一つ条件に対して、モデルの学習、評価、描画を行う.
    """
    # コマンドラインに条件を追加する
    # expand model_for, a dict(f_machine, f_section, f_target)
    f_machine = model_for['f_machine'] if model_for else None
    f_section = model_for['f_section'] if model_for else None
    f_target  = model_for['f_target'] if model_for else None
    # expand arg, a dict(aug_count, aug_gadd, aug_wcut)
    aug_count = data_aug['aug_count'] if data_aug else None
    aug_gadd  = data_aug['aug_gadd']  if data_aug else None
    aug_wcut  = data_aug['aug_wcut']  if data_aug else None
    # expand tl_arg, a dict(tl_aug_count, tl_aug_gadd, tl_aug_wcut)
    tl_aug_count = tl_aug['tl_aug_count'] if tl_aug else None
    tl_aug_gadd  = tl_aug['tl_aug_gadd']  if tl_aug else None
    tl_aug_wcut  = tl_aug['tl_aug_wcut']  if tl_aug else None
    sys.argv =  [  sys.argv[0], # script_path
                    *(['--nunits', *[str(x) for x in nunits]] if nunits else []), # as list
                    *(['--usable_machine_types', *[str(x) for x in usable_machine_types]] if usable_machine_types else []), # as list
                    * ['--n_frames', str(n_frames)] * bool(n_frames),
                    * ['--n_mels', str(n_mels)] * bool(n_mels),
                    * ['--n_hop_frames', str(n_hop_frames)] * bool(n_hop_frames),
                    * ['--epochs', str(epochs)] * bool(epochs),
                    * ['--kld_weight', str(kld_weight)] * bool(kld_weight),
                    * ['--lr', str(lr)] * bool(lr),
                    * ['--zdim', str(zdim)] * bool(zdim),
                    * ['--edim', str(edim)] * bool(edim),
                    * ['--n_enc_l', str(n_enc_l)] * bool(n_enc_l),
                    * ['--nhead', str(nhead)] * bool(nhead),
                    * ['--inp_layer', inp_layer] * bool(inp_layer),
                    * ['--out_layer', out_layer] * bool(out_layer),
                    * ['--pos_enc', pos_enc] * bool(pos_enc),
                    * ['--attn_type', attn_type] * bool(attn_type),
                    * ['--batch_size', str(batch_size)] * bool(batch_size),
                    * ['--data_size', str(data_size)] * bool(data_size),
                    * ['--aug_count', str(aug_count)] * bool(aug_count),
                    * ['--aug_gadd', str(aug_gadd)] * bool(aug_gadd),
                    * ['--aug_wcut', str(aug_wcut)] * bool(aug_wcut),
                    * ['--d_model', str(d_model)] * bool(d_model),
                    * ['--d_ff', str(d_ff)] * bool(d_ff),
                    * ['--dropout', str(dropout)] * (dropout is not None), # can be 0.0
                    *(['--w_loss', *[str(x) for x in w_loss]] if w_loss else []), # as list
                    * ['--save_freq', str(save_freq)] * bool(save_freq),
                    * ['--restart_train', str(restart_train)] * bool(restart_train),
                    * ['--basedata_memory', basedata_memory] * bool(basedata_memory),
                    * ['--eval_epoch', str(eval_epoch)] * bool(eval_epoch),
                    * ['--overwrite'] * bool(overwrite), # as flag
                    * ['--overwrite_test'] * bool(overwrite_test), # as flag
                    * ['--defaults', defaults] * bool(defaults),
                    * ['--config', config] * bool(config),
                    * ['--verbose'] * (verbose or 0), # as count
                    * ['--debug'] * bool(debug), # as flag
                    * ['--re_calc_score_distribution'] * bool(
                        re_calc_score_distribution), # as flag
                    * ['--re_make_result_dir', str(re_make_result_dir)] \
                        * bool(re_make_result_dir), # as str
                    * ['--f_machine', str(f_machine)] * (f_machine is not None), # can be 0
                    * ['--f_section', str(f_section)] * (f_section is not None), # can be 0
                    * ['--f_target', str(f_target)] * (f_target is not None), # can be 0
                    # ---------------- 転移学習 -------------------------
                    * ['--tl_lr', str(tl_lr)] * bool(tl_lr),
                    * ['--tl_dropout', str(tl_dropout)] * bool(tl_dropout),
                    * ['--tl_batch_size', str(tl_batch_size)] * bool(tl_batch_size),
                    * ['--tl_data_size', str(tl_data_size)] * bool(tl_data_size),
                    * ['--tl_shuffle', str(tl_shuffle)] * bool(tl_shuffle),
                    * ['--tl_val_split', str(tl_val_split)] * bool(tl_val_split),
                    * ['--tl_aug_count', str(tl_aug_count)] * bool(tl_aug_count),
                    * ['--tl_aug_gadd', str(tl_aug_gadd)] * bool(tl_aug_gadd),
                    * ['--tl_aug_wcut', str(tl_aug_wcut)] * bool(tl_aug_wcut),
                    * ['--tl_epochs', str(tl_epochs)] * bool(tl_epochs),
                    * ['--tl_eval_epoch', str(tl_eval_epoch)] * bool(tl_eval_epoch),
                    # ---------------------------------------------------

                    ]
    argv = sys.argv
    results = []
    for mode_ in mode: # モードの順に処理
        sys.argv = argv + ['--mode', str(mode_)]
        try:
            if kwds:
                # kwds(余ったkwargsのこと) があるときは、configurationの値と一致するはず
                for k,v in kwds.items():
                    val = com.find_param(k, param)
                    assert val==v,"hyper-param {}:{} differs from configuration:{}".format(k,v,val)

            # 処理を行う
            train_main()
            test_main()
            results.append( ('completed', sys.argv[1:]) ) # success

        except Exception as e:
            print('\x1b[93mError が起きました\nError:{}\x1b[91m\n\x1b[95margv:{}\n\x1b[91m{}\x1b[0m'.format(
                e, sys.argv,traceback.format_exc()))
            if '--debug' in sys.argv:
                raise
            results.append( ('failed', sys.argv[1:]) )# failed

    return results


def run_job(_hyper_params_, # ここから _rest_args_ までの間は f_run()に渡る

            # 組み合わせに含む条件リスト
            nunits=None,
            usable_machine_types=None, n_frames=None, n_mels=None,
            n_hop_frames=None,
            kld_weight=None, lr=None, zdim=None, edim=None,
            n_enc_l=None, nhead=None, inp_layer=None,
            out_layer=None, pos_enc=None, attn_type=None,
            batch_size=None, data_aug=None,
            data_size=None,
            d_model=None, d_ff=None,
            w_loss=None, dropout=None, model_for=None,
            eval_epoch=None,
            basedata_memory=None,
            restart_train=None,
            # ---------------- 転移学習 -------------------------
            tl_lr=None,
            tl_dropout=None,
            tl_batch_size=None,
            tl_aug=None, # a dict(tl_aug_count, tl_aug_gadd, tl_aug_wcut)
            tl_eval_epoch=None,
            # ---------------------------------------------------
            mode=None, # 開発モード('dev'), 評価モード('eval')
            _rest_args_=None, # f_fun()に渡るhyper paramsはここまで
            
            # ここから下は 条件の組合せには含まれない
            # 追加する場合は、 下の options にも追加せよ
            epochs=None, # epochsは組合せの外(昇順の逐次処理として実行)

            # ---------------- 転移学習 -------------------------
            tl_epochs=None, # tl_epochsは組合せの外(昇順の逐次処理として実行)
            # ---------------------------------------------------

            save_freq=None,

            n_jobs=7,                        # 並列処理するとき
            collect_all=None,                # すべての結果を集計する
            re_calc_score_distribution=None, # 異常スコア分布の再計算と再評価
            re_make_result_dir=None,         # 評価結果の再作成

            overwrite=None,
            overwrite_test=None,
            defaults=None,  # ハイパーパラメータの設定ファイル
            config=None,
            verbose=None,
            debug=None,

            # 現存するディレクトリを探索してハイパーパラメータの組合せを取得し用いる
            # この場合、この関数の引数で指定されるハイパーパラメータは無視される
            use_existing_combinations=None,
            execution_time=None,

            ):
    """
    _hyper_params_から_rest_args_までの引数は、hyper parameterのiterableであり、
    これら iterable の組み合わせに対して、 上記の 関数 f_run()を call する.

    f_run()は一つ条件に対して、モデルの学習、評価、描画を行う.
    """
    remote = '--remote' in sys.argv # GPUをつかって演算する(リモートで実行するべき)
    if remote:
        print('remote')
    local = '--local' in sys.argv # 結果を収集表示する(ローカルで実行するべき)
    if local:
        print('local')

    # ハイパーパラメータ外の条件(f_runに渡される)
    options = dict(save_freq=save_freq,
                   overwrite=overwrite,
                   overwrite_test=overwrite_test,
                   defaults=defaults,
                   config=config,
                   verbose=verbose,
                   debug=debug,
                   re_calc_score_distribution=re_calc_score_distribution,
                   re_make_result_dir=re_make_result_dir,
                   )

    if re_make_result_dir:# 評価結果を再作成するとき
        # timestampを作成する
        if not os.path.exists(re_make_result_dir):
            with open(re_make_result_dir,'w') as f:
                pass # timestamp のみ使用。内容は空である。
        # 現存するディレクトリを探索してハイパーパラメータの組合せを取得する
        # この関数の引数で指定されるハイパーパラメータは無視する
        com.command_line_chk(**options)
        combinations = search_combinations(com)
        tl_combinations = search_combinations(com, transfer_learning=True)
        combinations += tl_combinations

        # APIに合わせるため combination を修正する
        for c in combinations:
            # model_for を作成する
            #f_machine = model_for['f_machine'] if model_for else None
            #f_section = model_for['f_section'] if model_for else None
            #f_target  = model_for['f_target'] if model_for else None
            c['model_for'] = dict(f_machine=c['f_machine'],
                                  f_section=c['f_section'],
                                  f_target=c['f_target'])
            del c['f_machine'], c['f_section'], c['f_target']

            # data_aug を作成する
            #aug_count = data_aug['aug_count'] if data_aug else None
            #aug_gadd  = data_aug['aug_gadd']  if data_aug else None
            #aug_wcut  = data_aug['aug_wcut']  if data_aug else None
            c['data_aug'] = dict(aug_count=c['aug_count'],
                                 aug_gadd=c['aug_gadd'],
                                 aug_wcut=c['aug_wcut'])
            del c['aug_count'], c['aug_gadd'], c['aug_wcut']

            if 'tl_aug_count' in c:
                # tl_aug を作成する
                #tl_aug_count = tl_aug['tl_aug_count'] if tl_aug else None
                #tl_aug_gadd  = tl_aug['tl_aug_gadd']  if tl_aug else None
                #tl_aug_wcut  = tl_aug['tl_aug_wcut']  if tl_aug else None
                c['tl_aug'] = dict(tl_aug_count=c['tl_aug_count'],
                                   tl_aug_gadd=c['tl_aug_gadd'],
                                   tl_aug_wcut=c['tl_aug_wcut'])
                del c['tl_aug_count'], c['tl_aug_gadd'], c['tl_aug_wcut']

            # 'result_directory' からは復元できないものを追加
            # usable_machine_types を追加
            c['usable_machine_types'] = [c['machine_types']]

        param = com.param
    elif use_existing_combinations:# 現存するハイパーパラメータの組合せを使う
        assert False,'use_existing_combinations is not supported'
        # 現存するディレクトリを探索してハイパーパラメータの組合せを取得する
        # この関数の引数で指定されるハイパーパラメータは無視する
        com.command_line_chk(**options)
        combinations = search_combinations(com,
                                           transfer_learning=False,
                                           use_existing_combinations=True)
        tl_combinations = search_combinations(com,
                                              transfer_learning=True,
                                              use_existing_combinations=True)
        combinations += tl_combinations

        # APIに合わせるため combination を修正する
        for c in combinations:
            # model_for を作成する
            #f_machine = model_for['f_machine'] if model_for else None
            #f_section = model_for['f_section'] if model_for else None
            #f_target  = model_for['f_target'] if model_for else None
            c['model_for'] = dict(f_machine=c['f_machine'],
                                  f_section=c['f_section'],
                                  f_target=c['f_target'])
            del c['f_machine'], c['f_section'], c['f_target']

            # data_aug を作成する
            #aug_count = data_aug['aug_count'] if data_aug else None
            #aug_gadd  = data_aug['aug_gadd']  if data_aug else None
            #aug_wcut  = data_aug['aug_wcut']  if data_aug else None
            c['data_aug'] = dict(aug_count=c['aug_count'],
                                 aug_gadd=c['aug_gadd'],
                                 aug_wcut=c['aug_wcut'])
            del c['aug_count'], c['aug_gadd'], c['aug_wcut']

            if 'tl_aug_count' in c:
                # tl_aug を作成する
                #tl_aug_count = tl_aug['tl_aug_count'] if tl_aug else None
                #tl_aug_gadd  = tl_aug['tl_aug_gadd']  if tl_aug else None
                #tl_aug_wcut  = tl_aug['tl_aug_wcut']  if tl_aug else None
                c['tl_aug'] = dict(tl_aug_count=c['tl_aug_count'],
                                   tl_aug_gadd=c['tl_aug_gadd'],
                                   tl_aug_wcut=c['tl_aug_wcut'])
                del c['tl_aug_count'], c['tl_aug_gadd'], c['tl_aug_wcut']

        param = com.param
    else: # 通常の処理
        # この関数の引数'_hyper_params_'と'_rest_args_'に囲まれた
        # 引数（リスト型）の組合せを辞書として取得する
        kwd = locals().copy()
        keys = list(kwd.keys())
        keys = keys[keys.index('_hyper_params_')+1:keys.index('_rest_args_')]
        keys = [k for k in keys if kwd[k]] # 空でないkey のみを選択する
        keys.reverse() # 最初のほうが早く変化するようにするreveseする
        kwd = { k:kwd[k] for k in keys}
        # ハイパーパラメータの組合せを生成する
        combinations = product(*[v for k,v in kwd.items()])
        def to_kwd(*a):
            return { k:v for v,k in zip(a, keys) }
        combinations = [to_kwd(*a) for a in combinations] # 組合せた条件

        # machine_type 依存の場合はmachine_typeについても並列化する
        new_combinations = [] # ここにmachine_typeを展開した組合せを得る
        for c in combinations:
            if c['model_for']['f_machine']: # machine_type 依存の場合
                flow = get_flow_instance(c)
                for machine_type in flow.machine_types:
                    usable_machine_types = c.get('usable_machine_types')
                    if not usable_machine_types or machine_type in usable_machine_types:
                        new_c = c.copy()
                        # 'usable_machine_types'をつかってmachine_typeを限定
                        new_c['usable_machine_types'] = [machine_type]
                        new_combinations.append(new_c)
            else:
                new_combinations.append(c)
        combinations = new_combinations # 新しい組合せに置き換える

        param={}

    # f_run()を逐次または並列で実行する
    if remote or not local:
        results = []
        if re_make_result_dir or use_existing_combinations: # 評価結果を再作成するとき
            epochs = natsorted(set([c.get('epochs') for c in combinations]))
            tl_epochs = natsorted(set([c.get('tl_epochs') for c in combinations]))
            epoch_combinations = list(product(epochs, tl_epochs)) # 注意:list()必要
            #print('epochs=',epochs)
            #print('tl_epochs=',tl_epochs)
            print('epoch_combinations:',epoch_combinations)

            def is_same_epochs(c):
                """
                (c.epochs, c.tl_epochs) が一致するとき true を返すフィルタ関数
                """
                return c.get('epochs')==e and c.get('tl_epochs')==tl_e

            if not n_jobs or n_jobs==1:
                print('\x1b[93mWarning 逐次処理 です\x1b[0m')
                for e,tl_e in epoch_combinations:
                    com.print(2,'epochs={} tl_epochs={}'.format(e,tl_e))
                    ## 逐次処理
                    for c in filter(is_same_epochs, combinations):
                        results.extend(f_run(**c, **options, param=param))
            else:
                print('\x1b[92m並列処理 です\x1b[0m')
                for e, tl_e in epoch_combinations:
                    com.print(2,'epochs={} tl_epochs={}'.format(e,tl_e))
                    # 並列処理
                    results.extend(
                        Parallel(n_jobs=n_jobs)(
                            delayed(f_run)(**c, **options, param=param)
                            for c in filter(is_same_epochs, combinations)))


        else: # 通常の組合せ処理
            # epochsは昇順の逐次処理として実行
            epochs = epochs and sorted(epochs) or [None]

            # tl_epochsは昇順の逐次処理として実行
            tl_epochs = tl_epochs and sorted(tl_epochs) or [None]

            epoch_combinations = list(product(epochs, tl_epochs)) # 注意:list()必要

            if not n_jobs or n_jobs==1:
                print('\x1b[93mWarning 逐次処理 です\x1b[0m')
                for e,tl_e in epoch_combinations:
                    com.print(2,'epochs={} tl_epochs={}'.format(e,tl_e))
                    ## 逐次処理
                    for k in combinations:
                        results.append( f_run(**k, **options, param=param,
                                              epochs=e,
                                              tl_epochs=tl_e) )
            else:
                print('\x1b[92m並列処理 です\x1b[0m')
                for e, tl_e in epoch_combinations:
                    com.print(2,'epochs={} tl_epochs={}'.format(e,tl_e))
                    # 並列処理
                    results.extend(
                        Parallel(n_jobs=n_jobs)(
                            delayed(f_run)(**k,
                                           **options,
                                           param=param,
                                           epochs=e,
                                           tl_epochs=tl_e)
                            for k in combinations))

        # 実行結果(train(), test()の結果)をプリントする
        print('\n'+'='*55,'実行結果','='*55)
        for x in results:
            pprint(x, compact=True, width=120)
        print('='*120)

    if execution_time:
        return # skip below

    # 結果を集計する
    if local or not remote:
        print('wait for collecting ...')
        sys.argv += ['--mode', 'dev'] # 開発モードに設定
        if collect_all:
            combinations = None # collect all results

        # to ods
        save_to = 'csv' # view xlsx
        collect_main(combinations, **options, save_to=save_to)

        # 結果を表示
        if save_to == 'ods': # view ods
            ods_path = com.summary_ods_path()
            os.system(os.path.abspath(ods_path))
        elif save_to == 'xlsx': # view xlsx
            xlsx_path = com.summary_xlsx_path()
            os.system(os.path.abspath(xlsx_path))
        elif save_to == 'csv': # view csv
            csv_path = com.summary_csv_path()
            os.system(os.path.abspath(csv_path))
