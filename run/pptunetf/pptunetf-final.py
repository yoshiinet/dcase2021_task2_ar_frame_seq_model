# Copyright (c) 2021 ralabo.jp
# This software is released under the MIT License.
# see https://opensource.org/licenses/mit-license.php
# ====================================================
import sys
import time

from _04_run_job import run_job

if __name__=='__main__':

    # 探索するハイパーパラメータ
    hyper_params=dict(
        _hyper_params_ = None, # これは先頭にあり、ハイパーパラメータの開始マーカーとなる

        # ========================================================================
        # 組み合わせるハイパーパラメータをリストで指定する
        **( # domain tuning の条件
            # ------------------------------------
            # domain tuning の流れ
            # (1) sourceデータを使ってdomain依存モデルを学習
            # (2) domain依存モデルの出力層を新品に置き換える
            # (3) targetデータを使ってdomain依存モデルの出力層を学習
            dict(
                # domain tuning のハイパーパラメーター
                tl_lr = {         # 出力層の学習率
                    1.0e-05, #+final
                    },
                tl_batch_size = [ # batch size
                    32, #+final
                    ],
                tl_epochs = {     # 転移学習のエポック数(epochsは昇順の逐次処理として実行)
                    8, #+final
                    16, #+final
                    32, #+final
                    },
                tl_eval_epoch = [ # which model is used when evaluation
                    #'last',
                    'best',
                    ],
                tl_dropout = [    # tuning時のdropout rate
                    0.1, #+test
                    ],
                # ----- データ拡張(転移学習) ----------
                tl_aug = [
                    #dict(
                    #    # 水増しする倍数(この倍数だけファイルを水増しする)(0は水増ししない)
                    #    # 水増しは、以下の変形を組合せて行う
                    #    tl_aug_count = 10, # eg. 10
                    #    # 偏差0～aug_gaddの正規乱数を重畳する
                    #    tl_aug_gadd = 3e-4, # eg. 3e-4 
                    #    # 前後計この分の波形をランダムに削る(分析フレーム位置がずれる)
                    #    tl_aug_wcut = 1024,   # eg. 1024 
                    #    ),
                    dict(tl_aug_count=0, tl_aug_gadd=0.0, tl_aug_wcut=0,), # 水増し無し
                    #dict(tl_aug_count=10, tl_aug_gadd=3e-4, tl_aug_wcut=1024,), # ノイズ加算で水増し
                   ],
                )
            # ------------------------------------
            ),
        usable_machine_types = [
            # 使用する machine_type のリスト(for debug or trial)
            # 注意：
            # machine_type 独立単一モデルの場合、この違いはディレクトリ上で区別がつかない
            # この違いを区別するため、config パラメータを変えることで区別することはできる
            #['ToyCar', 'valve', ], #+test
            #['ToyCar','ToyTrain','fan', 'gearbox','pump','slider','valve'],
            # すべて(無指定[]と同じ)
            ],
        n_frames = [
            # ここではフレーム系列の長さの意味
            64, #+final # 64は63フレームから1フレームを予測する場合を意味する
            ],
        n_mels = [
            # 128, #+baseline
            256, #+final
            ],
        d_model = [# 埋め込み次元(nhead の整数倍であること)
            512, #+final
            ],
        d_ff = [ # d_model * 4 が適当らしい
            2048, #+final
            ],
        n_enc_l = [ # transformerの階層数
            6, #+final
            ],
        nhead = [# attentionのヘッド数(d_modelの因数であること)
            16, #+final
            ],
        attn_type = [
            # linear transformer のハイパーパラメータ
            # see https://fast-transformers.github.io/
            'linear', #+test
            #'causal-linear',
            ],
        inp_layer = [
            # 入力層 type, { 'linear', 'none', 'non-linear' }
            #'linear',    # linear
            #'none',      # 直結
            'non-linear', # layer normalization + dropout + relu
            ],
        out_layer = [
             # 出力層 type, { 'linear' }
            'linear',
            ],
        dropout = [
            0.1, #+final
            ],
        lr = [
            # learning rate of RAdam
            1e-05,#+final
            ],
        batch_size = [
            32, #+final
            ],
        pos_enc = {
            # position encoding, { 'embed', 'sin_cos', 'none' }
            #'embed',
            #'sin_cos',
            'none', # ノイズを入れるだけかもしれないのでオフとする
            },
        epochs = [
            # epochsは昇順の逐次処理として実行
            8, #+final
            16, #+final
            32, #+final
            ],
        #
        # ----- データ拡張(通常学習) ----------
        #
        data_aug = [
            #dict(
            #    # 水増しする倍数(この倍数だけファイルを水増しする)(0は水増ししない)
            #    # 水増しは、以下の変形を組合せて行う
            #    aug_count = 0, # eg. 10
            #    # 偏差0～aug_gaddの正規乱数を重畳する
            #    aug_gadd = 0.0, # eg. 3e-4 
            #    # 前後計この分の波形をランダムに削る(分析フレーム位置がずれる)
            #    aug_wcut = 0,   # eg. 1024 
            #    ),

            dict(aug_count=0, aug_gadd=0.0, aug_wcut=0,), # 水増し無し # base
            #dict(aug_count=5, aug_gadd=1e-4, aug_wcut=0,), # ノイズ加算で水増し(5倍) #+new
            ],
        eval_epoch = [
            # which model is used when evaluation
            'best',  # use the best model
            #'last', # use the last model
            ],
        basedata_memory = [
            #'compact', # compact:フレーム系列のみを保持(都度特徴ベクトルを生成)
            'extend', # 生の特徴ベクトルを保持(速い)
            ],
        restart_train = [
            # 1:途中から再開する(overwrite時は最初から)
            1, #+final
            ],
        model_for=[
            # モデルのmachine_type, section_number, target_domainへの依存性
            #dict(f_machine=0, f_section=0, f_target=0), # machine_type 独立(単一モデル)
            #dict(f_machine=1, f_section=0, f_target=0), # machine_type 依存(section, target 独立)
            #dict(f_machine=1, f_section=1, f_target=0), # machine_type,section 依存(target独立)
            # 転移学習
            #dict(f_machine=1, f_section=0, f_target=1), # machine_type,target 依存(section独立,転移学習)
            #dict(f_machine=1, f_section=1, f_target=1), # machine_type,section,target 依存(転移学習)
            # 前処理付加, 前処理後の系列に対する損失を最小にする学習
            #dict(f_machine=1, f_section=0, f_target='pp-raw-tf3'), # Transformer x 3 layer
            #dict(f_machine=1, f_section=1, f_target='pp-raw-tf3'), # Transformer x 3 layer
            dict(f_machine=1, f_section=1, f_target='pp-raw-tf6'), # Transformer x 6 layer
            #dict(f_machine=1, f_section=1, f_target='pp-raw-tf9'), # Transformer x 9 layer
            ],
        data_size = [
            # データ数の制限(-1なら無制限)
            -1, #+final
            ],
        mode = [
            # 開発モード, 評価モード のリスト
            #['dev'],        # 開発モードのみ
            ['dev','eval'], # 開発モードと評価モード(順に処理)
            ],
        # ========================================================================
        _rest_args_ = None, # ここまででハイパーパラメータの組合せは終わり

        # この下は条件の組合せには含まれない

        #debug = 1, # デバグ
        #execution_time = 1, # 実行時間計測(集計をスキップ)
        config = 'final',  #+final
        #overwrite = 1, # for debug(restart_trainより優先される)
        #overwrite_test = 1, # _01_test() を再実行する(結果を上書き)

        save_freq = 2, #+final

        defaults = './run/pptunetf/pptunetf-final.yaml', # ベースのハイパーパラメータ(final)
        verbose = 1,                           # (final)

        #n_jobs = 1,  # 内部をデバグするとき -> 逐次処理(final)
        n_jobs = 3, # 並列処理するとき(final)

        collect_all = 1, # すべての結果を集計する

        #re_calc_score_distribution = 1, # 異常スコア分布の再計算と再評価
        #use_existing_combinations = 1, # 現存するハイパーパラメータの組合せを使う
        #re_make_result_dir = './run/pptunetf/remake_debug_v0.1', # result_directoryの下の評価結果(anormaly_score*.csv, etc.)を再作成する
        )

    # 探索するハイパーパラメータ
    hyper_params_v2 = dict(
        _hyper_params_ = None, # これは先頭にあり、ハイパーパラメータの開始マーカーとなる

        # ========================================================================
        # 組み合わせるハイパーパラメータをリストで指定する
        **( # domain tuning の条件
            # ------------------------------------
            # domain tuning の流れ
            # (1) sourceデータを使ってdomain依存モデルを学習
            # (2) domain依存モデルの出力層を新品に置き換える
            # (3) targetデータを使ってdomain依存モデルの出力層を学習
            dict(
                # domain tuning のハイパーパラメーター
                tl_lr = {         # 出力層の学習率
                    1.0e-05, #+final
                    },
                tl_batch_size = [ # batch size
                    32, #+final
                    ],
                tl_epochs = {     # 転移学習のエポック数(epochsは昇順の逐次処理として実行)
                    8, #+final
                    #16, #+final
                    #32, #+final
                    },
                tl_eval_epoch = [ # which model is used when evaluation
                    #'last',
                    'best',
                    ],
                tl_dropout = [    # tuning時のdropout rate
                    0.1, #+test
                    ],
                # ----- データ拡張(転移学習) ----------
                tl_aug = [
                    #dict(
                    #    # 水増しする倍数(この倍数だけファイルを水増しする)(0は水増ししない)
                    #    # 水増しは、以下の変形を組合せて行う
                    #    tl_aug_count = 10, # eg. 10
                    #    # 偏差0～aug_gaddの正規乱数を重畳する
                    #    tl_aug_gadd = 3e-4, # eg. 3e-4 
                    #    # 前後計この分の波形をランダムに削る(分析フレーム位置がずれる)
                    #    tl_aug_wcut = 1024,   # eg. 1024 
                    #    ),
                    dict(tl_aug_count=0, tl_aug_gadd=0.0, tl_aug_wcut=0,), # 水増し無し
                    #dict(tl_aug_count=10, tl_aug_gadd=3e-4, tl_aug_wcut=1024,), # ノイズ加算で水増し
                   ],
                )
            # ------------------------------------
            ),
        usable_machine_types = [
            # 使用する machine_type のリスト(for debug or trial)
            # 注意：
            # machine_type 独立単一モデルの場合、この違いはディレクトリ上で区別がつかない
            # この違いを区別するため、config パラメータを変えることで区別することはできる
            #['ToyCar', 'valve', ], #+test
            #['ToyCar','ToyTrain','fan', 'gearbox','pump','slider','valve'],
            # すべて(無指定[]と同じ)
            ],
        n_frames = [
            # ここではフレーム系列の長さの意味
            64, #+final # 64は63フレームから1フレームを予測する場合を意味する
            ],
        n_mels = [
            # 128, #+baseline
            256, #+final
            ],
        d_model = [# 埋め込み次元(nhead の整数倍であること)
            512, #+final
            ],
        d_ff = [ # d_model * 4 が適当らしい
            2048, #+final
            ],
        n_enc_l = [ # transformerの階層数
            6, #+final
            ],
        nhead = [# attentionのヘッド数(d_modelの因数であること)
            16, #+final
            ],
        attn_type = [
            # linear transformer のハイパーパラメータ
            # see https://fast-transformers.github.io/
            'linear', #+test
            #'causal-linear',
            ],
        inp_layer = [
            # 入力層 type, { 'linear', 'none', 'non-linear' }
            #'linear',    # linear
            #'none',      # 直結
            'non-linear', # layer normalization + dropout + relu
            ],
        out_layer = [
             # 出力層 type, { 'linear' }
            'linear',
            ],
        dropout = [
            0.1, #+final
            ],
        lr = [
            # learning rate of RAdam
            1e-05,#+final
            ],
        batch_size = [
            32, #+final
            ],
        pos_enc = {
            # position encoding, { 'embed', 'sin_cos', 'none' }
            'embed',#+v2
            #'sin_cos',
            #'none', # ノイズを入れるだけかもしれないのでオフとする
            },
        epochs = [
            # epochsは昇順の逐次処理として実行
            8, #+final
            #16, #+final
            #32, #+final
            ],
        #
        # ----- データ拡張(通常学習) ----------
        #
        data_aug = [
            #dict(
            #    # 水増しする倍数(この倍数だけファイルを水増しする)(0は水増ししない)
            #    # 水増しは、以下の変形を組合せて行う
            #    aug_count = 0, # eg. 10
            #    # 偏差0～aug_gaddの正規乱数を重畳する
            #    aug_gadd = 0.0, # eg. 3e-4 
            #    # 前後計この分の波形をランダムに削る(分析フレーム位置がずれる)
            #    aug_wcut = 0,   # eg. 1024 
            #    ),

            dict(aug_count=0, aug_gadd=0.0, aug_wcut=0,), # 水増し無し # base
            #dict(aug_count=5, aug_gadd=1e-4, aug_wcut=0,), # ノイズ加算で水増し(5倍) #+new
            ],
        eval_epoch = [
            # which model is used when evaluation
            'best',  # use the best model
            #'last', # use the last model
            ],
        basedata_memory = [
            #'compact', # compact:フレーム系列のみを保持(都度特徴ベクトルを生成)
            'extend', # 生の特徴ベクトルを保持(速い)
            ],
        restart_train = [
            # 1:途中から再開する(overwrite時は最初から)
            1, #+final
            ],
        model_for=[
            # モデルのmachine_type, section_number, target_domainへの依存性
            #dict(f_machine=0, f_section=0, f_target=0), # machine_type 独立(単一モデル)
            #dict(f_machine=1, f_section=0, f_target=0), # machine_type 依存(section, target 独立)
            #dict(f_machine=1, f_section=1, f_target=0), # machine_type,section 依存(target独立)
            # 転移学習
            #dict(f_machine=1, f_section=0, f_target=1), # machine_type,target 依存(section独立,転移学習)
            #dict(f_machine=1, f_section=1, f_target=1), # machine_type,section,target 依存(転移学習)
            # 前処理付加, 前処理後の系列に対する損失を最小にする学習
            #dict(f_machine=1, f_section=0, f_target='pp-raw-tf3'), # Transformer x 3 layer
            #dict(f_machine=1, f_section=1, f_target='pp-raw-tf3'), # Transformer x 3 layer
            dict(f_machine=1, f_section=1, f_target='pp-raw-tf6'), # Transformer x 6 layer
            #dict(f_machine=1, f_section=1, f_target='pp-raw-tf9'), # Transformer x 9 layer
            ],
        data_size = [
            # データ数の制限(-1なら無制限)
            -1, #+final
            ],
        mode = [
            # 開発モード, 評価モード のリスト
            #['dev'],        # 開発モードのみ
            ['dev','eval'], # 開発モードと評価モード(順に処理)
            ],
        # ========================================================================
        _rest_args_ = None, # ここまででハイパーパラメータの組合せは終わり

        # この下は条件の組合せには含まれない

        #debug = 1, # デバグ
        #execution_time = 1, # 実行時間計測(集計をスキップ)
        config = 'final',  #+final
        #overwrite = 1, # for debug(restart_trainより優先される)
        #overwrite_test = 1, # _01_test() を再実行する(結果を上書き)

        save_freq = 2, #+final

        defaults = './run/pptunetf/pptunetf-final.yaml', # ベースのハイパーパラメータ(final)
        verbose = 1,                           # (final)

        #n_jobs = 1,  # 内部をデバグするとき -> 逐次処理(final)
        n_jobs = 3, # 並列処理するとき(final)

        collect_all = 1, # すべての結果を集計する
        )

    # 探索するハイパーパラメータ
    hyper_params_v3 = dict(
        _hyper_params_ = None, # これは先頭にあり、ハイパーパラメータの開始マーカーとなる

        # ========================================================================
        # 組み合わせるハイパーパラメータをリストで指定する
        **( # domain tuning の条件
            # ------------------------------------
            # domain tuning の流れ
            # (1) sourceデータを使ってdomain依存モデルを学習
            # (2) domain依存モデルの出力層を新品に置き換える
            # (3) targetデータを使ってdomain依存モデルの出力層を学習
            dict(
                # domain tuning のハイパーパラメーター
                tl_lr = {         # 出力層の学習率
                    1.0e-04, #+final
                    },
                tl_batch_size = [ # batch size
                    8, #+final
                    ],
                tl_epochs = {     # 転移学習のエポック数(epochsは昇順の逐次処理として実行)
                    8, #+final
                    #16, #+final
                    #32, #+final
                    },
                tl_eval_epoch = [ # which model is used when evaluation
                    #'last',
                    'best',
                    ],
                tl_dropout = [    # tuning時のdropout rate
                    0.1, #+test
                    ],
                # ----- データ拡張(転移学習) ----------
                tl_aug = [
                    #dict(
                    #    # 水増しする倍数(この倍数だけファイルを水増しする)(0は水増ししない)
                    #    # 水増しは、以下の変形を組合せて行う
                    #    tl_aug_count = 10, # eg. 10
                    #    # 偏差0～aug_gaddの正規乱数を重畳する
                    #    tl_aug_gadd = 3e-4, # eg. 3e-4 
                    #    # 前後計この分の波形をランダムに削る(分析フレーム位置がずれる)
                    #    tl_aug_wcut = 1024,   # eg. 1024 
                    #    ),
                    dict(tl_aug_count=0, tl_aug_gadd=0.0, tl_aug_wcut=0,), # 水増し無し
                    #dict(tl_aug_count=10, tl_aug_gadd=3e-4, tl_aug_wcut=1024,), # ノイズ加算で水増し
                   ],
                )
            # ------------------------------------
            ),
        usable_machine_types = [
            # 使用する machine_type のリスト(for debug or trial)
            # 注意：
            # machine_type 独立単一モデルの場合、この違いはディレクトリ上で区別がつかない
            # この違いを区別するため、config パラメータを変えることで区別することはできる
            #['ToyCar', 'valve', ], #+test
            #['ToyCar','ToyTrain','fan', 'gearbox','pump','slider','valve'],
            # すべて(無指定[]と同じ)
            ],
        n_frames = [
            # ここではフレーム系列の長さの意味
            32, # 31フレームから1フレームを予測する
            ],
        n_mels = [
            128, #+baseline
            ],
        d_model = [
            # 埋め込み次元(nhead の整数倍であること)
            128,
            ],
        d_ff = [
            # d_model * 4 が適当らしい
            512,
            ],
        n_enc_l = [ # transformerの階層数
            3, #+final
            ],
        nhead = [# attentionのヘッド数(d_modelの因数であること)
            8, #+final
            ],
        attn_type = [
            # linear transformer のハイパーパラメータ, { 'linear', 'causal-linear' }
            # see https://fast-transformers.github.io/
            'linear',
            ],
        inp_layer = [
            # 入力層 type, { 'linear', 'none', 'non-linear' }
            'non-linear', # layer normalization + dropout + relu
            ],
        out_layer = [
             # 出力層 type, { 'linear' }
            'linear',
            ],
        dropout = [
            0.1,
            ],
        lr = [
            # learning rate of RAdam
            1e-05,
            ],
        batch_size = [
            32, #+final
            ],
        pos_enc = {
            # position encoding, { 'embed', 'sin_cos', 'none' }
            'embed',
            },
        epochs = [
            # epochsは昇順の逐次処理として実行
            8,
            ],
        #
        # ----- データ拡張(通常学習) ----------
        #
        data_aug = [
            #dict(
            #    # 水増しする倍数(この倍数だけファイルを水増しする)(0は水増ししない)
            #    # 水増しは、以下の変形を組合せて行う
            #    aug_count = 0, # eg. 10
            #    # 偏差0～aug_gaddの正規乱数を重畳する
            #    aug_gadd = 0.0, # eg. 3e-4 
            #    # 前後計この分の波形をランダムに削る(分析フレーム位置がずれる)
            #    aug_wcut = 0,   # eg. 1024 
            #    ),

            dict(aug_count=0, aug_gadd=0.0, aug_wcut=0,), # 水増し無し # base
            #dict(aug_count=5, aug_gadd=1e-4, aug_wcut=0,), # ノイズ加算で水増し(5倍) #+new
            ],
        eval_epoch = [
            # which model is used when evaluation
            'best',  # use the best model
            #'last', # use the last model
            ],
        basedata_memory = [
            #'compact', # compact:フレーム系列のみを保持(都度特徴ベクトルを生成)
            'extend', # 生の特徴ベクトルを保持(速い)
            ],
        restart_train = [
            # 1:途中から再開する(overwrite時は最初から)
            1,
            ],
        model_for=[
            # モデルのmachine_type, section_number, target_domainへの依存性
            #dict(f_machine=0, f_section=0, f_target=0), # machine_type 独立(単一モデル)
            #dict(f_machine=1, f_section=0, f_target=0), # machine_type 依存(section, target 独立)
            #dict(f_machine=1, f_section=1, f_target=0), # machine_type,section 依存(target独立)
            # 転移学習
            #dict(f_machine=1, f_section=0, f_target=1), # machine_type,target 依存(section独立,転移学習)
            #dict(f_machine=1, f_section=1, f_target=1), # machine_type,section,target 依存(転移学習)
            # 前処理付加, 前処理後の系列に対する損失を最小にする学習
            #dict(f_machine=1, f_section=0, f_target='pp-raw-tf3'), # Transformer x 3 layer
            dict(f_machine=1, f_section=1, f_target='pp-raw-tf3'), # Transformer x 3 layer
            #dict(f_machine=1, f_section=1, f_target='pp-raw-tf6'), # Transformer x 6 layer
            #dict(f_machine=1, f_section=1, f_target='pp-raw-tf9'), # Transformer x 9 layer
            ],
        data_size = [
            # データ数の制限(-1なら無制限)
            -1, #+final
            ],
        mode = [
            # 開発モード, 評価モード のリスト
            #['dev'],        # 開発モードのみ
            ['dev','eval'], # 開発モードと評価モード(順に処理)
            ],
        # ========================================================================
        _rest_args_ = None, # ここまででハイパーパラメータの組合せは終わり

        # この下は条件の組合せには含まれない

        #debug = 1, # デバグ
        #execution_time = 1, # 実行時間計測(集計をスキップ)
        config = 'final',  #+final
        #overwrite = 1, # for debug(restart_trainより優先される)
        #overwrite_test = 1, # _01_test() を再実行する(結果を上書き)

        save_freq = 2, #+final

        defaults = './run/pptunetf/pptunetf-final.yaml', # ベースのハイパーパラメータ(final)
        verbose = 1,                           # (final)

        #n_jobs = 1,  # 内部をデバグするとき -> 逐次処理(final)
        n_jobs = 3, # 並列処理するとき(final)

        collect_all = 1, # すべての結果を集計する

        #re_calc_score_distribution = 1, # 異常スコア分布の再計算と再評価
        #use_existing_combinations = 1, # 現存するハイパーパラメータの組合せを使う
        #re_make_result_dir = './run/pptunetf/remake_debug_v0.1', # result_directoryの下の評価結果(anormaly_score*.csv, etc.)を再作成する
        )

    # 探索するハイパーパラメータ(data_size==-1, (f_machine=1, f_section=0, f_target=0))
    hyper_params_v4 = dict(
        _hyper_params_ = None, # これは先頭にあり、ハイパーパラメータの開始マーカーとなる

        # ========================================================================
        # 組み合わせるハイパーパラメータをリストで指定する
        **( # domain tuning の条件
            # ------------------------------------
            # domain tuning の流れ
            # (1) sourceデータを使ってdomain依存モデルを学習
            # (2) domain依存モデルの出力層を新品に置き換える
            # (3) targetデータを使ってdomain依存モデルの出力層を学習
            dict(
                # domain tuning のハイパーパラメーター
                tl_lr = {         # 出力層の学習率
                    1.0e-04, #+final
                    },
                tl_batch_size = [ # batch size
                    8, #+final
                    ],
                tl_epochs = {     # 転移学習のエポック数(epochsは昇順の逐次処理として実行)
                    8, #+final
                    #16, #+final
                    #32, #+final
                    },
                tl_eval_epoch = [ # which model is used when evaluation
                    #'last',
                    'best',
                    ],
                tl_dropout = [    # tuning時のdropout rate
                    0.1, #+test
                    ],
                # ----- データ拡張(転移学習) ----------
                tl_aug = [
                    #dict(
                    #    # 水増しする倍数(この倍数だけファイルを水増しする)(0は水増ししない)
                    #    # 水増しは、以下の変形を組合せて行う
                    #    tl_aug_count = 10, # eg. 10
                    #    # 偏差0～aug_gaddの正規乱数を重畳する
                    #    tl_aug_gadd = 3e-4, # eg. 3e-4 
                    #    # 前後計この分の波形をランダムに削る(分析フレーム位置がずれる)
                    #    tl_aug_wcut = 1024,   # eg. 1024 
                    #    ),
                    dict(tl_aug_count=0, tl_aug_gadd=0.0, tl_aug_wcut=0,), # 水増し無し
                    #dict(tl_aug_count=10, tl_aug_gadd=3e-4, tl_aug_wcut=1024,), # ノイズ加算で水増し
                   ],
                )
            # ------------------------------------
            ),
        usable_machine_types = [
            # 使用する machine_type のリスト(for debug or trial)
            # 注意：
            # machine_type 独立単一モデルの場合、この違いはディレクトリ上で区別がつかない
            # この違いを区別するため、config パラメータを変えることで区別することはできる
            #['ToyCar', 'valve', ], #+test
            #['ToyCar','ToyTrain','fan', 'gearbox','pump','slider','valve'],
            # すべて(無指定[]と同じ)
            ],
        n_frames = [
            # ここではフレーム系列の長さの意味
            64, # 63フレームから1フレームを予測する
            ],
        n_mels = [
            128, #+baseline
            ],
        d_model = [
            # 埋め込み次元(nhead の整数倍であること)
            256,
            ],
        d_ff = [
            # d_model * 4 が適当らしい
            1024,
            ],
        n_enc_l = [ # transformerの階層数
            6,
            ],
        nhead = [# attentionのヘッド数(d_modelの因数であること)
            8,
            ],
        attn_type = [
            # linear transformer のハイパーパラメータ, { 'linear', 'causal-linear' }
            # see https://fast-transformers.github.io/
            'linear',
            ],
        inp_layer = [
            # 入力層 type, { 'linear', 'none', 'non-linear' }
            'non-linear', # layer normalization + dropout + relu
            ],
        out_layer = [
             # 出力層 type, { 'linear' }
            'linear',
            ],
        dropout = [
            0.1,
            ],
        lr = [
            # learning rate of RAdam
            1e-05,
            ],
        batch_size = [
            32, #+final
            ],
        pos_enc = {
            # position encoding, { 'embed', 'sin_cos', 'none' }
            'embed',
            },
        epochs = [
            # epochsは昇順の逐次処理として実行
            8,
            ],
        #
        # ----- データ拡張(通常学習) ----------
        #
        data_aug = [
            #dict(
            #    # 水増しする倍数(この倍数だけファイルを水増しする)(0は水増ししない)
            #    # 水増しは、以下の変形を組合せて行う
            #    aug_count = 0, # eg. 10
            #    # 偏差0～aug_gaddの正規乱数を重畳する
            #    aug_gadd = 0.0, # eg. 3e-4 
            #    # 前後計この分の波形をランダムに削る(分析フレーム位置がずれる)
            #    aug_wcut = 0,   # eg. 1024 
            #    ),

            dict(aug_count=0, aug_gadd=0.0, aug_wcut=0,), # 水増し無し # base
            #dict(aug_count=5, aug_gadd=1e-4, aug_wcut=0,), # ノイズ加算で水増し(5倍) #+new
            ],
        eval_epoch = [
            # which model is used when evaluation
            'best',  # use the best model
            #'last', # use the last model
            ],
        basedata_memory = [
            #'compact', # compact:フレーム系列のみを保持(都度特徴ベクトルを生成)
            'extend', # 生の特徴ベクトルを保持(速い)
            ],
        restart_train = [
            # 1:途中から再開する(overwrite時は最初から)
            1,
            ],
        model_for=[
            # モデルのmachine_type, section_number, target_domainへの依存性
            #dict(f_machine=0, f_section=0, f_target=0), # machine_type 独立(単一モデル)
            dict(f_machine=1, f_section=0, f_target=0), # machine_type 依存(section, target 独立)
            #dict(f_machine=1, f_section=1, f_target=0), # machine_type,section 依存(target独立)
            # 転移学習
            #dict(f_machine=1, f_section=0, f_target=1), # machine_type,target 依存(section独立,転移学習)
            #dict(f_machine=1, f_section=1, f_target=1), # machine_type,section,target 依存(転移学習)
            # 前処理付加, 前処理後の系列に対する損失を最小にする学習
            #dict(f_machine=1, f_section=0, f_target='pp-raw-tf3'), # Transformer x 3 layer
            #dict(f_machine=1, f_section=1, f_target='pp-raw-tf3'), # Transformer x 3 layer
            #dict(f_machine=1, f_section=1, f_target='pp-raw-tf6'), # Transformer x 6 layer
            #dict(f_machine=1, f_section=1, f_target='pp-raw-tf9'), # Transformer x 9 layer
            ],
        data_size = [
            # データ数の制限(-1なら無制限)
            -1, #+final
            ],
        mode = [
            # 開発モード, 評価モード のリスト
            #['dev'],        # 開発モードのみ
            ['dev','eval'], # 開発モードと評価モード(順に処理)
            ],
        # ========================================================================
        _rest_args_ = None, # ここまででハイパーパラメータの組合せは終わり

        # この下は条件の組合せには含まれない

        #debug = 1, # デバグ
        #execution_time = 1, # 実行時間計測(集計をスキップ)
        config = 'final',  #+final
        #overwrite = 1, # for debug(restart_trainより優先される)
        #overwrite_test = 1, # _01_test() を再実行する(結果を上書き)

        save_freq = 2, #+final

        defaults = './run/pptunetf/pptunetf-final.yaml', # ベースのハイパーパラメータ(final)
        verbose = 1,                           # (final)

        #n_jobs = 1,  # 内部をデバグするとき -> 逐次処理(final)
        n_jobs = 3, # 並列処理するとき(final)

        collect_all = 1, # すべての結果を集計する

        #re_calc_score_distribution = 1, # 異常スコア分布の再計算と再評価
        #use_existing_combinations = 1, # 現存するハイパーパラメータの組合せを使う
        #re_make_result_dir = './run/pptunetf/remake_debug_v0.1', # result_directoryの下の評価結果(anormaly_score*.csv, etc.)を再作成する
        )

    # 探索するハイパーパラメータ(data_size==-2,(f_machine=1, f_section=0, f_target=0))
    hyper_params_v4a = dict(
        _hyper_params_ = None, # これは先頭にあり、ハイパーパラメータの開始マーカーとなる

        # ========================================================================
        # 組み合わせるハイパーパラメータをリストで指定する
        **( # domain tuning の条件
            # ------------------------------------
            # domain tuning の流れ
            # (1) sourceデータを使ってdomain依存モデルを学習
            # (2) domain依存モデルの出力層を新品に置き換える
            # (3) targetデータを使ってdomain依存モデルの出力層を学習
            dict(
                # domain tuning のハイパーパラメーター
                tl_lr = {         # 出力層の学習率
                    1.0e-04, #+final
                    },
                tl_batch_size = [ # batch size
                    8, #+final
                    ],
                tl_epochs = {     # 転移学習のエポック数(epochsは昇順の逐次処理として実行)
                    8, #+final
                    #16, #+final
                    #32, #+final
                    },
                tl_eval_epoch = [ # which model is used when evaluation
                    #'last',
                    'best',
                    ],
                tl_dropout = [    # tuning時のdropout rate
                    0.1, #+test
                    ],
                # ----- データ拡張(転移学習) ----------
                tl_aug = [
                    #dict(
                    #    # 水増しする倍数(この倍数だけファイルを水増しする)(0は水増ししない)
                    #    # 水増しは、以下の変形を組合せて行う
                    #    tl_aug_count = 10, # eg. 10
                    #    # 偏差0～aug_gaddの正規乱数を重畳する
                    #    tl_aug_gadd = 3e-4, # eg. 3e-4 
                    #    # 前後計この分の波形をランダムに削る(分析フレーム位置がずれる)
                    #    tl_aug_wcut = 1024,   # eg. 1024 
                    #    ),
                    dict(tl_aug_count=0, tl_aug_gadd=0.0, tl_aug_wcut=0,), # 水増し無し
                    #dict(tl_aug_count=10, tl_aug_gadd=3e-4, tl_aug_wcut=1024,), # ノイズ加算で水増し
                   ],
                )
            # ------------------------------------
            ),
        usable_machine_types = [
            # 使用する machine_type のリスト(for debug or trial)
            # 注意：
            # machine_type 独立単一モデルの場合、この違いはディレクトリ上で区別がつかない
            # この違いを区別するため、config パラメータを変えることで区別することはできる
            #['ToyCar', 'valve', ], #+test
            #['ToyCar','ToyTrain','fan', 'gearbox','pump','slider','valve'],
            # すべて(無指定[]と同じ)
            ],
        n_frames = [
            # ここではフレーム系列の長さの意味
            64, # 63フレームから1フレームを予測する
            ],
        n_mels = [
            128, #+baseline
            ],
        d_model = [
            # 埋め込み次元(nhead の整数倍であること)
            256,
            ],
        d_ff = [
            # d_model * 4 が適当らしい
            1024,
            ],
        n_enc_l = [ # transformerの階層数
            6,
            ],
        nhead = [# attentionのヘッド数(d_modelの因数であること)
            8,
            ],
        attn_type = [
            # linear transformer のハイパーパラメータ, { 'linear', 'causal-linear' }
            # see https://fast-transformers.github.io/
            'linear',
            ],
        inp_layer = [
            # 入力層 type, { 'linear', 'none', 'non-linear' }
            'non-linear', # layer normalization + dropout + relu
            ],
        out_layer = [
             # 出力層 type, { 'linear' }
            'linear',
            ],
        dropout = [
            0.1,
            ],
        lr = [
            # learning rate of RAdam
            1e-05,
            ],
        batch_size = [
            32, #+final
            ],
        pos_enc = {
            # position encoding, { 'embed', 'sin_cos', 'none' }
            'embed',
            },
        epochs = [
            # epochsは昇順の逐次処理として実行
            8,
            ],
        #
        # ----- データ拡張(通常学習) ----------
        #
        data_aug = [
            #dict(
            #    # 水増しする倍数(この倍数だけファイルを水増しする)(0は水増ししない)
            #    # 水増しは、以下の変形を組合せて行う
            #    aug_count = 0, # eg. 10
            #    # 偏差0～aug_gaddの正規乱数を重畳する
            #    aug_gadd = 0.0, # eg. 3e-4 
            #    # 前後計この分の波形をランダムに削る(分析フレーム位置がずれる)
            #    aug_wcut = 0,   # eg. 1024 
            #    ),

            dict(aug_count=0, aug_gadd=0.0, aug_wcut=0,), # 水増し無し # base
            #dict(aug_count=5, aug_gadd=1e-4, aug_wcut=0,), # ノイズ加算で水増し(5倍) #+new
            ],
        eval_epoch = [
            # which model is used when evaluation
            'best',  # use the best model
            #'last', # use the last model
            ],
        basedata_memory = [
            #'compact', # compact:フレーム系列のみを保持(都度特徴ベクトルを生成)
            'extend', # 生の特徴ベクトルを保持(速い)
            ],
        restart_train = [
            # 1:途中から再開する(overwrite時は最初から)
            1,
            ],
        model_for=[
            # モデルのmachine_type, section_number, target_domainへの依存性
            #dict(f_machine=0, f_section=0, f_target=0), # machine_type 独立(単一モデル)
            dict(f_machine=1, f_section=0, f_target=0), # machine_type 依存(section, target 独立)
            #dict(f_machine=1, f_section=1, f_target=0), # machine_type,section 依存(target独立)
            # 転移学習
            #dict(f_machine=1, f_section=0, f_target=1), # machine_type,target 依存(section独立,転移学習)
            #dict(f_machine=1, f_section=1, f_target=1), # machine_type,section,target 依存(転移学習)
            # 前処理付加, 前処理後の系列に対する損失を最小にする学習
            #dict(f_machine=1, f_section=0, f_target='pp-raw-tf3'), # Transformer x 3 layer
            #dict(f_machine=1, f_section=1, f_target='pp-raw-tf3'), # Transformer x 3 layer
            #dict(f_machine=1, f_section=1, f_target='pp-raw-tf6'), # Transformer x 6 layer
            #dict(f_machine=1, f_section=1, f_target='pp-raw-tf9'), # Transformer x 9 layer
            ],
        data_size = [
            # データ数の制限(-1なら無制限)
            # (-1): development train data のみを学習に使う
            # (-2): evaluation train data も学習に含める
            # (int): データファイル数を最大intに制限する
            #-1, #+v4
            -2, #+v4a
            ],
        mode = [
            # 開発モード, 評価モード のリスト
            #['dev'],        # 開発モードのみ
            ['dev','eval'], # 開発モードと評価モード(順に処理)
            ],
        # ========================================================================
        _rest_args_ = None, # ここまででハイパーパラメータの組合せは終わり

        # この下は条件の組合せには含まれない

        #debug = 1, # デバグ
        #execution_time = 1, # 実行時間計測(集計をスキップ)
        config = 'final',  #+final
        #overwrite = 1, # for debug(restart_trainより優先される)
        #overwrite_test = 1, # _01_test() を再実行する(結果を上書き)

        save_freq = 2, #+final

        defaults = './run/pptunetf/pptunetf-final.yaml', # ベースのハイパーパラメータ(final)
        verbose = 1,                           # (final)

        #n_jobs = 1,  # 内部をデバグするとき -> 逐次処理(final)
        n_jobs = 3, # 並列処理するとき(final)

        collect_all = 1, # すべての結果を集計する

        #re_calc_score_distribution = 1, # 異常スコア分布の再計算と再評価
        #use_existing_combinations = 1, # 現存するハイパーパラメータの組合せを使う
        #re_make_result_dir = './run/pptunetf/remake_debug_v0.1', # result_directoryの下の評価結果(anormaly_score*.csv, etc.)を再作成する
        )

    # 探索するハイパーパラメータ((f_machine=1, f_section=1, f_target='pp-raw-tf6'))
    hyper_params_v5 = dict(
        _hyper_params_ = None, # これは先頭にあり、ハイパーパラメータの開始マーカーとなる

        # ========================================================================
        # 組み合わせるハイパーパラメータをリストで指定する
        **( # domain tuning の条件
            # ------------------------------------
            # domain tuning の流れ
            # (1) sourceデータを使ってdomain依存モデルを学習
            # (2) domain依存モデルの出力層を新品に置き換える
            # (3) targetデータを使ってdomain依存モデルの出力層を学習
            dict(
                # domain tuning のハイパーパラメーター
                tl_lr = {         # 出力層の学習率
                    1.0e-04, #+final
                    },
                tl_batch_size = [ # batch size
                    8, #+final
                    ],
                tl_epochs = {     # 転移学習のエポック数(epochsは昇順の逐次処理として実行)
                    #8, #+final
                    #16, #+final
                    #32, #+final
                    64, #+final
                    },
                tl_eval_epoch = [ # which model is used when evaluation
                    #'last',
                    'best',
                    ],
                tl_dropout = [    # tuning時のdropout rate
                    0.1, #+test
                    ],
                # ----- データ拡張(転移学習) ----------
                tl_aug = [
                    #dict(
                    #    # 水増しする倍数(この倍数だけファイルを水増しする)(0は水増ししない)
                    #    # 水増しは、以下の変形を組合せて行う
                    #    tl_aug_count = 10, # eg. 10
                    #    # 偏差0～aug_gaddの正規乱数を重畳する
                    #    tl_aug_gadd = 3e-4, # eg. 3e-4 
                    #    # 前後計この分の波形をランダムに削る(分析フレーム位置がずれる)
                    #    tl_aug_wcut = 1024,   # eg. 1024 
                    #    ),
                    dict(tl_aug_count=0, tl_aug_gadd=0.0, tl_aug_wcut=0,), # 水増し無し
                    #dict(tl_aug_count=10, tl_aug_gadd=3e-4, tl_aug_wcut=1024,), # ノイズ加算で水増し
                   ],
                )
            # ------------------------------------
            ),
        usable_machine_types = [
            # 使用する machine_type のリスト(for debug or trial)
            # 注意：
            # machine_type 独立単一モデルの場合、この違いはディレクトリ上で区別がつかない
            # この違いを区別するため、config パラメータを変えることで区別することはできる
            #['ToyCar', 'valve', ], #+test
            #['ToyCar','ToyTrain','fan', 'gearbox','pump','slider','valve'],
            # すべて(無指定[]と同じ)
            ],
        n_frames = [
            # ここではフレーム系列の長さの意味
            64, # 63フレームから1フレームを予測する
            ],
        n_mels = [
            128, #+baseline
            ],
        d_model = [
            # 埋め込み次元(nhead の整数倍であること)
            256,
            ],
        d_ff = [
            # d_model * 4 が適当らしい
            1024,
            ],
        n_enc_l = [ # transformerの階層数
            6,
            ],
        nhead = [# attentionのヘッド数(d_modelの因数であること)
            8,
            ],
        attn_type = [
            # linear transformer のハイパーパラメータ, { 'linear', 'causal-linear' }
            # see https://fast-transformers.github.io/
            'linear',
            ],
        inp_layer = [
            # 入力層 type, { 'linear', 'none', 'non-linear' }
            'non-linear', # layer normalization + dropout + relu
            ],
        out_layer = [
             # 出力層 type, { 'linear' }
            'linear',
            ],
        dropout = [
            0.1,
            ],
        lr = [
            # learning rate of RAdam
            1e-05,
            ],
        batch_size = [
            32, #+final
            ],
        pos_enc = {
            # position encoding, { 'embed', 'sin_cos', 'none' }
            'embed',
            },
        epochs = [
            # epochsは昇順の逐次処理として実行
            8,
            ],
        #
        # ----- データ拡張(通常学習) ----------
        #
        data_aug = [
            #dict(
            #    # 水増しする倍数(この倍数だけファイルを水増しする)(0は水増ししない)
            #    # 水増しは、以下の変形を組合せて行う
            #    aug_count = 0, # eg. 10
            #    # 偏差0～aug_gaddの正規乱数を重畳する
            #    aug_gadd = 0.0, # eg. 3e-4 
            #    # 前後計この分の波形をランダムに削る(分析フレーム位置がずれる)
            #    aug_wcut = 0,   # eg. 1024 
            #    ),

            dict(aug_count=0, aug_gadd=0.0, aug_wcut=0,), # 水増し無し # base
            #dict(aug_count=5, aug_gadd=1e-4, aug_wcut=0,), # ノイズ加算で水増し(5倍) #+new
            ],
        eval_epoch = [
            # which model is used when evaluation
            'best',  # use the best model
            #'last', # use the last model
            ],
        basedata_memory = [
            #'compact', # compact:フレーム系列のみを保持(都度特徴ベクトルを生成)
            'extend', # 生の特徴ベクトルを保持(速い)
            ],
        restart_train = [
            # 1:途中から再開する(overwrite時は最初から)
            1,
            ],
        model_for=[
            # モデルのmachine_type, section_number, target_domainへの依存性
            #dict(f_machine=0, f_section=0, f_target=0), # machine_type 独立(単一モデル)
            #dict(f_machine=1, f_section=0, f_target=0), # machine_type 依存(section, target 独立)
            #dict(f_machine=1, f_section=1, f_target=0), # machine_type,section 依存(target独立)
            # 転移学習
            #dict(f_machine=1, f_section=0, f_target=1), # machine_type,target 依存(section独立,転移学習)
            #dict(f_machine=1, f_section=1, f_target=1), # machine_type,section,target 依存(転移学習)
            # 前処理付加, 前処理後の系列に対する損失を最小にする学習
            #dict(f_machine=1, f_section=0, f_target='pp-raw-tf3'), # Transformer x 3 layer
            #dict(f_machine=1, f_section=1, f_target='pp-raw-tf3'), # Transformer x 3 layer
            dict(f_machine=1, f_section=1, f_target='pp-raw-tf6'), # Transformer x 6 layer
            #dict(f_machine=1, f_section=1, f_target='pp-raw-tf9'), # Transformer x 9 layer
            ],
        data_size = [
            # データ数の制限(-1なら無制限)
            -1, #+final
            ],
        mode = [
            # 開発モード, 評価モード のリスト
            #['dev'],        # 開発モードのみ
            ['dev','eval'], # 開発モードと評価モード(順に処理)
            ],
        # ========================================================================
        _rest_args_ = None, # ここまででハイパーパラメータの組合せは終わり

        # この下は条件の組合せには含まれない

        #debug = 1, # デバグ
        #execution_time = 1, # 実行時間計測(集計をスキップ)
        config = 'final',  #+final
        #overwrite = 1, # for debug(restart_trainより優先される)
        #overwrite_test = 1, # _01_test() を再実行する(結果を上書き)

        save_freq = 2, #+final

        defaults = './run/pptunetf/pptunetf-final.yaml', # ベースのハイパーパラメータ(final)
        verbose = 1,                           # (final)

        #n_jobs = 1,  # 内部をデバグするとき -> 逐次処理(final)
        n_jobs = 3, # 並列処理するとき(final)

        collect_all = 1, # すべての結果を集計する

        #re_calc_score_distribution = 1, # 異常スコア分布の再計算と再評価
        #use_existing_combinations = 1, # 現存するハイパーパラメータの組合せを使う
        #re_make_result_dir = './run/pptunetf/remake_debug_v0.1', # result_directoryの下の評価結果(anormaly_score*.csv, etc.)を再作成する
        )

    t0 = time.time()
    #run_job(**hyper_params) # pos-enc == 'none'
    #run_job(**hyper_params_v2)
    #run_job(**hyper_params_v3)
    #run_job(**hyper_params_v4) # data_size==-2, (f_machine=1, f_section=0, f_target=0)
    run_job(**hyper_params_v4a) # data_size==-2, (f_machine=1, f_section=0, f_target=0)
    #run_job(**hyper_params_v5) # model_for=(f_machine=1, f_section=1, f_target='pp-raw-tf6')

    print('total elapsed: {:g}s'.format(time.time()-t0))
