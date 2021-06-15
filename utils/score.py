# Copyright (c) 2021 ralabo.jp
# This software is released under the MIT License.
# see https://opensource.org/licenses/mit-license.php
# ====================================================
import os
import json
import torch
import numpy as np
import matplotlib.pylab as plt

from common import com
from scipy.stats import gamma

def gamma_ppf(decision_thr, gamma_params):
    """
    gamma_params : a dict(a, loc, scale)
    return decision_thr
    """
    return gamma.ppf(q=decision_thr,
                        a=gamma_params['a'],
                        loc=gamma_params['loc'],
                        scale=gamma_params['scale'])

def gamma_pdf(x, gamma_params):
    """
    x : array
    return pdf at x
    """
    return gamma.pdf(x,  
                        a=gamma_params['a'],
                        loc=gamma_params['loc'],
                        scale=gamma_params['scale'])

def gamma_fit(y_pred):
    return gamma.fit(y_pred)

def calc_data_anomaly_scores(model, dl_tra):
    model.eval()
    with torch.no_grad():
        # calculate y_pred for fitting anomaly score distribution
        y_pred = []
        wav_path = []
        for blocks, _, waves in dl_tra:
            blocks = blocks.to(model.device)
            anomaly_scores = calc_batch_anomaly_scores(model, blocks) # a list
            y_pred.extend( anomaly_scores )
            wav_path.extend( waves )

    return y_pred, wav_path

def calc_batch_anomaly_scores(model, blocks):
    """
    学習、評価で呼ばれる共通のコア関数

    return Mean Squared Error for each block in a batch
    """
    if hasattr(model,'calc_anomaly_scores'):
        # GANomaly モデルの場合
        anomaly_scores = model.calc_anomaly_scores(blocks)
    else:
        # model を適用する
        outputs = model( blocks.view( (-1, blocks.shape[-1]) ) )
        if isinstance(outputs,tuple): # VAE の場合は x, mu, logvar のタプル
            outputs = outputs[0] # VAEの場合 x が予測値
        # outputs の形状を blocks の形状に合わせる
        outputs = outputs.view_as( blocks ) 
        # block毎の異常スコアを計算する
        anomaly_scores = ( blocks - outputs ).square().mean([1,2])
        
    # cpuに移動して list に変換する(gpuから離脱)
    anomaly_scores = anomaly_scores.tolist()
    return anomaly_scores

def plot_score_distribution(y_pred, gamma_params, decision_thr, score_distr_fig_path):
    """
    gamma_params : a dict(a, loc, scale) 
    """
    thresh = gamma_ppf(decision_thr, gamma_params)

    plt.title(os.path.basename(score_distr_fig_path))
    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.hist(y_pred, density=True)

    x = np.linspace(min(y_pred), max(y_pred), 50)
    plt.plot(x, gamma_pdf(x, gamma_params), 'r-', lw=5, alpha=0.6, label='gamma pdf')
    plt.axvline(thresh, color='green', lw=2, label='threshold')
    plt.legend()
    #if com.debug:
    #    plt.show()
    plt.savefig(score_distr_fig_path)
    plt.clf() # clear figure
    return

def estimate_score_distribution(model, dl_tra, param,
                                score_distr_param_path,
                                score_distr_csv_path,
                                score_distr_fig_path):
    """
    異常スコア分布パラメータを推定する
    """
    # 異常スコアの分布とガンマ分布パラメータを取得
    # 異常スコアは、フレーム連続を入力して、計算する
 
    # calculate y_pred for fitting anomaly score distribution
    y_pred, wav_path = calc_data_anomaly_scores(model, dl_tra)

    try:
        # fit anomaly score distribution
        gamma_params = gamma_fit(y_pred)

    except RuntimeError as e:
        # The data contains non-finite values.
        y_pred_new = x[~np.isnan(y_pred)]
        print("Warning gamma.fit:{}\nRetry removing nan. len:{} --> len:{}"\
              .format(e, len(y_pred), len(y_pred_new)))
        y_pred = y_pred_new
        gamma_params = gamma_fit(y_pred)

    a, loc, scale = gamma_params
    gamma_params = dict(a=a, loc=loc, scale=scale)

    # 判定閾値を計算し、gamma_paramsに追加する
    thresh = gamma_ppf(com.param['decision']["decision_thr"], gamma_params)
    gamma_params['thresh'] = thresh

    # ガンマ分布パラメータを保存
    with open(score_distr_param_path,'w') as f:
        json.dump(gamma_params, f, indent=2)
    com.print(2, 'save',score_distr_param_path)


    # 異常スコアのヒストグラムとガンマ分布フィットを描画する
    plot_score_distribution(y_pred, gamma_params,
                            param['decision']['decision_thr'],
                            score_distr_fig_path)
    com.print(2, 'save',score_distr_fig_path)

    # 学習データにおける 異常スコアと波形ファイルパスをcsvに記録する
    csv_lines = [[os.path.basename(w),y] for w, y in zip(wav_path, y_pred)]
    com.to_csv(csv_lines, score_distr_csv_path)
    com.print(2,'save',score_distr_csv_path)
