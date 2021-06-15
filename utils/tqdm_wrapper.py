# Copyright (c) 2021 ralabo.jp
# This software is released under the MIT License.
# see https://opensource.org/licenses/mit-license.php
# ====================================================
"""
リモートマシンの実行結果を、ローカルのVT端末で美しく表示するためのラッパー
"""
# Virtual Terminal Sequences(ANSI escape sequences) を有効化する
from ctypes import windll, wintypes, byref
from functools import reduce
import sys

def enable():
  INVALID_HANDLE_VALUE = -1
  STD_INPUT_HANDLE = -10
  STD_OUTPUT_HANDLE = -11
  STD_ERROR_HANDLE = -12
  ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
  ENABLE_LVB_GRID_WORLDWIDE = 0x0010

  hOut = windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
  if hOut == INVALID_HANDLE_VALUE:
    return False
  dwMode = wintypes.DWORD()
  if windll.kernel32.GetConsoleMode(hOut, byref(dwMode)) == 0:
    return False
  dwMode.value |= ENABLE_VIRTUAL_TERMINAL_PROCESSING
  # dwMode.value |= ENABLE_LVB_GRID_WORLDWIDE
  if windll.kernel32.SetConsoleMode(hOut, dwMode) == 0:
    return False
  return dwMode

ok=enable()
# print('enable ansi=',ok)

import subprocess
p=subprocess.run('powershell write-host $host.name',stdout=subprocess.PIPE)
console_host = p.stdout.strip().lower()
# print('console_host=',console_host)
console_host = ( console_host == 'ConsoleHost'.lower() )
if not console_host:

    # tqdm をカストマイズする
    # '\r' をエスケープシーケンスに置換
    from tempfile import TemporaryFile
    from tqdm import tqdm as tqdm_base # class tqdm を別名tadm_baseとしてインポート

    # 別名tqdm_baseでインポートしたtqdmを基底クラスとした派生クラスを定義
    class tqdm(tqdm_base):
        def __init__(self,*args,file='ignored',**kwds):
            # 書き込み可能な file を作って
            newfile = TemporaryFile()
            # write 属性を write_ansi に置き換え
            newfile.write = self.write_ansi
            # tqdm を初期化する
            super().__init__(*args,**kwds,file=newfile)

        def write_ansi(self,s):
            # 改行文字を エスケープシーケンスに置換する
            s = s.replace('\r','\x1b[1G') # カーソルを行の先頭に
                
            print(s) # printで改行することにより、フラッシュ

            # printによる改行をキャンセルするためカーソルを1行上に移動
            print('\x1b[1A',end='') # １行上に

else:
    from tqdm import tqdm
