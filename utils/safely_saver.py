# safely_saver.py
#
# Copyright (c) 2021 ralabo.jp
#
# This software is released under the MIT License.
# see https://opensource.org/licenses/mit-license.php
# ====================================================
import os

class SafelySaver():
    """
    (1) with SafelySaver(path,verbose=0,keep_bak=False) as tmp_path:
            # process using tmp_path
            # where tmp_paths is a tuple of tmp_path0,tmp_path1,...

    (2) with SafelySaver((path0[,path1,...]),verbose=0,keep_bak=False) as (tmp_path0[,tmp_path1,...]):
            # process using tmp_path0, tmp_path1, ...

    tmp_path(拡張子はpathと同じ)を提供する
    tmp_path に対してファイル処理を行い, scopeから外れるとき、
    tmp_path を path に改名する
    古い path があれば bak_path として残す
    path のディレクトリがなければ作成する
    tmp_path, bak_path は path から次のように合成される
        path     : dirname/path.ext
        tmp_path : dirname/path_$tmp.ext
        bak_path : dirname/path_$bak.ext

    ARGS
    keep_bak : 古いいファイル <body>.<ext> があれば <body>_$bak.<ext> として残す

    """
    def __init__(self,paths,verbose=0,keep_bak=False):
        self.verbose=verbose
        self.keep_bak=keep_bak

        self.is_paths_iterable=isinstance(paths,(list,tuple))

        # pathsがiterableでなければ(即ちpathsがsingletonならば)iterableにする
        if not self.is_paths_iterable:
            paths=tuple([paths])
        self.paths=paths
        
        # paths のディレクトリがなければ作成する
        for path in self.paths:
            dir=os.path.dirname(path) or '.'
            os.makedirs(dir,exist_ok=True)

        # tmp_paths, bak_paths を作成する
        self.tmp_paths=[]
        self.bak_paths=[]
        for path in self.paths:
            body,ext=os.path.splitext(path)
            self.tmp_paths.append(body+'_$tmp'+ext)
            self.bak_paths.append(body+'_$bak'+ext)

    def __enter__(self):
        """
        with ... as ... 構文における　as ... の部分を返す
        """
        if self.is_paths_iterable:
            return tuple(self.tmp_paths)
        return self.tmp_paths[0]

    def __exit__(self,exc_type, exc_value, traceback):
        for path,tmp_path,bak_path in zip(self.paths,self.tmp_paths,self.bak_paths):
            if os.path.exists(path):
                if os.path.exists(bak_path):
                    os.remove(bak_path)
                os.rename(path,bak_path)
            if os.path.exists(tmp_path):
                os.rename(tmp_path,path)
                if os.path.exists(bak_path):# bak ファイルがあるとき
                    if self.keep_bak:
                       pass # bak ファイルを残す
                    else:
                        os.remove(bak_path)# bak ファイルを削除する
                if self.verbose:
                    print('saved safely',path)


if __name__=='__main__':
    """test"""

    from utils.safely_saver import SafelySaver
    
    # case-1; single file
    with SafelySaver('tmp1.tmp') as tmp1:
        print('case-1',tmp1)

    # case-2; multiple files
    with SafelySaver(('tmp1.tmp','tmp2.tmp')) as (tmp1,tmp2):
        print('case-2',tmp1,tmp2)

