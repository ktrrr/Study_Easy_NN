"""This is a test program."""

import numpy as np
###################以下はfunction###################

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):#x(241, 3) x.ndim=2,x.T(3,241)
    if x.ndim == 2:#なぜ転置にしたか全然わからん！！！
        x = x.T
        x = x - np.max(x, axis=0)#列方向
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
# 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)#横の中で最大
            
    batch_size = y.shape[0]#y.shape:(241,2)
    #y[241,2]*t[241,2]t=1となる列のyの和を求めればいい
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


def num_grad_func(f, x):#数学的な微分。NN用ではない。fはxの関数
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)#xと同様のshapeを持つ、0を要素とするndarrayが返される。

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])#nditerはN-Dimentionイテレーション。N次元データの要素をWhileなどで取得可能
    while not it.finished:
        idx = it.multi_index#(0,0,0)などのような要素を示すタプル
        tmp_val = x[idx]
        """
        注：Wはダミー変数なのになぜf(x)の実行結果が異なるか。
        ⇒引数は変えてないが、net.param{}の値を変えている。
        loss_W=lambda W : self.loss(x,t)
        のxとtは変えていないが、NNのインスタンス変数params{"W1b1W2b2"}を変えているので
        f(x)の実行結果が異なる。
        
        教訓：関数の引数に安易に値を代入してはいけない。
        """
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)#x[idx]だけ微小量大きくしてほかのx[]は変えずに代入
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)    
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
    
    return grad