import sys,os
import numpy as np

class TwoLayerEasyNet:

    def __init__(self,input_size,hidden_size,output_size,weight_init_std=1/1.732):
        #重み初期化
        self.params={}
        self.params['W1']=weight_init_std*np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2']=weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros(output_size)

    def predict(self,x):
        W1,W2=self.params['W1'],self.params['W2']
        b1,b2=self.params['b1'],self.params['b2']

        a1=np.dot(x,W1)+b1
        z1=self.sigmoid(a1)
        a2=np.dot(z1,W2)+b2
        y=self.softmax(a2)

        return y

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def softmax(self,x):#x(241, 3) x.ndim=2,x.T(3,241)
        if x.ndim == 2:#なぜ転置にしたか全然わからん！！！
            x = x.T
            x = x - np.max(x, axis=0)#列方向
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x = x - np.max(x) # オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x))





    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)



###sigmoid()は当然n次元に対応している。
# abc=[[[0,2,3],[4,5,6]],[[0,2,3],[4,5,6]]]
# abc=np.array(abc)
# print(sigmoid(abc))



    




"""
def forward(network,x):
    W1,W2=network['W1'],network['W2']
    b1,b2=network['b1'],network['b2']

    a1=np.dot(x,W1)+b1
    #print(a1)
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=a2
    y=softmax(z2)
    return y

def _change_one_hot_label(X):#0か１かの２値：スカラをワンホット表現にする
    T = np.zeros((X.size, 2))
    for idx, row in enumerate(T):#ループの際にインデックス付きで要素を得る。#rowは別に予約語ではない。
        row[X[idx]] = 1
#        print(X[idx])
    return T

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)#横の中で最大
             
    batch_size = y.shape[0]#y.shape:(241,2)
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size#y[241,2]*t[241,2]t=1となる列のyの和を求めればいい
"""