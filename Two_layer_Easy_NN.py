import numpy as np
from function import *
from layers import *
from collections import OrderedDict

class TwoLayerEasyNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=1/1.732):
        #重み初期化
        self.params = {}
        self.params['W1'] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        #レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():#value:オブジェクトをlayerに代入してる
            x = layer.forward(x)

        return x




    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)#列方向
        t = np.argmax(t, axis=1)

        accuraccy = np.sum(y == t)/float(x.shape[0])
        return accuraccy


    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)



    def numerical_gradient(self,x,t):#fはCross_Entropy_Error，xは重みW
        loss_W=lambda W : self.loss(x,t)
        grads={}
        grads['W1']=num_grad_func(loss_W,self.params['W1'])
        grads['b1']=num_grad_func(loss_W,self.params['b1'])
        grads['W2']=num_grad_func(loss_W,self.params['W2'])
        grads['b2']=num_grad_func(loss_W,self.params['b2'])
        return grads

    def gradient(self, x, t):
        #forward
        self.loss(x,t)

        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()#listクラスのメソッド
        for layer in layers:
            dout = layer.backward(dout)

        #勾配の設定
        grads = {}
        grads['W1']=self.layers['Affine1'].dW
        grads['b1']=self.layers['Affine1'].db
        grads['W2']=self.layers['Affine2'].dW
        grads['b2']=self.layers['Affine2'].db

        return grads


# # lambda関数のお勉強
# x=1
# t=2
# loss_W=lambda W: x*2+t*2#Wは何の数字でもよい。Wの値に関わらず、loss_Wはきまる
# print(loss_W((0,0)))

"""
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad
"""

    




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
    # def predict(self, x):
    #     W1, W2 = self.params['W1'], self.params['W2']
    #     b1, b2 = self.params['b1'], self.params['b2']

    #     a1 = np.dot(x, W1)+b1
    #     z1 = sigmoid(a1)
    #     a2 = np.dot(z1, W2)+b2
    #     y = softmax(a2)

    #     return y
"""