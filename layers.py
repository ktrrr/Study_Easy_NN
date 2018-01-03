import numpy as np
from function import *

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):#doutはベクトル
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1/(1+np.exp(-x))
        self.out = out
        return out

    def backward(self,dout):#doutはスカラ
        dx=dout*(1.0-self.out)*self.out
        return dx


class Affine:
    def __init__(self,W,b):
        self.W=W
        self.b=b
        self.x=None
        self.dW=None#学習に使用するdL/dW
        self.db=None#学習に使用するdL/db
    def forward(self,x):
        self.x=x
        out=np.dot(x,self.W)+self.b
        return out
    
    def backward(self,dout):#doutはスカラ
        dx=np.dot(dout,self.W.T)
        self.dW=np.dot(self.x.T,dout)#forwardを実行したあとでないとエラー吐く
        self.db=np.sum(dout,axis=0)

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss=None#損失関数
        self.y=None#softmaxの出力
        self.t=None#教師データ（one-hot-表現）
    def forward(self,x,t):
        self.t=t
        self.y=softmax(x)
        self.loss=cross_entropy_error(self.y,self.t)
        return self.loss
    def backward(self,dout=1):#ソフトマックスのdoutは1
        batch_size=self.t.shape[0]
        dx=(self.y-self.t)/batch_size
        return dx





#"""sigmoidテスト

#Affineテスト。
W=np.array([[1,2,3],[4,5,6]])
b=np.array([1,2,3])
t=np.array([1,1])
x=np.array([1,2])
# net=Affine(W,b)

# net.forward(x)
# print(net.backward(1))

#####Sigmoidテスト
Sig=Sigmoid()
# print(Sig.forward(x))
# print(Sig.backward(0.1))

# Sft=SoftmaxWithLoss()
# print(Sft.forward(x,t))
# print(Sft.backward())
# #"""
# x=np.array([-1,10,10,-2])
# Rel = Relu()
# out = Rel.forward(x)

# print(Rel.mask)
# print(out)
# print(Rel.backward(out))
