import pandas as pd
import numpy as np
df = pd.read_csv("data9022_plus.csv")
#print(len(df))#245#データは246行まで存在する。１行めはラベル用len()は行の数を数える。
#print(len(df.iloc[0]))#18#１列目(index)も含めて数える
# ilocを使った列選択
# 文法 ：iloc[rows番号, columns番号]の形で書く
df = df.sort_values(by=["index"],ascending=False)#ascenmding=False:indexの値が大きい順に並ぶ：古い日付順
df=df.iloc[0:len(df)-1]
#print(df)
df_train=df.iloc[1:len(df)-1]#一番古いDateは捨てる。前日のデータがないので〜
df_test=df.iloc[len(df)-1:len(df)]#テスト用に分ける#次回目標テスト用データは2017のcsvを成形して作る？
#print(df_test)

xlist=[
        "diff_9020",
        "diff_9021",
        "diff_9048",
        ]

x_train=[]#リスト
y_train=[]
#print(df_train.shape)
for s in range(0,len(df_train)-1):
    #print("x_train :",df_train["Date"].iloc[s])
    #print("y_train :",df_train["Date"].iloc[s+1])
    #print("")
    #print(df_train[xlist].iloc[s])
    #print(df_train["diff_9020"].iloc[s])
    x_train.append(df_train[xlist].iloc[s])
    if df_train["Close"].iloc[s+1]>df_train["Close"].iloc[s]:
        y_train.append(1)
    else:
        y_train.append(0)



def sigmoid(x):
    return 1/(1+np.exp(-x))

###sigmoid()は当然n次元に対応している。
# abc=[[[0,2,3],[4,5,6]],[[0,2,3],[4,5,6]]]
# abc=np.array(abc)
# print(sigmoid(abc))

#だめなソフトマックス。
# def softmax(a):
#     c=np.max(a)
#     exp_a=np.exp(a-c)
#     sum_exp_a=np.sum(exp_a)
#     y=exp_a/sum_exp_a
#     return y

def softmax(x):#x(241, 3) x.ndim=2,x.T(3,241)
    if x.ndim == 2:#なぜ転置にしたか全然わからん！！！
        x = x.T
        x = x - np.max(x, axis=0)#列方向
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))




def network(input_size,hidden_size,output_size,weight_init_std=1/1.732):
    network={}
    network['W1']=weight_init_std*np.random.randn(input_size,hidden_size)
    #print(network['W1'])
    network['b1']=np.zeros(hidden_size)
    network['W2']=weight_init_std*np.random.randn(hidden_size,output_size)
    network['b2']=np.zeros(output_size)
    return network

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

########enumerateの練習#########
# X=[0,0,1,1,1,0]
# print(X)
# print(_change_one_hot_label(np.array(X)))
# T=_change_one_hot_label(np.array(X))
# for idx,row in enumerate(T):
#     print(row)
#     print(idx)
#     row[0]=idx
# print(T)
#########argmaxの勉強
#axisを指定しなければ、1次元配列に強制的に直されてそのうちはじめに出てくる最大値を出すインデックスが帰ってくる。
#axis=0、行方向（行が増える方向、縦）に最大値を求めて、そのインデックスが返ってくる。よって返り値は配列となる。

# def cross_entropy_error(y,t):
#     delta=1e-7
#     return -np.sum(t*np.log(y+delta))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)#横の中で最大
             
    batch_size = y.shape[0]#y.shape:(241,2)
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size#y[241,2]*t[241,2]t=1となる列のyの和を求めればいい



def numerical_gradient(f, x):#fはCross_Entropy_Error，xは重みW
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)#xと同様のshapeを持つ、0を要素とするndarrayが返される。
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])#nditerはN-Dimentionイテレーション。N次元データの要素をWhileなどで取得可能
    while not it.finished:
        idx = it.multi_index#(0,0,0)などのような要素を示すタプル
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)#x[idx]だけ微小量大きくしてほかのx[]は変えずに代入
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad
#print(x_train)#listにsizeなんて属性はない
#df_x_train=pd.DataFrame(x_train)
#rint(df_x_train.iloc[0])
#print(y_train)
x=np.array(x_train)#問題がおこるよ
#print(x.T.shape)
t=np.array(y_train)
#print(t[0].shape)
#print(x)
#t=np.array(y_train)
#print(np.array(y_train).shape)
#print(x.shape)

network=network(3,4,2)
# y=forward(network,x)
# p=np.argmax(y,axis=1)

#print(y_train==p)
#print(y)
#print(np.sum(y_train==p)/len(y_train))

#print(len(x))
accuracy_cnt=0
#y=np.zeros((len(x), 2))
#print(y)
t=_change_one_hot_label(t)
batch_size=1
for i in range(0,len(x),batch_size):
    x_batch=x[i:i+batch_size]
    y_batch=forward(network,x_batch)
#    d=cross_entropy_error(y_batch,t)
    print(y_batch)
    p=np.argmax(y_batch,axis=1)
    accuracy_cnt+=np.sum(p==t[i:i+batch_size])
#print(y)
#print(y.shape)#(241,2)
# print(p)
#print(accuracy_cnt/len(x))

#print(t)
# d=cross_entropy_error(y,t)
# print(d)
