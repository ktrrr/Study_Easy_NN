import pandas as pd
import numpy as np
import time

from Two_layer_Easy_NN import TwoLayerEasyNet
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

def _change_one_hot_label(X):#0か１かの２値：スカラをワンホット表現にする
    T = np.zeros((X.size, 2))
    for idx, row in enumerate(T):#ループの際にインデックス付きで要素を得る。#rowは別に予約語ではない。
        row[X[idx]] = 1
#        print(X[idx])
    return T



# def cross_entropy_error(y,t):
#     delta=1e-7
#     return -np.sum(t*np.log(y+delta))







    

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

net=TwoLayerEasyNet(3,4,2)
print(net.params['W1'])
print(net.params['b1'])
print(net.params['W2'])
print(net.params['b2'])
#print(np.sum(y_train==p)/len(y_train))

#print(len(x))
accuracy_cnt=0
#y=np.zeros((len(x), 2))
#print(y)
t=_change_one_hot_label(t)

train_loss_list=[]
train_accuracy_list=[]
train_size=x.shape[0]
#print(train_size)
batch_size=100
learning_rate=0.5
iters_num=10000

#処理前の時刻
t1=time.time()

for i in range(iters_num):
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch=x[batch_mask]
    #y_batch=net.predict(x_batch)
    t_batch=t[batch_mask]
    #grad=net.numerical_gradient(x_batch,t_batch)
    grad=net.gradient(x_batch,t_batch)
    for key in ('W1','b1','W2','b2'):
        net.params[key]-=learning_rate*grad[key]
    loss=net.loss(x_batch,t_batch)
    acc=net.accuracy(x_batch,t_batch)
    train_loss_list.append(loss)
    train_accuracy_list.append(acc)
    #numerical_gradient(d,network['W1'])#できないのでclassを用いる！！！network.Wとかのインスタンス変数を用意！

#処理後の時刻
t2=time.time()

#経過時間
print(f"{t2-t1}秒")

print(net.params['W1'])
print(net.params['b1'])
print(net.params['W2'])
print(net.params['b2'])
#print(train_loss_list)
train_loss_list=pd.DataFrame({"Loss":train_loss_list,"Accuracy":train_accuracy_list})#,"Time":past_time}
train_loss_list.to_csv("Loss_List_0_5.csv")
#print(y)
#print(y.shape)#(241,2)
# print(p)
#print(accuracy_cnt/len(x))

#print(t)
# d=cross_entropy_error(y,t)
# print(d)




"""以下はTwo_layer_Easy_NN.pyにクラスとして移動させた。

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