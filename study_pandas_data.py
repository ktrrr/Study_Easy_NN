import pandas as pd

####Pandasのデータ構造の扱い方####
####20171115####

#http://www.madopro.net/entry/2016/10/14/022723
#ここを参照。すごいわかりやすい

####Seriesのお勉強####

#s=[1,2,3]#リスト
#s= pd.Series([1,2,3])#Series(1次元データ)
s= pd.Series([1,2,3],index=[10,20,30])#Series(1次元データ)indexを指定

t={10:4,20:5,30:6}#dictionary型もおｋ
#print(t.shape)#エラーが出る。
t=pd.Series(t)#sと同じ結果
t=pd.Series(t,name='a')#name属性
#print(t.name)
t=t.rename('abc')#これで名前変更。変更したものを代入するところまで
#print(t)
#print(t.shape)#(3,)

####DataFrameのお勉強####

df=pd.DataFrame({
    's':s,
    't':t
    })
#print(df)
#print(df.shape)#indexが10,20,30と同じなので(3,2)となる。indexが違ったらs,tのデータが結合されず、最大(6,2)となる。

df.index=[i for i in range(len(df))]#こんな感じでindexは変えられる。len()は行数
df.columns=['a','b']#columnも変えられる
#print(df)
#print(df['b'])#列にアクセスできる。戻り値はSeries
#print(df.iloc[1])#行にもアクセスできる。戻り値がSeries
#print(df['b'].shape)#(3,)
#print(df.iloc[1].shape)#(2,)

####Panel:３次元データのお勉強はまた今度####

####pandas の loc、iloc、ix の違い – python http://ailaby.com/lox_iloc_ix/
####わかりやすい
df=pd.DataFrame([[1,2,3],
              [10,20,30],
              [100,200,300],
              [1000,2000,3000]],
              index=['row_0', 'row_1','row_2','row_3'],
              columns=['col_0','col_1','col_2'])
#print(df.loc[['row_2','row_3']])#indexかcolumnsのラベルを指定して表示.index,columnsも表示されている。
#print(df.iloc[2])#2行（0から数えて）の値を抽出。columnsも表示されている。

#下記の違いに注意
#print(df.iloc[1,2])#1行2列の要素にアクセス。20
#print(df.iloc[[1,2]])#1〜2行の全ての列の値にアクセス。
#print(df.iloc[:,[1,2]])#1〜2列の全ての行の値にアクセス
#ixは上のどちらでも指定出来る。扱いちゅうい！
#print(df.loc[[False,False,True,True],[False,True,True]])#取得するラベルをTrue