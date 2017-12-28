import pandas as pd

data9022=pd.read_csv("stocks_9022-T_1d_2016.csv",header=0)
print(data9022)
data9022.columns=["Date","Open","High","Low","Close","Volume","Trading Value"]
data9022["index"]=[i for i in range(len(data9022))]

#print(data9022.head(10))#全てのkeyの0〜9列まででてくる
#print(data9022["Date"])#Dateが全て出てくる
#print(data9022.values)#"Date"〜"Trading Value"以外の全てでてくる
#print(data9022.keys)#[245 rows x 8 columns]>

stock_list=[
            9020,#東日本旅客鉄道
            9021,#西日本旅客鉄道
            9048,#名古屋鉄道
            ]
for stock in stock_list:

    df_stock=pd.read_csv("stocks_"+str(stock)+"-T_1d_2016.csv",header=0)
    df_stock.columns=["Date","Open","High","Low","Close","Volume","Trading Value"]

    dates=[]
    closeis=[]

    for d in data9022["Date"]:
        date=df_stock.loc[(df_stock.Date==d),"Date"]#行はdf_stock.Date==dのTrueを指定。列は"Date"の値を指定
        yesterday_date=date.values[0]
        #print(date.values[0])
        dates.append(date.values[0])
        #print(dates)

        close=df_stock.loc[(df_stock.Date==d),"Close"]
        if str(close.values[0])!=str("nan"):
            yesterday_close=close.values[0]
            closeis.append(close.values[0])
            #print(closeis)
        else:
            #print(nan)#nanはない
            closeis.append(yesterday_close)

    df_stock_2=pd.DataFrame({"Date_"+str(stock):dates,"Close_"+str(stock):closeis})
    #print(df_stock.shape)
    dates_array=pd.DataFrame({"sample":dates})
    #print(dates_array)
    data9022=pd.concat([data9022,df_stock_2],axis=1)
    data9022["diff_"+str(stock)]=(data9022["Close_"+str(stock)]/data9022["Close_"+str(stock)].shift(-1))-1
    #print(data9022)

data9022.to_csv("data9022_plus.csv")