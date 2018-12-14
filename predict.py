# -*- coding: utf-8 -*-
# テスト4, 体重、身長の重回帰分析。
# 学習ロード　＞評価


import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd

# 可視化モジュール
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
# 機械学習モジュール
import sklearn
import pickle

#Load
# モデルを保存する
# 保存したモデルをロードする
filename = 'model.pkl'
model = pickle.load(open(filename, 'rb'))
#quit()
#
# 学習データ
wdata = pd.read_csv("data.csv" )
wdata.columns =["no", "price","siki_price", "rei_price" ,"menseki" ,"nensu" ,"toho" ,"madori" ,"houi" ,"kouzou" ]
print(wdata.head() )
#print(wdata["NO"][: 10 ] )

# conv=> num
sub_data = wdata[[ "no","price","siki_price", "rei_price" ,"menseki" ,"nensu" ,"toho" ] ]
sub_data = sub_data.assign(price=pd.to_numeric( sub_data.price))

#print(sub_data["price"][: 10])
##quit()
#

# データの分割（学習データとテストデータに分ける）
from sklearn.model_selection import train_test_split
# モデル
from sklearn import linear_model
# モデルのインスタンス
#l_model = linear_model.LinearRegression() 
# 説明変数に "price" 以外を利用
X = sub_data.drop("price", axis=1)
print(X.shape )
#print( type( X) )
#print(X[: 10 ] )
# 目的変数
Y = sub_data["price"]
# 学習データとテストデータに分ける
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25 ,random_state=0)
print(X_train.shape , y_train.shape  )
print(X_test.shape , y_test.shape  )
#print( type( X_test ) )
#predict
#tdat =X_test[1: 2]
tdat =X_test[0: 5 ]
#print(tdat )
pred = model.predict(tdat )
#print(pred.shape )
print(pred )
#print(pred[: 10])
quit()
d  = np.array(pred )
frame1 = DataFrame(d )
print(frame1.shape)
print(frame1.head() )

#print(type(pred ))
#X_test.head(10 )
#print(X_test[: 10])
