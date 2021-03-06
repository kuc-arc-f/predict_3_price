# -*- coding: utf-8 -*-
# テスト4, 体重、身長の重回帰分析。
#


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
import time 

#
# 学習データ
global_start_time = time.time()
wdata = pd.read_csv("data.csv" )
#wdata.columns =["no","addr","price","siki_price", "rei_price" ,"menseki" ,"nensu" ,"toho" ,"madori" ,"houi" ,"kouzou" ]
wdata.columns =["no", "price","siki_price", "rei_price" ,"menseki" ,"nensu" ,"toho" ,"madori" ,"houi" ,"kouzou" ]
print(wdata.head() )
#print(wdata["NO"][: 10 ] )

# conv=> num
sub_data = wdata[[ "no","price","siki_price", "rei_price" ,"menseki" ,"nensu" ,"toho" ] ]
sub_data = sub_data.assign(price=pd.to_numeric( sub_data.price))

print(sub_data["price"][: 10])
##quit()

#
# データの分割（学習データとテストデータに分ける）
# sklearnのバージョンによっては train_test_splitはsklearn.cross_validationにしか入ってない場合があります
from sklearn.model_selection import train_test_split

# モデル
from sklearn import linear_model

# モデルのインスタンス
l_model = linear_model.LinearRegression()
 
# 説明変数に "price" 以外を利用
X = sub_data.drop("price", axis=1)

print(X.shape )
#print( type( X) )
#print(X[: 10 ] )
# 目的変数
Y = sub_data["price"]

# 学習データとテストデータに分ける
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5,random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25 ,random_state=0)
print(X_train.shape , y_train.shape  )
print(X_test.shape , y_test.shape  )
#print( type( X_test ) )
#quit()

# fit
clf = l_model.fit(X_train,y_train)
print("train:",clf.__class__.__name__ ,clf.score(X_train,y_train))
print("test:",clf.__class__.__name__ , clf.score(X_test,y_test))
 
# 偏回帰係数
print(pd.DataFrame({"Name":X.columns,
                    "Coefficients":clf.coef_}).sort_values(by='Coefficients') )
 
# 切片 
print(clf.intercept_)

# モデルを保存する
filename = 'model.pkl'
pickle.dump( l_model , open(filename, 'wb'))
print("model save, complete !!")
print ('time : ', time.time() - global_start_time)
