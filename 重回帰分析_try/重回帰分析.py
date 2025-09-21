import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#データセットの準備
from sklearn.datasets import fetch_california_housing
dataset = fetch_california_housing()

#データフレームに変換
columns = dataset.feature_names
df = pd.DataFrame(dataset.data,columns=dataset.feature_names)
df["t"] = dataset.target

#説明変数と目的変数に分割
X = df.drop("t",axis=1)
y = df["t"]

#学習用とテスト用に分ける
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


#特徴量の標準化
"""
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.fit_transform(X_test)
"""

#モデルの作成と学習
MRA = LinearRegression() #インスタンス化
MRA.fit(X_train,y_train)

#重み表示
plt.figure(figsize=(10,7))
plt.bar(x=columns,height=MRA.coef_)
#plt.show()

#推論
s = MRA.predict(X_test)
y_test_np = y_test.to_numpy() 
print(y_test_np[:5])
print(s[:5])








