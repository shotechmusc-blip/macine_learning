import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#説明変数、目的変数を含むデータフレームの作成
wine =  load_wine()
df = pd.DataFrame(wine.data,columns=wine.feature_names)
df["t"] = wine.target


#確認
#print(df.head())

#特徴量とラベルに分割
X = df.drop("t",axis=1)#ラベルだけはいらない
y = df["t"]


#学習用とテスト用に分ける
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


#説明変数の標準化をする　
scaler = StandardScaler()   #インスタンス化
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


#knnモデル作成
knn = KNeighborsClassifier(n_neighbors=5)   #インスタンス化
knn.fit(X_train_scaled,y_train)


#精度の確認
score = knn.score(X_test_scaled,y_test)
#print(f"テストデータの精度： {score:.3f}")


#新しい入力からyを予想する
new_data = df.loc[55].drop("t").values.reshape(1,-1)
#新しい説明変数を標準化
new_data_scaled = scaler.transform(new_data)
#predictでyを予想する
prediction = knn.predict(new_data_scaled)

print(f"予測されたクラス:{wine.target_names[prediction[0]]}")

