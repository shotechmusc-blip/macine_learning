from sklearn.datasets import load_iris

#データセットの作成
X,y = load_iris(return_X_y=True)

#学習用とテスト用に分ける
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

#KNNクラスの呼び出し
from sklearn.neighbors import KNeighborsClassifier as KN

#インスタンスの作成
clf = KN(n_neighbors=3)

#インスタンスに引数を渡す
clf.fit(X_train,y_train)

#テストデータを引数に結果を表示
accuracy = clf.score(X_test,y_test)
print("結果 : {:.2f}".format(accuracy))

#学習終了　モデルはclfで呼び出そう

#ユーザー入力からyを予想する
input_data = [[5.1,3.5,1.4,0.2]]
prediction = clf.predict(input_data)
print("予測結果",prediction[0])
