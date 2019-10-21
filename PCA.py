#(1030-研究方法)PCA矩陣做轉換(選完才能用sklearn)
#1
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
#from sklearn import preprocessing,tree
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

#載入數據集
avocado=pd.read_csv("avocado.csv")

x=pd.DataFrame([avocado["Total Volume"],
                avocado["Total Bags"],
                avocado["AveragePrice"],
                avocado["Small Bags"],
                avocado["Large Bags"],
                avocado["XLarge Bags"],]).T
y=avocado["type"]

#切割成75%訓練集，25%測試集
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

#描述性分析
#standardizing the data
sc=StandardScaler()
X_train_std=sc.fit_transform(X_train)
X_test_std=sc.fit_transform(X_test)

#Eigenvalue分解
cov_mat=np.cov(X_train_std.T)
eigen_vals,eigen_vecs=np.linalg.eig(cov_mat)
print('\nEigenvalues :\n%s' % eigen_vals)

#相關矩陣
x_s = scale(x, with_mean=True, with_std=True, axis=0)
x_c = np.corrcoef(x_s.T)
print('相關矩陣:\n',x_c)

#特徵向量與特徵值
eig_val, eig_vec = np.linalg.eig(cov_mat)
print("特徵向量=\n",eig_vec)
print("特徵值=\n",eig_val)

#共變異數矩陣
#Eigendecomposition of the convariance matrix(eigenvalues->lamda)
cov_mat=np.cov(X_train_std.T)
eigen_vals,eigen_vecs=np.linalg.eig(cov_mat)
print("共變異係數矩陣.shape=",cov_mat.shape)
print("共變異係數矩陣=\n",cov_mat)

#Feature transformation(將資料轉換到新座標軸,特徵做投影->投影矩陣)
#Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(reverse=True)
W = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Projection matrix W:\n', W)

#散佈圖(兩兩變數)
#visualize it using (pca_scatter)
# Z-normalize data
sc = StandardScaler()
Z = sc.fit_transform(x)
Z_pca = Z.dot(W)

colors = ['y', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y.values), colors, markers):
    plt.scatter(Z_pca[y.values==l, 0], 
                Z_pca[y.values==l, 1], 
                c=c, label=l, marker=m)
plt.title('pca')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

#找出主成分矩陣
#選出主成分比較不同數目主成分的MSE
eig_val, eig_vec = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
eig_pairs.sort(reverse=True)
feature=eig_pairs[0][1]
print('matrix=\n',eig_pairs)
print('前k個特徵向量=\n',feature)




#2
#直接用sklearn的PCA和回歸分析
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import classification_report  
#載入數據集
avocado=pd.read_csv("avocado.csv")

x=pd.DataFrame([avocado["Total Volume"],
                avocado["Total Bags"],
                avocado["AveragePrice"],
                avocado["Small Bags"],
                avocado["Large Bags"],
                avocado["XLarge Bags"],]).T
y=avocado["type"]

#用交叉檢驗建立訓練集測試集,在訓練集上用PCA
X_train,X_test,y_train,y_test=train_test_split(x,y)
pca = PCA(n_components=2) #降成二維 

#standardizing the data
sc=StandardScaler()
X_train_std=sc.fit_transform(X_train)
X_test_std=sc.fit_transform(X_test)

#把所有樣本降到二維,訓練一個邏輯回歸分類器
X_train_reduced = pca.fit_transform(X_train)  
X_test_reduced = pca.transform(X_test)  
print('訓練集原始度：{}'.format(X_train.shape))  
print('PCA降維後數據集：{}'.format(X_train_reduced.shape))  
classifier = LogisticRegression()  
accuracies = cross_val_score(classifier, X_train_reduced, y_train) 

#選出主成分,比較不同數目主成分
pca = PCA(n_components=2)  
reduced_x = pca.fit_transform(x)  
print('降維後特徵維度數目:',pca.n_components)
print('降維後矩陣:\n',reduced_x)

#說明解釋量Eigenvalue和MSE的關係
print('主成分解釋數據變異比率:',pca.explained_variance_ratio_)
#(說明:第一個主成分能解釋原數據 99.2%的變異,第二個主成分能解釋原數據 0.71%的變異)
print('主成分解釋數據變異量:',pca.explained_variance_)
#(說明:主成分解釋變異量皆大於1)


#最後用交叉驗證和測試集評估分類器性能
print('\n交叉驗證準確率：{}\n{}'.format(np.mean(accuracies), accuracies))  
classifier.fit(X_train_reduced, y_train)  
predictions = classifier.predict(X_test_reduced)  
print(classification_report(y_test, predictions))  






