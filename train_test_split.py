
# coding: utf-8

# # train_test_split概要 

# train_test_splitは、テストデータと学習データに分けることができる。<br>
# 引数として次を与えることができる<br>
# <ul>
#     <li>random_state:乱数のシード値（この値を記憶することでどの環境でも同じように分けれる）</li>
#     <li>shuffle:データをランダムに分ける（デフォルト:True）</li>
#     <li>stratify:目的変数を指定すると、学習データとテストデータの目的変数の出現頻度が元のデータの出現人と同じになる</li>
# </ul>

# ## stratifyを指定したときの出現頻度の確認 

# irisのデータで学習データ(元のデータの80%)、テストデータ(元のデータの30%)、元のデータの出現頻度を比較する

# ### stratifyを利用しない場合

# In[33]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()
#3列目と4列目をデータとして選択する
X = iris.data[:,[2,3]]
#クラスラベルの取得
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,shuffle=True)


# In[34]:


#元データのラベルを確認
print('origin data')
for label in set(y):
    print('label' + str(label) + ':' + str(len(y[y == label])))


# In[35]:


#学習データのラベルを確認
print('train data')
for label in set(y_train):
    print('label' + str(label) + ':' + str(len(y_train[y_train == label])))


# In[36]:


#テストのラベルを確認
print('test data')
for label in set(y_test):
    print('label' + str(label) + ':' + str(len(y_test[y_test == label])))


# ### stratifyを利用した場合 

# In[23]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()
#3列目と4列目をデータとして選択する
X = iris.data[:,[2,3]]
#クラスラベルの取得
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,shuffle=True,stratify = y)


# In[30]:


#元データのラベルを確認
print('origin data')
for label in set(y):
    print('label' + str(label) + ':' + str(len(y[y == label])))


# In[31]:


#学習データのラベルを確認
print('train data')
for label in set(y_train):
    print('label' + str(label) + ':' + str(len(y_train[y_train == label])))


# In[32]:


#テストのラベルを確認
print('test data')
for label in set(y_test):
    print('label' + str(label) + ':' + str(len(y_test[y_test == label])))

