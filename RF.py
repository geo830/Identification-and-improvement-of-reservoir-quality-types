# 导入必要的库
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib.ticker import FixedLocator
from imblearn.over_sampling import SMOTE
import seaborn as sns
from imblearn.over_sampling import ADASYN
from collections import Counter
import random
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# 加载示例数据集（这里以鸢尾花数据集为例）
data = pd.read_excel(r"11.11.xlsx") #读取excel表格数据，如果表格在默认文件夹直接写表格名称就行
X1 = data.drop('class', axis=1) # 特征矩阵（drop的意思是去掉class这一列）
y1= data['class']                     # 目标变量
print('Original dataset shape %s' % Counter(y1))  #显示最初的各类数据得个数

 # 目标变量
#mapping = {1: 0, 2: 1, 3: 2, 4: 3}
#y3 = np.array([mapping[i] for i in y1])
#print('Original dataset shape %s' % Counter(y3))  #显示最初的各类数据得个数



"""
使用SMOTE进行重采样（不用管，需要用直接把#删掉就行）(如果要用，把下边两行#删掉就行了，用ctrl+/ 两个键就去掉了）（要用的话在上边X、y后边加1）
"""
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X1, y1)
print('Resampled dataset shape %s' % Counter(y))


# ADASYN重采样(如果要用把上一行的X_train，Y_train后边加个1）
#ada = ADASYN(random_state=42)
#X, y = ada.fit_resample(X1, y1)
#print('Resampled dataset shape %s' % Counter(y))

"""
新添加内容
"""

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义参数范围（以便网格搜索得到最优参数）
#param_grid = {
    #'n_estimators':range(10,101,10),
   # 'max_depth':range(1, 50, 10),
   # 'min_samples_split': [2, 5, 10],
    #'min_samples_leaf': [2,3,4,5]
#}

#param_grid = {
  # 'n_estimators':range(1, 100, 10),
  # 'max_depth': range(1, 50, 10),
  # 'min_samples_split': range(1, 5, 1),
  #  'min_samples_leaf':range(2, 10, 1)

#}

param_grid = {
    'n_estimators':[25],
    'max_depth': [15],
   'min_samples_split': [7],
   'min_samples_leaf': [2]

}

# 实例化随机森林模型
rf =RandomForestClassifier()
# 实例化网格搜索
grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=5)


# 模型训练
grid_search.fit(X_train_scaled,y_train)
# 输出最佳参数组合
print(grid_search.best_params_)
# 使用最佳参数组合构建随机森林模型
best_rf = grid_search.best_estimator_
# 模型训练
best_rf.fit(X_train_scaled,y_train)
# 模型预测
y_pred = best_rf.predict(X_test_scaled)
# 模型评估
train_predict = best_rf.predict(X_train_scaled)
test_predict = best_rf.predict(X_test_scaled)     #改这里了
# accuracy = metrics.accuracy_score(y_test,y_pred)
print('训练集正确率:',metrics.accuracy_score(y_train,train_predict))
print('测试集正确率:',metrics.accuracy_score(y_test,test_predict))


from sklearn import metrics

# print('训练集正确率:',metrics.accuracy_score(y_train,train_predict))
# print('测试集正确率:',metrics.accuracy_score(y_test,test_predict))
#
## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)
cm_percentage = confusion_matrix_result.astype('float') / confusion_matrix_result.sum(axis=1)[:, np.newaxis]
percent = np.round(cm_percentage, decimals=2)
# percent = np.char.add(np.char.mod('%.1f', percentage), '%')
print('The confusion matrix result:\n',confusion_matrix_result)
print('The confusion matrix result:\n',cm_percentage)
 # 先不用

# 利用热力图对于结果进行可视化
# plt.figure(figsize=(8, 6))
# sns.heatmap(percent, fmt='.20g',annot=True, cmap='Purples',square=True,xticklabels=['1', '2', '3','4'],yticklabels=['1', '2', '3','4'])
# # sns.heatmap(cm_percentage, fmt='g',annot=True, cmap='Purples',square=True)#出百分比的混淆矩阵
# plt.xlabel('Predicted labels')
# plt.ylabel('True labels')
# plt.show()
"""
另外一种显示
"""
# 出数字
# plt.matshow(confusion_matrix_result, cmap=plt.colormaps['Blues'])
# plt.colorbar()
# for i in range(len(confusion_matrix_result)):
#     for j in range(len(confusion_matrix_result)):
#         plt.annotate(confusion_matrix_result[i,j], xy=(i, j), horizontalalignment='center', verticalalignment='center')
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()

# 出百分比
# plt.figure(figsize=(8, 6))
plt.matshow(percent, cmap=plt.colormaps['Blues'])
plt.colorbar()
for i in range(len(percent)):
    for j in range(len(percent)):
        if i == j:
            plt.annotate(percent[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center',
                         color='white')
        else:
            plt.annotate(percent[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')


plt.ylabel('True label')
plt.xlabel('Predicted label')

ax = plt.gca()
ax.xaxis.set_major_locator(FixedLocator(range(4)))
ax.yaxis.set_major_locator(FixedLocator(range(4)))
ax.set_xticklabels(['1', '2', '3', '4'])
ax.set_yticklabels(['1', '2', '3', '4'])
plt.show()



print("\t\t\tClassification Report")
print(classification_report(y_test, test_predict))



train_result = str(train_predict)
test_result = str(test_predict)
# 打开一个名为results.txt的文件，如果不存在则创建一个新文件
with open("resultsrf.txt", "w", encoding="utf-8") as f:
    # 将训练集结果写入文件
    f.write("训练集结果：")
    f.write(train_result)
    f.write("")

    # 将测试集结果写入文件
    f.write("测试集结果：")
    f.write(test_result)


X_train.to_csv('outputrf1.txt', sep='\t', index=True )
X_test.to_csv('outputrf2.txt', sep='\t', index=True )

# 二值化标签
lb = LabelBinarizer()
y_train_bin = lb.fit_transform(y_train)
y_test_bin = lb.transform(y_test)

# 训练模型并预测概率
best_rf = RandomForestClassifier(**grid_search.best_params_).fit(X_train_scaled, y_train_bin)
y_pred_proba = best_rf.predict_proba(X_test_scaled)


