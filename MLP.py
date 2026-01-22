import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import numpy as np
from matplotlib.ticker import FixedLocator
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from collections import Counter
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.preprocessing import label_binarize



data = pd.read_excel(r"11.11.xlsx")
X1 = data.drop('class', axis=1)
y1 = data['class']
print('Original dataset shape %s' % Counter(y1))  #显示最初的各类数据得个数


"""
使用SMOTE进行重采样（不用管，需要用直接把#删掉就行）(如果要用，把下边两行#删掉就行了，用ctrl+/ 两个键就去掉了）（要用的话在上边X、y后边加1）
"""
#smote = SMOTE(random_state=42)
#X, y = smote.fit_resample(X1, y1)
#print('Resampled dataset shape %s' % Counter(y))

#ada = ADASYN(random_state=42)
#X, y = ada.fit_resample(X1, y1)
#print('Resampled dataset shape %s' % Counter(y))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

'''
标准化
'''
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ADASYN重采样(如果要用把上一行的X_train，Y_train后边加个1）
# ada = ADASYN(random_state=42)
# X_train, y_train = ada.fit_resample(X_train1, y_train1)
# print('Resampled dataset shape %s' % Counter(y_train))

# 设置BP算法参数范围
param_grid = {'hidden_layer_sizes': [(10,),(50,),(100,),(50,50)],
              'activation': ['relu', 'tanh'],
              'solver': ['adam', 'sgd'],
              'alpha':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],  #这个是L2正则化
             # 'alpha': [1e-5],  # 这个是L2正则化
              #'learning_rate_init':[0.001],
              #'max_iter':[0,10,20,30,40,40,50,100],
              # 'momentum':[0.001],
              # 'n_iter_no_change':[10]
              }
# 设置参数，其中alpha为L2正则化系数，hidden_layer_sizes为隐藏层神经元数量，solver为优化算法（默认为'adam'）
#clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(10,), solver='lbfgs')

# 使用GridSearchCV进行参数调优
grid_search = GridSearchCV(MLPClassifier(max_iter=1000), param_grid, refit=True, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# 输出最优参数
print("最优参数：", grid_search.best_params_)

# 使用最优参数的BP算法模型进行分类
best_bp = grid_search.best_estimator_
best_bp.fit(X_train_scaled, y_train)


# 输出分类报告
train_predict = best_bp.predict(X_train_scaled)
test_predict = best_bp.predict(X_test_scaled)
confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)
cm_percentage = confusion_matrix_result.astype('float') / confusion_matrix_result.sum(axis=1)[:, np.newaxis]
percent = np.round(cm_percentage, decimals=2)

print('训练集正确率:',metrics.accuracy_score(y_train,train_predict))
print('测试集正确率:',metrics.accuracy_score(y_test,test_predict))
print('The confusion matrix result:\n',confusion_matrix_result)

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
with open("resultsbp.txt", "w", encoding="utf-8") as f:
    # 将训练集结果写入文件
    f.write("训练集结果：")
    f.write(train_result)
    f.write("")

    # 将测试集结果写入文件
    f.write("测试集结果：")
    f.write(test_result)


X_train.to_csv('outputbp1.txt', sep='\t', index=True )
X_test.to_csv('outputbp2.txt', sep='\t', index=True )

