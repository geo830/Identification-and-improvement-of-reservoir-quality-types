import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
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
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.preprocessing import label_binarize



data = pd.read_excel(r"11.11.xlsx")
X1 = data.drop('class', axis=1)
y1 = data['class']
print('Original dataset shape %s' % Counter(y1))  #显示最初的各类数据得个数

 # 目标变量
#mapping = {1: 0, 2: 1, 3: 2, 4: 3}
#y3 = np.array([mapping[i] for i in y1])
#print('Original dataset shape %s' % Counter(y1))  #显示最初的各类数据得个数

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X1, y1)
print('Resampled dataset shape %s' % Counter(y))


#ada = ADASYN(random_state=42)
#X, y = ada.fit_resample(X1, y1)
#print('Resampled dataset shape %s' % Counter(y))


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 设置SVM参数范围（C便是L2正则化参数）
param_grid = { 'C':range(3,10,1),
             #'C':range(1,10,1),
             'kernel': ['linear', 'rbf'],
              'gamma': [10,1,0.1, 0.01, 0.001],
             'degree':[0,1,2,3],
            #'coef0':[0,1,2,3,4],
            #  'tol':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
             # 'cache_size':[200],
             # 'verbose':[0,1,2],
            #  'max_iter':[-1]

             }


# 设置SVM参数范围（C便是L2正则化参数）
#param_grid = {'C':[1],
          #    'kernel': ['linear', 'rbf'],
           #   'gamma': [10,1,0.1, 0.01, 0.001],
           #  'degree':[0,1,2,3],
             # 'coef0':[0,1,2,3,4],
             # 'tol':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
             # 'cache_size':[200],
             # 'verbose':[0,1,2],
             # 'max_iter':[-1]

          #    }



# 使用GridSearchCV进行参数调优
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# 输出最优参数
print("最优参数：", grid_search.best_params_)

# 使用最优参数的SVM模型进行分类
best_svm = grid_search.best_estimator_
best_svm.fit(X_train_scaled, y_train)


# 输出分类报告
train_predict = best_svm.predict(X_train_scaled)
test_predict = best_svm.predict(X_test_scaled)
confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)
cm_percentage = confusion_matrix_result.astype('float') / confusion_matrix_result.sum(axis=1)[:, np.newaxis]
percent = np.round(cm_percentage, decimals=2)

print("Accuracy score: ", best_svm.score(X_test_scaled, y_test))
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
with open("resultssvm.txt", "w", encoding="utf-8") as f:
    # 将训练集结果写入文件
    f.write("训练集结果：")
    f.write(train_result)
    f.write("")

    # 将测试集结果写入文件
    f.write("测试集结果：")
    f.write(test_result)


X_train.to_csv('outputsvm1.txt', sep='\t', index=True )
X_test.to_csv('outputsvm2.txt', sep='\t', index=True )


# 预测概率
#y_pred_proba = best_svm.predict_proba(X_test_scaled)[:, 1]


