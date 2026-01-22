import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FixedLocator
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from collections import Counter
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.preprocessing import label_binarize


# 读取Excel文件
data = pd.read_excel(r"11.11.xlsx")
X1 = np.array(data.drop('class', axis=1))
y1 = np.array(data['class'])
#mapping = {1: 0, 2: 1, 3: 2, 4: 3,5:4,6:5}
# 6类
mapping = {1: 0, 2: 1, 3: 2, 4: 3}
y2 = np.array([mapping[i] for i in y1])
print('Original dataset shape %s' % Counter(y1))  #显示最初的各类数据得个数
print('Original dataset shape %s' % Counter(y2))  #显示最初的各类数据得个数

#smote = SMOTE(random_state=42)
#X, y = smote.fit_resample(X1, y2)

ada = ADASYN(random_state=42)
X, y = ada.fit_resample(X1, y2)
print('Resampled dataset shape %s' % Counter(y))


# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义XGBoost分类器
xgb_clf = xgb.XGBClassifier()

# 设置要调整的参数范围
param_grid = {
    'n_estimators': range(10,101,10),
    'max_depth': range(5,7,1),
     # 'max_depth': range(1,10,1),
    'learning_rate': [0.01, 0.1, 0.2],
   'subsample': [0.8, 0.9, 1],
   'colsample_bytree': [0.8, 0.9, 1],
   # 'reg_alpha':[3],
   'reg_alpha':range(3, 5, 1)
}
#param_grid = {
  #  'n_estimators': range(10,101,10),
   # 'max_depth': [3,5,7],
   # 'learning_rate': [0.01, 0.1, 0.2],
   # 'subsample': [0.8, 0.9, 1],
  #  'colsample_bytree': [0.8, 0.9, 1],
  #  'reg_alpha':range(0, 5, 1)
#}

#param_grid = {
  #  'n_estimators': [60],
   # 'max_depth': [3],
   # 'learning_rate': [0.1],
   # 'subsample': [0.8],
   # 'colsample_bytree': [0.8, 0.9, 1],
  #  'reg_alpha':[3]
#}

# 使用网格搜索自动调参数
grid_search = GridSearchCV(xgb_clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# 输出最佳参数组合
print("Best parameters found: ", grid_search.best_params_)

# 使用最佳参数组合的模型进行预测
best_clf = grid_search.best_estimator_
best_clf.fit(X_train_scaled, y_train)

# 模型评估
train_predict = best_clf.predict(X_train_scaled)
test_predict = best_clf.predict(X_test_scaled)
confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)
cm_percentage = confusion_matrix_result.astype('float') / confusion_matrix_result.sum(axis=1)[:, np.newaxis]
percent = np.round(cm_percentage, decimals=2)
# accuracy = metrics.accuracy_score(y_test,y_pred)
print('训练集正确率:',metrics.accuracy_score(y_train,train_predict))
print('测试集正确率:',metrics.accuracy_score(y_test,test_predict))


print('The confusion matrix result:\n',confusion_matrix_result)
 # 利用热力图对于结果进行可视化
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix_result, fmt='.20g',annot=True, cmap='Blues',square=True)
# plt.xlabel('Predicted labels')
# plt.ylabel('True labels')
# plt.show()


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
#ax.xaxis.set_major_locator(FixedLocator(range(6)))
#ax.yaxis.set_major_locator(FixedLocator(range(6)))
#ax.set_xticklabels(['1', '2', '3', '4','5','6'])
#ax.set_yticklabels(['1', '2', '3', '4','5','6'])
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
with open("resultsxg.txt", "w", encoding="utf-8") as f:
    # 将训练集结果写入文件
    f.write("训练集结果：")
    f.write(train_result)
    f.write("")

    # 将测试集结果写入文件
    f.write("测试集结果：")
    f.write(test_result)


#X_train.to_csv('outputxg1.txt', sep='\t', index=True )
#X_test.to_csv('outputxg1.txt', sep='\t', index=True )


np.savetxt("outputxgb3.csv", X_train, delimiter=",", fmt="%.2f")
np.savetxt("outputxgb4.csv",X_test, delimiter=",", fmt="%.2f")
# pd_train = pd.DataFrame(X_train)
# pd_test = pd.DataFrame(X_test)
#
# X_train.to_csv('outputxgb1.txt', sep='\t', index=True )
# X_test.to_csv('outputxgb2.txt', sep='\t', index=True )
