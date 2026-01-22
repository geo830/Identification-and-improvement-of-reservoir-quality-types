from matplotlib.ticker import FixedLocator
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from collections import Counter
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.preprocessing import label_binarize



# 加载示例数据集
data = pd.read_excel(r"11.11.xlsx")  # 读取excel表格数据，如果表格在默认文件夹直接写表格名称就行
X2 = data.drop('class', axis=1)  # 特征矩阵
y2 = data['class']  # 目标变量
mapping = {1: 0, 2: 1, 3: 2, 4: 3}
y1 = np.array([mapping[i] for i in y2])
print('Original dataset shape %s' % Counter(y1))  # 显示最初的各类数据得个数

# scaler = StandardScaler()
# X1 = scaler.fit_transform(X2)

# 使用SMOTE进行重采样（不用管，需要用直接把#删掉就行）
smote = SMOTE(random_state=42)
X1, y = smote.fit_resample(X2, y1)

#ada = ADASYN(random_state=42)
#X1, y = ada.fit_resample(X2, y1)

print('Resampled dataset shape %s' % Counter(y))

scaler = StandardScaler()
X = scaler.fit_transform(X1)
np.savetxt("outputscale1r.csv", X1,  delimiter=",", fmt="%.2f")
np.savetxt("outputscaler2.csv", X, delimiter=",", fmt="%.2f")


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
np.savetxt("cgoutput1.csv", X_train, delimiter=",", fmt="%.2f")
np.savetxt("cgoutput2.csv",X_test, delimiter=",", fmt="%.2f")


param_grid = {
     'n_estimators': range(1, 100, 10),
     'max_depth': range(1, 50, 10),
    'min_samples_split': range(1, 5, 1),
    'min_samples_leaf': range(3, 10, 1)

}

# 实例化随机森林模型
rf = RandomForestClassifier()
# 实例化网格搜索
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train,y_train)
# 输出最佳参数组合
print(grid_search.best_params_)
# 使用最佳参数组合构建随机森林模型
best_rf = grid_search.best_estimator_


# 定义XGBoost分类器
xgb_clf = xgb.XGBClassifier()
# 设置要调整的参数范围
param_grid = {
     'n_estimators': range(10,101,10),
   # 'max_depth': [3, 5, 7],
    'max_depth': range(5, 7, 1),
    'learning_rate': [0.01, 0.1, 0.2],
     'subsample': [0.8, 0.9, 1],
     'colsample_bytree': [0.8, 0.9, 1],
    # 'reg_lambda': [5],
    'reg_alpha': [3],
}

# 使用网格搜索自动调参数
grid_search = GridSearchCV(xgb_clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
# 输出最佳参数组合
print(grid_search.best_params_)
best_xgb = grid_search.best_estimator_



# 设置SVM参数范围（C便是L2正则化参数）
param_grid = {
              'C': range(2, 10, 1),#smote方法
              # 'C': range(1, 10, 1),ada方法
              'kernel': ['linear', 'rbf'],
               'gamma': [10,1,0.1, 0.01, 0.001],
             # 'degree':[0,1,2,3],
              }

# 使用GridSearchCV进行参数调优
grid_search = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=2,cv=5)
grid_search.fit(X_train, y_train)
# 输出最佳参数组合
print(grid_search.best_params_)
best_svm = grid_search.best_estimator_



# 设置BP算法参数范围
param_grid = {'hidden_layer_sizes': [(10,),(50,), (100,), (50, 50)],
               'activation': ['relu', 'tanh'],
               'solver': ['adam', 'sgd'],
              # 'alpha':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]ada
             'alpha':[1e-1]
              }
grid_search = GridSearchCV(MLPClassifier(max_iter=1000), param_grid, refit=True, verbose=2,cv=5)
grid_search.fit(X_train, y_train)

# 输出最佳参数组合
print(grid_search.best_params_)
best_bp = grid_search.best_estimator_


for clf in (best_rf, best_xgb, best_svm, best_bp):
    clf.fit(X_train, y_train)
    y_pred1 = clf.predict(X_train)
    y_pred = clf.predict(X_test)
    print("训练集",clf.__class__.__name__, '=', accuracy_score(y_train, y_pred1))
    print("测试集" ,clf.__class__.__name__, '=', accuracy_score(y_test, y_pred))
for clf in (best_rf, best_xgb, best_svm, best_bp):
    clf.fit(X_train, y_train)
    y_pred1 = clf.predict(X_train)
    y_pred = clf.predict(X_test)
    print("训练集",clf.__class__.__name__, '=', accuracy_score(y_train, y_pred1))
    print("测试集" ,clf.__class__.__name__, '=', accuracy_score(y_test, y_pred))
    confusion_matrix_result = metrics.confusion_matrix(y_pred, y_test)
    cm_percentage = confusion_matrix_result.astype('float') / confusion_matrix_result.sum(axis=1)[:, np.newaxis]
    percent = np.round(cm_percentage, decimals=2)
    print('The confusion matrix result:\n', confusion_matrix_result)
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
    train_result = str(y_pred1)
    test_result = str(y_pred)



# 软投票
voting_soft = VotingClassifier(estimators=[('rf', best_rf), ('xgb', best_xgb), ('svm', best_svm), ('bp', best_bp)], voting='soft')
'''
# 定义10倍交叉验证
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
# 存储每次迭代的性能指标
scores1 = []
scores2 = []
# 创建一个空的DataFrame对象，用于存储训练集和测试集的数据
data = pd.DataFrame(columns=['train_index', 'test_index', 'X_train', 'X_test', 'y_train', 'y_test'])
# 进行10次迭代
for i,(train_index, test_index) in enumerate(kfold.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 训练投票算法
    voting_soft.fit(X_train, y_train)
    # 将每次迭代的训练集和测试集数据添加到DataFrame对象中
    data.loc[i] = [train_index, test_index, X_train, X_test, y_train, y_test]


    # 评估性能指标
    score1 = voting_soft.score(X_train, y_train)
    score2 = voting_soft.score(X_test, y_test)
    scores1.append(score1)
    scores2.append(score2)
    train_predict = best_rf.predict(X_train)
    test_predict = best_rf.predict(X_test)
    train_result = str(train_predict)
    test_result = str(test_predict)
    # 打开一个名为results.txt的文件，如果不存在则创建一个新文件
    with open("softresults.txt", "a", encoding="utf-8") as f:
        # 将训练集结果写入文件
        f.write("训练集结果：")
        f.write(train_result)
        f.write("")

        # 将测试集结果写入文件
        f.write("测试集结果：")
        f.write(test_result)

# 将DataFrame对象输出到Excel文件
data.to_excel('soft_kfold_data.xlsx', index=True)

# 计算平均性能指标
mean_score1 = sum(scores1) / len(scores1)
mean_score2 = sum(scores2) / len(scores2)
print("软投票训练集平均性能指标：", mean_score1)
print("软投票测试集平均性能指标：", mean_score2)
voting_soft.fit(X_train, y_train)
'''

voting_soft.fit(X_train, y_train)
score1 = voting_soft.score(X_train, y_train)
score2 = voting_soft.score(X_test, y_test)
# voting_soft.fit(X_train, y_train)
y_pred = voting_soft.predict(X_train)
y_pred1 = voting_soft.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred1)
print("软投票训练集：", score1)
print("软投票测试集：", score2)
print("软投票Accuracy1: ", accuracy1)
print('软投票混淆矩阵')




confusion_matrix_result = metrics.confusion_matrix(y_pred1, y_test)
cm_percentage = confusion_matrix_result.astype('float') / confusion_matrix_result.sum(axis=1)[:, np.newaxis]
percent = np.round(cm_percentage, decimals=2)
print('The confusion matrix result:\n', confusion_matrix_result)
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

print("\t\t\tClassification Report")
print(classification_report(y_test, y_pred1))

plt.show()
train_result = str(y_pred)
test_result = str(y_pred1)
with open("softresults.txt", "w", encoding="utf-8") as f:
    # 将训练集结果写入文件
    f.write("训练集结果：")
    f.write(train_result)
    f.write("")

    # 将测试集结果写入文件
    f.write("测试集结果：")
    f.write(test_result)

# 硬投票
voting_hard = VotingClassifier(estimators=[('rf', best_rf), ('xgb', best_xgb), ('svm', best_svm), ('bp', best_bp)], voting='hard')
'''
# 定义10倍交叉验证
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
# 存储每次迭代的性能指标
scores1 = []
scores2 = []
# 创建一个空的DataFrame对象，用于存储训练集和测试集的数据
data = pd.DataFrame(columns=['train_index', 'test_index', 'X_train', 'X_test', 'y_train', 'y_test'])
# 进行10次迭代
for i, (train_index, test_index) in enumerate(kfold.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 训练投票算法
    voting_soft.fit(X_train, y_train)
    # 将每次迭代的训练集和测试集数据添加到DataFrame对象中
    data.loc[i] = [train_index, test_index, X_train, X_test, y_train, y_test]

    # 评估性能指标
    score1 = voting_soft.score(X_train, y_train)
    score2 = voting_soft.score(X_test, y_test)
    scores1.append(score1)
    scores2.append(score2)
    train_predict = best_rf.predict(X_train)
    test_predict = best_rf.predict(X_test)
    train_result = str(train_predict)
    test_result = str(test_predict)
    # 打开一个名为results.txt的文件，如果不存在则创建一个新文件
    with open("hardresults.txt", "a", encoding="utf-8") as f:
        # 将训练集结果写入文件
        f.write("训练集结果：")
        f.write(train_result)
        f.write("")

        # 将测试集结果写入文件
        f.write("测试集结果：")
        f.write(test_result)

# 将DataFrame对象输出到Excel文件
data.to_excel('hard_kfold_data.xlsx', index=True)

# 计算平均性能指标
mean_score3 = sum(scores1) / len(scores1)
mean_score4 = sum(scores2) / len(scores2)
print("硬投票训练集平均性能指标：", mean_score3)
print("硬投票测试集平均性能指标：", mean_score4)
# voting_soft.fit(X_train, y_train)
# voting_hard.fit(X_train, y_train)
'''
'''
voting_hard.fit(X_train, y_train)
y_pred = voting_hard.predict(X_train)
y_pred1 = voting_hard.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred1)
print("Accuracy2: ", accuracy2)
print('硬投票混淆矩阵')
'''
voting_hard.fit(X_train, y_train)
score3 = voting_hard.score(X_train, y_train)
score4 = voting_hard.score(X_test, y_test)
# voting_hard.fit(X_train, y_train)
y_pred = voting_hard.predict(X_train)
y_pred1 = voting_hard.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred1)
print("硬投票训练集：", score3)
print("硬投票测试集：", score4)
print("Accuracy2: ", accuracy2)
print('硬投票混淆矩阵')





confusion_matrix_result = metrics.confusion_matrix(y_pred1, y_test)
cm_percentage = confusion_matrix_result.astype('float') / confusion_matrix_result.sum(axis=1)[:, np.newaxis]
percent = np.round(cm_percentage, decimals=2)
print('The confusion matrix result:\n', confusion_matrix_result)
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

print("\t\t\tClassification Report")
print(classification_report(y_test, y_pred1))

plt.show()
train_result = str(y_pred)
test_result = str(y_pred1)
with open("hardresults.txt", "w", encoding="utf-8") as f:
    # 将训练集结果写入文件
    f.write("训练集结果：")
    f.write(train_result)
    f.write("")

    # 将测试集结果写入文件
    f.write("测试集结果：")
    f.write(test_result)


for clf in (best_rf, best_xgb, best_svm, best_bp):
    clf.fit(X_train, y_train)
    y_pred1 = clf.predict(X_train)
    y_pred = clf.predict(X_test)
    print("训练集",clf.__class__.__name__, '=', accuracy_score(y_train, y_pred1))
    print("测试集" ,clf.__class__.__name__, '=', accuracy_score(y_test, y_pred))
    confusion_matrix_result = metrics.confusion_matrix(y_pred, y_test)
    cm_percentage = confusion_matrix_result.astype('float') / confusion_matrix_result.sum(axis=1)[:, np.newaxis]
    percent = np.round(cm_percentage, decimals=2)
    print('The confusion matrix result:\n', confusion_matrix_result)
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

    print("\t\t\tClassification Report")
    print(classification_report(y_test, y_pred1))



    plt.show()
    train_result = str(y_pred1)
    test_result = str(y_pred)
    # 打开一个名为results.txt的文件，如果不存在则创建一个新文件
    with open("cgresults.txt", "a", encoding="utf-8") as f:
        # 将训练集结果写入文件
        f.write("训练集结果：")
        f.write(train_result)
        f.write("")

        # 将测试集结果写入文件
        f.write("测试集结果：")
        f.write(test_result)



# 加权值的软投票

voting_soft2 = VotingClassifier(estimators=[('rf', best_rf), ('xgb', best_xgb), ('svm', best_svm), ('bp', best_bp)], voting='hard',
                                weights=[0.83870,0.825806,0.896774, 0.877419])

# 计算平均性能指标
voting_soft2.fit(X_train, y_train)
# mean_score5 = sum(scores1) / len(scores1)
# mean_score6 = sum(scores2) / len(scores2)
score5 = voting_soft2.score(X_train, y_train)
score6 = voting_soft2.score(X_test, y_test)
y_pred = voting_hard.predict(X_train)
y_pred1 = voting_hard.predict(X_test)
print("加权软投票训练集平均性能指标：", score5)
print("加权软投票测试集平均性能指标：", score6)
print('加权值软投票混淆矩阵')

confusion_matrix_result = metrics.confusion_matrix(y_pred1, y_test)
cm_percentage = confusion_matrix_result.astype('float') / confusion_matrix_result.sum(axis=1)[:, np.newaxis]
percent = np.round(cm_percentage, decimals=2)
print('The confusion matrix result:\n', confusion_matrix_result)
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
train_result = str(y_pred)
test_result = str(y_pred1)
with open("soft2results.txt", "w", encoding="utf-8") as f:
    # 将训练集结果写入文件
    f.write("训练集结果：")
    f.write(train_result)
    f.write("")

    # 将测试集结果写入文件
    f.write("测试集结果：")
    f.write(test_result)


for clf in (best_rf, best_xgb, best_svm, best_bp):
    clf.fit(X_train, y_train)
    y_pred1 = clf.predict(X_train)
    y_pred = clf.predict(X_test)
    print("训练集",clf.__class__.__name__, '=', accuracy_score(y_train, y_pred1))
    print("测试集" ,clf.__class__.__name__, '=', accuracy_score(y_test, y_pred))

