import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os
import pickle as pkl
from copy import deepcopy
# 倾向评分预计算_整体可解释 保存后可加速后续操作
features = ['可用额度','年龄','逾期30-59天笔数','负债率','月收入','开放式信贷数量','逾期90天笔数','固定贷款量','逾期60-89天笔数','家属数量']
paths = {}
for f in features:
    paths[f]='App/static/pps_models/{}.pkl'.format(f)

def entire_propensity_score(train_feature,feature_input,train_feature_value_minmaxScaler,seed):
    propensitys = pd.DataFrame(columns=feature_input)
    for feature in feature_input:
        input_x = train_feature.columns.get_loc(feature)
        # 抽取当前feature作为label
        train = np.delete(train_feature_value_minmaxScaler, input_x, axis=1)
        label = train_feature_value_minmaxScaler[:, input_x]
        # MLP计算法
        if not os.path.exists(paths[feature]):
            clf = MLPRegressor(hidden_layer_sizes=10, verbose=0, random_state=seed)
            parameter_grid = {'alpha': [0.1, 0.05, 0.01], }
            gridsearch = GridSearchCV(clf, param_grid=parameter_grid)
            gridsearch.fit(train, label)
            pkl.dump(gridsearch,open(paths[feature],'wb'))
        else:
            gridsearch = pkl.load(open(paths[feature],'rb'))
        # 总体倾向分
        propensity = gridsearch.predict(train)
        propensitys[feature] = propensity

    propensitys.to_csv('App/static/propensitys/entire_propensitys.csv',index=False)

    return propensitys

# 同一模型中不同属性的解释性分析
def entire_interpretability_1(x,feature_input,model_name, train_features,propensitys, predictions):
    train_feature = train_features.copy(deep=True)
    yy_predict_alls = []
    xx = np.linspace(0, 1, 100)
    for feature in feature_input:
        # 等距分组cut、等频分组qcut
        cut_num = 5
        # cut1 = pd.qcut(propensitys[feature], cut_num)  不加values的这个不对，有空值
        cut1 = pd.qcut(propensitys[feature].values, cut_num)
        d1 = train_feature
        d1["Bucket"] = cut1
        d1['propensity'] = propensitys[feature]
        # 对多个模型遍历，一个图中每个模型一条线
        #d1['model{}'.format(x)] = predictions.iloc[:, x].values
        d1[model_name] = predictions[model_name].values
        d2 = d1.groupby('Bucket', as_index=True)
        # print(d1.describe())
        coef = np.zeros(shape=(1, 3))
        intercept = 0
        for name, group in d2:
            #label = group['model{}'.format(x)].values
            label = group[model_name].values
            propensity_treatment = group[[feature]]  # 'propensity',
            propensity_treatment_minmax = preprocessing.MinMaxScaler().fit_transform(propensity_treatment.values)
            # 多项式建模 degree=幂次
            quadratic_featurizer = PolynomialFeatures(degree=2)
            X_train_quadratic = quadratic_featurizer.fit_transform(propensity_treatment_minmax)
            regression = LinearRegression()
            regression.fit(X_train_quadratic, label)
            coef += regression.coef_
            intercept += regression.intercept_
            print('二次回归     r-squared', regression.score(X_train_quadratic, label))

        # 五分组总体 画图
        coef /= cut_num
        intercept /= cut_num
        regression_new = LinearRegression()
        regression_new.coef_ = coef
        regression_new.intercept_ = intercept

        quadratic_featurizer = PolynomialFeatures(degree=2)
        xx_quadratic = quadratic_featurizer.fit_transform(xx.reshape(xx.shape[0], 1))
        yy_predict_all = regression_new.predict(xx_quadratic) - intercept
        yy_predict_alls.append(yy_predict_all[:,0].tolist())
    return list(xx), yy_predict_alls

# 同一属性在不同模型中的解释性分析
def entire_interpretability_2(x,feature_input,model_name, train_features,propensitys,predictions):
    train_feature = train_features.copy(deep=True)
    # 等距分组cut、等频分组qcut
    cut_num = 5
    # cut1 = pd.qcut(propensitys[feature], cut_num)  不加values的这个不对，有空值
    cut1 = pd.qcut(propensitys[feature_input[x]].values, cut_num)
    d1 = train_feature
    d1["Bucket"] = cut1
    d1['propensity'] = propensitys[feature_input[x]]
    xx = np.linspace(0, 1, 100)
    yy_predict_alls = []
    for c_name in predictions:
        d1[c_name] = predictions[c_name]
        d2 = d1.groupby('Bucket', as_index=True)
        coef = np.zeros(shape=(1, 3))
        intercept = 0
        for name, group in d2:
            #print('group {}: '.format(num), name)
            label = group[c_name]
            propensity_treatment = group[[feature_input[x]]]  # 'propensity',
            propensity_treatment_minmax = preprocessing.MinMaxScaler().fit_transform(propensity_treatment.values)
            # 多项式建模 degree=幂次
            quadratic_featurizer = PolynomialFeatures(degree=2)
            X_train_quadratic = quadratic_featurizer.fit_transform(propensity_treatment_minmax)
            regression = LinearRegression()
            regression.fit(X_train_quadratic, label)
            coef += regression.coef_
            intercept += regression.intercept_
            print('二次回归     r-squared', regression.score(X_train_quadratic, label))
        # 五分组总体 画图
        coef /= cut_num
        intercept /= cut_num
        regression_new = LinearRegression()
        regression_new.coef_ = coef
        regression_new.intercept_ = intercept

        quadratic_featurizer = PolynomialFeatures(degree=2)
        xx_quadratic = quadratic_featurizer.fit_transform(xx.reshape(xx.shape[0], 1))
        yy_predict_all = regression_new.predict(xx_quadratic) - intercept
        yy_predict_alls.append(yy_predict_all[:,0].tolist())
    return list(xx),yy_predict_alls

# 倾向评分预计算_个体可解释
def individual_propensity_score(X,train_feature, feature_input, train_feature_value_minmaxScaler, seed,is_entire=False):
    print(feature_input)
    propensitys, propensitys_x = pd.DataFrame(columns=feature_input), pd.DataFrame(columns=feature_input)
    for feature in feature_input:
        input_x = train_feature.columns.get_loc(feature)

        # 总体 训练
        train = np.delete(train_feature_value_minmaxScaler, input_x, axis=1)
        label = train_feature_value_minmaxScaler[:, input_x]
        # 个体 拿来预测
        x_train = np.delete(X, input_x, axis=1)
        # MLP计算法
        if not os.path.exists(paths[feature]):
            clf = MLPRegressor(hidden_layer_sizes=10, verbose=0, random_state=seed)
            parameter_grid = {'alpha': [0.1, 0.05, 0.01], }
            gridsearch = GridSearchCV(clf, param_grid=parameter_grid)
            gridsearch.fit(train, label)
            pkl.dump(gridsearch,open(paths[feature],'wb'))
        else:
            print(feature)
            gridsearch = pkl.load(open(paths[feature],'rb'))
            print(gridsearch)
        # 总体倾向分
        if not is_entire:
            propensity = gridsearch.predict(train)
            propensitys[feature] = propensity
            propensitys.to_csv('App/static/propensitys/entire_propensitys.csv', index=False)
        else:
            propensitys = pd.read_csv('App/static/propensitys/entire_propensitys.csv')

        propensity_x = gridsearch.predict(x_train)
        propensitys_x[feature] = propensity_x
    propensitys_x.to_csv('App/static/propensitys/single_propensitys.csv', index=False)

    return propensitys,propensitys_x

# 输入：X只有1个，feature_input只有1个，model_name有多个
def individual_interpretability_1(x,y,feature_input,model_name,train_features,propensitys_read, propensitys_x_read,predictions):

    train_feature = train_features.copy(deep=True)
    propensitys, propensitys_x = propensitys_read, propensitys_x_read
    # 找最近的1000个样本group d1
    # 1.排序
    # 2.查找分一样的点
    # 3.上下窗口

    # 选择第几个样本

    input_y = propensitys.columns.get_loc(feature_input[y])

    p = propensitys_x.iat[x, input_y]
    print(p)
    train_feature['propensity'] = propensitys[feature_input[y]].values
    # for i in range(predictions.shape[1]):
    #     train_feature['model{}'.format(i)] = predictions.iloc[:, i].values
    #
    for c_name in predictions:
        train_feature[c_name] = predictions[c_name]
    # 排序
    train_feature_sort = train_feature.sort_values(by=['propensity'], ascending=False)
    # 查找
    dis = 999
    pos = 0
    for d in range(0, len(train_feature_sort)):
        if abs(p - train_feature_sort.iloc[d]['propensity']) < dis:
            dis = abs(p - train_feature_sort.iloc[d]['propensity'])
            pos = d
    # 上下窗口选取
    total = len(train_feature) / 20
    if pos - total / 2 < 0:
        front = 0
        after = front + total
    elif pos + total / 2 > len(train_feature) - 1:
        after = len(train_feature) - 1
        front = after - total
    else:
        front = pos - total / 2
        after = pos + total / 2
    front = int(front)
    after = int(after)
    print(pos, front, after)
    train_x = train_feature_sort[front:after]
    train_x = shuffle(train_x)

    #for i in range(predictions.shape[1]):
    xx = np.linspace(0, 1, 100)
    yy_predict_alls = []
    for c_name in predictions:
        coef = np.zeros(shape=(1, 3))
        intercept = 0
        #label = train_x['model{}'.format(i)].values
        label = train_x[c_name].values
        propensity_treatment = train_x[[feature_input[y]]]  # 'propensity',
        propensity_treatment_minmax = preprocessing.MinMaxScaler().fit_transform(propensity_treatment.values)
        quadratic_featurizer = PolynomialFeatures(degree=2)
        X_train_quadratic = quadratic_featurizer.fit_transform(propensity_treatment_minmax)
        regression = LinearRegression()
        regression.fit(X_train_quadratic, label)
        coef = regression.coef_
        intercept = regression.intercept_
        print('二次回归_1     r-squared', regression.score(X_train_quadratic, label))

        # 五分组总体 画图
        regression_new = LinearRegression()
        regression_new.coef_ = coef
        regression_new.intercept_ = intercept

        quadratic_featurizer = PolynomialFeatures(degree=2)
        xx_quadratic = quadratic_featurizer.fit_transform(xx.reshape(xx.shape[0], 1))
        yy_predict_all = regression_new.predict(xx_quadratic) - intercept
        yy_predict_alls.append(yy_predict_all.tolist())
    return list(xx), yy_predict_alls

# 单一个体-单一模型-不同属性的分析
# 输入：X只有1个，model_name只有1个，feature_input有多个
def individual_interpretability_2(x,y,feature_input,model_name,train_features,propensitys_read, propensitys_x_read,predictions):
    train_feature = train_features.copy(deep=True)
    propensitys, propensitys_x = propensitys_read, propensitys_x_read
    xx = np.linspace(0, 1, 100)
    yy_predict_alls = []
    for feature in feature_input:

        # 选择第几个样本
        input_x = propensitys_x.columns.get_loc(feature)
        p = propensitys_x.iat[x, input_x]
        train_feature['propensity'] = propensitys[feature].values
        #for i in range(predictions.shape[1]):
        #['model{}'.format(y)] = predictions.iloc[:, y].values
        train_feature[model_name] = predictions[model_name].values
        # 排序
        train_feature_sort = train_feature.sort_values(by=['propensity'], ascending=False)
        # 查找
        dis = 999
        pos = 0
        for d in range(0, len(train_feature_sort)):
            if abs(p - train_feature_sort.iloc[d]['propensity']) < dis:
                dis = abs(p - train_feature_sort.iloc[d]['propensity'])
                pos = d
        # 上下窗口选取  选取总体数据的 1/50
        total = len(train_feature) / 50
        if pos - total / 2 < 0:
            front = 0
            after = front + total
        elif pos + total / 2 > len(train_feature) - 1:
            after = len(train_feature) - 1
            front = after - total
        else:
            front = pos - total / 2
            after = pos + total / 2
        front = int(front)
        after = int(after)
        print(pos, front, after)
        train_x = train_feature_sort[front:after]
        train_x = shuffle(train_x)


        coef = np.zeros(shape=(1, 3))
        intercept = 0
        #label = train_x['model{}'.format(y)].values
        label = train_x[model_name].values
        propensity_treatment = train_x[[feature]]  # 'propensity',
        propensity_treatment_minmax = preprocessing.MinMaxScaler().fit_transform(propensity_treatment.values)
        quadratic_featurizer = PolynomialFeatures(degree=2)
        X_train_quadratic = quadratic_featurizer.fit_transform(propensity_treatment_minmax)
        regression = LinearRegression()
        regression.fit(X_train_quadratic, label)
        coef = regression.coef_
        intercept = regression.intercept_
        print('二次回归_2     r-squared', regression.score(X_train_quadratic, label))

        # 五分组总体 画图
        regression_new = LinearRegression()
        regression_new.coef_ = coef
        regression_new.intercept_ = intercept

        quadratic_featurizer = PolynomialFeatures(degree=2)
        xx_quadratic = quadratic_featurizer.fit_transform(xx.reshape(xx.shape[0], 1))
        yy_predict_all = regression_new.predict(xx_quadratic) - intercept
        yy_predict_alls.append(yy_predict_all.tolist())
    return list(xx), yy_predict_alls

# 不同个体的对比解释性分析
# 输入:多个个体，固定模型，固定属性

def individual_interpretability_3(x,y,feature_input,model_name,train_features,propensitys_read, propensitys_x_read,predictions):
    train_feature = train_features.copy(deep=True)
    propensitys, propensitys_x = propensitys_read, propensitys_x_read

    # 选择第几个样本
    xx = np.linspace(0, 1, 100)
    yy_predict_alls = []
    for i in range(propensitys_x_read.shape[0]):

        p = propensitys_x.iat[i, propensitys_x.columns.get_loc(feature_input[y])]

        train_feature['propensity'] = propensitys[feature_input[y]].values
        #for i in range(predictions.shape[1]):
        #train_feature['model{}'.format(x)] = predictions.iloc[:, x].values
        train_feature[model_name] = predictions[model_name].values
        # 排序
        train_feature_sort = train_feature.sort_values(by=['propensity'], ascending=False)
        # 查找
        dis = 999
        pos = 0
        for d in range(0, len(train_feature_sort)):
            if abs(p - train_feature_sort.iloc[d]['propensity']) < dis:
                dis = abs(p - train_feature_sort.iloc[d]['propensity'])
                pos = d
        # 上下窗口选取
        # 选取总体数据的 1/100 ；多个个体容易曲线重合，需要较小窗口
        total = len(train_feature) / 50
        if pos - total / 2 < 0:
            front = 0
            after = front + total
        elif pos + total / 2 > len(train_feature) - 1:
            after = len(train_feature) - 1
            front = after - total
        else:
            front = pos - total / 2
            after = pos + total / 2
        front = int(front)
        after = int(after)
        print(pos, front, after)
        train_x = train_feature_sort[front:after]
        train_x = shuffle(train_x)

        coef = np.zeros(shape=(1, 3))
        intercept = 0
        label = train_x[model_name].values
        propensity_treatment = train_x[[feature_input[y]]]  # 'propensity',
        propensity_treatment_minmax = preprocessing.MinMaxScaler().fit_transform(propensity_treatment.values)
        quadratic_featurizer = PolynomialFeatures(degree=2)
        X_train_quadratic = quadratic_featurizer.fit_transform(propensity_treatment_minmax)
        regression = LinearRegression()
        regression.fit(X_train_quadratic, label)
        coef = regression.coef_
        intercept = regression.intercept_
        print('二次回归_3     r-squared', regression.score(X_train_quadratic, label))

        # 五分组总体 画图
        regression_new = LinearRegression()
        regression_new.coef_ = coef
        regression_new.intercept_ = intercept

        quadratic_featurizer = PolynomialFeatures(degree=2)
        xx_quadratic = quadratic_featurizer.fit_transform(xx.reshape(xx.shape[0], 1))
        yy_predict_all = regression_new.predict(xx_quadratic) - intercept
        yy_predict_alls.append(yy_predict_all.tolist())
    return list(xx), yy_predict_alls

