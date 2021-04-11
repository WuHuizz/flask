from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif,mutual_info_classif,f_regression,mutual_info_regression
from sklearn import preprocessing
from scipy.stats import pearsonr
import numpy as np
#personr
def rank_to_dict(score, feature_names, order=1):
    nd_array = order * np.array([score]).T  # [score]为二维行向量,但是fit_transform是以列来进行规范化的,所以需要转置
    ranks = preprocessing.MinMaxScaler().fit_transform(nd_array).T[0]   # T[0]返回ndarray的第一列,为1维数组;.T[0]等价于[:, 0]
    ranks = list(map(lambda x: round(x, 2), ranks))   # 注意在python3中，map函数得到的是对象，需要用list()转化才能得到list中的数据
    return ranks

def rankByPearonr(data,label,feature_names):
    corrs = []
    print(type(data))
    for i in range(data.shape[1]):
        corr, pval = pearsonr(data[:, i], label)  # pearsonr的系数是两个一维ndarray数组,也可以是list
        corrs.append(abs(corr))
    corrs = rank_to_dict(corrs,feature_names)
    return corrs

def rankByFC(data,label,feature_names):
    f, pval = f_classif(data, label)

    rank = rank_to_dict(f,feature_names=feature_names)
    return rank

def rankByMIC(data,label,feature_names):
    data_feature1 = data[:10000]
    data_label1 = label[:10000]

    f = mutual_info_regression(data_feature1, data_label1)
    # ranks["mutual_info_classif"] = rank_to_dict(f, feature_names)
    rank = rank_to_dict(f, feature_names)
    return rank

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from itertools import chain
rf = RandomForestClassifier(n_estimators=5, max_depth=4)
lr = LogisticRegression()
liner = LinearRegression()
lasso = Lasso(alpha=0.001)

def rankByRFE(data,label,feature_names):
    data_feature1 = data[:10000]
    data_label1 = label[:10000]
    rfe = RFE(lr, n_features_to_select=1)
    rfe.fit(data_feature1, data_label1)
    return rank_to_dict(list(map(float, rfe.ranking_)), feature_names, order=-1)

def rankByL1(data,label,feature_names):

    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False)
    lsvc.fit(data, label)
    return rank_to_dict(np.abs(list(chain.from_iterable(lsvc.coef_))), feature_names)

def rankByL2(data,label,feature_names):
    lsvc = LinearSVC(C=0.01, penalty="l2", dual=False)
    lasso = Lasso(alpha=.05)
    lsvc.fit(data, label)
    return rank_to_dict(np.abs(list(chain.from_iterable(lsvc.coef_))), feature_names)

def allRank(data,label,feature_names):
    ranks = dict()
    ranks['Pearsonr'] =rankByPearonr(data,label,feature_names)
    ranks['f_classif'] =rankByFC(data,label,feature_names)
    ranks['MIC'] = rankByMIC(data, label, feature_names)
    ranks['RFE_lr'] = rankByRFE(data, label, feature_names)
    ranks['L1'] = rankByL1(data, label, feature_names)
    ranks['L2'] = rankByL2(data, label, feature_names)
    from pandas.core.frame import DataFrame

    df = DataFrame(ranks)
    df['mean_score'] = df.mean(axis=1)
    all =  df['mean_score'].values.tolist()
    all = [ round(i,2) for i in all]
    df['mean_score'] = all
    ranks['all'] = all
    return ranks,df.values.tolist(),df.columns.tolist()





