import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing

def load_data(path,is_first=True):
    train_data = pd.read_csv(path)
    # 此行只针对cs-training数据集
    if is_first:
        train_data = train_data.drop(['Unnamed: 0'], axis=1)

        column = {'SeriousDlqin2yrs': '好坏客户',
                  'RevolvingUtilizationOfUnsecuredLines': '可用额度',
                  'age': '年龄',
                  'NumberOfTime30-59DaysPastDueNotWorse': '逾期30-59天笔数',
                  'DebtRatio': '负债率',
                  'MonthlyIncome': '月收入',
                  'NumberOfOpenCreditLinesAndLoans': '开放式信贷数量',
                  'NumberOfTimes90DaysLate': '逾期90天笔数',
                  'NumberRealEstateLoansOrLines': '固定贷款量',
                  'NumberOfTime60-89DaysPastDueNotWorse': '逾期60-89天笔数',
                  'NumberOfDependents': '家属数量'}
        train_data.rename(columns=column, inplace=True)
        train_data.to_csv(path,index=False)

    feature_name = [column for column in train_data]
    return train_data, feature_name

def data_d():
    column = {'标签': '好坏客户',
              '特征A': '可用额度',
              '特征B': '年龄',
              '特征C': '逾期30-59天笔数',
              '特征D': '负债率',
              '特征E': '月收入',
              '特征F': '开放式信贷数量',
              '特征G': '逾期90天笔数',
              '特征H': '固定贷款量',
              '特征I': '逾期60-89天笔数',
              '特征J': '家属数量'}
    return column

def show_ratio(data):
    ratio_data = data['好坏客户'].value_counts().to_dict()
    return ratio_data

def show_miss(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum() / data.isnull().count() * 100)
    tt = pd.concat([total, percent], axis=1, keys=['Total_null', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    miss_data = np.transpose(tt)
    index = miss_data.index.tolist()
    data = miss_data.values.tolist()
    return data, index

def show_vt_raletion(train_data):
    d1, iv1, cut1, woe1 = mono_bin(train_data['好坏客户'], train_data['可用额度'])
    d2, iv2, cut2, woe2 = mono_bin(train_data['好坏客户'], train_data['年龄'])
    d3, iv3, cut3, woe3 = mono_bin(train_data['好坏客户'], train_data['负债率'])
    d4, iv4, cut4, woe4 = mono_bin(train_data['好坏客户'], train_data['月收入'])
    c1 = binning_cate(train_data, '逾期30-59天笔数', '好坏客户')
    c2 = binning_cate(train_data, '开放式信贷数量', '好坏客户')
    c3 = binning_cate(train_data, '逾期90天笔数', '好坏客户')
    c4 = binning_cate(train_data, '固定贷款量', '好坏客户')
    c5 = binning_cate(train_data, '逾期60-89天笔数', '好坏客户')
    c6 = binning_cate(train_data, '家属数量', '好坏客户')
    informationValue = [iv1, iv2, c4['IV'].mean(), iv3, iv4, c1['IV'].mean(), c3['IV'].mean(), c5['IV'].mean(),
                        c2['IV'].mean(), c6['IV'].mean()]
    index = ['可用额度', '年龄', '固定贷款量', '负债率', '月收入', '逾期30-59天笔数', '逾期90天笔数', '逾期60-89天笔数', '开放式信贷数量', '家属数量']
    Vt_feature = [covert_feature(feature) for feature in index]
    return "",Vt_feature,informationValue

def MissingHandler(df):
    DataMissing = df.isnull().sum()*100/len(df)
    DataMissingByColumn = pd.DataFrame({'Percentage_Nulls':DataMissing})
    DataMissingByColumn.sort_values(by='Percentage_Nulls',ascending=False,inplace=True)
    DataMissingByColumn = DataMissingByColumn.reset_index()
    DataMissingByColumn.rename(columns={'index':'Column_Name'},inplace=True)
    return DataMissingByColumn

def process_miss(df):
    type_dict = dict()
    for col in df.columns.to_list():
        type_dict[col] = str(df[col].dtypes)
    DataMiss = MissingHandler(df)
    columns = DataMiss.Column_Name.to_list()
    percentage = DataMiss.Percentage_Nulls.to_list()
    for i in range(len(percentage)):
        if percentage[i] > 0:
            if 'int' in type_dict[columns[i]]:  ### 中位数填充
                df[columns[i]].fillna(df[columns[i]].median(), inplace=True)
                print("median ：{}".format(columns[i]))
            elif 'float' in type_dict[columns[i]]:  ### 均值填充
                df[columns[i]].fillna(df[columns[i]].mean(), inplace=True)
                print("mean ：{}".format(columns[i]))
            else:  ### 众数填充
                df[columns[i]].fillna(df[columns[i]].mode(), inplace=True)
                print("mode ：{}".format(columns[i]))
    return df

def process_outlier(train_data):
    train_data = train_data[train_data.loc[:, '逾期30-59天笔数'] < 90]
    train_data = train_data[train_data['年龄'] != 0]
    train_data = train_data[train_data['可用额度'] < 10]
    train_data = train_data[train_data['负债率'] < 50000]
    train_data = train_data[train_data['固定贷款量'] < 50]
    return train_data

def fill_up_missing(train_data):
    ###填充 家属数量
    train_data['家属数量'].fillna(train_data['家属数量'].median(), inplace=True)

    ##填充 月收入
    process_df = train_data.iloc[:, [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]]
    known = process_df[process_df.月收入.notnull()].values
    unknown = process_df[process_df.月收入.isnull()].values
    X = known[:, 1:]
    y = known[:, 0]

    rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
    rfr.fit(X, y)

    predicted = rfr.predict(unknown[:, 1:]).round(0)
    train_data.loc[(train_data.月收入.isnull()), '月收入'] = predicted
    return train_data

def show_map(df):
    pass

def show_raletion(df):
    x_cols = [col for col in df.columns if col not in ['好坏客户'] if df[col].dtype != 'object']  # 处理目标的其他所有特征
    labels = []
    values = []
    drop_cols_bycorr = []
    for col in x_cols:
        labels.append(col)
        values.append(np.corrcoef(df[col].values, df['好坏客户'], values)[0, 1])
        if abs(values[-1]) < 0.01:
            drop_cols_bycorr.append(col)
    corr_df = pd.DataFrame({'col_labels': labels, 'corr_values': values})
    corr_data = corr_df.corr_values.tolist()
    corr_feature = [covert_feature(feature) for feature in corr_df.col_labels.tolist()]
    return drop_cols_bycorr,corr_data,corr_feature

def covert_feature(feature):
    column = {'好坏客户': '标签',
              '可用额度': '特征A',
              '年龄': '特征B',
              '逾期30-59天笔数': '特征C',
              '负债率': '特征D',
              '月收入': '特征E',
              '开放式信贷数量': '特征F',
              '逾期90天笔数': '特征G',
              '固定贷款量': '特征H',
              '逾期60-89天笔数': '特征I',
              '家属数量': '特征J'}
    return column[feature]

def scaler_data(train_data):
    feature_name = [column for column in train_data]
    label = feature_name[0]
    feature_name.remove(label)
    train_feature = train_data[feature_name]
    train_label = train_data[label]
    train_feature_value = train_feature.values
    train_label_value = train_label.values
    scaler = preprocessing.MinMaxScaler()
    train_feature_value_minmaxScaler = scaler.fit_transform(train_feature_value)
    return train_feature_value_minmaxScaler, train_label_value, train_feature, train_label,scaler

def show_sample(data):
    sample = data.sample(n=4,axis=0,replace=False)
    sample_feature = [c for c in sample]
    sample_data = sample.values
    return sample_feature, sample_data

import scipy.stats as stats

# 定义自动分箱函数
def mono_bin(Y, X, n=5):
    r = 0
    bad = Y.sum()
    good = Y.count()-bad
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)})
        d2 = d1.groupby('Bucket', as_index=True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    d3 = pd.DataFrame(d2.X.min(), columns=['min'])
    d3['min'] = d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.count().Y-d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] =(d2.count().Y-d2.sum().Y)/d2.count().Y
    d3['woe'] = np.log((d3['rate']/(1-d3['rate']))/(good/bad))
    d3['goodattribute'] = d3['sum']/good
    d3['badattribute'] = (d3['total']-d3['sum'])/bad
    iv = ((d3['goodattribute']-d3['badattribute'])*d3['woe']).sum()
    d4 = (d3.sort_values(by='min'))
    print("=" * 60)
    print(d4)
    cut = []
    cut.append(float('-inf'))
    for i in range(1, n+1):
        qua = X.quantile(i/(n+1))
        cut.append(round(qua, 4))
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))
    return d4, iv, cut, woe


def binning_cate(df, col, target):
    total = df[target].count()
    bad = df[target].sum()
    good = total - bad
    d1 = df.groupby([col], as_index=True)
    d2 = pd.DataFrame()
    d2['样本数'] = d1[target].count()
    d2['黑样本数'] = d1[target].sum()
    d2['白样本数'] = d2['样本数'] - d2['黑样本数']
    d2['逾期用户占比'] = d2['黑样本数'] / d2['样本数']
    d2['badattr'] = d2['黑样本数'] / bad
    d2['goodattr'] = d2['白样本数'] / good
    d2['WOE'] = np.log(d2['badattr'] / d2['goodattr'])

    where_are_inf = np.isinf(d2['WOE'])
    d2['WOE'][where_are_inf] = 0
    d2['bin_iv'] = (d2['badattr'] - d2['goodattr']) * d2['WOE']
    d2['IV'] = d2['bin_iv'].sum()

    bin_df = d2.reset_index()
    bin_df.drop(['badattr', 'goodattr', ], axis=1, inplace=True)
    bin_df.rename(columns={col: '分箱结果'}, inplace=True)
    bin_df['特征名'] = col
    bin_df = pd.concat([bin_df['特征名'], bin_df.iloc[:, :-1]], axis=1)
    return bin_df

def show_corr(train_data):
    corr = train_data.corr(method='pearson')
    features = [c for c in corr]
    first_value = corr['好坏客户'].values.tolist() [1:]
    corr_data = corr.values.tolist()
    num = len(features)
    data = []
    for i in range(num):
        for j in range(num):
            data.append([i,j,round(corr_data[i][j],4)])
    features = [covert_feature(f) for f in features]
    return features[1:], first_value, features, data
    #values = corr.values.tolist()


