from flask import render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from flask import Blueprint
from .utils.dataprocess import (load_data,data_d,show_miss,show_ratio,show_vt_raletion,
                                process_miss,process_outlier,show_corr,covert_feature)
import pandas as pd
blue = Blueprint('blue', __name__)
data, feature_name = None, None
from .utils.featrue_select import allRank
import pickle as pkl
@blue.route('/')
def hello_world():
    return render_template('index.html')

@blue.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        upload_path = os.path.join('App/static/uploads', secure_filename('train.csv'))
        f.save(upload_path)
        global data, feature_name
        data, feature_name = load_data(upload_path)
        return redirect(url_for('blue.upload'))
    return render_template('upload.html')

@blue.route('/showdata', methods=['POST', 'GET'])
def showdata():
    global feature_name,data
    if feature_name is None :
        data,feature_name = load_data('App/static/uploads/train.csv',is_first=False)
    head_10 = data.head(10).values.tolist()
    feature_d = data_d()
    miss_data,miss_index=show_miss(data)
    ratio_data = show_ratio(data)
    ##process data
    data = process_miss(data)
    data = process_outlier(data)
    upload_path = os.path.join('App/static/uploads', secure_filename('train.csv'))
    data.to_csv(upload_path,index=False)
    drop_cols_byVt, Vt_feature, Vt_value = show_vt_raletion(data)
    corr_feature,corr_data,full_corr_feature,full_corr_data=show_corr(data)
    return render_template('showdata.html',data=head_10,feature=feature_name,feature_d=feature_d,
                           miss_data=miss_data,miss_index=miss_index,ratio_data=ratio_data,
                           Vt_feature=Vt_feature, Vt_value=Vt_value,corr_data=corr_data, corr_feature=corr_feature,
                           full_corr_feature=full_corr_feature,full_corr_data=full_corr_data)

@blue.route('/showfeatrue',methods=['POST','GET'])
def show_feature():
    if request.method == 'POST':
        _features = data_d()
        features = request.form.getlist('boxes')
        data = [_features[f] for f in features]
        path = os.path.join('App/static/datas', 'features.pkl')

        pkl.dump(data,open(path,'wb'))
        print(pkl.load(open(path,'rb')))

        ret = "筛选特征为："
        for f in features:
            ret += "(" + f + ":" + _features[f] + "), "
        return {"ret":ret}

    else:
        path = os.path.join('App/static/uploads', 'train.csv')
        if os.path.exists(path):
            data = pd.read_csv(path)
            data_feature = data.drop(['好坏客户'], axis=1)
            data_label = data['好坏客户']
            feature_names = data_feature.columns.values.tolist()
            feature_names = [covert_feature(f) for f in feature_names]
            data_feature = data_feature.values
            data_label = data_label.values
            rank_dic,df_scores,df_col = allRank(data_feature,data_label,feature_names)
            print(feature_names)
            print(rank_dic)
            return render_template('choose_feature.html',rank_dic=rank_dic,feature_names=feature_names,
                                   df_scores=df_scores,df_col=df_col)


