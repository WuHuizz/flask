from flask import Blueprint,request,render_template,jsonify
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression,Lasso
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from .utils.dataprocess import scaler_data, load_data, process_miss, process_outlier,show_sample
from .utils.model_exp import (entire_propensity_score,entire_interpretability_1,entire_interpretability_2,
                              individual_propensity_score,individual_interpretability_1,individual_interpretability_2,individual_interpretability_3)
import pickle as pkl
import os
import numpy  as np
import pandas as pd
from glob import glob
from time import sleep
seed = 328
Model_list = {
    'lr':LogisticRegression(penalty='l2',random_state=seed),
    'dt':DecisionTreeClassifier(random_state=seed,max_depth=5,max_features=5),
    'mlp':MLPClassifier(hidden_layer_sizes=10,random_state=seed),
    'rf':RandomForestClassifier(random_state=seed,max_depth=5,max_features=5)
}
model2name = {
    'lr': "逻辑回归",
    'dt': "决策树",
    'mlp': "神经网络",
    'rf': "随机森林"
}
name2model = {value:key for (key,value) in model2name.items()}
third_blue = Blueprint('third_blue',__name__)

scaler_x, scaler_y, scaler,origin_x, origin_y, feature_name, sample_data,sample_feature = None, None, None, None, None,None,None,None
entire_propensitys,predictions,scaler,single_propensitys= None,None,None,None
expline_feature = []
def print_x():
    global scaler_x, scaler_y, origin_x, origin_y, feature_name, sample_data,sample_feature
    global entire_propensitys,predictions,scaler,single_propensitys,scaler
    print("scaler_x:{}, scaler_y:{}, origin_x:{}, origin_y:{}, feature_name:{}, sample_data,sample_feature:{}".format(scaler_x, scaler_y, origin_x, origin_y, feature_name, sample_data,sample_feature))
    print("entire_propensitys:{},predictions:{},scaler:{},single_propensitys:{}".format(entire_propensitys,predictions,scaler,single_propensitys))
    print("scaler",scaler)
    print("expline_feature",expline_feature)
def get_exp_f():
    global expline_feature
    if os.path.exists(os.path.join('App/static/datas', 'features.pkl')):
        expline_feature = pkl.load(open(os.path.join('App/static/datas', 'features.pkl'), 'rb'))
        print(expline_feature)
    else:
        expline_feature = ['可用额度', '年龄', '逾期30-59天笔数', '逾期90天笔数', '逾期60-89天笔数']
    return expline_feature

entire_process, single_process = 0, 0

def start():
    global scaler_x, scaler_y, origin_x, origin_y, feature_name,entire_propensitys,seed,predictions
    global scaler,sample_feature,sample_data,expline_feature
    expline_feature = get_exp_f()
    scaler_x_path = 'App/static/datas/train_x_scaler.pkl'
    scaler_y_path = 'App/static/datas/train_y_scaler.pkl'
    origin_x_path = 'App/static/datas/train_x_origin.pkl'
    origin_y_path = 'App/static/datas/train_y_origin.pkl'
    scaler_path = 'App/static/datas/scaler.pkl'
    data, feature_name = load_data('App/static/uploads/train.csv', is_first=False)
    label = feature_name[0]
    feature_name.remove(label)
    if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path) or not os.path.exists(scaler_path):
        data = process_miss(data)
        data = process_outlier(data)
        scaler_x, scaler_y, origin_x, origin_y, scaler = scaler_data(data)
        pkl.dump(scaler_x, open(scaler_x_path, 'wb'))
        pkl.dump(scaler_y, open(scaler_y_path, 'wb'))
        pkl.dump(origin_x, open(origin_x_path, 'wb'))
        pkl.dump(origin_y, open(origin_y_path, 'wb'))
        pkl.dump(scaler, open(scaler_path, 'wb'))
    else:
        scaler_x = pkl.load(open(scaler_x_path, 'rb'))
        scaler_y = pkl.load(open(scaler_y_path, 'rb'))
        origin_x = pkl.load(open(origin_x_path, 'rb'))
        origin_y = pkl.load(open(origin_x_path, 'rb'))
        scaler = pkl.load(open(scaler_path, 'rb'))
    
    predictions = pkl.load(open('App/static/predictions/total_predictions.pkl','rb'))
    sample_feature,sample_data = show_sample(origin_x)
@third_blue.route('/explain',methods=['POST','GET'])
def explain():
    start()
    print_x()
    path = 'App/static/models/*'
    dirlist = glob(path)
    model_name = []
    for dir in dirlist:
        name = os.path.split(dir)[-1].split('.')[0]
        model_name.append(model2name[name])
    global sample_data,sample_feature,expline_feature
    return render_template('explain.html',model_name=model_name,expline_feature=expline_feature,
                            sample_feature=sample_feature,sample_data=sample_data)


@third_blue.route('/explain_s',methods=['POST','GET'])
def explain_s():
    start()
    print_x()
    if os.path.exists("App/static/propensitys/single_propensitys.csv"):
        os.remove("App/static/propensitys/single_propensitys.csv")
    path = 'App/static/models/*'
    dirlist = glob(path)
    model_name = []
    for dir in dirlist:
        name = os.path.split(dir)[-1].split('.')[0]
        model_name.append(model2name[name])
    global sample_data,sample_feature,expline_feature

    return render_template('explain_s.html',model_name=model_name,expline_feature=expline_feature,
                            sample_feature=sample_feature,sample_data=sample_data)

@third_blue.route('/show_explain/entire_pps',methods=['POST','GET'])
def entire_pps():
    global entire_process
    global entire_propensitys,origin_x,feature_name,scaler_x,seed
    entire_process = 0
    propensitys_path = 'App/static/propensitys/entire_propensitys.csv'
    if not os.path.exists(propensitys_path):
        entire_propensitys = entire_propensity_score(train_feature=origin_x, feature_input=feature_name, train_feature_value_minmaxScaler=scaler_x,seed=seed)
    else:
        entire_propensitys = pd.read_csv(propensitys_path)
    entire_process = 100
    return jsonify({'res': entire_process})

@third_blue.route('/show_explain/entire_pps_show',methods=['POST','GET'])
def show_pps_process():
    global entire_process
    if os.path.exists('App/static/propensitys/entire_propensitys.csv'):
        entire_process = 100
    else:
        entire_process += 1
        entire_process = 99 if entire_process>=100 else entire_process
    return jsonify({'res': entire_process})

@third_blue.route('/show_explain/entire_1',methods=['POST','GET'])
def expline_entire_1():
    global expline_feature,origin_x,scaler_x,entire_propensitys,predictions
    if len(expline_feature) == 0 or expline_feature is None:
        expline_feature = get_exp_f()
    if entire_propensitys is None:
        entire_propensitys = pd.read_csv('App/static/propensitys/entire_propensitys.csv')
    if predictions is None:
        predictions = pkl.load(open('App/static/predictions/total_predictions.pkl','rb'))
    if origin_x is None:
        scaler_x = pkl.load(open("App/static/datas/train_x_scaler.pkl", 'rb'))
        origin_x = pkl.load(open("App/static/datas/train_x_origin.pkl", 'rb'))
    model_name = request.values['model_name']
    model_name = name2model[model_name]
    xx, yy = entire_interpretability_1(0,expline_feature,model_name, origin_x,entire_propensitys,predictions)
    data = [['x']+xx]
    for i in range(len(expline_feature)):
        data.append([expline_feature[i]]+yy[i])
    return jsonify({"data":data,"lines":len(expline_feature)})

@third_blue.route('/show_explain/entire_2',methods=['POST','GET'])
def expline_entire_2():
    global expline_feature, origin_x, scaler_x, entire_propensitys, predictions
    if len(expline_feature) == 0 or expline_feature is None:
        expline_feature = get_exp_f()
    if entire_propensitys is None:
        entire_propensitys = pd.read_csv('App/static/propensitys/entire_propensitys.csv')
    if predictions is None:
        predictions = pkl.load(open('App/static/predictions/total_predictions.pkl','rb'))
    if origin_x is None:
        scaler_x = pkl.load(open("App/static/datas/train_x_scaler.pkl", 'rb'))
        origin_x = pkl.load(open("App/static/datas/train_x_origin.pkl", 'rb'))
    models = [c for c in predictions]
    feature_name = request.values['feature_name']
    model_name = '逻辑回归'
    model_name = name2model[model_name]
    x = expline_feature.index(feature_name)
    xx, yy = entire_interpretability_2(x, expline_feature, model_name, origin_x, entire_propensitys, predictions)
    data = [['x'] + xx]
    for i in range(len(models)):
        data.append([models[i]]+yy[i])
    return jsonify({"data":data,"lines":len(models)})

@third_blue.route('/show_explain/single_pps',methods=['POST','GET'])
def single_pps():
    global scaler_x, origin_x, expline_feature, entire_propensitys, predictions,sample_data
    global scaler,feature_name,entire_propensitys,single_propensitys,single_process
    single_process = 0
    is_entire = False
    if entire_propensitys is not None:
        is_entire = True
    X = sample_data
    if scaler is None:
        scaler = pkl.load(open('App/static/datas/scaler.pkl', 'rb'))
    X = scaler.transform(X)
    entire_propensitys, single_propensitys = individual_propensity_score(X,origin_x,feature_name,scaler_x,seed,is_entire=is_entire)
    single_process = 100
    sleep(1)
    return jsonify({'res': single_process})

@third_blue.route('/show_explain/single_pps_show',methods=['POST','GET'])
def single_pps_process():
    global single_process
    single_process += 1
    single_process = 99 if single_process >= 99 else single_process
    return jsonify({'res': single_process})

# 1.单一个体-单一属性-不同模型的分析
@third_blue.route('/show_explain/single_1',methods=['POST','GET'])
def explain_single_1():
    global scaler, feature_name, entire_propensitys, single_propensitys, single_process,expline_feature,origin_x,predictions,origin_x
    print("expline_feature",expline_feature)
    if len(expline_feature) == 0 or expline_feature is None:
        expline_feature = get_exp_f()
    if entire_propensitys is None:
        entire_propensitys = pd.read_csv('App/static/propensitys/entire_propensitys.csv')
    if single_propensitys is None:
        single_propensitys = pd.read_csv('App/static/propensitys/single_propensitys.csv')
    if predictions is None:
        predictions = pkl.load(open('App/static/predictions/total_predictions.pkl','rb'))
    if origin_x is None:
        origin_x = pkl.load(open(origin_x_path, 'rb'))
    idx = request.values['example-select']
    modelname = 'lr'
    feature_name = request.values['feature_name']
    feature_index = expline_feature.index(feature_name)

    xx,yy = individual_interpretability_1(int(idx),feature_index,expline_feature,modelname,origin_x,entire_propensitys,single_propensitys,predictions)
    data = [['x']+xx]
    models = [c for c in predictions]
    for i in range(len(models)):
        data.append([models[i]] + yy[i])
    print(data)
    return jsonify({"data":data,"lines":len(models)})

#单一个体-单一模型-不同属性的分析
@third_blue.route('/show_explain/single_2',methods=['POST','GET'])
def explain_single_2():
    print("--2--开始分析...")
    global scaler, feature_name, entire_propensitys, single_propensitys, single_process,expline_feature,origin_x,predictions
    if len(expline_feature) == 0 or expline_feature is None:
        expline_feature = get_exp_f()
    if entire_propensitys is None:
        entire_propensitys = pd.read_csv('App/static/propensitys/entire_propensitys.csv')
    if single_propensitys is None:
        single_propensitys = pd.read_csv('App/static/propensitys/single_propensitys.csv')
    if predictions is None:
        predictions = pkl.load(open('App/static/predictions/total_predictions.pkl','rb'))
    if origin_x is None:
        origin_x = pkl.load(open(origin_x_path, 'rb'))
    idx = request.values['example-select']
    model_name = name2model[request.values['model_name']]
    feature_name = ""
    xx,yy = individual_interpretability_2(int(idx),0,expline_feature,model_name,origin_x,entire_propensitys,single_propensitys,predictions)
    data = [['x']+xx]
    for i in range(len(expline_feature)):
        data.append([expline_feature[i]]+yy[i])
    print("--2--结束分析...")
    return jsonify({"data": data, "lines": len(expline_feature)})


@third_blue.route('/show_explain/single_3',methods=['POST','GET'])
def explain_single_3():
    global scaler, feature_name, entire_propensitys, single_propensitys, single_process,expline_feature,predictions,origin_x
    if len(expline_feature) == 0 or expline_feature is None:
        expline_feature = get_exp_f()
    if entire_propensitys is None:
        entire_propensitys = pd.read_csv('App/static/propensitys/entire_propensitys.csv')
    if single_propensitys is None:
        single_propensitys = pd.read_csv('App/static/propensitys/single_propensitys.csv')
    if predictions is None:
        predictions = pkl.load(open('App/static/predictions/total_predictions.pkl','rb'))
    if origin_x is None:
        origin_x = pkl.load(open(origin_x_path, 'rb'))
    model_name = name2model[request.values['model_name']]
    feature_name = request.values['feature_name']
    xx, yy = individual_interpretability_3(0,expline_feature.index(feature_name),expline_feature,model_name,origin_x,entire_propensitys,single_propensitys,predictions)
    data = [['x'] + xx]
    for i in range(4):
        data.append(['第{}个样本'.format(i+1)] + yy[i])
    return jsonify({"data": data, "lines": 4})

@third_blue.route('/show_explain/reflash',methods=['POST','GET'])
def reflash():
    global origin_x,sample_data
    origin_x_path = 'App/static/datas/train_x_origin.pkl'
    if origin_x is None:
        origin_x = pkl.load(open(origin_x_path, 'rb'))
    f,sample_data = show_sample(origin_x)
    if os.path.exists('App/static/propensitys/single_propensitys.csv'):
        os.remove('App/static/propensitys/single_propensitys.csv')
    return jsonify({"feature":f,"data":sample_data.tolist()})



