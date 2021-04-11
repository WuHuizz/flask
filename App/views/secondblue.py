from flask import Blueprint,request,render_template,jsonify
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression,Lasso
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from .utils.dataprocess import scaler_data, load_data, process_miss, process_outlier,fill_up_missing
import pickle as pkl
import os
import pandas as pd
from glob import glob
seed = 2021
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

second = Blueprint('second_blue',__name__)
model_dict = {}
@second.route('/hello')
def hello():
    return "Hello"

@second.route('/show_process/<string:name>/')
def show_process(name):
    global model_dict
    if name not in model_dict:
        model_dict[name] = [None, 0]
    elif model_dict[name][0] is not None:
        model_dict[name][1] = 100
    elif model_dict[name][0] is None:
        model_dict[name][1] = 99 if model_dict[name][1] + 1 >= 100 else model_dict[name][1] + 1
    print("{}:{}".format( name,model_dict[name][1]))
    return jsonify({'res': model_dict[name][1]})

@second.route('/train_process/<string:name>')
def train_process(name):
    model_dict[name] = [None, 0]
    if name not in Model_list:
        return jsonify({'res':'null'})
    train(Model_list[name],name)
    return jsonify({'res':model_dict[name][1]})

def train(model,name):
    prediction_path=os.path.join('App/static/predictions/','{}_prediction.pkl'.format(name))
    global model_dict
    model_dict[name] = [None,0]
    x_path = 'App/static/datas/train_x_scaler.pkl'
    y_path = 'App/static/datas/train_y_scaler.pkl'
    origin_x_path='App/static/datas/train_x_origin.pkl'
    origin_y_path = 'App/static/datas/train_y_origin.pkl'
    scaler_path = 'App/static/datas/scaler.pkl'
    if not os.path.exists(x_path) or not os.path.exists(y_path) or not os.path.exists(scaler_path):
        data,feature_name = load_data('App/static/uploads/train.csv',is_first=False)
        data = process_miss(data)
        data = process_outlier(data)
        X,y,origin_X,origin_y , scaler= scaler_data(data)
        pkl.dump(X,open(x_path,'wb'))
        pkl.dump(y, open(y_path, 'wb'))
        pkl.dump(origin_X,open(origin_x_path,'wb'))
        pkl.dump(origin_y, open(origin_y_path, 'wb'))
        pkl.dump(scaler, open(scaler_path, 'wb'))
    else:
        X = pkl.load(open(x_path,'rb'))
        y = pkl.load(open(y_path,'rb'))
    model.fit(X,y)
    prediction = model.predict_proba(X)[:, 1]
    pkl.dump(prediction,open(prediction_path,'wb'))
    print('model:{} AUC Score : '.format(name), (metrics.roc_auc_score(y, prediction)))
    path = os.path.join('App/static/models','{}.pkl'.format(name))
    pkl.dump(model,open(path,'wb'))
    merge_prediction()
    model_dict[name][0] = name

def merge_prediction():
    path = 'App/static/predictions/*'
    dirlist = glob(path)
    dict = {}
    names = []
    for dir in dirlist:
        name = os.path.split(dir)[-1].split('.')[0].split('_')[0]
        if name != "total":
            dict[name] = pkl.load(open(dir,'rb'))
            names.append(name)
    predictions = pd.DataFrame(columns=(names))
    for name in names:
        predictions[name]=dict[name]
    path = 'App/static/predictions/total_predictions.pkl'
    pkl.dump(predictions,open(path,'wb'))

@second.route('/trainmodel', methods=['POST', 'GET'])
def trainmodel():
    return render_template('trainmodel.html')






