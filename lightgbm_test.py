#############
######使用第一轮训练好的模型
##############

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import numpy as np

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

train_features = train_df.drop(['target', 'ID_code'], axis=1)
test_features = test_df.drop(['ID_code'], axis=1)
train_target = train_df['target']

scaler = StandardScaler().fit(train_features)
train_features = scaler.transform(train_features)
test_features = scaler.transform(test_features)

splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=1).split(train_features, train_target))

x_train = np.array(train_features)
y_train = np.array(train_target)

path = '../santander_customer_transaction_prediction_save/2019-03-08-19-03-24_4512'

for i, ()

for i in range(1, 6):
    filename = f'lgbm_{i}.txt'
    clf = lgb.Booster(model_file=f'{path}/{filename}')
    predictions = clf.predict(train_features)
    res = metrics.roc_auc_score(train_target, predictions)
    print(res)

