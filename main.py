import pandas as pd
import numpy as np
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn import metrics

import os
import shutil
from datetime import datetime
from Logger import Logger

def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y

save_path = '../santander_customer_transaction_prediction_save'
if not os.path.exists(save_path):
    os.mkdir(save_path)
time_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
save_path = os.path.join(save_path, f'{time_str}_{os.getpid()}')
os.makedirs(save_path)
current_dir = os.path.abspath(os.path.dirname(__file__))
shutil.copytree(current_dir, os.path.join(save_path, 'code'))

logger = Logger(os.path.join(save_path, 'log.txt'))
logger.info(f'create dir {save_path}')

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

train_features = train_df.drop(['target', 'ID_code'], axis=1)
test_features = test_df.drop(['ID_code'], axis=1)
train_target = train_df['target']

for col in train_features.columns:
    train_features[col + '_err'] = train_features[col] - train_features[col].mean()
    train_features[col + '_abserr'] = abs(train_features[col] - train_features[col].mean())
print(train_features.shape)

for col in test_features.columns:
    test_features[col + '_err'] = test_features[col] - test_features[col].mean()
    test_features[col + '_abserr'] = abs(test_features[col] - test_features[col].mean())

features = train_features.columns

scaler = StandardScaler().fit(train_features)
train_features = scaler.transform(train_features)
test_features = scaler.transform(test_features)

n_splits = 11 # Number of K-folder Splits
random_state = 42
np.random.seed(random_state)
splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(train_features, train_target))
#
# lgbm_param = {
#     'num_leaves': 7,
#     'learning_rate': 0.01,
#     'feature_fraction': 0.04,
#     'max_depth': 17,
#     'objective': 'binary',
#     'boosting_type': 'gbdt',
#     'metric': 'auc'
# }
lgbm_param = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 13,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 80,
    "min_sum_hessian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    #"lambda_l1" : 5,
    #"lambda_l2" : 5,
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "seed": random_state
    "num_threads" : 32
}

oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()

models = []

for i, (train_idx, valid_idx) in enumerate(splits):
    logger.info(f'Fold {i+1}')
    # x_train = np.array(train_features)
    # y_train = np.array(train_target)
    x_train, y_train = train_features[train_idx], train_target[train_idx]
    x_valid, y_valid = train_features[valid_idx], train_target[valid_idx]
    print(x_train.shape)
    x_tr, y_tr = augment(x_train, y_train)
    print(x_tr.shape)
    x_tr = pd.DataFrame(x_tr)

    train_data = lgb.Dataset(x_tr, label=y_tr)
    valid_data = lgb.Dataset(x_valid, label=y_valid)

    num_round = 100000
    clf = lgb.train(lgbm_param, train_data, num_round, valid_sets=[valid_data],\
                    verbose_eval=100, early_stopping_rounds=1000)

    models.append(clf)

    logger.info(f'best score {clf.best_score}')
    logger.info(f'best_iteration {clf.best_iteration}')

    clf.save_model(os.path.join(save_path, f'lgbm_{i+1}.txt'), num_iteration=clf.best_iteration)

    oof[valid_idx] = clf.predict(x_valid, num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df['feature'] = features
    fold_importance_df['importance'] = clf.feature_importance()
    fold_importance_df['fold'] = i + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(test_features, num_iteration=clf.best_iteration) / n_splits

id_code_test = test_df['ID_code']
submission = pd.DataFrame({'ID_code': id_code_test, 'target': predictions})
submission.to_csv(os.path.join(save_path, 'submission.csv'), index=False, header=True)

print(feature_importance_df)
logger.info(metrics.roc_auc_score(train_target.values, oof))

res = models[0].predict(test_features, num_iteration=models[0].best_iteration)
