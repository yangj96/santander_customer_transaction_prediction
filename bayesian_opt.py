import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn import metrics

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

train_features = train_df.drop(['target', 'ID_code'], axis=1)
test_features = test_df.drop(['ID_code'], axis=1)
train_target = train_df['target']

bayesian_tr_index, bayesian_val_index = list(
    StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
        .split(train_features, train_target)
)[0]


def LGB_bayesian(
        num_leaves,  # int
        min_data_in_leaf,  # int
        learning_rate,
        min_sum_hessian_in_leaf,  # int
        feature_fraction,
        lambda_l1,
        lambda_l2,
        min_gain_to_split,
        max_depth,
        seed = 666):
    # LightGBM expects next three parameters need to be integer. So we make them integer
    num_leaves = int(num_leaves)
    min_data_in_leaf = int(min_data_in_leaf)
    max_depth = int(max_depth)

    assert type(num_leaves) == int
    assert type(min_data_in_leaf) == int
    assert type(max_depth) == int

    param = {
        'num_leaves': num_leaves,
        'max_bin': 63,
        'min_data_in_leaf': min_data_in_leaf,
        'learning_rate': learning_rate,
        'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
        'bagging_fraction': 1.0,
        'bagging_freq': 5,
        'feature_fraction': feature_fraction,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'min_gain_to_split': min_gain_to_split,
        'max_depth': max_depth,
        'save_binary': True,
        'seed': seed,
        'feature_fraction_seed': seed,
        'bagging_seed': seed,
        'drop_seed': seed,
        'data_random_seed': seed,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False,

    }

    xg_train = lgb.Dataset(train_features.iloc[bayesian_tr_index],
                           label=train_target.iloc[bayesian_tr_index],
                           )
    xg_valid = lgb.Dataset(train_features.iloc[bayesian_val_index],
                           label=train_target.iloc[bayesian_val_index],
                           )

    num_round = 5000
    clf = lgb.train(param, xg_train, num_round, valid_sets=[xg_valid], verbose_eval=250, early_stopping_rounds=50)

    predictions = clf.predict(train_features.iloc[bayesian_val_index].values, num_iteration=clf.best_iteration)

    score = metrics.roc_auc_score(train_target.iloc[bayesian_val_index].values, predictions)

    return score

# Bounded region of parameter space
bounds_LGB = {
    'num_leaves': (2, 20),
    'min_data_in_leaf': (2, 20),
    'learning_rate': (0.01, 0.3),
    'min_sum_hessian_in_leaf': (0.000001, 0.01),
    'feature_fraction': (0.01, 1),
    'lambda_l1': (0, 5.0),
    'lambda_l2': (0, 5.0),
    'min_gain_to_split': (0, 1.0),
    'max_depth':(2,30),
}

from bayes_opt import BayesianOptimization

LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=13)

print(LGB_BO.space.keys)

init_points = 6
n_iter = 5

LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)

print(LGB_BO.max['params'])

# {'feature_fraction': 0.050000005250353315, 'lambda_l1': 0.0, 'lambda_l2': 4.284331626977193,
# 'learning_rate': 0.0703523686996351,
# 'max_depth': 3.0175120350617677,
# 'min_data_in_leaf': 5.017857679853695,
# 'min_gain_to_split': 0.007958419918558482,
# 'min_sum_hessian_in_leaf': 1e-05, 'num_leaves': 5.0}
