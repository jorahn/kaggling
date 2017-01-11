import pandas as pd
import numpy as np
import feather
import logging
import warnings
import sys
from datetime import datetime
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%y/%m/%d %H:%M:%S', level=logging.DEBUG)

def rsme(y, y_pred):
    return mean_squared_error(y, y_pred)**0.5

def load_data():
    logging.info('Loading Data')
    train = feather.read_dataframe('../santander/data/x_train.feather')
    test = feather.read_dataframe('../santander/data/x_test.feather')
    labels = feather.read_dataframe('../santander/data/y_train.feather')['TARGET'].values
    sub_ix = test['ID']
    sub_col = 'TARGET'
    return train, test, labels, sub_ix, sub_col

def remove_equals(train, test):
    logging.info('Dropping Equal Columns')
    remove = []
    c = train.columns
    for i in range(len(c)):
        v = train[c[i]].values
        for j in range(i+1, len(c)):
            if np.array_equal(v, train[c[j]].values):
                remove.append(c[j])
    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)
    return train, test

def remove_constants(train, test):
    logging.info('Dropping Constant Columns')
    remove = []
    for col in train.columns:
        if train[col].std() == 0:
            remove.append(col)
    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)
    return train, test

def add_pca(train, test, n_components):
    logging.info('Adding PCA Columns')
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    features = train.columns
    pca = PCA(n_components=n_components)
    x_train_projected = pca.fit_transform(normalize(train[features], axis=0))
    x_test_projected = pca.transform(normalize(test[features], axis=0))
    for i in range(n_components):
        train.insert(1, 'PCA_{}'.format(i), x_train_projected[:, i])
        test.insert(1, 'PCA_{}'.format(i), x_test_projected[:, i])
    return train, test

def add_sum_zeros(train, test):
    logging.info('Adding Zero-Sum Column')
    cols = train.columns
    train.insert(1, 'Sum_Zeros', (train[cols] == 0).astype(int).sum(axis=1))
    test.insert(1, 'Sum_Zeros', (test[cols] == 0).astype(int).sum(axis=1))
    return train, test

def impute_scale(X):
    logging.info('Scaling Data')
    from sklearn.preprocessing import StandardScaler, Imputer
    X = Imputer().fit_transform(X)
    X = StandardScaler().fit_transform(X)
    return X

def label_encode(train, test):
    logging.info('Label Encoding')
    from sklearn.preprocessing import LabelEncoder
    X = pd.concat([train, test], axis=0)
    for c in X.select_dtypes(exclude=['int','float']).columns:
        X[c] = LabelEncoder().fit_transform(X[c].fillna('missing'))
    return X[:len(train)], X[len(train):]


def predict_rfc(train, test, labels, params, proba=True):
    logging.info('Training Random Forest')
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(**params)
    clf.fit(train, labels)
    if proba:
        return clf.predict_proba(train)[:, 1], clf.predict_proba(test)[:, 1]
    else:
        return clf.predict(train)[:, 1], clf.predict(test)[:, 1]

def predict_rfr(train, test, labels, params, proba=False):
    logging.info('Training Random Forest')
    from sklearn.ensemble import RandomForestRegressor
    clf = RandomForestRegressor(**params)
    clf.fit(train, labels)
    if proba:
        return clf.predict_proba(train), clf.predict_proba(test)
    else:
        return clf.predict(train), clf.predict(test)

def predict_gbc(train, test, labels, params, proba=True):
    logging.info('Training Gradient Boosting')
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(**params)
    clf.fit(train, labels)
    if proba:
        return clf.predict_proba(train)[:, 1], clf.predict_proba(test)[:, 1]
    else:
        return clf.predict(train)[:, 1], clf.predict(test)[:, 1]

def predict_gbr(train, test, labels, params, proba=False):
    logging.info('Training Gradient Boosting')
    from sklearn.ensemble import GradientBoostingRegressor
    clf = GradientBoostingRegressor(**params)
    clf.fit(train, labels)
    if proba:
        return clf.predict_proba(train), clf.predict_proba(test)
    else:
        return clf.predict(train), clf.predict(test)

def predict_xgb(train, test, labels, params, proba=True):
    logging.info('Training Extreme Gradient Boosting')
    params = params.copy()
    import xgboost as xgb
    dtrain = xgb.DMatrix(train, label=labels)
    dtest = xgb.DMatrix(test)
    num_round = params.pop('num_round')
    if not params.get('booster'):
        params['booster'] = 'gbtree'
    if not params.get('silent'):
        params['silent'] = 1
    bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_round)
    return bst.predict(dtrain), bst.predict(dtest)

def predict_nn(train, test, labels, params, proba=True):
    logging.info('Training Neural Network')
    params = params.copy()
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.utils import np_utils
    input_dim = train.shape[1]
    output_dim = params.pop('output_dim')
    init = params.pop('init')
    act = params.pop('activation')
    loss = params.pop('loss')
    opt = params.pop('optimizer')
    dropout = params.pop('dropout')
    nb_epoch = params.pop('nb_epoch')
    layers = params.pop('layers')
    model = Sequential()
    model.add(Dense(layers[0], input_dim=input_dim, init=init))
    model.add(Activation(act))
    model.add(Dropout(dropout))
    for nodes in layers[1:]:
        model.add(Dense(nodes))
        model.add(Activation(act))
        model.add(Dropout(dropout))
    model.add(Dense(output_dim))
    model.add(Activation('softmax')) # linear
    model.compile(loss=loss, optimizer=opt)
    if 'categorical' in loss:
        labels = np_utils.to_categorical(labels)
    model.fit(train, labels, nb_epoch=nb_epoch, batch_size=16, verbose=0)
    if proba:
        return model.predict_proba(train, verbose=0)[:, 1], model.predict_proba(test, verbose=0)[:, 1]
    else:
        return model.predict(train, verbose=0)[:, 0], model.predict(test, verbose=0)[:, 0]

def predict(model, params, X_train, X_test, y_train, proba):
    if model == 'rfc':
        pred_train, pred_test = predict_rfc(X_train, X_test, y_train, params, proba)
    elif model == 'rfr':
        pred_train, pred_test = predict_rfr(X_train, X_test, y_train, params, proba)
    elif model == 'gbc':
        pred_train, pred_test = predict_gbc(X_train, X_test, y_train, params, proba)
    elif model == 'gbr':
        pred_train, pred_test = predict_gbr(X_train, X_test, y_train, params, proba)
    elif model == 'xgb':
        pred_train, pred_test = predict_xgb(X_train, X_test, y_train, params, proba)
    elif model == 'nn':
        pred_train, pred_test = predict_nn(X_train, X_test, y_train, params, proba)
    return pred_train, pred_test
        
def blend_weights(scorer, labels, preds, greater_is_better=True):
    logging.info('Optimizing Weights')
    from itertools import permutations
    digits, n = 2, len(preds)
    weight_combinations = np.array([e for e in permutations(range(10**digits + 1), n) if sum(e) == 10**digits]) / 10**digits
    best_score = -10 if greater_is_better else 10
    best_weights = [1 / len(preds)] * len(preds)
    for w in weight_combinations:
        score = scorer(labels, np.dot(np.array(w), preds))
        if greater_is_better and score > best_score:
            best_score = score
            best_weights = w
        elif not greater_is_better and score < best_score:
            best_score = score
            best_weights = w
    logging.info('Blended Test Score: {:.4f} / Weights: {}'.format(best_score, best_weights))

def cv(train, labels, models, scorer, folds=5, stratified=False, shuffle=True, proba=True):
    logging.info('Cross Validation')
    from sklearn.cross_validation import StratifiedKFold, KFold
    n = 1
    if stratified:
        split = StratifiedKFold(labels, n_folds=folds, shuffle=shuffle)
    else:
        split = KFold(len(labels), n_folds=folds, shuffle=shuffle)
    for train_index, test_index in split:
        logging.info('Fold {}'.format(n))
        preds = []
        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        for model, params in models:
            pred_train, pred_test = predict(model, params, X_train, X_test, y_train, proba)
            train_score = scorer(y_train, pred_train)
            test_score = scorer(y_test, pred_test)
            logging.info('Train Score: {:.4f} / Test Score: {:.4f}'.format(train_score, test_score))
            preds.append(pred_test)
        if len(preds) > 1:
            blend_weights(scorer, y_test, preds)
        n += 1

def submission(train, test, labels, sub_ix, sub_col, models, weights, scorer, proba=True):
    logging.info('Creating Submission')
    if len(weights) != len(models):
        raise ValueError('Number of Models ({}) mismatches number of Weights ({})'.format(len(models), len(weights)))
    preds = []
    for model, params in models:
        pred_train, pred_test = predict(model, params, train, test, labels, proba)
        logging.info('Train Score: {:.4f}'.format(scorer(labels, pred_train)))
        preds.append(pred_test)
    result = pd.DataFrame(sub_ix)
    result[sub_col] = np.dot(np.array(weights), preds)
    # result[sub_col][result[sub_col] > 3] = 3.
    result.to_csv('submission_new.csv', index=False)

def main(submit=False):
    logging.info('Starting')
    
    scorer = roc_auc_score
    shuffle = True
    
    rfc_params = {
        'n_estimators': 500,                # 10
        'criterion': 'gini',                # gini, entropy
        'max_features': 'auto',             # auto, sqrt, log2, None, int, float
        'max_depth': None,                  # None, int
        'min_samples_split': 2,             # 2, int
        'min_samples_leaf': 1,              # 1, int
        'min_weight_fraction_leaf': 0.,     # 0., float
        'max_leaf_nodes': None,             # None, int
        'bootstrap': True,                  # True, False
        'oob_score': False,                 # False, True
        'random_state': None,               # None, int
        'n_jobs': -1,
    }

    rfr_params = {
        'n_estimators': 100,                # 10
        'criterion': 'mse',                 # gini, entropy
        'max_features': 'auto',             # auto, sqrt, log2, None, int, float
        'max_depth': None,                  # None, int
        'min_samples_split': 2,             # 2, int
        'min_samples_leaf': 1,              # 1, int
        'min_weight_fraction_leaf': 0.,     # 0., float
        'max_leaf_nodes': None,             # None, int
        'bootstrap': True,                  # True, False
        'oob_score': False,                 # False, True
        'random_state': None,               # None, int
        'n_jobs': -1,
    }
    
    gbc_params = {
        'loss': 'deviance',                 # deviance, exponential
        'learning_rate': 0.07,              # 0.1, float
        'n_estimators': 100,                # 100, int
        'max_depth': 5,                     # 3, int
        'min_samples_split': 2,             # 2, int
        'min_samples_leaf': 1,              # 1, int
        'min_weight_fraction_leaf': 0.,     # 0., float
        'max_features': None,               # None, auto, sqrt, log2, int, float
        'max_leaf_nodes': None,             # None, int
        'subsample': 1,                     # 1., float
        'presort': 'auto',                  # auto, True, False
        'random_state': None,               # None, int
    }
    
    gbr_params = {
        'loss': 'ls',                       # deviance, exponential
        'learning_rate': 0.07,              # 0.1, float
        'n_estimators': 30,                 # 100, int
        'max_depth': 5,                     # 3, int
        'min_samples_split': 2,             # 2, int
        'min_samples_leaf': 1,              # 1, int
        'min_weight_fraction_leaf': 0.,     # 0., float
        'max_features': None,               # None, auto, sqrt, log2, int, float
        'max_leaf_nodes': None,             # None, int
        'subsample': 1,                     # 1., float
        'presort': 'auto',                  # auto, True, False
        'random_state': None,               # None, int
    }
    
    xgb_params = {
        'objective': 'binary:logistic',     # reg:linear, reg:logistic, binary:logistic, binary:logitraw
                                            # count:poisson, multi:softmax, multi:softprob, rank:pairwise
        'num_round': 350,                   # 
        'eta': 0.03,                        # .3, float (0-1)
        'gamma': 0,                         # 0, (0-inf)
        'max_depth': 5,                     # 6, (1-inf)
        'min_child_weight': 1,              # 1, (0-inf)
        'max_delta_step': 0,                # 0, (0-inf)
        'subsample': 0.8,                   # 1., (0.-1.)
        'colsample_bytree': 0.7,            # 1., (0.-1.)
        'colsample_bylevel': 1,             # 1., (0.-1.)
        'lambda': 1,                        # 1
        'alpha': 0,                         # 0
        'tree_method': 'auto',              # auto, exact, approx
        
    }
    
    nn_params = {
        'layers': [128, 64, 32],            # list of int
        'output_dim': 2,                    # 1 for regr, no. classes for classification
        'nb_epoch': 150,                    # int
        'dropout': 0.1,                     # float
        'loss': 'categorical_crossentropy', # mse, mae, mape, msle, binary_crossentropy, hinge
        'optimizer': 'adamax',              # sgd, rmsprop, adagrad, adadelta, adam, adamax
        'init': 'glorot_uniform',           # zero, normal, glorot_normal, he_normal
        'activation': 'relu',               # tanh, relu, linear, sigmoid, softmax, softplus
    }
    
    models = [('rfc', rfc_params), ('gbc', gbc_params), ('xgb', xgb_params), ('nn', nn_params)]
    weights = [0.25, 0.25, 0.25, 0.25]
    
    features = ['var38', 'var15', 'saldo_medio_var5_hace2', 'saldo_var30',
    'saldo_medio_var5_ult1', 'saldo_medio_var5_hace3', 'num_var45_hace3',
    'saldo_medio_var5_ult3', 'num_var22_ult3', 'num_var22_hace3', 'num_var45_hace2',
    'saldo_var5', 'imp_op_var41_ult1', 'imp_op_var41_efect_ult3', 'num_var45_ult3', 
    'num_var22_ult1', 'imp_op_var41_efect_ult1', 'num_meses_var39_vig_ult3',
    'num_var22_hace2', 'num_var45_ult1', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1',
    'var3', 'imp_op_var39_comer_ult3', 'saldo_var37', 'imp_op_var39_ult1', 'var36',
    'num_med_var45_ult3', 'num_op_var41_efect_ult1', 'num_op_var41_ult3',
    'imp_trans_var37_ult1', 'imp_var43_emit_ult1', 'num_var42_0', 'saldo_var8',
    'ind_var8_0', 'num_op_var41_efect_ult3', 'num_ent_var16_ult1', 'saldo_var26',
    'num_op_var41_hace2', 'num_meses_var5_ult3', 'imp_op_var41_comer_ult1',
    'saldo_medio_var8_ult1', 'imp_sal_var16_ult1', 'num_op_var41_hace3', 'ind_var5_0',
    'num_op_var41_ult1', 'num_var5_0', 'num_var37_0', 'num_op_var39_comer_ult1', 'num_var4', 
    'num_var43_emit_ult1', 'num_op_var41_comer_ult1', 'num_var37_med_ult2', 'saldo_var31',
    'ind_var39_0', 'num_op_var39_comer_ult3', 'saldo_var42', 'imp_op_var41_comer_ult3',
    'num_op_var39_efect_ult1', 'ind_var30', 'num_op_var39_ult1', 'ind_var1_0',
    'saldo_medio_var8_hace3', 'num_op_var40_comer_ult3', 'num_op_var41_comer_ult3',
    'num_var43_recib_ult1']
    
    train, test, labels, sub_ix, sub_col = load_data()
    train = train[features]
    test = test[features]
    train, test = remove_equals(train, test)
    train, test = remove_constants(train, test)
    train, test = add_sum_zeros(train, test)
    train, test = add_pca(train, test, 2)
    train, test = label_encode(train, test)
    train, test = impute_scale(train), impute_scale(test)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if submit:
            submission(train, test, labels, sub_ix, sub_col, models, weights, scorer, proba=True)
        else:
            cv(train, labels, models, scorer, shuffle=shuffle, stratified=False, proba=True)

if __name__ == '__main__':
    args = sys.argv
    submit = 'submit' in args
    main(submit)
