import pandas as pd
import numpy as np
import feather
import logging
import scorers
from datetime import datetime

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%y/%m/%d %H:%M:%S', level=logging.DEBUG)

class Dataset(object):
    def __init__(self, train, test, labels):
        for name, val in [('train', train), ('test', test), ('labels', labels)]:
            test = False
            if isinstance(val, pd.DataFrame):
                setattr(self, name, val)
                test = True
            else:
                if isinstance(val, str):
                    result = feather.read_dataframe(val)
                    if isinstance(result, pd.DataFrame):
                        setattr(self, name, result)
                        test = True
                else:
                    df = pd.DataFrame(val)
                    if isinstance(df, pd.DataFrame):
                        setattr(self, name, df)
                        test = True
            if not test:
                raise Exception('Invalid source', name,
                    '- must be DataFrame or path to feather file.')
        if len(self.train.columns) != len(self.test.columns):
            raise Exception('Different number of Train and Test columns.')
        if len(self.train) != len(self.labels):
            print(self.train)
            print(self.labels)
            raise Exception('Different number of Train and Label samples.')
        self.test_ix = self.test.columns[0]
        self.labels_col = self.labels.columns[-1]
        self.labels = self.labels[self.labels_col].values
        
    def __repr__(self):
        return '<Dataset train: {}, test: {}, labels: {}>'.format(self.train.shape,
            self.test.shape, self.labels.shape)


class Project(object):
    def __init__(self, name, data, method, scorer, proba=True, shuffle=True, stratified=False):
        self.name = name
        
        if not isinstance(data, Dataset):
            raise Exception('Invalid data', repr(data),
                '- must be type Dataset')
        self.data = data
        
        method = method.lower()
        if not method in ['classification', 'regression']:
            raise Exception('Invalid method', method,
                '- must be classification or regression')
        self.method = method
        self.classification = method == 'classification'
        self.regression = method == 'regression'
        
        self.scorer = scorer
        self.proba = proba
        self.shuffle = shuffle
        self.stratified = stratified
        
    def __repr__(self):
        return '<Project {} data: {}, method: {}>'.format(self.name, repr(self.data), 
            self.method)
    
    def preprocess(self, pipeline):
        for task in pipeline:
            self.train, self.test = prep(self.train, self.test, task)
        
    def cv(self, models, folds=5):
        logging.info('Cross Validation')
        from sklearn.cross_validation import StratifiedKFold, KFold
        n = 1
        if self.stratified:
            split = StratifiedKFold(self.data.labels, n_folds=folds, shuffle=self.shuffle)
        else:
            split = KFold(len(self.data.labels), n_folds=folds, shuffle=self.shuffle)
        for train_index, test_index in split:
            logging.info('Fold {}'.format(n))
            preds = []
            X_train, X_test = self.data.train[train_index], self.data.train[test_index]
            y_train, y_test = self.data.labels[train_index], self.data.labels[test_index]
            for model, params in models:
                pred_train, pred_test = predict(model, params, X_train, X_test, y_train, 
                    self.proba)
                train_score = self.scorer(y_train, pred_train)
                test_score = self.scorer(y_test, pred_test)
                logging.info('Train Score: {:.4f} / Test Score: {:.4f}'.format(
                    train_score, test_score))
                preds.append(pred_test)
            if len(preds) > 1:
                blend_weights(self.scorer, y_test, preds)
            n += 1
    
    def submit(self, models, weights, value_range=False):
        logging.info('Creating Submission')
        if len(weights) != len(models):
            raise ValueError(
                'Number of Models ({}) mismatches number of Weights ({})'.format(
                len(models), len(weights)))
        preds = []
        for model, params in models:
            pred_train, pred_test = predict(model, params, self.data.train, self.data.test, 
                self.data.labels, self.data.proba)
            logging.info('Train Score: {:.4f}'.format(self.scorer(self.data.labels, 
                pred_train)))
            preds.append(pred_test)
        result = pd.DataFrame(self.data.test_ix)
        result[self.data.labels_col] = np.dot(np.array(weights), preds)
        if value_range:
            vr = value_range
            result[self.data.labels_col][result[self.data.labels_col] < vr[0]] = vr[0]
            result[self.data.labels_col][result[self.data.labels_col] > vr[1]] = vr[1]
        fn = 'subm_{}_{}.csv'.format(self.name, datetime.now().strftime('%y%m%d-%H%M'))
        result.to_csv(fn, index=False)


### PREPROCESSING ###

def remove_duplicates(train, test):
    logging.info('Dropping Duplicate Columns')
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

def add_sum_col(train, test, value):
    logging.info('Adding Sum Column')
    cols = train.columns
    train.insert(1, 'Sum_Col', (train[cols] == value).astype(int).sum(axis=1))
    test.insert(1, 'Sum_Col', (test[cols] == value).astype(int).sum(axis=1))
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

def prep(train, test, task):
    if task == 'remove_duplicates':
        train, test = remove_duplicates(train, test)
    elif task == 'remove_constants':
        train, test = remove_constants(train, test)
    elif task == 'add_sum_col':
        train, test = add_sum_col(train, test, 0)
    elif task == 'add_pca':
        train, test = add_pca(train, test, 2)
    elif task == 'label_encode':
        train, test = label_encode(train, test)
    elif task == 'impute_scale':
        train, test = impute_scale(train), impute_scale(test)
    return train, test

### PREDICTION ###

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

### BLENDING ###

def blend_weights(scorer, labels, preds, digits=2, greater_is_better=True):
    logging.info('Optimizing Weights')
    from itertools import permutations
    n = len(preds)
    weight_combinations = np.array([e for e in permutations(range(10**digits + 1), n) if sum(e) == 10**digits]) / 10**digits
    best_score = -10 if greater_is_better else 10
    best_weights = [1/n] * n
    for w in weight_combinations:
        score = scorer(labels, np.dot(np.array(w), preds))
        if greater_is_better and score > best_score:
            best_score = score
            best_weights = w
        elif not greater_is_better and score < best_score:
            best_score = score
            best_weights = w
    logging.info('Blended Test Score: {:.4f} / Weights: {}'.format(best_score, best_weights))
