import numpy as np
import pandas as pd
import feather
from glob import glob
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from sklearn.metrics import matthews_corrcoef

from keras import backend as K

def matthews_correlation(y_true, y_pred):
    ''' Matthews correlation coefficient
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(1 - y_neg * y_pred_pos)
    fn = K.sum(1 - y_pos * y_pred_neg)
    
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

weights = {0: 1.0058451754997162, 1: 172.0812618113098}
input_dim = 4264

output_dim = 1                    # 1 for regr and binary class., no. classes for multi-class
layers = [128, 64, 32]            # list of int
dropout = 0.5                     # float
loss = 'binary_crossentropy'      # mse, mae, mape, msle, categorical_crossentropy, hinge
opt = 'rmsprop'                   # sgd, rmsprop, adagrad, adadelta, adam, adamax
init = 'zero'                     # zero, normal, glorot_normal, he_normal, glorot_uniform
act = 'relu'                      # tanh, relu, linear, sigmoid, softmax, softplus

def create_model1():
    model = Sequential()
    model.add(Dense(layers[0], input_dim=input_dim, init=init))
    model.add(Activation(act))
    model.add(Dropout(dropout))
    for nodes in layers[1:]:
        model.add(Dense(nodes))
        model.add(Activation(act))
        model.add(Dropout(dropout))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))  # linear
    model.compile(optimizer=opt, loss=loss, metrics=[matthews_correlation])
    return model

def train_generator():
    for n, f in enumerate(glob('data/train_*.f')):
        if (n % 4 != 0):
            X = feather.read_dataframe(f)
            y = X.pop('TARGET').values
            yield X, y

def test_generator():
    for n, f in enumerate(glob('data/train_*.f')):
        if (n % 4 == 0):
            X = feather.read_dataframe(f)
            y = X.pop('TARGET').values
            yield X, y

def pred_generator():
    for f in glob('data/test_*.f'):
        X = feather.read_dataframe(f)
        yield X

def test_index():
    return pd.read_csv('data/train_numeric.csv.zip', usecols=['Id'])['Id']

print('building model ...')
model = create_model1()

# training
print('training ...')
model.fit_generator(train_generator(), 
                    nb_epoch=15, #150, 
                    samples_per_epoch=300000, #800000
                    class_weight=weights)

# scoring
scores = []
print('scoring test data ', end='')
for X_test, y_test in test_generator():
    print('.', end='')
    y_pred = model.predict_on_batch(X_test) #[:, 0]
    scores.append(matthews_corrcoef(y_test, y_pred))
print()
print('mean score:', np.mean(scores))

# prediction
if False:
    y_pred = pd.Series()
    print('predicting chunks ', end='')
    for X_test in pred_generator():
        print('.', end='')
        y_pred = y_pred.append(pd.Series(model.predict_on_batch(X_test)[:, 0]))
    print()
    y_pred.index = test_index()
    y_pred.to_csv('data/submission03.csv', index=True)
    print('predicting', y_pred.sum(), 'of', len(y_pred), 'as class 1', y_pred.sum()/len(y_pred))
