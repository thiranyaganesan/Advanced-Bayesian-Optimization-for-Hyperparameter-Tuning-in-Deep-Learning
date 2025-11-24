# Optional example: scikit-optimize (skopt) based tuning
# Usage: install scikit-optimize and sklearn, then adapt and run.
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
import numpy as np
from data import load_cifar10
from model import build_cnn

def _train_eval(params):
    # params ordering: n_conv_blocks, kernel_size, dense_units, dropout_rate, lr, batch_size
    n_conv_blocks = params[0]
    conv_filters = [params[1 + i] for i in range(n_conv_blocks)]
    kernel_size = params[4]
    dense_units = params[5]
    dropout_rate = params[6]
    lr = params[7]
    batch_size = params[8]
    # Build param dict and train for a few epochs
    param_dict = {
        'conv_filters': conv_filters,
        'kernel_size': kernel_size,
        'dense_units': dense_units,
        'dropout_rate': dropout_rate,
        'lr': lr,
        'optimizer': 'adam'
    }
    x_train, y_train, x_test, y_test = load_cifar10()
    # small split
    idx = np.arange(len(x_train))
    np.random.shuffle(idx)
    tr, val = idx[:45000], idx[45000:50000]
    x_tr, y_tr = x_train[tr], y_train[tr]
    x_val, y_val = x_train[val], y_train[val]
    model = build_cnn(input_shape=x_train.shape[1:], num_classes=y_train.shape[1], params=param_dict)
    model.fit(x_tr, y_tr, validation_data=(x_val, y_val), epochs=8, batch_size=batch_size, verbose=0)
    val_acc = model.evaluate(x_val, y_val, verbose=0)[1]
    # gp_minimize minimizes, so return negative accuracy
    return -val_acc

def run_skopt(n_calls=20):
    space = [
        Integer(1, 3, name='n_conv_blocks'),
        Categorical([16,32,48,64], name='f0'),
        Categorical([16,32,48,64], name='f1'),
        Categorical([16,32,48,64], name='f2'),
        Categorical([3,5], name='kernel_size'),
        Categorical([64,128,256], name='dense_units'),
        Real(0.2, 0.6, name='dropout_rate'),
        Real(1e-4, 1e-2, 'log-uniform', name='lr'),
        Categorical([64,128], name='batch_size')
    ]
    res = gp_minimize(_train_eval, space, n_calls=n_calls, random_state=42)
    print('Best negative accuracy:', res.fun)
    print('Best params:', res.x)
    return res
