import optuna, json, os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from data import load_cifar10
from model import build_cnn

def objective(trial):
    # Suggest hyperparameters
    n_conv_blocks = trial.suggest_int('n_conv_blocks', 1, 3)
    conv_filters = []
    for i in range(n_conv_blocks):
        conv_filters.append(trial.suggest_categorical(f'filters_{i}', [16, 32, 48, 64]))
    kernel_size = trial.suggest_categorical('kernel_size', [3,5])
    dense_units = trial.suggest_categorical('dense_units', [64, 128, 256])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.6)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    batch_size = trial.suggest_categorical('batch_size', [64, 128])
    epochs = 10  # short by default for tuning; increase for final training

    params = {
        'conv_filters': conv_filters,
        'kernel_size': kernel_size,
        'dense_units': dense_units,
        'dropout_rate': dropout_rate,
        'lr': lr,
        'optimizer': optimizer
    }

    x_train, y_train, x_test, y_test = load_cifar10()

    # Use a small validation split to speed up tuning
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    split = int(0.9 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    x_tr, y_tr = x_train[train_idx], y_train[train_idx]
    x_val, y_val = x_train[val_idx], y_train[val_idx]

    model = build_cnn(input_shape=x_train.shape[1:], num_classes=y_train.shape[1], params=params)
    es = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=0)
    history = model.fit(x_tr, y_tr, validation_data=(x_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)

    val_acc = max(history.history['val_accuracy'])
    # Optuna tries to **maximize** the objective by default
    return val_acc

def run_optuna(n_trials=20, storage=None, study_name='optuna_bo_study', seed=42):
    if storage is None:
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    else:
        study = optuna.create_study(direction='maximize', storage=storage, study_name=study_name, load_if_exists=True,
                                    sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    # Save best params
    with open('best_params_optuna.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print('Best value:', study.best_value)
    print('Best params saved to best_params_optuna.json')
    return study
