"""Neural experiments requested by the JADS reviewers (manuscript #1664).

Same data snapshot, chronological split, Sharpe-type loss, callbacks and scaling as the published
notebook (lstm_core.py, unchanged). New here:
  arch : LSTM, GRU, MLP and a linear softmax policy, 10 seeds each, published split (final 100 returns)
  wf   : walk-forward retraining, model refitted every 20 test returns on all earlier data, 5 seeds per model
  pen  : LSTM trained with a turnover penalty in the loss, lambda in {0.1, 0.3, 1.0}, 10 seeds each
  ref  : LSTM, seed 42, published split (reproduction check of the submitted run)
  size : LSTM and GRU with 16 or 64 hidden units, 5 seeds each (sensitivity to network size)
Each fit saves its test-period target weights and a metadata row; jads_evaluate.py does the accounting.
Usage: python jads_neural.py [arch|wf|pen|ref|size|all]
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('OMP_NUM_THREADS', '1')

OUT = os.environ.get('JADS_OUT', 'neural')
os.makedirs(OUT, exist_ok=True)
TEST = 100
BLOCK = 20


def fit_predict(kind, split_idx, horizon, seed, lam=0.0, lookback=20, units=32, dropout=0.1, lr=1e-3, epochs=20, batch=64, save_to=None):
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import lstm_core as core

    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    returns = core.returns
    n_assets = returns.shape[1]
    train_raw = returns.iloc[:split_idx].copy()
    test_raw = returns.iloc[split_idx:split_idx + horizon].copy()

    # scaling fitted on rows before the first validation target, as in the notebook
    n_seq = len(train_raw) - lookback
    val_size = max(1, int(0.2 * n_seq))
    fit_end = len(train_raw) - val_size
    _, train_scaled, mu, sigma = core.standardize_train_test(train_raw.iloc[:fit_end], train_raw)
    test_scaled = ((test_raw - mu) / (sigma + 1e-8)).astype(np.float32)
    X_full, _ = core.create_sequence_dataset(train_scaled.values, lookback)
    y_full = train_raw.values[lookback:].astype(np.float32)
    val_size = max(1, int(0.2 * len(X_full)))
    X_tr, y_tr, X_val, y_val = X_full[:-val_size], y_full[:-val_size], X_full[-val_size:], y_full[-val_size:]

    layers = [Input(shape=(lookback, n_assets))]
    if kind == 'lstm':
        layers += [LSTM(units), Dropout(dropout)]
    elif kind == 'gru':
        layers += [GRU(units), Dropout(dropout)]
    elif kind == 'mlp':
        layers += [Flatten(), Dense(units, activation='relu'), Dropout(dropout)]
    elif kind == 'linear':
        layers += [Flatten()]
    else:
        raise ValueError(kind)
    layers += [Dense(n_assets, activation='softmax')]
    model = Sequential(layers)

    def loss(y_true, y_pred):
        port = tf.reduce_sum(y_true * y_pred, axis=1)
        sharpe = tf.reduce_mean(port) / (tf.math.reduce_std(port) + 1e-8)
        if lam == 0.0:
            return -sharpe
        # mean absolute change of consecutive target weights inside the ordered batch
        turn = tf.reduce_mean(tf.reduce_sum(tf.abs(y_pred[1:] - y_pred[:-1]), axis=1))
        return -sharpe + lam * turn

    model.compile(optimizer=Adam(learning_rate=lr), loss=loss)
    cb = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
          ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)]
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch,
                     shuffle=False, callbacks=cb, verbose=0)

    full_for_test = pd.concat([train_scaled.iloc[-lookback:], test_scaled], axis=0)
    X_test, _ = core.create_sequence_dataset(full_for_test.values, lookback)
    if save_to:
        model.save_weights(save_to)
    w = model.predict(X_test[:horizon], verbose=0)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
    vl = hist.history['val_loss']
    return w, dict(epochs_run=len(vl), best_epoch=int(np.argmin(vl)) + 1, best_val_loss=float(np.min(vl)),
                   params=int(model.count_params()), loss=[float(x) for x in hist.history['loss']], val_loss=[float(x) for x in vl])


def task(spec):
    import numpy as np
    t0 = time.time()
    tag = spec['tag']
    w, meta = fit_predict(spec['kind'], spec['split_idx'], spec['horizon'], spec['seed'], spec.get('lam', 0.0),
                          units=spec.get('units', 32), save_to=os.path.join(OUT, tag + '.weights.h5'))
    np.save(os.path.join(OUT, tag + '.npy'), w)
    meta.update(spec, seconds=round(time.time() - t0, 1))
    with open(os.path.join(OUT, tag + '.json'), 'w') as f:
        json.dump(meta, f)
    return tag


def specs(which):
    import lstm_core as core
    n = len(core.returns)
    s0 = n - TEST
    out = []
    if which in ('ref', 'all'):
        out.append(dict(tag='ref_lstm_s42', exp='ref', kind='lstm', seed=42, split_idx=s0, horizon=TEST))
    if which in ('arch', 'all'):
        for kind in ('lstm', 'gru', 'mlp', 'linear'):
            for seed in range(1, 11):
                out.append(dict(tag=f'arch_{kind}_s{seed}', exp='arch', kind=kind, seed=seed, split_idx=s0, horizon=TEST))
    if which in ('wf', 'all'):
        for kind in ('lstm', 'gru', 'mlp', 'linear'):
            for seed in range(1, 6):
                for b in range(TEST // BLOCK):
                    out.append(dict(tag=f'wf_{kind}_s{seed}_b{b}', exp='wf', kind=kind, seed=seed, block=b,
                                    split_idx=s0 + b * BLOCK, horizon=BLOCK))
    if which in ('pen', 'all'):
        for lam in (0.1, 0.3, 1.0):
            for seed in range(1, 11):
                out.append(dict(tag=f'pen_lstm_l{lam}_s{seed}', exp='pen', kind='lstm', seed=seed, lam=lam, split_idx=s0, horizon=TEST))
    if which in ('size', 'all'):
        for kind in ('lstm', 'gru'):
            for units in (16, 64):
                for seed in range(1, 6):
                    out.append(dict(tag=f'size_{kind}_u{units}_s{seed}', exp='size', kind=kind, units=units, seed=seed, split_idx=s0, horizon=TEST))
    return [s for s in out if not os.path.exists(os.path.join(OUT, s['tag'] + '.json'))]


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    todo = specs(which)
    print(f'{len(todo)} fits to run', flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, tag in enumerate(ex.map(task, todo), 1):
            if i % 10 == 0 or i == len(todo):
                print(f'{i}/{len(todo)} done ({tag})', flush=True)
