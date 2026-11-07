"""Accounting for the JADS revision: classical benchmarks, neural runs, cap and lag variants.

Same returns, test window (final 100 returns) and per-period cost simulation as the published
notebook (simulate_portfolio copied unchanged from lstm_core.py). Run after jads_neural.py.
Usage: py jads_evaluate.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ASSETS = ['CFG', 'ATW', 'BCP', 'CIH', 'BOA', 'BCI']
P = pd.read_csv('prices_snapshot.csv', index_col=0, parse_dates=True)[ASSETS]
R = P.pct_change().dropna().astype(np.float32)
n, k = R.shape
TEST = 100
split = n - TEST
r = R.iloc[split:].values.astype(np.float64)
os.makedirs('results', exist_ok=True)


def simulate_portfolio(asset_returns, target_weights, cost_bps=0.0):
    """Unchanged from lstm_core.py: drift-aware turnover, self-financing proportional costs."""
    r_ = np.asarray(asset_returns, dtype=np.float64)
    w = np.asarray(target_weights, dtype=np.float64)
    w = np.maximum(w, 0)
    w /= w.sum(axis=1, keepdims=True)
    c = cost_bps / 10000.0
    previous = np.zeros(w.shape[1])
    net, turnover = [], []
    for target, ret in zip(w, r_):
        low, high = 0.0, 1.0
        for _ in range(60):
            q = (low + high) / 2
            if q + c * np.abs(q * target - previous).sum() > 1:
                high = q
            else:
                low = q
        q = (low + high) / 2
        traded = np.abs(q * target - previous).sum()
        gross = np.dot(target, ret)
        net.append(q * (1 + gross) - 1)
        turnover.append(traded)
        previous = target * (1 + ret) / (1 + gross)
    return np.array(net), np.array(turnover)


def cum(x):
    return float(np.prod(1 + x) - 1)


def solve_cost(f, lo=0.0, hi=300.0):
    """Smallest one-way cost (bps) at which f(cost) <= 0; nan if never within [lo, hi]."""
    if f(lo) <= 0:
        return 0.0
    if f(hi) > 0:
        return float('nan')
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def metrics(w, ret=r, ref=None):
    g, tau = simulate_portfolio(ret, w, 0.0)
    W = np.cumprod(1 + g)
    dd = W / np.maximum.accumulate(np.r_[1.0, W])[1:] - 1
    out = dict(cum=cum(g), sharpe=float(np.sqrt(252) * g.mean() / g.std()), vol=float(g.std() * np.sqrt(252)),
               mdd=float(dd.min()), worst=float(g.min()), turnover=float(tau.mean()),
               max_weight=float(np.max(w)), mean_hhi=float(np.mean((w ** 2).sum(axis=1))),
               sortino=float(np.sqrt(252) * g.mean() / (np.sqrt(np.mean(np.minimum(g, 0) ** 2)) + 1e-12)))
    for bps in (10, 25, 50):
        out[f'net{bps}'] = cum(simulate_portfolio(ret, w, bps)[0])
    out['be_zero'] = solve_cost(lambda c: cum(simulate_portfolio(ret, w, c)[0]))
    if ref is not None:
        for name, wr in ref.items():
            out[f'be_vs_{name}'] = solve_cost(lambda c: cum(simulate_portfolio(ret, w, c)[0]) - cum(simulate_portfolio(ret, wr, c)[0]))
            out[f'sharpe_gap_{name}'] = out['sharpe'] - metrics(wr, ret)['sharpe']
            out[f'gap_{name}'] = out['cum'] - cum(simulate_portfolio(ret, wr, 0.0)[0])
    return out


# ---------------- classical benchmarks (estimated from returns before each target date) ----------------
def heuristic_mk(h):
    mu = h.mean(axis=0); S = np.cov(h, rowvar=False) + 1e-5 * np.eye(k)
    try:
        w = np.linalg.solve(S, mu)
    except np.linalg.LinAlgError:
        w = mu.copy()
    w = np.maximum(w, 0); s = w.sum()
    return np.ones(k) / k if s <= 1e-12 else w / s


def ledoit_wolf(h):
    X = h - h.mean(axis=0); T, N = X.shape
    emp = X.T @ X / T
    mu = np.trace(emp) / N
    X2 = X ** 2
    beta_ = (X2.T @ X2).sum() / T
    delta_ = ((X.T @ X) ** 2).sum() / T ** 2
    beta = (beta_ / T - delta_ / T) / N
    delta = (delta_ - 2 * mu * np.trace(emp) + N * mu ** 2) / N
    s = 0.0 if delta == 0 else min(beta, delta) / delta
    return (1 - s) * emp + s * mu * np.eye(N), s


def long_only_gmv(S):
    res = minimize(lambda w: w @ S @ w, np.ones(k) / k, method='SLSQP', bounds=[(0, 1)] * k,
                   constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}], options={'ftol': 1e-14, 'maxiter': 500})
    w = np.maximum(res.x, 0)
    return w / w.sum()


def exact_tangency(h):
    mu = h.mean(axis=0); S = np.cov(h, rowvar=False) + 1e-5 * np.eye(k)
    if mu.max() <= 0:
        return np.ones(k) / k
    best = None
    for x0 in [np.ones(k) / k] + [np.eye(k)[i] * 0.94 + 0.01 for i in range(k)]:
        res = minimize(lambda w: -(w @ mu) / np.sqrt(w @ S @ w), x0, method='SLSQP', bounds=[(0, 1)] * k,
                       constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}], options={'ftol': 1e-12, 'maxiter': 500})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    w = np.maximum(best.x, 0)
    return w / w.sum()


A = R.values.astype(np.float64)
bench = {'EW': [], 'MK': [], 'MK_exact': [], 'MinVar_LW': [], 'InvVol': [], 'Momentum': []}
shrink = []
for t in range(split, n):
    h = A[:t]
    bench['EW'].append(np.ones(k) / k)
    bench['MK'].append(heuristic_mk(h))
    bench['MK_exact'].append(exact_tangency(h))
    S_lw, s = ledoit_wolf(h); shrink.append(s)
    bench['MinVar_LW'].append(long_only_gmv(S_lw))
    vol = A[t - 60:t].std(axis=0)
    bench['InvVol'].append((1 / vol) / (1 / vol).sum())
    mom = np.prod(1 + A[t - 20:t], axis=0) - 1
    pos = np.maximum(mom, 0)
    bench['Momentum'].append(pos / pos.sum() if pos.sum() > 0 else np.ones(k) / k)
bench = {name: np.array(w) for name, w in bench.items()}
check = pd.read_csv('../corrected_lstm_test_daily_returns.csv')['Markowitz'].values
assert np.abs(simulate_portfolio(r, bench['MK'])[0] - check).max() < 1e-6, 'Markowitz heuristic does not reproduce the paper'

refs = {'EW': bench['EW'], 'MK': bench['MK']}
rows = []
for name, w in bench.items():
    rows.append(dict(exp='bench', model=name, seed=None, variant='base', **metrics(w, ref=refs)))

# ---------------- neural runs ----------------
meta = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob('neural/*.json')}


def cap_weights(w, cap=0.40):
    w = w.copy()
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w - cap).clip(min=0).sum(axis=1, keepdims=True)
        w = np.minimum(w, cap)
        free = (w < cap - 1e-12)
        share = np.where(free, w, 0)
        share = share / np.maximum(share.sum(axis=1, keepdims=True), 1e-12)
        w = w + excess * share
    return w


def add_run(exp, model, seed, w, extra=None):
    rows.append(dict(exp=exp, model=model, seed=seed, variant='base', **metrics(w, ref=refs), **(extra or {})))
    if model == 'lstm' and exp in ('arch', 'ref'):
        rows.append(dict(exp=exp, model=model, seed=seed, variant='cap40', **metrics(cap_weights(w), ref=refs)))
        # one additional observation of execution lag: weights decided for t are applied at t+1
        rows.append(dict(exp=exp, model=model, seed=seed, variant='lag1',
                         **metrics(w[:-1], ret=r[1:], ref={nm: wr[:-1] for nm, wr in refs.items()})))
        rows.append(dict(exp=exp, model=model, seed=seed, variant='nolag_same99',
                         **metrics(w[1:], ret=r[1:], ref={nm: wr[1:] for nm, wr in refs.items()})))


for tag, m in meta.items():
    if m['exp'] in ('arch', 'ref'):
        add_run(m['exp'], m['kind'], m['seed'], np.load(f'neural/{tag}.npy'),
                dict(params=m['params'], epochs_run=m['epochs_run'], best_epoch=m['best_epoch']))
    elif m['exp'] == 'size':
        add_run('size', f"{m['kind']}_u{m['units']}", m['seed'], np.load(f'neural/{tag}.npy'),
                dict(params=m['params'], epochs_run=m['epochs_run'], best_epoch=m['best_epoch']))
    elif m['exp'] == 'pen':
        add_run('pen', f"lstm_lam{m['lam']}", m['seed'], np.load(f'neural/{tag}.npy'),
                dict(params=m['params'], epochs_run=m['epochs_run'], best_epoch=m['best_epoch']))
for kind in ('lstm', 'gru', 'mlp', 'linear'):
    for seed in range(1, 6):
        tags = [f'wf_{kind}_s{seed}_b{b}' for b in range(TEST // 20)]
        if all(t in meta for t in tags):
            w = np.vstack([np.load(f'neural/{t}.npy') for t in tags])
            add_run('wf', kind, seed, w, dict(epochs_run=np.mean([meta[t]['epochs_run'] for t in tags])))

df = pd.DataFrame(rows)
df.to_csv('results/all_runs.csv', index=False)

# ---------------- summaries ----------------
def summarise(g):
    return pd.Series({
        'fits': len(g),
        'cum_median': g['cum'].median(), 'cum_min': g['cum'].min(), 'cum_max': g['cum'].max(),
        'sharpe_median': g['sharpe'].median(),
        'ahead_of_EW_and_MK': int(((g['gap_EW'] > 0) & (g['gap_MK'] > 0)).sum()),
        'sharpe_gap_MK_median': g['sharpe_gap_MK'].median(),
        'turnover_median': g['turnover'].median(), 'max_weight_median': g['max_weight'].median(),
        'net25_median': g['net25'].median(), 'be_zero_median': g['be_zero'].median(),
        'be_vs_EW_median': g['be_vs_EW'].median(), 'be_vs_MK_median': g['be_vs_MK'].median(),
    })


summary = df.groupby(['exp', 'model', 'variant']).apply(summarise, include_groups=False).reset_index()
summary.to_csv('results/summary.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(summary.round(4).to_string(index=False))
print('Ledoit-Wolf shrinkage intensity over the test: min %.3f, median %.3f, max %.3f' % (min(shrink), np.median(shrink), max(shrink)))
params = df.dropna(subset=['params']).groupby('model')['params'].first()
print('trainable parameters:', params.to_dict())
