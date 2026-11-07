"""Tables and the cost-curve figure for the JADS revision, from results/all_runs.csv and the saved weights.
Usage: py jads_tables.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jads_evaluate as ev

df = pd.read_csv('results/all_runs.csv')
base = df[df.variant == 'base']
best_classical = base[base.exp == 'bench']['cum'].max()


def row(g):
    return pd.Series({'fits': len(g), 'cum_med': g.cum.median(), 'cum_min': g.cum.min(), 'cum_max': g.cum.max(),
                      'sharpe_med': g.sharpe.median(), 'turn_med': g.turnover.median(), 'peak_med': g.max_weight.median(),
                      'hhi_med': g.mean_hhi.median(), 'net25_med': g.net25.median(), 'be_zero_med': g.be_zero.median(),
                      'ahead_all_classical': int((g.cum > best_classical).sum())})


t4 = base[base.exp.isin(['arch', 'wf', 'bench', 'ref'])].groupby(['exp', 'model']).apply(row, include_groups=False)
t4.to_csv('results/table_models.csv')
pen = pd.concat([base[(base.exp == 'arch') & (base.model == 'lstm')].assign(model='lstm_lam0'), base[base.exp == 'pen']])
t_pen = pen.groupby('model').apply(row, include_groups=False)
t_pen.to_csv('results/table_penalty.csv')
v = df[(df.exp == 'arch') & (df.model == 'lstm')]
t_var = v.groupby('variant').apply(row, include_groups=False)
t_var.to_csv('results/table_variants.csv')
gap = base[base.exp.isin(['arch', 'wf'])].groupby(['exp', 'model'])['sharpe_gap_MK'].describe()
gap.to_csv('results/sharpe_gap_mk.csv')
pd.set_option('display.width', 250)
print(t4.round(4).to_string()); print(t_pen.round(4).to_string()); print(t_var.round(4).to_string()); print(gap.round(2).to_string())

# overfitting: training vs validation loss of the 40 fixed-split fits
meta = [json.load(open(p)) for p in glob.glob('neural/arch_*.json')]
of = pd.DataFrame([dict(model=m['kind'], train_loss=m['loss'][-1], best_val=min(m['val_loss'])) for m in meta])
print(of.groupby('model').median().round(3).to_string())
of.groupby('model').median().to_csv('results/table_losses.csv')

# cost curve: median net cumulative return over seeds against the one-way cost
costs = np.arange(0, 61, 2.5)
def curve(ws):
    return np.median([[ev.cum(ev.simulate_portfolio(ev.r, w, c)[0]) for c in costs] for w in ws], axis=0)
groups = {}
for kind in ('lstm', 'gru'):
    groups[f'{kind.upper()}, 10 seeds'] = [np.load(p) for p in sorted(glob.glob(f'neural/arch_{kind}_s*.npy'))]
groups['LSTM, walk-forward'] = [np.vstack([np.load(f'neural/wf_lstm_s{s}_b{b}.npy') for b in range(5)]) for s in range(1, 6)]
for name, key in (('Equal weight', 'EW'), ('Markowitz-style', 'MK'), ('Minimum variance (shrinkage)', 'MinVar_LW')):
    groups[name] = [ev.bench[key]]
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=300)
styles = {'LSTM, 10 seeds': ('#1f4e79', '-'), 'GRU, 10 seeds': ('#2e7d32', '-'), 'LSTM, walk-forward': ('#1f4e79', '--'),
          'Equal weight': ('#555555', ':'), 'Markowitz-style': ('#c55a11', ':'), 'Minimum variance (shrinkage)': ('#7f3f98', '-.')}
out = {}
for name, ws in groups.items():
    y = curve(ws) * 100; out[name] = y
    ax.plot(costs / 100, y, color=styles[name][0], linestyle=styles[name][1], linewidth=1.4, label=name)
ax.axhline(0, color='black', linewidth=0.6)
ax.set_xlabel('One-way cost per unit traded (%)'); ax.set_ylabel('Net cumulative test return (%)')
ax.legend(fontsize=7, frameon=False, ncol=2); ax.grid(alpha=0.3, linewidth=0.5)
fig.tight_layout(); fig.savefig('results/figure_cost_curve.png');
pd.DataFrame(out, index=costs / 100).to_csv('results/cost_curve.csv')
print(pd.DataFrame(out, index=costs / 100).iloc[[0, 4, 8, 10, 12, 16, 20, 24]].round(2).to_string())
