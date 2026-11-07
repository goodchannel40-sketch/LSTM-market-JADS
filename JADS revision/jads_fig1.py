"""Figure 1 of the JADS revision: test cumulative returns of every fit, by network and design."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d = pd.read_csv('results/all_runs.csv'); b = d[d.variant == 'base']
groups = [('LSTM', 'arch', 'lstm'), ('GRU', 'arch', 'gru'), ('Feedforward', 'arch', 'mlp'), ('Linear', 'arch', 'linear'),
          ('LSTM\nwalk-forward', 'wf', 'lstm'), ('GRU\nwalk-forward', 'wf', 'gru'), ('Feedforward\nwalk-forward', 'wf', 'mlp'), ('Linear\nwalk-forward', 'wf', 'linear')]
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=300)
rng = np.random.default_rng(0)
for i, (lab, exp, m) in enumerate(groups):
    y = b[(b.exp == exp) & (b.model == m)].cum.values * 100
    ax.scatter(i + rng.uniform(-0.12, 0.12, len(y)), y, s=14, color='#1f4e79' if m in ('lstm', 'gru') else '#999999', zorder=3)
    ax.hlines(np.median(y), i - 0.25, i + 0.25, color='black', linewidth=1.2, zorder=4)
for name, key, ls in (('Minimum variance (shrinkage)', 'MinVar_LW', '-.'), ('Markowitz-style', 'MK', ':'), ('Equal weight', 'EW', '--')):
    ax.axhline(b[(b.exp == 'bench') & (b.model == key)].cum.iloc[0] * 100, color='#c55a11', linestyle=ls, linewidth=0.9, label=name)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=7)
ax.set_ylabel('Cumulative test return (%)'); ax.set_ylim(top=14.5); ax.legend(fontsize=7, frameon=False, loc='upper center', ncol=3); ax.grid(axis='y', alpha=0.3, linewidth=0.5)
fig.tight_layout(); fig.savefig('results/figure_seed_distribution.png')
