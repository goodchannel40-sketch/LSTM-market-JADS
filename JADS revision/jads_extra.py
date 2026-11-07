"""Data checks for the JADS revision on the new runs: 2026 dividends added back, and returns that span
several sessions. Uses the saved weights of jads_neural.py and the benchmarks of jads_evaluate.py.
Usage: py jads_extra.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

import jads_evaluate as ev  # recomputes the benchmark weights (prints its summary)

P, R, r, split = ev.P, ev.R, ev.r, ev.split
dates = R.index[split:]
# 2026 cash dividends (MAD per share) and ex-dates, exchange financial calendar
DIV = {'CFG': ('2026-06-10', 4.0), 'CIH': ('2026-07-01', 14.0), 'ATW': ('2026-07-08', 22.0),
       'BCP': ('2026-07-09', 11.0), 'BOA': ('2026-07-16', 5.0), 'BCI': ('2026-07-17', 14.0)}
r_div = r.copy()
prev_close = P.shift(1).loc[dates]
for j, a in enumerate(ev.ASSETS):
    ex, amt = DIV[a]
    pos = np.searchsorted(dates.values, np.datetime64(ex))  # first retained date on or after the ex-date
    r_div[pos, j] += amt / prev_close[a].iloc[pos]

gap_days = np.diff(np.r_[R.index[split - 1:split].values, dates.values]).astype('timedelta64[D]').astype(int)
weekdays = np.array([np.busday_count(R.index[split - 1 + i].date(), dates[i].date()) for i in range(len(dates))])
one_day = weekdays == 1


def runs():
    meta = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob('neural/*.json')}
    for tag, m in meta.items():
        if m['exp'] == 'arch':
            yield m['kind'], m['seed'], np.load(f'neural/{tag}.npy')
    for name, w in ev.bench.items():
        yield name, None, w


rows = []
for model, seed, w in runs():
    g, _ = ev.simulate_portfolio(r, w, 0.0)
    gd, _ = ev.simulate_portfolio(r_div, w, 0.0)
    rows.append(dict(model=model, seed=seed, cum=ev.cum(g), cum_div=ev.cum(gd),
                     be_zero_div=ev.solve_cost(lambda c: ev.cum(ev.simulate_portfolio(r_div, w, c)[0])),
                     vol_all=g.std() * np.sqrt(252), vol_1day=g[one_day].std() * np.sqrt(252),
                     sharpe_all=np.sqrt(252) * g.mean() / g.std(), sharpe_1day=np.sqrt(252) * g[one_day].mean() / g[one_day].std()))
df = pd.DataFrame(rows)
df.to_csv('results/data_checks.csv', index=False)
print('test returns spanning more than one weekday:', int((~one_day).sum()), 'of', len(one_day))
print(df.groupby('model')[['cum', 'cum_div', 'be_zero_div', 'vol_all', 'vol_1day', 'sharpe_all', 'sharpe_1day']].median().round(4).to_string())
