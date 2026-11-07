# JADS revision: multi-model experiments

Code and results for the revised manuscript #1664, Journal of Applied Data Sciences: "Recurrent Neural Networks for Direct Portfolio Allocation on Moroccan Bank Equities: A Multi-Model Walk-Forward Evaluation".

All 191 fits use the same six bank stocks (CFG, ATW, BCP, CIH, BOA and BMCI, ticker BCI), the same chronological split and the same return-to-volatility loss as the original notebook. `lstm_core.py` is the unchanged copy found at the top level of the repository.

## Contents

| File | What it is |
|---|---|
| `jads_neural.py` | Fits every network and saves each fit. Experiments: `ref` (LSTM, seed 42, reproduction of the original run), `arch` (LSTM, GRU, feedforward and linear policy, 10 seeds each), `wf` (walk-forward, refit every 20 test returns, 5 seeds per network), `pen` (LSTM with a turnover penalty of 0.1, 0.3 or 1.0, 10 seeds each), `size` (LSTM and GRU with 16 or 64 units, 5 seeds each) |
| `jads_evaluate.py` | Classical rules (equal weighting, Markowitz-style rule, exact long-only tangency, Ledoit-Wolf minimum variance, inverse volatility, momentum), cost accounting, 40% cap and one-observation delay. Writes `results/all_runs.csv` and `results/summary.csv` |
| `jads_extra.py` | Data checks: the six 2026 dividends added back, and one-day returns only. Writes `results/data_checks.csv` |
| `jads_tables.py` | Tables 4 and 5, training and validation losses, Sharpe gaps against the Markowitz-style rule, and the cost curve of Figure 2 |
| `jads_fig1.py` | Figure 1, the test return of every fit |
| `lstm_core.py` | Data loading, model, loss, training rules and trading accounting from the original notebook |
| `requirements_training.txt`, `requirements_evaluation.txt` | Exact package versions of the training and evaluation environments |
| `results/` | Every result table of the paper as CSV, plus Figures 1 and 2 |
| `neural.zip` | The 191 fits: network parameters (`.weights.h5`), settings and training history (`.json`) and test-period weights (`.npy`) |

## How to reproduce

1. Training runs under Python 3.13 with `requirements_training.txt` (TensorFlow 2.21). Evaluation runs under Python 3.14 with `requirements_evaluation.txt`.
2. Put the price file `prices_snapshot.csv` in this folder. It is not included because TradingView's terms do not allow redistribution. When the file is missing, `lstm_core.py` downloads the closes through tvDatafeed and saves them. A new download returns the most recent sessions, so keep only the 603 aligned dates from 18 December 2023 to 25 September 2026 to match the paper.
3. Unzip `neural.zip` in this folder (it creates `neural/`) to use the saved fits, or refit them with `python jads_neural.py all`. A fit takes about five seconds on a laptop processor. In our second complete run the test weights of the 171 main fits were identical, and those of the 20 network-size fits agreed to within 3e-7.
4. Then run `python jads_evaluate.py`, `python jads_extra.py`, `python jads_tables.py` and `python jads_fig1.py`.

## Licence

Code released under the MIT licence. Results files may be reused with citation of the article.
