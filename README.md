# Direct LSTM portfolio allocation on Moroccan bank equities: code and results

Replication material for the study of direct portfolio allocation with recurrent networks on six bank stocks listed in Casablanca (CFG, ATW, BCP, CIH, BOA and BMCI).

- The files at the top level reproduce the original single-network study: the notebook `code_corrected.ipynb` and its reference run with seed 42.
- The folder [`JADS revision`](JADS%20revision) holds the experiments of the revised manuscript #1664 for the Journal of Applied Data Sciences: four networks under ten seeds, walk-forward refitting, six classical rules, transaction costs, an issuer cap, a turnover penalty and an execution-delay test. It has its own README.

## Contents of the top level

| File | What it is |
|---|---|
| `code_corrected.ipynb` | Notebook that produced the reference run and its tables and figures |
| `lstm_core.py` | Data loading, model, loss, training rules, Markowitz-style rule, equal weighting and trading accounting, copied unchanged from the notebook |
| `robustness.py` | Robustness experiments: 31 random seeds, two earlier test windows, a 3 x 3 settings grid |
| `markowitz_exact.py` | Exact long-only tangency and minimum-variance portfolios, compared with the heuristic rule |
| `robustness_runs.csv` | One row per fitted model (98 fits): test and training performance, turnover, concentration, exact cost break-evens |
| `test_weights_all_runs.csv` | Test-period portfolio weights of every fit |
| `training_histories.json` | Training and validation loss by epoch for every fit |
| `models.zip` | Trained network weights of every fit (Keras `.weights.h5`) |

## How to reproduce the top-level results

1. Python 3.13 with TensorFlow 2.21, numpy, pandas, matplotlib and openpyxl; scipy for `markowitz_exact.py`.
2. Place the price file `prices_snapshot.csv` (daily closes of CFG, ATW, BCP, CIH, BOA and BMCI, 18 Dec 2023 to 25 Sep 2026) in the same folder. The prices come from TradingView through the tvDatafeed interface and are not redistributed here because of the provider's terms. The notebook downloads them again if the file is absent.
3. Run the notebook. Seed 42 reproduces the reference run exactly (cumulative test return 4.3517%, Sharpe ratio 0.6995).
4. Run `python robustness.py all` for the robustness experiments (about one minute on a 20-core CPU) and `python markowitz_exact.py` for the Markowitz comparison.

## Licence

Code released under the MIT licence. Results files may be reused with citation of the article.
