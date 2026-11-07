"""Model, rules and accounting extracted verbatim from code_corrected.ipynb (cells 1 and 3, up to the fixed-model run)."""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import numpy as np
import pandas as pd

SNAPSHOT = Path("prices_snapshot.csv")
ASSETS = ["CFG", "ATW", "BCP", "CIH", "BOA", "BCI"]
# Optional fixed cutoff, selected before viewing results; None keeps all data.
END_DATE = None
if SNAPSHOT.exists():
    prices = pd.read_csv(SNAPSHOT, index_col=0, parse_dates=True)[ASSETS]
else:
    from tvDatafeed import TvDatafeed, Interval
    tv = TvDatafeed()
    series = {}
    for symbol in ASSETS:
        downloaded = tv.get_hist(symbol=symbol, exchange="CSEMA",
                                 interval=Interval.in_daily, n_bars=1000)
        if downloaded is None or downloaded.empty:
            raise ValueError(f"No data returned for {symbol}")
        close = downloaded["close"].copy()
        # Join daily observations by calendar date, not differing intraday stamps.
        close.index = pd.to_datetime(close.index).normalize()
        if close.index.has_duplicates:
            raise ValueError(f"Duplicate daily dates for {symbol}")
        series[symbol] = close
    prices = pd.concat(series, axis=1).sort_index().dropna()
    prices.to_csv(SNAPSHOT)

prices.index = pd.to_datetime(prices.index).normalize()
prices = prices.sort_index()
if prices.index.has_duplicates or prices.isna().any().any():
    raise ValueError("Snapshot contains duplicate dates or missing prices")
if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
    raise ValueError("Prices must be finite and positive")
if END_DATE is not None:
    prices = prices.loc[:END_DATE]
print(f"Price sample: {len(prices)} observations, {prices.index.min()} to {prices.index.max()}")
print("No final observations are silently removed.")


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from numpy.lib.stride_tricks import sliding_window_view

# ============================================================
# 0) USER DATA
# prices must already exist:
# rows = dates
# columns = assets
# ============================================================
returns = prices.pct_change().dropna().astype(np.float32)
n_assets = returns.shape[1]

# ============================================================
# 1) GLOBAL SETTINGS
# ============================================================
SEED = 42
TEST_SIZE = 100
BATCH_SIZE = 64
TRADING_DAYS = 252
PLOT_FINAL = True
SAVE_RESULTS = True
OUTPUT_PREFIX = "corrected_lstm"
# No trading fees: gross portfolio returns, as requested.
COST_BPS = 0.0
COST_SENSITIVITY_BPS = [0.0]

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 2) SAVED MODEL CONFIGURATION, ONE RUN
# ============================================================
LOOKBACK = 20
MAX_EPOCHS = 20
LEARNING_RATE = 0.001
LSTM_UNITS = 32
DROPOUT_RATE = 0.1

# ============================================================
# 3) HELPERS
# ============================================================
def standardize_train_test(train_df, test_df, eps=1e-8):
    mu = train_df.mean(axis=0)
    sigma = train_df.std(axis=0).replace(0, 1.0)

    train_scaled = (train_df - mu) / (sigma + eps)
    test_scaled = (test_df - mu) / (sigma + eps)

    return train_scaled.astype(np.float32), test_scaled.astype(np.float32), mu, sigma


def create_sequence_dataset(data_array, lookback):
    if len(data_array) <= lookback:
        raise ValueError(f"Not enough rows for lookback={lookback}")

    X = sliding_window_view(data_array, window_shape=lookback, axis=0)[:-1]
    X = np.transpose(X, (0, 2, 1))
    y = data_array[lookback:]
    return X.astype(np.float32), y.astype(np.float32)


def compute_markowitz_weights(hist_returns, eps=1e-5):
    mu = hist_returns.mean(axis=0).values.astype(np.float64)
    sigma = hist_returns.cov().values.astype(np.float64)
    sigma = sigma + eps * np.eye(len(mu), dtype=np.float64)

    try:
        w = np.linalg.solve(sigma, mu)
    except np.linalg.LinAlgError:
        w = mu.copy()

    w = np.maximum(w, 0.0)
    s = w.sum()

    if s <= 1e-12:
        w = np.ones_like(w) / len(w)
    else:
        w = w / s

    return w.astype(np.float32)


def compute_expanding_markowitz(returns_df, start_idx):
    weights = np.zeros((len(returns_df) - start_idx, returns_df.shape[1]), dtype=np.float32)

    for i, t in enumerate(range(start_idx, len(returns_df))):
        hist_df = returns_df.iloc[:t]
        if len(hist_df) > 2:
            weights[i] = compute_markowitz_weights(hist_df)
        else:
            weights[i] = np.ones(returns_df.shape[1], dtype=np.float32) / returns_df.shape[1]

    return weights


def sharpe_loss(y_true, y_pred):
    port_ret = tf.reduce_sum(y_true * y_pred, axis=1)
    mean_ret = tf.reduce_mean(port_ret)
    std_ret = tf.math.reduce_std(port_ret) + 1e-8
    return -mean_ret / std_ret


def build_model(lookback, n_assets, units=32, dropout=0.1, lr=1e-3):
    model = Sequential([
        Input(shape=(lookback, n_assets)),
        LSTM(units),
        Dropout(dropout),
        Dense(n_assets, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=sharpe_loss
    )
    return model


def simulate_portfolio(asset_returns, target_weights, cost_bps=0.0):
    """Daily rebalance, with drift-aware turnover and self-financing costs.

    Targets are set before each row's returns. Cost is charged per unit of
    equity bought or sold (sum of absolute trades), with no extra half factor.
    Each evaluated segment starts in cash and pays initial entry costs.
    The final position is not liquidated. Cash earns zero; no borrowing.
    Execution at the previous observed close is assumed by this close-only data.
    """
    r = np.asarray(asset_returns, dtype=np.float64)
    w = np.asarray(target_weights, dtype=np.float64)
    if r.shape != w.shape or r.ndim != 2 or not np.isfinite(r).all():
        raise ValueError("Returns and weights must be finite matching matrices")
    if (r <= -1).any() or not np.isfinite(w).all() or (w < -1e-8).any():
        raise ValueError("Invalid simple returns or long-only weights")
    if not np.allclose(w.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Targets must sum to one")
    if not 0 <= cost_bps < 10000:
        raise ValueError("Cost must be between 0 and 10000 basis points")
    w = np.maximum(w, 0)
    w /= w.sum(axis=1, keepdims=True)
    c = cost_bps / 10000.0
    previous = np.zeros(w.shape[1])
    net, turnover, costs = [], [], []
    for target, ret in zip(w, r):
        # q is wealth remaining after paying costs. Solve
        # q + c * sum(abs(q * target - previous)) = 1.
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
        costs.append(1 - q)
        previous = target * (1 + ret) / (1 + gross)
    return np.array(net), np.array(turnover), np.array(costs)


def compute_drawdown(return_series):
    wealth = (1 + return_series).cumprod()
    running_max = wealth.cummax().clip(lower=1.0)
    drawdown = wealth / running_max - 1.0
    return drawdown


def safe_skew(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan
    mean = x.mean()
    std = x.std(ddof=0)
    if std < 1e-12:
        return 0.0
    return np.mean(((x - mean) / std) ** 3)


def safe_kurtosis(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) < 4:
        return np.nan
    mean = x.mean()
    std = x.std(ddof=0)
    if std < 1e-12:
        return 0.0
    return np.mean(((x - mean) / std) ** 4) - 3.0


def performance_stats(series, turnover_series=None, trading_days=252):
    s = pd.Series(series).dropna()
    if len(s) == 0:
        return {}

    mean_daily = s.mean()
    std_daily = s.std(ddof=0)

    # Zero target return; include all observations in downside deviation.
    downside_std = np.sqrt(np.mean(np.minimum(s.to_numpy(), 0.0) ** 2))

    annual_return = (1 + s).prod() ** (trading_days / len(s)) - 1
    annual_vol = std_daily * np.sqrt(trading_days)
    sharpe = mean_daily / (std_daily + 1e-12) * np.sqrt(trading_days)
    sortino = mean_daily / (downside_std + 1e-12) * np.sqrt(trading_days)

    cumulative_return = (1 + s).prod() - 1
    drawdown = compute_drawdown(s)
    max_drawdown = drawdown.min()
    calmar = annual_return / (abs(max_drawdown) + 1e-12)

    win_rate = (s > 0).mean()
    best_day = s.max()
    worst_day = s.min()

    var_95 = s.quantile(0.05)
    cvar_95 = s[s <= var_95].mean() if (s <= var_95).sum() > 0 else var_95

    skewness = safe_skew(s.values)
    kurt = safe_kurtosis(s.values)

    avg_turnover = np.nan
    if turnover_series is not None:
        t = pd.Series(turnover_series).dropna()
        if len(t) > 0:
            avg_turnover = t.mean()

    return {
        "n_obs": len(s),
        "mean_daily_return": mean_daily,
        "std_daily_return": std_daily,
        "annualized_return": annual_return,
        "annualized_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "cumulative_return": cumulative_return,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "best_day": best_day,
        "worst_day": worst_day,
        "skewness": skewness,
        "kurtosis": kurt,
        "VaR_95": var_95,
        "CVaR_95": cvar_95,
        "avg_turnover": avg_turnover
    }


def build_performance_table(df_returns, turnover_map=None, dataset_name="train", trading_days=252):
    rows = []

    for col in ["LSTM", "Markowitz", "EqualWeight"]:
        turnover_series = None
        if turnover_map is not None and col in turnover_map:
            turnover_series = turnover_map[col]

        st = performance_stats(
            df_returns[col],
            turnover_series=turnover_series,
            trading_days=trading_days
        )

        row = {"dataset": dataset_name, "strategy": col}
        row.update(st)
        rows.append(row)

    return pd.DataFrame(rows)


def build_weights_summary(weights_df, strategy_name, dataset_name):
    mean_w = weights_df.mean()
    std_w = weights_df.std(ddof=0)
    min_w = weights_df.min()
    max_w = weights_df.max()

    out = pd.DataFrame({
        "asset": [c.replace("LSTM_", "").replace("MK_", "").replace("EQ_", "") for c in weights_df.columns],
        "mean_weight": mean_w.values,
        "std_weight": std_w.values,
        "min_weight": min_w.values,
        "max_weight": max_w.values
    })
    out.insert(0, "strategy", strategy_name)
    out.insert(0, "dataset", dataset_name)
    return out


def print_table(title, df, decimals=6):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 200,
                           "display.float_format", lambda x: f"{x:.{decimals}f}"):
        print(df)


def build_perf_dataframe(index, lstm_daily, mk_daily, eq_daily):
    df = pd.DataFrame({
        "LSTM": lstm_daily,
        "Markowitz": mk_daily,
        "EqualWeight": eq_daily
    }, index=index)

    df["cum_LSTM"] = (1 + df["LSTM"]).cumprod() - 1
    df["cum_Markowitz"] = (1 + df["Markowitz"]).cumprod() - 1
    df["cum_EqualWeight"] = (1 + df["EqualWeight"]).cumprod() - 1

    df["dd_LSTM"] = compute_drawdown(df["LSTM"])
    df["dd_Markowitz"] = compute_drawdown(df["Markowitz"])
    df["dd_EqualWeight"] = compute_drawdown(df["EqualWeight"])

    return df


# ============================================================
# 4) MAIN EXPERIMENT
# ============================================================
def run_experiment(
    returns,
    lookback=20,
    test_size=100,
    epochs=20,
    batch_size=64,
    lr=1e-3,
    units=32,
    dropout=0.1,
    seed=42,
    plot=False,
    trading_days=252,
    cost_bps=COST_BPS
):
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    n_assets = returns.shape[1]
    n = len(returns)

    if n <= lookback + test_size + 20:
        raise ValueError(f"Not enough data: n={n}, lookback={lookback}, test_size={test_size}")

    split_idx = n - test_size
    train_returns_raw = returns.iloc[:split_idx].copy()
    test_returns_raw = returns.iloc[split_idx:].copy()

    # Reserve the final 20% of pre-test sequences for chronological validation.
    # Fit preprocessing only on rows preceding the first validation target.
    n_sequences = len(train_returns_raw) - lookback
    val_size = max(1, int(0.2 * n_sequences))
    fit_end = len(train_returns_raw) - val_size
    scaler_fit = train_returns_raw.iloc[:fit_end]
    _, train_scaled, scaler_mu, scaler_sigma = standardize_train_test(
        scaler_fit, train_returns_raw
    )
    test_scaled = ((test_returns_raw - scaler_mu) / (scaler_sigma + 1e-8)).astype(np.float32)

    full_scaled_for_test = pd.concat([
        train_scaled.iloc[-lookback:],
        test_scaled
    ], axis=0)

    # --------------------------------------------------------
    # TRAIN DATASET
    # --------------------------------------------------------
    X_train_full, _ = create_sequence_dataset(train_scaled.values, lookback)
    # The loss must see actual investable returns, not standardized returns.
    y_train_full = train_returns_raw.values[lookback:].astype(np.float32)

    val_size = max(1, int(0.2 * len(X_train_full)))
    X_tr = X_train_full[:-val_size]
    y_tr = y_train_full[:-val_size]
    X_val = X_train_full[-val_size:]
    y_val = y_train_full[-val_size:]

    model = build_model(
        lookback=lookback,
        n_assets=n_assets,
        units=units,
        dropout=dropout,
        lr=lr
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5
        )
    ]

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=0
    )

    # --------------------------------------------------------
    # TRAIN PREDICTIONS
    # --------------------------------------------------------
    train_lstm_weights = model.predict(X_train_full, verbose=0)
    train_lstm_weights = train_lstm_weights / (
        train_lstm_weights.sum(axis=1, keepdims=True) + 1e-12
    )

    train_realized_returns = train_returns_raw.values[lookback:]
    train_lstm_daily, train_lstm_turnover, _ = simulate_portfolio(
        train_realized_returns, train_lstm_weights, cost_bps
    )

    train_mk_weights = compute_expanding_markowitz(train_returns_raw, start_idx=lookback)
    train_mk_daily, train_mk_turnover, _ = simulate_portfolio(
        train_realized_returns, train_mk_weights, cost_bps
    )

    train_eq_weights = np.ones((len(train_realized_returns), n_assets), dtype=np.float32) / n_assets
    train_eq_daily, train_eq_turnover, _ = simulate_portfolio(
        train_realized_returns, train_eq_weights, cost_bps
    )

    train_index = train_returns_raw.index[lookback:]
    train_perf = build_perf_dataframe(
        train_index,
        train_lstm_daily,
        train_mk_daily,
        train_eq_daily
    )

    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------
    X_test, _ = create_sequence_dataset(full_scaled_for_test.values, lookback)
    X_test = X_test[:test_size]

    test_lstm_weights = model.predict(X_test, verbose=0)
    test_lstm_weights = test_lstm_weights / (
        test_lstm_weights.sum(axis=1, keepdims=True) + 1e-12
    )

    test_realized_returns = test_returns_raw.values
    test_lstm_daily, test_lstm_turnover, _ = simulate_portfolio(
        test_realized_returns, test_lstm_weights, cost_bps
    )

    test_history_df = returns.iloc[:split_idx + test_size]
    test_mk_weights = compute_expanding_markowitz(test_history_df, start_idx=split_idx)
    test_mk_daily, test_mk_turnover, _ = simulate_portfolio(
        test_realized_returns, test_mk_weights, cost_bps
    )

    test_eq_weights = np.ones((len(test_realized_returns), n_assets), dtype=np.float32) / n_assets
    test_eq_daily, test_eq_turnover, _ = simulate_portfolio(
        test_realized_returns, test_eq_weights, cost_bps
    )

    test_index = test_returns_raw.index
    test_perf = build_perf_dataframe(
        test_index,
        test_lstm_daily,
        test_mk_daily,
        test_eq_daily
    )

    # --------------------------------------------------------
    # WEIGHTS DATAFRAMES
    # --------------------------------------------------------
    train_lstm_weights_df = pd.DataFrame(
        train_lstm_weights,
        columns=[f"LSTM_{c}" for c in returns.columns],
        index=train_index
    )

    train_mk_weights_df = pd.DataFrame(
        train_mk_weights,
        columns=[f"MK_{c}" for c in returns.columns],
        index=train_index
    )

    train_eq_weights_df = pd.DataFrame(
        train_eq_weights,
        columns=[f"EQ_{c}" for c in returns.columns],
        index=train_index
    )

    test_lstm_weights_df = pd.DataFrame(
        test_lstm_weights,
        columns=[f"LSTM_{c}" for c in returns.columns],
        index=test_index
    )

    test_mk_weights_df = pd.DataFrame(
        test_mk_weights,
        columns=[f"MK_{c}" for c in returns.columns],
        index=test_index
    )

    test_eq_weights_df = pd.DataFrame(
        test_eq_weights,
        columns=[f"EQ_{c}" for c in returns.columns],
        index=test_index
    )

    # --------------------------------------------------------
    # PERFORMANCE TABLES
    # --------------------------------------------------------
    train_table = build_performance_table(
        train_perf[["LSTM", "Markowitz", "EqualWeight"]],
        turnover_map={
            "LSTM": train_lstm_turnover,
            "Markowitz": train_mk_turnover,
            "EqualWeight": train_eq_turnover
        },
        dataset_name="train",
        trading_days=trading_days
    )

    test_table = build_performance_table(
        test_perf[["LSTM", "Markowitz", "EqualWeight"]],
        turnover_map={
            "LSTM": test_lstm_turnover,
            "Markowitz": test_mk_turnover,
            "EqualWeight": test_eq_turnover
        },
        dataset_name="test",
        trading_days=trading_days
    )

    summary_all = pd.concat([train_table, test_table], ignore_index=True)

    # --------------------------------------------------------
    # WEIGHT SUMMARY TABLES
    # --------------------------------------------------------
    train_weights_summary = pd.concat([
        build_weights_summary(train_lstm_weights_df, "LSTM", "train"),
        build_weights_summary(train_mk_weights_df, "Markowitz", "train"),
        build_weights_summary(train_eq_weights_df, "EqualWeight", "train")
    ], ignore_index=True)

    test_weights_summary = pd.concat([
        build_weights_summary(test_lstm_weights_df, "LSTM", "test"),
        build_weights_summary(test_mk_weights_df, "Markowitz", "test"),
        build_weights_summary(test_eq_weights_df, "EqualWeight", "test")
    ], ignore_index=True)

    # --------------------------------------------------------
    # GAP TABLE
    # --------------------------------------------------------
    gap_rows = []
    for strat in ["LSTM", "Markowitz", "EqualWeight"]:
        tr = train_table.loc[train_table["strategy"] == strat].iloc[0]
        te = test_table.loc[test_table["strategy"] == strat].iloc[0]

        gap_rows.append({
            "strategy": strat,
            "train_sharpe": tr["sharpe_ratio"],
            "test_sharpe": te["sharpe_ratio"],
            "gap_sharpe": tr["sharpe_ratio"] - te["sharpe_ratio"],
            "train_cumulative_return": tr["cumulative_return"],
            "test_cumulative_return": te["cumulative_return"],
            "gap_cumulative_return": tr["cumulative_return"] - te["cumulative_return"],
            "train_max_drawdown": tr["max_drawdown"],
            "test_max_drawdown": te["max_drawdown"],
            "train_avg_turnover": tr["avg_turnover"],
            "test_avg_turnover": te["avg_turnover"]
        })

    gap_table = pd.DataFrame(gap_rows)

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------
    if plot:
        plt.figure(figsize=(11, 5))
        plt.plot(train_perf.index, train_perf["cum_LSTM"], label="Train LSTM")
        plt.plot(train_perf.index, train_perf["cum_Markowitz"], label="Train Markowitz")
        plt.plot(train_perf.index, train_perf["cum_EqualWeight"], label="Train EqualWeight")
        plt.title("Train Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(11, 5))
        plt.plot(test_perf.index, test_perf["cum_LSTM"], label="Test LSTM")
        plt.plot(test_perf.index, test_perf["cum_Markowitz"], label="Test Markowitz")
        plt.plot(test_perf.index, test_perf["cum_EqualWeight"], label="Test EqualWeight")
        plt.title("Test Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(11, 5))
        plt.plot(train_perf.index, train_perf["dd_LSTM"], label="Train LSTM DD")
        plt.plot(train_perf.index, train_perf["dd_Markowitz"], label="Train Markowitz DD")
        plt.plot(train_perf.index, train_perf["dd_EqualWeight"], label="Train EqualWeight DD")
        plt.title("Train Drawdowns")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(11, 5))
        plt.plot(test_perf.index, test_perf["dd_LSTM"], label="Test LSTM DD")
        plt.plot(test_perf.index, test_perf["dd_Markowitz"], label="Test Markowitz DD")
        plt.plot(test_perf.index, test_perf["dd_EqualWeight"], label="Test EqualWeight DD")
        plt.title("Test Drawdowns")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    sensitivity_rows = []
    for rate in COST_SENSITIVITY_BPS:
        for name, weights in [("LSTM", test_lstm_weights),
                              ("Markowitz", test_mk_weights),
                              ("EqualWeight", test_eq_weights)]:
            net, traded, _ = simulate_portfolio(test_realized_returns, weights, rate)
            sensitivity_rows.append({"cost_bps": rate, "strategy": name,
                                     **performance_stats(net, traded, trading_days)})

    result = {
        "params": {
            "lookback": lookback,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "units": units,
            "dropout": dropout,
            "test_size": test_size,
            "trading_days": trading_days,
            "cost_bps": cost_bps
        },
        "cost_sensitivity": pd.DataFrame(sensitivity_rows),
        "history": history.history,
        "model": model,
        "train_perf": train_perf,
        "test_perf": test_perf,
        "train_table": train_table,
        "test_table": test_table,
        "summary_all": summary_all,
        "gap_table": gap_table,
        "train_weights_summary": train_weights_summary,
        "test_weights_summary": test_weights_summary,
        "train_lstm_weights_df": train_lstm_weights_df,
        "train_mk_weights_df": train_mk_weights_df,
        "train_eq_weights_df": train_eq_weights_df,
        "test_lstm_weights_df": test_lstm_weights_df,
        "test_mk_weights_df": test_mk_weights_df,
        "test_eq_weights_df": test_eq_weights_df
    }

    return result


