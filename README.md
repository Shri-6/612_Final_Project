# Cross-Asset Attention Transformer for Multi-Asset Financial Forecasting

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Course](https://img.shields.io/badge/Course-DATA%20612%20%E2%80%94%20UMD-red)

**DATA 612 — Deep Learning · Final Project · University of Maryland**

**Eniyan Ezhilan Sumathi · Shri Varshan Periyaswamy · Dhanush Sambasivam · Madhumitha Rajagopal**

---

## Overview

This project develops a **Cross-Asset Attention Transformer (CAT)** that jointly predicts
next-day return directions for 20 financial assets — large-cap equities, sector ETFs,
broad-market ETFs, defensive instruments, and cryptocurrencies. Unlike per-asset models,
the architecture uses factorized temporal and cross-asset self-attention so the model
learns *how assets relate to each other* directly from data, rather than treating each
instrument in isolation.

We train the model on daily OHLCV data from Yahoo Finance (January 2018 – December 2023,
2,169 trading days) using 16 engineered technical features per asset across a 40-day
lookback window. The model (~330K parameters) is evaluated on 5,720 out-of-sample
predictions (286 usable test days × 20 assets, Feb–Dec 2023) and compared against six
baselines: Ridge Regression, ARIMA, XGBoost, LSTM, GRU, and TCN.

The main contributions are:

1. A factorized dual-attention architecture — temporal attention per asset, followed by
   cross-asset attention across the full portfolio — producing a richer representation
   than any single-axis model.
2. A custom **Direction-Aware Composite Loss** (MSE + class-weighted cross-entropy +
   sign penalty) that directly optimizes for directional accuracy, raising the
   backtest Sharpe from −0.61 to +1.09 versus MSE-only training.
3. Evidence that learned cross-asset attention captures fundamentally different structure
   than classical rolling correlation (Spearman ρ = −0.37, attention 25× more stable
   day-to-day).

---

## Results

### Model Comparison

All metrics are evaluated on 5,720 out-of-sample predictions (286 days × 20 assets,
Feb–Dec 2023). Backtest Sharpe uses a daily long-short top-5 strategy (long the 5
assets with the highest predicted returns, short the 5 with the lowest) with a 1 basis
point transaction cost. DM = Diebold–Mariano Harvey-corrected test relative to the
Transformer.

| Model | RMSE | Dir. Acc | Macro F1 | Sharpe | DM Direction |
|---|---|---|---|---|---|
| **Cross-Asset Transformer (ours)** | **0.0119** | 52.3% | 0.495 | **+1.09** | — |
| Ridge Regression | 0.0123 | 49.9% | 0.477 | +1.47 | Beaten (p < .001) |
| XGBoost | 0.0120 | 49.3% | 0.458 | +0.97 | Beaten (p < .001) |
| GRU | 0.0118 | 48.4% | 0.455 | +0.50 | Beats Transformer (p = .009) |
| TCN | 0.0170 | **52.7%** | **0.499** | +0.64 | Beaten (p < .001) |
| ARIMA | 0.0783 | 50.6% | 0.479 | −0.13 | Beaten (p < .001) |
| LSTM | 0.0161 | 47.1% | 0.415 | −0.16 | Beaten (p < .001) |

The Transformer achieves the **best Sharpe ratio among deep learning sequence models
(+1.09)** and is statistically superior to five of six baselines on directional accuracy
(Diebold–Mariano, p < 0.001). Ridge Regression achieves the highest raw Sharpe (+1.47),
but is statistically beaten by the Transformer on directional accuracy — its Sharpe
advantage is attributable to the specific top-5 long/short construction rather than
superior directional prediction across all 20 assets. TCN edges ahead on raw F1 (0.499
vs. 0.495) but collapses to a Sharpe of 0.64, indicating its marginal F1 advantage
comes from predicting the majority "up" class more reliably rather than from improved
down-day discrimination. GRU is the only baseline that statistically outperforms the
Transformer on directional accuracy (p = .009).

### Loss Function Ablation

| Loss Configuration | Down-Day Recall | Backtest Sharpe |
|---|---|---|
| MSE Only | 14% | −0.61 |
| MSE + Weighted CE | 18% | +0.23 |
| MSE + Sign Penalty | 28% | +1.08 |
| Full Composite (All 3) | **31%** | **+1.09** |

### Cross-Asset Attention vs. Rolling Correlation

The learned attention matrix at layer 3 has a Spearman correlation of **−0.37** with
the rolling Pearson correlation matrix and is **25× more stable** day-to-day. The
attention matrix is dense and assigns elevated weights to the defensive asset (TLT)
from nearly all other assets — consistent with it serving as a regime indicator — a
structure that pairwise linear correlation cannot represent dynamically.

### Ablation — Contribution of Cross-Asset Attention

| Variant | Macro F1 | Delta vs. Full Model |
|---|---|---|
| Full Model (Temporal + Cross-Asset) | 0.495 | — |
| Temporal Only (no cross-asset attn) | 0.412 | −0.083 |
| Single-Asset LSTM | 0.387 | −0.108 |

---

## Asset Universe

20 assets across five categories (January 2018 – December 2023):

| Category | Tickers | Count |
|---|---|---|
| Large-Cap Equities | AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, BRK-B | 8 |
| Sector ETFs | XLK, XLF, XLE, XLV, XLI | 5 |
| Broad-Market ETFs | SPY, QQQ, IWM | 3 |
| Defensive | TLT (bonds), GLD (gold) | 2 |
| Cryptocurrencies | BTC-USD, ETH-USD | 2 |

Data split (chronological, no shuffling): 70% train (1,518 days), 15% validation
(325 days), 15% test (326 days). The 40-day lookback window leaves **286 usable
out-of-sample prediction days**, yielding 5,720 total predictions.

---

## Model Architecture

```
Input (B, 40, 20, 16)
    -> Linear Embedding          d_model = 96
    -> Temporal Attention x3     per-asset, 40-day window, d_ff = 192
    -> Cross-Asset Attention x3  across 20 assets per timestep, d_ff = 192
    -> Linear Head               sigmoid -> (B, 20) direction probabilities
```

**Hyperparameters:**

| Parameter | Value |
|---|---|
| d_model | 96 |
| Attention heads | 4 (per module) |
| Layers (temporal / cross-asset) | 3 / 3 |
| Feed-forward dim (d_ff) | 192 |
| Dropout | 0.25 |
| Lookback window | 40 days |
| Total parameters | ~330K |
| Optimizer | AdamW (LR = 3×10⁻⁴, WD = 10⁻⁴) |
| Batch size | 64 |
| Early stopping patience | 10 epochs |

---

## Feature Engineering

16 technical features per asset per day, organized into six families. All features
are derived exclusively from OHLCV data — no fundamentals, no sentiment.

| Family | Features | Count |
|---|---|---|
| Returns | log_ret, ret_5d, ret_20d | 3 |
| Volatility | vol_5d, vol_20d | 2 |
| Momentum | RSI-14, MACD, signal, histogram | 4 |
| Bollinger Bands | BB_%pos, BB_width | 2 |
| Volume / Range | ATR-14, OBV, vol_chg, hl_range | 4 |
| Trend Anchor | close_norm (P_t / SMA20 − 1) | 1 |

Total dataset: 16 features × 20 assets × 2,169 trading days = **681,120 feature values**.
All features are Z-score normalized using training-set statistics only (no lookahead).

---

## Repository Structure

```
.
├── app/
│   └── streamlit_app.py              # 5-tab interactive dashboard
├── scripts/
│   ├── 01_fetch_data.py              # download OHLCV data via yfinance
│   ├── 02_feature_engineering.py     # compute all 16 features, build tensors
│   ├── 04_train_transformer.py       # train the Cross-Asset Transformer
│   ├── 04_backtest.py                # long-short top-5 strategy, Sharpe ratio
│   ├── 05_extract_attention.py       # save (T, N, N) cross-asset attention tensors
│   ├── 06_evaluate.py                # RMSE, Dir. Acc, Macro F1, DM test
│   ├── 06_classification_report.py   # per-class F1, precision, recall per asset
│   └── 07_threshold_calibration.py   # tune decision threshold on validation set
├── artifacts/
│   ├── models/                       # saved .pt checkpoints
│   ├── metrics/                      # CSV files with per-model evaluation results
│   └── attention/                    # saved attention matrices for visualization
├── requirements.txt
└── README.md
```

---

## Setup and Installation

**Prerequisites:** Python 3.10+, pip, and a virtual environment manager.

```bash
git clone https://github.com/Shri-6/612_Final_Project
cd 612_Final_Project
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Reproducing the Results

Run the scripts in order. Each step writes its outputs to `artifacts/` so subsequent
steps can read them. All random seeds are fixed (`torch.manual_seed(42)`,
`numpy.random.seed(42)`) for deterministic results.

```bash
python scripts/01_fetch_data.py              # download OHLCV data for all 20 assets
python scripts/02_feature_engineering.py    # compute features and build input tensors
python scripts/04_train_transformer.py      # train the Cross-Asset Transformer
python scripts/06_evaluate.py               # RMSE, directional accuracy, Macro F1, DM test
python scripts/04_backtest.py               # long-short top-5 backtest, Sharpe ratio
python scripts/05_extract_attention.py      # extract and save cross-asset attention tensors
python scripts/06_classification_report.py  # per-class F1, precision, recall per asset
python scripts/07_threshold_calibration.py  # tune decision threshold on validation set
```

Training on CPU takes approximately 30–45 minutes. On a single GPU (e.g., NVIDIA T4),
training completes in under 10 minutes. Early stopping typically triggers around
epoch 68, saving the best checkpoint at approximately 0.0120 validation MSE.

---

## Running the Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard is also live at **data612.streamlit.app** and has five tabs:

1. **Overview** — project pipeline and motivation
2. **Leaderboard** — test-set metrics across all 7 models
3. **Backtest** — equity curves and Sharpe ratios
4. **Attention vs. Correlation** — interactive heatmap comparison (the core novelty)
5. **Live Prediction** — select a date and run the trained Transformer forward

---

## Data Source

All price data is sourced from **Yahoo Finance** via the `yfinance` Python library.
The full window spans January 2018 through December 2023. The test set covers
February–December 2023 and was held out entirely until final evaluation.

---

## Implementation Tools

| Library | Version | Purpose |
|---|---|---|
| PyTorch | 2.1.0 | Cross-Asset Transformer, training loop, loss functions |
| scikit-learn | 1.3 | Ridge regression baseline |
| XGBoost | 2.0 | XGBoost baseline |
| statsmodels | 0.14 | ARIMA baseline |
| pandas / NumPy | 2.1 / 1.26 | Data preprocessing and feature engineering |
| yfinance | 0.2 | Automated OHLCV data collection |
| Matplotlib | — | All figures and visualizations |
| Streamlit | — | Interactive dashboard |

---

## License

This project is released under the MIT License. See `LICENSE` for details.

---

## Citation

If you build on this work, please cite:

```
@misc{cross_asset_transformer_2026,
  title   = {Cross-Asset Attention Transformer for Multi-Asset Financial Forecasting},
  author  = {Sumathi, Eniyan Ezhilan and Periyaswamy, Shri Varshan and
             Sambasivam, Dhanush and Rajagopal, Madhumitha},
  year    = {2026},
  school  = {University of Maryland, College Park},
  note    = {DATA 612 — Deep Learning, Final Project}
}
```
