# CryptoCast: Multi-Horizon Bitcoin Price Forecasting Using Deep Learning

CryptoCast benchmarks four deep learning architectures—**1D-CNN**, **Simple RNN**, **LSTM**, and **Transformer**—to forecast Bitcoin closing prices across **1-day (1D)**, **3-day (3D)**, and **7-day (7D)** horizons using 60-day historical sliding windows.

## Problem Statement & Use Cases
Cryptocurrency markets exhibit high volatility and non-linear temporal dependencies that challenge traditional statistical forecasting methods. Multi-horizon price forecasting supports:
- Short-term trading signals and entry/exit timing (`1D`)
- Multi-day trend positioning and risk management (`3D` and `7D`)
- Financial analytics and research on sequence modeling trade-offs

## Dataset & Preprocessing Pipeline
- **Input Features:** `Price` (target), `Open`, `High`, `Low`, `Vol.`, and `Change %`
- **Chronological Ordering & Split:** Dates are parsed (`dayfirst=True`), sorted ascending, and split **80% train / 20% test** without shuffling to prevent temporal data leakage
- **Feature Scaling:** `MinMaxScaler(feature_range=(0, 1))` is fit strictly on the training split and applied to the test split
- **Sliding Window & Targets:** Uses a **60-day** lookback window (`X` shape `(N, 60, 6)`) to predict scaled target `Price` at `t + 1` (`1D`), `t + 3` (`3D`), and `t + 7` (`7D`) horizons

## Model Architectures & Training Setup
- 1D-CNN (`Conv1D` + `MaxPooling1D` + `Dense`):** Captures local short-term patterns across time steps
- **Simple RNN (`SimpleRNN` + `Dense`):** Baseline recurrent model for sequential dependencies
- **LSTM (`LSTM` + `Dropout` + `Dense`):** Gated recurrent network for longer temporal context
- **Transformer (`MultiHeadAttention` + `LayerNormalization` + `GlobalAveragePooling1D` + `Dense`):** Self-attention block modeling sequence relationships
- **Apple Silicon (M1) Execution Note:** Recurrent layers (`SimpleRNN` and `LSTM`) can be routed to `/CPU:0` (`clipnorm=1.0`, `tanh`) while convolutional and attention layers run on `/GPU:0` (`tensorflow-metal`) to avoid per-step Metal GPU overhead

## Multi-Horizon Benchmark Results
Metrics below are evaluated on the scaled test sets across `1D`, `3D`, and `7D` horizons (`outputs/model_comparison_metrics.csv`):

| Horizon | Model | MAE | RMSE | MAPE (%) |
| :--- | :--- | :---: | :---: | :---: |
| **1D** | CNN | 0.1375 | 0.4071 | 30.40% |
| **1D** | RNN | 0.1119 | 0.3755 | 18.03% |
| **1D** | LSTM | 0.0707 | 0.2016 | 11.88% |
| **1D** | Transformer | 0.3309 | 0.3858 | 80.94% |
| **3D** | CNN | 0.2016 | 0.6007 | 41.05% |
| **3D** | RNN | 0.0881 | 0.2651 | 14.65% |
| **3D** | LSTM | 0.0766 | 0.1995 | 13.22% |
| **3D** | Transformer | 0.2173 | 0.2636 | 49.58% |
| **7D** | CNN | 0.3464 | 1.0617 | 71.92% |
| **7D** | RNN | 0.0581 | 0.1032 | 10.58% |
| **7D** | LSTM | 0.0879 | 0.2050 | 15.98% |
| **7D** | Transformer | 0.2520 | 0.2896 | 63.91% |

## Saved Models & Visualizations
- **Trained Models (`models/`):** Saved as `btc_{cnn,rnn,lstm,transformer}_{1d,3d,7d}.keras`
- **Metrics CSV (`outputs/model_comparison_metrics.csv`):** Full comparison table across horizons and architectures
- **Evaluation Charts (`outputs/`):**
  - `final_comparison.png` — Actual vs. predicted `1D` price curves across all four models
  - `loss_curves.png` — `1D` error comparison across architectures
  - `error_distribution.png` — Prediction error distributions (`Actual - Predicted`) on `1D`
  - `horizon_comparison.png` — `RMSE` comparison across `1D`, `3D`, and `7D` horizons

## Repository Structure
```text
cryptocast_project/
├── data/
│   ├── Bitcoin Historical Data.csv
│   ├── bitcoin_raw.csv
│   ├── bitcoin_processed.csv
│   └── {X,y}_{train,test}_{1d,3d,7d}.npy
├── models/
│   └── btc_{cnn,rnn,lstm,transformer}_{1d,3d,7d}.keras
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   └── 02_model_training.ipynb
├── outputs/
│   ├── model_comparison_metrics.csv
│   ├── final_comparison.png
│   ├── loss_curves.png
│   ├── error_distribution.png
│   └── horizon_comparison.png
└── README.md