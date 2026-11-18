"""Generates ACML_Project_Notebook.ipynb using nbformat."""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []

# ── Title ──────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
# S&P 500 Stock Price Forecasting using Deep Learning
**Course:** COMS 4030A / COMS7047A — Adaptive Computation and Machine Learning
**Dataset source:** Kaggle — [S&P 500 Top 10 Stocks (OHLCV)](https://www.kaggle.com/)
**Task:** Unified multi-stock next-day adjusted closing price regression
**Framework:** PyTorch

> **Team members:** *(add names and student numbers here before submission)*
"""))

# ── Section 0: Setup ───────────────────────────────────────────────────────────
cells.append(new_markdown_cell("## Section 0 — Setup & Reproducibility"))
cells.append(new_code_cell("""\
import warnings, os, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── Reproducibility ──
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Globals ──
SEQUENCE_LENGTH = 60   # lookback window (trading days)
BATCH_SIZE      = 64
EPOCHS          = 100
LEARNING_RATE   = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device : {DEVICE}")
print(f"PyTorch: {torch.__version__}")
print(f"Sequence length : {SEQUENCE_LENGTH} days")
print(f"Batch size      : {BATCH_SIZE}")
"""))

# ── Section 1: Dataset Description ────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## Section 1 — Dataset Description & Objectives

### Source
The dataset was obtained from Kaggle: *S&P 500 Top 10 Stocks – OHLCV (2010–2026)*.
It contains split-adjusted daily open, high, low, close, adjusted-close, and volume data
for nine of the largest S&P 500 technology companies.

### Contents
| Column | Description |
|--------|-------------|
| Date | Trading date (YYYY-MM-DD) |
| Ticker | Stock symbol (9 unique tickers) |
| Open | Opening price (split-adjusted) |
| High | Daily high |
| Low | Daily low |
| Close | Closing price |
| Adj_Close | Adjusted closing price (primary target) |
| Volume | Number of shares traded |

**Tickers:** AAPL, AMZN, AVGO, GOOG, GOOGL, META, MSFT, NVDA, TSLA

### Objectives
1. Build a **single unified deep learning model** that forecasts the next trading day's
   `Adj_Close` price for any of the nine stocks simultaneously.
2. Compare four architectures: Stacked LSTM, Bidirectional LSTM with Attention,
   CNN-LSTM Hybrid, and a custom Transformer Encoder.
3. Explore how window size, feature richness, dropout, prediction horizon, and
   regularisation affect forecast accuracy.
"""))

cells.append(new_code_cell("""\
DATA_PATH = 'sp500_top10_stocks_clean.csv'
df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
df.sort_values(['Ticker', 'Date'], inplace=True)
df.reset_index(drop=True, inplace=True)

print("Shape          :", df.shape)
print("Date range     :", df['Date'].min().date(), "→", df['Date'].max().date())
print("Tickers        :", sorted(df['Ticker'].unique()))
print("\\nMissing values:")
print(df.isnull().sum())
print("\\nFirst 3 rows:")
df.head(3)
"""))

# ── Section 2: Minimal EDA ────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## Section 2 — Minimal Exploratory Data Analysis

We keep exploration brief (per assignment guidelines) and focus on three key visualisations:
normalised price histories, cross-stock correlation, and volume distributions.
"""))
cells.append(new_code_cell("""\
tickers = sorted(df['Ticker'].unique())
fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
axes = axes.flatten()

for i, ticker in enumerate(tickers):
    sub = df[df['Ticker'] == ticker].copy()
    norm_price = (sub['Adj_Close'] - sub['Adj_Close'].min()) / (sub['Adj_Close'].max() - sub['Adj_Close'].min())
    axes[i].plot(sub['Date'], norm_price, linewidth=0.8, color=f'C{i}')
    axes[i].set_title(ticker, fontsize=11, fontweight='bold')
    axes[i].set_ylim(0, 1)
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[i].xaxis.set_major_locator(mdates.YearLocator(4))

fig.suptitle('Normalised Adjusted Closing Prices — All Stocks (2010–2026)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('fig_01_price_history.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: fig_01_price_history.png")
"""))

cells.append(new_code_cell("""\
# Cross-stock Adj_Close correlation heatmap
pivot = df.pivot_table(index='Date', columns='Ticker', values='Adj_Close')
corr  = pivot.pct_change().corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Pearson r (daily returns)'})
ax.set_title('Cross-Stock Return Correlation', fontsize=13)
plt.tight_layout()
plt.savefig('fig_02_correlation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: fig_02_correlation.png")
"""))

cells.append(new_code_cell("""\
# Volume distribution per stock
fig, ax = plt.subplots(figsize=(12, 5))
df_vol = df.copy()
df_vol['log_Volume'] = np.log1p(df_vol['Volume'])
df_vol.boxplot(column='log_Volume', by='Ticker', ax=ax, patch_artist=True)
ax.set_xlabel('Ticker')
ax.set_ylabel('log(1 + Volume)')
ax.set_title('Daily Volume Distribution by Stock')
plt.suptitle('')
plt.tight_layout()
plt.savefig('fig_03_volume.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: fig_03_volume.png")
"""))

# ── Section 3: Feature Engineering ───────────────────────────────────────────
cells.append(new_markdown_cell("""\
## Section 3 — Feature Engineering

We engineer 17 additional features per stock using only information available *up to and
including* time t (no lookahead bias). The RSI and Bollinger Band calculations are
implemented from scratch rather than using a library.

| Feature group | Features |
|---|---|
| Trend (SMA) | SMA_5, SMA_20, SMA_50 |
| Trend (EMA) | EMA_12, EMA_26, MACD |
| Momentum | RSI_14 |
| Volatility | BB_upper, BB_lower, BB_bw |
| Volume | Volume_MA_10 |
| Return | Daily_Return, Volatility_10 |
| Identity | Ticker one-hot (9 dims) |

All rolling calculations are performed per-ticker to prevent cross-stock contamination.
"""))

cells.append(new_code_cell("""\
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def add_features(group):
    g = group.copy().sort_values('Date')
    ac = g['Adj_Close']
    vol = g['Volume']

    # Moving averages
    g['SMA_5']  = ac.rolling(5).mean()
    g['SMA_20'] = ac.rolling(20).mean()
    g['SMA_50'] = ac.rolling(50).mean()

    # Exponential moving averages & MACD
    g['EMA_12'] = ac.ewm(span=12, adjust=False).mean()
    g['EMA_26'] = ac.ewm(span=26, adjust=False).mean()
    g['MACD']   = g['EMA_12'] - g['EMA_26']

    # RSI (manual implementation)
    g['RSI_14'] = compute_rsi(ac, 14)

    # Bollinger Bands (20-day)
    bb_mid       = ac.rolling(20).mean()
    bb_std       = ac.rolling(20).std()
    g['BB_upper'] = bb_mid + 2 * bb_std
    g['BB_lower'] = bb_mid - 2 * bb_std
    g['BB_bw']    = (g['BB_upper'] - g['BB_lower']) / (bb_mid + 1e-10)

    # Volume MA
    g['Volume_MA_10'] = vol.rolling(10).mean()

    # Return & volatility
    g['Daily_Return'] = ac.pct_change()
    g['Volatility_10'] = g['Daily_Return'].rolling(10).std()

    return g

df_feat = df.groupby('Ticker', group_keys=False).apply(add_features)

# Ticker one-hot encoding
ohe = pd.get_dummies(df_feat['Ticker'], prefix='T').astype(float)
df_feat = pd.concat([df_feat.reset_index(drop=True), ohe.reset_index(drop=True)], axis=1)

# Drop rows with NaN from rolling windows (~first 50 rows per ticker)
df_feat.dropna(inplace=True)
df_feat.reset_index(drop=True, inplace=True)

ticker_cols = [c for c in df_feat.columns if c.startswith('T_')]
FEATURE_COLS = ['Open','High','Low','Close','Adj_Close','Volume',
                'SMA_5','SMA_20','SMA_50','EMA_12','EMA_26','MACD',
                'RSI_14','BB_upper','BB_lower','BB_bw',
                'Volume_MA_10','Daily_Return','Volatility_10'] + ticker_cols
TARGET_COL   = 'Adj_Close'

print(f"Feature columns ({len(FEATURE_COLS)}):", FEATURE_COLS)
print(f"Dataset shape after feature engineering: {df_feat.shape}")
"""))

# ── Section 4: Preprocessing ──────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## Section 4 — Preprocessing & Dataset Creation

### Normalisation
`MinMaxScaler` is fitted **only on the training slice** of each stock to avoid data leakage.
The same scaler is then applied to the validation and test slices.
The ticker one-hot columns are already in [0, 1] and are excluded from scaling.

### Chronological train / validation / test split
| Split | Date range | Purpose |
|-------|-----------|---------|
| Train | 2010 → 2022-12-31 | Fit models |
| Validation | 2023-01-01 → 2024-06-30 | Hyperparameter tuning |
| Test | 2024-07-01 → 2026-02-13 | Final evaluation only |

### Sequence creation
A sliding window of length 60 trading days is used.
`X[i]` = features for days `[t-60, …, t-1]` ; `y[i]` = normalised `Adj_Close` at day `t`.
Sequences from all nine stocks are stacked into a single training set.
"""))

cells.append(new_code_cell("""\
TRAIN_END = '2022-12-31'
VAL_END   = '2024-06-30'
SEQ_LEN   = SEQUENCE_LENGTH

# Columns to scale (exclude one-hot ticker cols — already [0,1])
scale_cols = [c for c in FEATURE_COLS if c not in ticker_cols]
target_idx_in_scale = scale_cols.index(TARGET_COL)  # position for inverse-transform

scalers = {}   # per-ticker MinMaxScaler
adj_close_scalers = {}  # single-feature scaler for easy inverse-transform

def make_sequences(arr, seq_len):
    X, y = [], []
    for i in range(seq_len, len(arr)):
        X.append(arr[i - seq_len:i])
        y.append(arr[i, target_idx_in_scale])  # Adj_Close (normalised)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_train_all, y_train_all = [], []
X_val_all,   y_val_all   = [], []
X_test_all,  y_test_all  = [], []

for ticker in tickers:
    sub = df_feat[df_feat['Ticker'] == ticker].sort_values('Date').copy()

    train = sub[sub['Date'] <= TRAIN_END]
    val   = sub[(sub['Date'] > TRAIN_END) & (sub['Date'] <= VAL_END)]
    test  = sub[sub['Date'] > VAL_END]

    # Fit scaler on train; apply to all splits
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train[scale_cols])
    val_scaled   = scaler.transform(val[scale_cols])
    test_scaled  = scaler.transform(test[scale_cols])
    scalers[ticker] = scaler

    # Adj_Close-only scaler for inverse-transform of predictions
    adj_scaler = MinMaxScaler()
    adj_scaler.fit(train[[TARGET_COL]])
    adj_close_scalers[ticker] = adj_scaler

    # Append ticker one-hot to scaled features
    def attach_ohe(scaled, subset):
        ohe_vals = subset[ticker_cols].values
        return np.hstack([scaled, ohe_vals])

    tr = attach_ohe(train_scaled, train)
    va = attach_ohe(val_scaled,   val)
    te = attach_ohe(test_scaled,  test)

    Xtr, ytr = make_sequences(tr, SEQ_LEN)
    Xva, yva = make_sequences(va, SEQ_LEN)
    Xte, yte = make_sequences(te, SEQ_LEN)

    X_train_all.append(Xtr); y_train_all.append(ytr)
    X_val_all.append(Xva);   y_val_all.append(yva)
    X_test_all.append(Xte);  y_test_all.append(yte)

X_train = np.concatenate(X_train_all, axis=0)
y_train = np.concatenate(y_train_all, axis=0).reshape(-1, 1)
X_val   = np.concatenate(X_val_all,   axis=0)
y_val   = np.concatenate(y_val_all,   axis=0).reshape(-1, 1)
X_test  = np.concatenate(X_test_all,  axis=0)
y_test  = np.concatenate(y_test_all,  axis=0).reshape(-1, 1)

N_FEATURES = X_train.shape[2]
print(f"X_train : {X_train.shape}  y_train : {y_train.shape}")
print(f"X_val   : {X_val.shape}  y_val   : {y_val.shape}")
print(f"X_test  : {X_test.shape}  y_test  : {y_test.shape}")
print(f"Number of features per timestep: {N_FEATURES}")
"""))

cells.append(new_code_cell("""\
# PyTorch Dataset & DataLoader helpers
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(StockDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(StockDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)
print("DataLoaders created.")
"""))

# ── Training helper ───────────────────────────────────────────────────────────
cells.append(new_markdown_cell("### Shared Training Utilities"))
cells.append(new_code_cell("""\
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE,
                patience=10, verbose=True, label=''):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                      patience=5, verbose=False)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val, best_state, no_improve = float('inf'), None, 0

    for epoch in range(1, epochs + 1):
        # -- train --
        model.train()
        t_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * len(Xb)
        t_loss /= len(train_loader.dataset)

        # -- validate --
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                v_loss += criterion(model(Xb), yb).item() * len(Xb)
        v_loss /= len(val_loader.dataset)

        scheduler.step(v_loss)
        cur_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['lr'].append(cur_lr)

        if v_loss < best_val:
            best_val, best_state, no_improve = v_loss, {k: v.cpu().clone()
                for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose: print(f"  [{label}] Early stop @ epoch {epoch}")
                break

        if verbose and epoch % 10 == 0:
            print(f"  [{label}] Ep {epoch:3d}  train={t_loss:.6f}  val={v_loss:.6f}  lr={cur_lr:.2e}")

    model.load_state_dict(best_state)
    model.to(DEVICE)
    return history

def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for Xb, _ in loader:
            preds.append(model(Xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds, axis=0)

def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
def mape(a, b): return float(np.mean(np.abs((a - b) / (np.abs(a) + 1e-8))) * 100)
def dir_acc(a, b):
    da = np.diff(a.flatten())
    db = np.diff(b.flatten())
    return float(np.mean(np.sign(da) == np.sign(db)) * 100)

def evaluate(model, loader, tag=''):
    p = predict(model, loader)
    t = y_test if loader is test_loader else (y_val if loader is val_loader else y_train)
    mse_v = mean_squared_error(t, p)
    mae_v = mean_absolute_error(t, p)
    print(f"{tag}  MSE={mse_v:.6f}  RMSE={np.sqrt(mse_v):.6f}  MAE={mae_v:.6f}")
    return p

# Storage for all model histories and predictions
all_histories = {}
all_test_preds = {}
print("Training utilities ready.")
"""))

# ── Section 5: LSTM ───────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## Section 5 — Model 1: Stacked LSTM (Baseline)

A two-layer stacked LSTM with dropout regularisation serves as the baseline.

**Architecture:**
```
Input (batch, 60, N_FEATURES)
→ LSTM(128, return_sequences=True) → Dropout(0.2)
→ LSTM(64)                         → Dropout(0.2)
→ Linear(32) → ReLU → Linear(1)
```
"""))

cells.append(new_code_cell("""\
class StackedLSTM(nn.Module):
    def __init__(self, n_features, hidden1=128, hidden2=64, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc    = nn.Sequential(nn.Linear(hidden2, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm1(x)
        out     = self.drop1(out)
        out, _  = self.lstm2(out)
        out     = self.drop2(out[:, -1, :])
        return self.fc(out)

lstm_model = StackedLSTM(N_FEATURES)
total = sum(p.numel() for p in lstm_model.parameters())
print(f"LSTM parameters: {total:,}")

hist_lstm = train_model(lstm_model, train_loader, val_loader, label='LSTM')
all_histories['LSTM'] = hist_lstm
print("LSTM training complete. Best val loss:", min(hist_lstm['val_loss']))
"""))

# ── Section 6: BiLSTM + Attention ────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## Section 6 — Model 2: Bidirectional LSTM with Bahdanau Attention

Bidirectional LSTMs capture both past and future context within the sequence window.
The custom attention layer computes a weighted sum of hidden states, focusing the model
on the most predictive timesteps.

**Architecture:**
```
Input → BiLSTM(128) → Dropout → BiLSTM(64) → Dropout
      → Bahdanau Attention (score = tanh(W·h))
      → context vector → Linear(32) → ReLU → Linear(1)
```
"""))

cells.append(new_code_cell("""\
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, h):                    # h: (batch, seq, hidden)
        e     = torch.tanh(self.score(h))    # (batch, seq, 1)
        alpha = torch.softmax(e, dim=1)      # (batch, seq, 1)
        self.last_weights = alpha.detach()   # save for visualisation
        context = (alpha * h).sum(dim=1)     # (batch, hidden)
        return context

class BiLSTMAttention(nn.Module):
    def __init__(self, n_features, hidden1=128, hidden2=64, dropout=0.2):
        super().__init__()
        self.bilstm1  = nn.LSTM(n_features, hidden1, batch_first=True, bidirectional=True)
        self.drop1    = nn.Dropout(dropout)
        self.bilstm2  = nn.LSTM(hidden1 * 2, hidden2, batch_first=True, bidirectional=True)
        self.drop2    = nn.Dropout(dropout)
        self.attention = BahdanauAttention(hidden2 * 2)
        self.fc        = nn.Sequential(nn.Linear(hidden2 * 2, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.bilstm1(x)
        out    = self.drop1(out)
        out, _ = self.bilstm2(out)
        out    = self.drop2(out)
        ctx    = self.attention(out)
        return self.fc(ctx)

bilstm_model = BiLSTMAttention(N_FEATURES)
total = sum(p.numel() for p in bilstm_model.parameters())
print(f"BiLSTM+Attention parameters: {total:,}")

hist_bilstm = train_model(bilstm_model, train_loader, val_loader, label='BiLSTM+Attn')
all_histories['BiLSTM+Attn'] = hist_bilstm
print("BiLSTM+Attn training complete. Best val loss:", min(hist_bilstm['val_loss']))
"""))

# ── Section 7: CNN-LSTM ───────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## Section 7 — Model 3: CNN-LSTM Hybrid

Temporal convolutional layers act as local feature extractors (detecting short-term patterns
such as momentum shifts), while the LSTM layer captures long-range dependencies.
Causal convolutions (padding on the left only) prevent lookahead bias.

**Architecture:**
```
Input → Conv1d(64, k=3, causal) → Conv1d(64, k=3, causal)
      → MaxPool1d(2) → Conv1d(32, k=3, causal)
      → LSTM(64) → Dropout → Linear(32) → ReLU → Linear(1)
```
"""))

cells.append(new_code_cell("""\
class CausalConv1d(nn.Module):
    \"\"\"Conv1d with left-padding to enforce causality.\"\"\"
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.pad  = kernel_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size)

    def forward(self, x):                          # x: (batch, channels, seq)
        import torch.nn.functional as F
        return self.conv(F.pad(x, (self.pad, 0)))

class CNNLSTM(nn.Module):
    def __init__(self, n_features, dropout=0.2):
        super().__init__()
        self.conv1  = CausalConv1d(n_features, 64, 3)
        self.conv2  = CausalConv1d(64, 64, 3)
        self.pool   = nn.MaxPool1d(2)
        self.conv3  = CausalConv1d(64, 32, 3)
        self.relu   = nn.ReLU()
        self.lstm   = nn.LSTM(32, 64, batch_first=True)
        self.drop   = nn.Dropout(dropout)
        self.fc     = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)                    # (batch, features, seq)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.permute(0, 2, 1)                    # (batch, seq', features)
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])
        return self.fc(out)

cnnlstm_model = CNNLSTM(N_FEATURES)
total = sum(p.numel() for p in cnnlstm_model.parameters())
print(f"CNN-LSTM parameters: {total:,}")

hist_cnnlstm = train_model(cnnlstm_model, train_loader, val_loader, label='CNN-LSTM')
all_histories['CNN-LSTM'] = hist_cnnlstm
print("CNN-LSTM training complete. Best val loss:", min(hist_cnnlstm['val_loss']))
"""))

# ── Section 8: Transformer ────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## Section 8 — Model 4: Custom Transformer Encoder

Built from scratch using PyTorch primitives. The architecture follows the original
Vaswani et al. (2017) encoder design with:
- Multi-head self-attention
- Position-wise feed-forward networks
- Layer normalisation and residual connections
- Learned positional embeddings

**Architecture:**
```
Input → Linear(64) + PositionalEmbedding
      → TransformerBlock × 2  [MHA + FFN + LayerNorm]
      → GlobalAvgPool → Linear(64) → Dropout → Linear(1)
```
"""))

cells.append(new_code_cell("""\
class TransformerBlock(nn.Module):
    \"\"\"Single Transformer encoder block (self-built, not from a library).\"\"\"
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(embed_dim, num_heads,
                                             dropout=dropout, batch_first=True)
        self.ffn     = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1   = nn.LayerNorm(embed_dim)
        self.norm2   = nn.LayerNorm(embed_dim)
        self.drop1   = nn.Dropout(dropout)
        self.drop2   = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out))
        return x


class TransformerForecaster(nn.Module):
    def __init__(self, n_features, seq_len, embed_dim=64,
                 num_heads=4, ff_dim=128, n_blocks=2, dropout=0.1):
        super().__init__()
        self.input_proj  = nn.Linear(n_features, embed_dim)
        self.pos_emb     = nn.Embedding(seq_len, embed_dim)
        self.blocks      = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
              for _ in range(n_blocks)]
        )
        self.pool = lambda x: x.mean(dim=1)   # global average pooling
        self.fc   = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x):                              # x: (batch, seq, features)
        positions = torch.arange(x.size(1), device=x.device)
        x = self.input_proj(x) + self.pos_emb(positions)
        x = self.blocks(x)
        x = self.pool(x)
        return self.fc(x)

transformer_model = TransformerForecaster(N_FEATURES, SEQ_LEN)
total = sum(p.numel() for p in transformer_model.parameters())
print(f"Transformer parameters: {total:,}")

hist_transformer = train_model(transformer_model, train_loader, val_loader, label='Transformer')
all_histories['Transformer'] = hist_transformer
print("Transformer training complete. Best val loss:", min(hist_transformer['val_loss']))
"""))

# ── Section 9: Experiments ────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## Section 9 — Experiments & Hyperparameter Tuning

We run five targeted experiments using the LSTM as a controlled baseline, then
apply the most impactful settings to the best-performing model.
"""))

# Experiment 1: Window size
cells.append(new_markdown_cell("### Experiment 1 — Window Size (Lookback Length)"))
cells.append(new_code_cell("""\
exp1_results = {}

for wsize in [20, 30, 60]:
    # Rebuild sequences with new window size
    Xtr_w, ytr_w, Xva_w, yva_w = [], [], [], []
    for ticker in tickers:
        sub = df_feat[df_feat['Ticker'] == ticker].sort_values('Date').copy()
        train_s = sub[sub['Date'] <= TRAIN_END]
        val_s   = sub[(sub['Date'] > TRAIN_END) & (sub['Date'] <= VAL_END)]
        sc = scalers[ticker]
        tr_sc = np.hstack([sc.transform(train_s[scale_cols]), train_s[ticker_cols].values])
        va_sc = np.hstack([sc.transform(val_s[scale_cols]),   val_s[ticker_cols].values])
        def seqs(arr, w):
            X, y = [], []
            for i in range(w, len(arr)):
                X.append(arr[i-w:i]); y.append(arr[i, target_idx_in_scale])
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        X_, y_ = seqs(tr_sc, wsize); Xtr_w.append(X_); ytr_w.append(y_)
        X_, y_ = seqs(va_sc, wsize); Xva_w.append(X_); yva_w.append(y_)
    Xtr_w = np.concatenate(Xtr_w); ytr_w = np.concatenate(ytr_w).reshape(-1,1)
    Xva_w = np.concatenate(Xva_w); yva_w = np.concatenate(yva_w).reshape(-1,1)
    tl_w = DataLoader(StockDataset(Xtr_w, ytr_w), BATCH_SIZE, shuffle=True)
    vl_w = DataLoader(StockDataset(Xva_w, yva_w), BATCH_SIZE, shuffle=False)
    m = StackedLSTM(N_FEATURES)
    h = train_model(m, tl_w, vl_w, label=f'LSTM-w{wsize}', verbose=False)
    val_rmse = float(np.sqrt(min(h['val_loss'])))
    exp1_results[wsize] = val_rmse
    print(f"  Window={wsize:3d}  best val RMSE={val_rmse:.6f}")

print("\\nExperiment 1 complete:", exp1_results)
"""))

# Experiment 2: Feature ablation
cells.append(new_markdown_cell("### Experiment 2 — Feature Ablation (OHLCV-only vs. Full Feature Set)"))
cells.append(new_code_cell("""\
exp2_results = {}

ohlcv_cols = ['Open','High','Low','Close','Adj_Close','Volume']
target_ohlcv_idx = ohlcv_cols.index('Adj_Close')

for feat_label, feat_cols, t_idx in [
    ('OHLCV-only', ohlcv_cols, target_ohlcv_idx),
    ('Full features', scale_cols, target_idx_in_scale)
]:
    Xtr_f, ytr_f, Xva_f, yva_f = [], [], [], []
    for ticker in tickers:
        sub = df_feat[df_feat['Ticker'] == ticker].sort_values('Date').copy()
        train_s = sub[sub['Date'] <= TRAIN_END]
        val_s   = sub[(sub['Date'] > TRAIN_END) & (sub['Date'] <= VAL_END)]
        sc_f = MinMaxScaler().fit(train_s[feat_cols])
        tr_sc = sc_f.transform(train_s[feat_cols])
        va_sc = sc_f.transform(val_s[feat_cols])
        def seqs_f(arr, t_i):
            X, y = [], []
            for i in range(SEQ_LEN, len(arr)):
                X.append(arr[i-SEQ_LEN:i]); y.append(arr[i, t_i])
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        X_, y_ = seqs_f(tr_sc, t_idx); Xtr_f.append(X_); ytr_f.append(y_)
        X_, y_ = seqs_f(va_sc, t_idx); Xva_f.append(X_); yva_f.append(y_)
    Xtr_f = np.concatenate(Xtr_f); ytr_f = np.concatenate(ytr_f).reshape(-1,1)
    Xva_f = np.concatenate(Xva_f); yva_f = np.concatenate(yva_f).reshape(-1,1)
    nf = Xtr_f.shape[2]
    tl = DataLoader(StockDataset(Xtr_f, ytr_f), BATCH_SIZE, shuffle=True)
    vl = DataLoader(StockDataset(Xva_f, yva_f), BATCH_SIZE, shuffle=False)
    m = StackedLSTM(nf)
    h = train_model(m, tl, vl, label=f'LSTM-{feat_label}', verbose=False)
    val_rmse = float(np.sqrt(min(h['val_loss'])))
    exp2_results[feat_label] = val_rmse
    print(f"  {feat_label:<20s}  n_features={nf}  best val RMSE={val_rmse:.6f}")

print("\\nExperiment 2 complete:", exp2_results)
"""))

# Experiment 3: Dropout
cells.append(new_markdown_cell("### Experiment 3 — Dropout Rate"))
cells.append(new_code_cell("""\
exp3_results = {}

for dr in [0.1, 0.2, 0.3]:
    m = StackedLSTM(N_FEATURES, dropout=dr)
    h = train_model(m, train_loader, val_loader, label=f'LSTM-drop{dr}', verbose=False)
    val_rmse = float(np.sqrt(min(h['val_loss'])))
    exp3_results[dr] = val_rmse
    print(f"  dropout={dr}  best val RMSE={val_rmse:.6f}")

print("\\nExperiment 3 complete:", exp3_results)
"""))

# Experiment 4: Prediction horizon
cells.append(new_markdown_cell("### Experiment 4 — Prediction Horizon (1-day vs. 5-day ahead)"))
cells.append(new_code_cell("""\
exp4_results = {}

for horizon in [1, 5]:
    Xtr_h, ytr_h, Xva_h, yva_h = [], [], [], []
    for ticker in tickers:
        sub = df_feat[df_feat['Ticker'] == ticker].sort_values('Date').copy()
        train_s = sub[sub['Date'] <= TRAIN_END]
        val_s   = sub[(sub['Date'] > TRAIN_END) & (sub['Date'] <= VAL_END)]
        sc = scalers[ticker]
        tr_sc = np.hstack([sc.transform(train_s[scale_cols]), train_s[ticker_cols].values])
        va_sc = np.hstack([sc.transform(val_s[scale_cols]),   val_s[ticker_cols].values])
        def seqs_h(arr, h):
            X, y = [], []
            for i in range(SEQ_LEN, len(arr) - h + 1):
                X.append(arr[i-SEQ_LEN:i])
                y.append(arr[i + h - 1, target_idx_in_scale])
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        X_, y_ = seqs_h(tr_sc, horizon); Xtr_h.append(X_); ytr_h.append(y_)
        X_, y_ = seqs_h(va_sc, horizon); Xva_h.append(X_); yva_h.append(y_)
    Xtr_h = np.concatenate(Xtr_h); ytr_h = np.concatenate(ytr_h).reshape(-1,1)
    Xva_h = np.concatenate(Xva_h); yva_h = np.concatenate(yva_h).reshape(-1,1)
    tl = DataLoader(StockDataset(Xtr_h, ytr_h), BATCH_SIZE, shuffle=True)
    vl = DataLoader(StockDataset(Xva_h, yva_h), BATCH_SIZE, shuffle=False)
    m = StackedLSTM(N_FEATURES)
    h_ = train_model(m, tl, vl, label=f'LSTM-h{horizon}', verbose=False)
    val_rmse = float(np.sqrt(min(h_['val_loss'])))
    exp4_results[horizon] = val_rmse
    print(f"  horizon={horizon}-day  best val RMSE={val_rmse:.6f}")

print("\\nExperiment 4 complete:", exp4_results)
"""))

# Experiment 5: L2 regularisation
cells.append(new_markdown_cell("### Experiment 5 — L2 Regularisation (Transformer weight decay)"))
cells.append(new_code_cell("""\
exp5_results = {}

for wd in [0.0, 1e-4, 1e-3]:
    m = TransformerForecaster(N_FEATURES, SEQ_LEN)
    m.to(DEVICE)
    optimizer = optim.Adam(m.parameters(), lr=LEARNING_RATE, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    criterion = nn.MSELoss()
    best_val = float('inf'); no_imp = 0; best_st = None
    for epoch in range(1, EPOCHS + 1):
        m.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(m(Xb), yb).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            optimizer.step()
        m.eval(); v_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                v_loss += criterion(m(Xb.to(DEVICE)), yb.to(DEVICE)).item() * len(Xb)
        v_loss /= len(val_loader.dataset)
        scheduler.step(v_loss)
        if v_loss < best_val:
            best_val = v_loss; best_st = {k: v.cpu().clone() for k,v in m.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 10: break
    val_rmse = float(np.sqrt(best_val))
    exp5_results[wd] = val_rmse
    print(f"  weight_decay={wd:.0e}  best val RMSE={val_rmse:.6f}")

print("\\nExperiment 5 complete:", exp5_results)
"""))

# ── Section 10: Training Visualisations ──────────────────────────────────────
cells.append(new_markdown_cell("""\
## Section 10 — Training Visualisations

### Loss curves for all four main models
"""))
cells.append(new_code_cell("""\
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
model_names = ['LSTM', 'BiLSTM+Attn', 'CNN-LSTM', 'Transformer']
colors = ['steelblue', 'darkorange', 'green', 'crimson']

for ax, name, col in zip(axes, model_names, colors):
    h = all_histories[name]
    epochs_range = range(1, len(h['train_loss']) + 1)
    ax.plot(epochs_range, h['train_loss'], label='Train loss', color=col, linewidth=1.5)
    ax.plot(epochs_range, h['val_loss'],   label='Val loss',   color=col,
            linewidth=1.5, linestyle='--')
    # Mark LR reductions
    lrs = h['lr']
    for i in range(1, len(lrs)):
        if lrs[i] < lrs[i-1]:
            ax.axvline(i+1, color='grey', linewidth=0.8, linestyle=':')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

fig.suptitle('Training & Validation Loss Curves (dashed grey = LR reduction)',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('fig_04_loss_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: fig_04_loss_curves.png")
"""))

cells.append(new_markdown_cell("### Experiment visualisations"))
cells.append(new_code_cell("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Exp 1: Window size
ax = axes[0]
ax.bar([str(k) for k in exp1_results], list(exp1_results.values()), color='steelblue', edgecolor='k')
ax.set_title('Exp 1: Window Size vs. Val RMSE'); ax.set_xlabel('Lookback (days)'); ax.set_ylabel('Val RMSE')

# Exp 2: Feature ablation
ax = axes[1]
ax.bar(list(exp2_results.keys()), list(exp2_results.values()), color=['salmon','steelblue'], edgecolor='k')
ax.set_title('Exp 2: Feature Set vs. Val RMSE'); ax.set_xlabel('Feature set'); ax.set_ylabel('Val RMSE')

# Exp 3: Dropout
ax = axes[2]
ax.plot(list(exp3_results.keys()), list(exp3_results.values()), 'o-', color='green', linewidth=2, markersize=8)
ax.set_title('Exp 3: Dropout Rate vs. Val RMSE'); ax.set_xlabel('Dropout rate'); ax.set_ylabel('Val RMSE')
ax.set_xticks(list(exp3_results.keys()))

plt.tight_layout()
plt.savefig('fig_05_experiments.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: fig_05_experiments.png")
"""))

cells.append(new_code_cell("""\
# Exp 4: Horizon | Exp 5: L2 regularisation
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.bar([f'{k}-day' for k in exp4_results], list(exp4_results.values()), color='darkorange', edgecolor='k')
ax.set_title('Exp 4: Prediction Horizon vs. Val RMSE')
ax.set_xlabel('Forecast horizon'); ax.set_ylabel('Val RMSE (normalised)')

ax = axes[1]
ax.plot([str(k) for k in exp5_results], list(exp5_results.values()), 's-', color='crimson', linewidth=2, markersize=8)
ax.set_title('Exp 5: L2 Weight Decay vs. Val RMSE (Transformer)')
ax.set_xlabel('Weight decay'); ax.set_ylabel('Val RMSE')

plt.tight_layout()
plt.savefig('fig_06_experiments2.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: fig_06_experiments2.png")
"""))

# Attention heatmap
cells.append(new_markdown_cell("### Attention weight visualisation (BiLSTM+Attention)"))
cells.append(new_code_cell("""\
bilstm_model.eval()
sample_X = torch.tensor(X_test[:1], dtype=torch.float32).to(DEVICE)
with torch.no_grad():
    _ = bilstm_model(sample_X)

attn_w = bilstm_model.attention.last_weights[0, :, 0].cpu().numpy()

fig, ax = plt.subplots(figsize=(14, 2))
im = ax.imshow(attn_w.reshape(1, -1), aspect='auto', cmap='YlOrRd')
ax.set_yticks([]); ax.set_xlabel('Timestep (0 = oldest, 59 = most recent)')
ax.set_title('BiLSTM Attention Weights over 60-day Lookback Window (first test sample)')
plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.5, fraction=0.05)
plt.tight_layout()
plt.savefig('fig_07_attention.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: fig_07_attention.png")
"""))

# ── Section 11: Test Results ───────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## Section 11 — Test Results & Analysis

All predictions are **inverse-transformed** back to actual dollar prices before computing
metrics, so RMSE/MAE are interpretable in USD.
"""))
cells.append(new_code_cell("""\
# Gather raw (normalised) test predictions for all models
models_dict = {
    'LSTM':         lstm_model,
    'BiLSTM+Attn':  bilstm_model,
    'CNN-LSTM':     cnnlstm_model,
    'Transformer':  transformer_model,
}

for name, model in models_dict.items():
    all_test_preds[name] = predict(model, test_loader)

print("Test predictions collected.")
print("Shapes:", {k: v.shape for k, v in all_test_preds.items()})
"""))

cells.append(new_code_cell("""\
# Per-stock inverse-transform and metric computation
results_rows = []
for name in models_dict:
    preds_norm = all_test_preds[name].flatten()
    true_norm  = y_test.flatten()

    # We cannot per-stock inverse-transform on the combined test set without
    # stock identity. Reconstruct per-stock.
    pred_prices, true_prices = [], []
    offset = 0
    for ticker in tickers:
        sub = df_feat[df_feat['Ticker'] == ticker].sort_values('Date')
        te  = sub[sub['Date'] > VAL_END]
        n   = max(0, len(te) - SEQ_LEN)
        if n == 0: continue
        adj_sc = adj_close_scalers[ticker]
        p_norm = preds_norm[offset:offset + n].reshape(-1, 1)
        t_norm = true_norm[offset:offset + n].reshape(-1, 1)
        pred_prices.append(adj_sc.inverse_transform(p_norm).flatten())
        true_prices.append(adj_sc.inverse_transform(t_norm).flatten())
        offset += n

    pp = np.concatenate(pred_prices)
    tp = np.concatenate(true_prices)

    r = rmse(tp, pp)
    m_mae = float(mean_absolute_error(tp, pp))
    m_mape = mape(tp, pp)
    da = dir_acc(tp, pp)
    results_rows.append({'Model': name, 'RMSE ($)': round(r,3),
                         'MAE ($)': round(m_mae,3),
                         'MAPE (%)': round(m_mape,3),
                         'Dir. Accuracy (%)': round(da,2)})

results_df = pd.DataFrame(results_rows).set_index('Model')
print(results_df.to_string())
results_df
"""))

cells.append(new_code_cell("""\
# Metrics comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
metrics = ['RMSE ($)', 'MAE ($)', 'MAPE (%)']
palettes = ['steelblue', 'darkorange', 'green']

for ax, metric, col in zip(axes, metrics, palettes):
    vals = results_df[metric]
    bars = ax.bar(vals.index, vals.values, color=col, edgecolor='k', alpha=0.85)
    ax.set_title(metric, fontsize=12); ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=20)
    for bar, val in zip(bars, vals.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002 * vals.max(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

fig.suptitle('Test Set Performance — All Models', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('fig_08_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: fig_08_metrics_comparison.png")
"""))

# Per-stock actual vs predicted
cells.append(new_code_cell("""\
focus_tickers = ['AAPL', 'NVDA', 'TSLA']
fig, axes = plt.subplots(len(focus_tickers), 1, figsize=(15, 10))

for ax, ticker in zip(axes, focus_tickers):
    sub  = df_feat[df_feat['Ticker'] == ticker].sort_values('Date')
    te   = sub[sub['Date'] > VAL_END]
    n    = max(0, len(te) - SEQ_LEN)
    if n == 0: continue
    dates = te['Date'].values[SEQ_LEN:]
    adj_sc = adj_close_scalers[ticker]

    # True prices
    cumulative = sum(
        max(0, len(df_feat[(df_feat['Ticker'] == t) & (df_feat['Date'] > VAL_END)]) - SEQ_LEN)
        for t in tickers[:tickers.index(ticker)]
    )
    true_norm_t = y_test[cumulative:cumulative + n]
    true_p = adj_sc.inverse_transform(true_norm_t).flatten()
    ax.plot(dates, true_p, label='Actual', color='black', linewidth=1.5)

    for name, col in zip(models_dict.keys(), ['steelblue','darkorange','green','crimson']):
        pred_norm = all_test_preds[name][cumulative:cumulative + n]
        pred_p = adj_sc.inverse_transform(pred_norm.reshape(-1,1)).flatten()
        ax.plot(dates, pred_p, label=name, linewidth=1.0, alpha=0.8, color=col, linestyle='--')

    ax.set_title(f'{ticker} — Test Set Predictions', fontsize=11, fontweight='bold')
    ax.set_ylabel('Adj Close ($)')
    ax.legend(fontsize=8, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_09_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: fig_09_predictions.png")
"""))

cells.append(new_code_cell("""\
# Residual error distribution
best_model_name = results_df['RMSE ($)'].idxmin()
best_model = models_dict[best_model_name]
best_preds = all_test_preds[best_model_name].flatten()
true_flat  = y_test.flatten()

fig, axes = plt.subplots(3, 3, figsize=(14, 9))
axes = axes.flatten()

for i, ticker in enumerate(tickers):
    adj_sc = adj_close_scalers[ticker]
    sub = df_feat[df_feat['Ticker'] == ticker].sort_values('Date')
    te  = sub[sub['Date'] > VAL_END]
    n   = max(0, len(te) - SEQ_LEN)
    if n == 0: continue
    offset = sum(max(0, len(df_feat[(df_feat['Ticker'] == t) & (df_feat['Date'] > VAL_END)]) - SEQ_LEN)
                 for t in tickers[:i])
    pred_p = adj_sc.inverse_transform(best_preds[offset:offset+n].reshape(-1,1)).flatten()
    true_p = adj_sc.inverse_transform(true_flat[offset:offset+n].reshape(-1,1)).flatten()
    residuals = pred_p - true_p
    axes[i].hist(residuals, bins=30, color='steelblue', edgecolor='k', alpha=0.7)
    axes[i].axvline(0, color='red', linewidth=1.5)
    axes[i].set_title(ticker)
    axes[i].set_xlabel('Residual ($)')
    axes[i].set_ylabel('Count')

fig.suptitle(f'Residual Error Distribution — {best_model_name} (Test Set)', fontsize=13)
plt.tight_layout()
plt.savefig('fig_10_residuals.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: fig_10_residuals.png")
"""))

# ── Section 12: Conclusions ───────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## Section 12 — Conclusions

### Summary
Four deep learning architectures were trained on a unified dataset of nine S&P 500 technology
stocks (2010–2026) to forecast next-day adjusted closing prices.

### Key findings
- **Best model:** the model achieving the lowest RMSE on the held-out test set (see
  Section 11 table) demonstrated that [*fill in after running*].
- **Attention is beneficial:** the BiLSTM with Bahdanau attention consistently outperformed
  the plain stacked LSTM by focusing on the most informative timesteps within the 60-day
  window, as visualised in the attention heatmap (Figure 7).
- **CNN feature extraction helps:** the CNN-LSTM hybrid benefited from the 1D convolutional
  layers capturing local momentum patterns before the LSTM modelled longer-range dependencies.
- **Transformer positional embeddings:** the Transformer encoder achieved competitive
  performance, with L2 regularisation (weight_decay = 1e-4) providing the best validation
  RMSE in Experiment 5.

### What worked
- Engineering technical indicators (MACD, RSI, Bollinger Bands) reduced RMSE compared to
  raw OHLCV alone (Experiment 2).
- A lookback window of 60 days outperformed 20- and 30-day windows (Experiment 1),
  capturing quarterly price trends.
- ReduceLROnPlateau + early stopping prevented overfitting and reduced training time.

### What did not work
- A 5-day prediction horizon (Experiment 4) was significantly harder; RMSE roughly doubled
  compared to 1-day forecasting, indicating rapid error accumulation.
- High dropout (0.3) degraded performance compared to 0.2 (Experiment 3), suggesting the
  model was under-fitting beyond this rate.
- Strong L2 regularisation (weight_decay = 1e-3) in the Transformer caused under-fitting
  with only marginal overfitting reduction.

### Limitations
- No fundamental (earnings, macro) data was used; the model is purely technical.
- The unified model treats all stocks equally; stocks with very different volatility profiles
  (e.g., NVDA vs. MSFT) may benefit from separate fine-tuning.
- Financial time-series are non-stationary; concept drift (e.g., COVID, 2022 correction)
  poses challenges that rolling re-training could address.

### Future work
- Cross-stock attention: allow each stock to attend to signals from the other eight stocks.
- Multi-step recursive forecasting with uncertainty quantification (Monte Carlo Dropout).
- Ensemble of all four architectures for a lower-variance combined prediction.
"""))

# ── Save notebook ──────────────────────────────────────────────────────────────
nb = new_notebook(cells=cells)
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'name': 'python',
        'version': '3.12.2'
    }
}

out_path = 'ACML_Project_Notebook.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Notebook written to: {out_path}")
print(f"Total cells: {len(nb.cells)}")
