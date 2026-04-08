# Demand Forecasting Project — Full Build Prompt

## Phase 0: Data Inventory

You have two datasets downloaded locally:

**Walmart Sales Dataset** — ~45 stores, ~100 departments, weekly sales, plus markdowns, temperature, fuel price, CPI, unemployment, holiday flags. Roughly 2010–2012.

**M5 Competition Dataset** — ~30,000 SKU-level products across 10 Walmart stores in 3 states (CA, TX, WI). Daily sales from 2011–2016. Includes `calendar.csv` (events, SNAP days), `sell_prices.csv`, and `sales_train_validation.csv`.

The M5 dataset is the backbone — it's larger, daily granularity, longer time range, and has price data. The Walmart Sales dataset adds macro features (CPI, unemployment, fuel price, temperature) at the store-week level. You'll merge them to get a richer feature set than either alone.

---

## Phase 1: Data Loading and Merging

1. Load M5's `sales_train_validation.csv`. It's wide format (one column per day). Melt it to long format: columns = `[item_id, store_id, date, sales]`.
2. Load `calendar.csv`. Merge on `date` to get `weekday`, `month`, `year`, `event_name_1`, `event_name_2`, `snap_CA`, `snap_TX`, `snap_WI`.
3. Load `sell_prices.csv`. Merge on `[store_id, item_id, wm_yr_wk]` to get the selling price per item per week.
4. Load Walmart Sales dataset. It has `Store`, `Date`, `Weekly_Sales`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`, `IsHoliday`. The M5 stores are numbered 1–10 and correspond to specific Walmart stores. You won't get a perfect 1:1 store match, so aggregate the Walmart Sales macro features (Temperature, Fuel_Price, CPI, Unemployment) by **state and week**. M5 tells you which state each store is in (`store_id` starts with `CA_`, `TX_`, `WI_`). Map Walmart Sales stores to states using store metadata or just use the national average per week as a proxy.
5. Merge these macro features into your M5 long-format dataframe by `[state, wm_yr_wk]`.

**Result:** A single dataframe with daily item-store sales, price, calendar features, event flags, SNAP flags, and macro indicators.

---

## Phase 2: Feature Engineering

This is where you win or lose on accuracy.

### Time Features
- `day_of_week`, `day_of_month`, `week_of_year`, `month`, `quarter`, `year`
- `is_weekend` (binary)
- `is_month_start`, `is_month_end`
- Days since last event, days until next event

### Lag Features (Critical for LSTM and Tree Models)
- `lag_7`, `lag_14`, `lag_28` (sales 7/14/28 days ago)
- `rolling_mean_7`, `rolling_mean_28`, `rolling_std_7`, `rolling_std_28`
- `rolling_mean_7 / rolling_mean_28` (short-term vs long-term momentum ratio)
- `expanding_mean` (cumulative average up to that point)

### Price Features
- `price_change_pct` (week-over-week price change)
- `price_relative_to_mean` (current price / historical mean price for that item)
- `price_relative_to_category` (item price / average price in its category)

### Event/Holiday Encoding
- One-hot or target-encode `event_name_1` and `event_name_2`
- Binary `has_event`
- SNAP day flag (use the one matching the store's state)

### Macro Features (From Walmart Sales Merge)
- `CPI`, `unemployment`, `fuel_price`, `temperature`
- Lag and difference versions of these (e.g., `CPI_change_month`)

### Category Encodings
- M5 items have `cat_id` (FOODS, HOUSEHOLD, HOBBIES) and `dept_id`. Target-encode these or use label encoding.

---

## Phase 3: Aggregation Level Decision

Running 30,000 individual SKU models is computationally brutal and most will be noisy (lots of zero-sale days). Do this instead:

**Primary modeling level:** Aggregate to **department × store × day**. That gives you roughly 70 time series (7 departments × 10 stores). Each one has enough volume to show clear patterns.

**Secondary (stretch goal):** Pick 5–10 high-volume individual items and model those separately to show you can handle sparse, noisy data.

---

## Phase 4: Train/Test Split

Use a **temporal split**, never random. Last 28 days of data = test set. Everything before = training. For validation during tuning, hold out the 28 days before the test set.

```
|-------- Training --------|-- Validation (28d) --|-- Test (28d) --|
```

---

## Phase 5: Model 1 — SARIMA (Statistical Baseline)

For each of your ~70 department-store series:

1. Plot the series. Visually confirm trend and seasonality.
2. Run ADF test (`statsmodels.tsa.stattools.adfuller`) to check stationarity. If non-stationary, difference once (d=1). If seasonal non-stationarity, seasonal difference (D=1, m=7 for weekly seasonality).
3. Plot ACF and PACF on the differenced series. Use these to pick starting values for p, q, P, Q.
4. Use `pmdarima.auto_arima` with `seasonal=True, m=7` to auto-search. This saves you from manually tuning all 70 series. Set `stepwise=True` for speed.
5. Forecast 28 days. Store predictions.

**Expect:** SARIMA will do okay on smooth, high-volume series and poorly on noisy ones. That's the point — you're establishing a classical baseline.

---

## Phase 6: Model 2 — Prophet (Quick Win)

For each series:

```python
from prophet import Prophet
import pandas as pd

df_prophet = series_df[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model.add_country_holidays(country_name='US')
model.fit(df_prophet)
future = model.make_future_dataframe(periods=28)
forecast = model.predict(future)
```

Add regressors for extra power:

```python
model.add_regressor('snap_flag')
model.add_regressor('has_event')
model.add_regressor('price_relative_to_mean')
```

These go into both the training df and the future df.

**Expect:** Prophet will beat SARIMA on series with strong holiday effects and multiple seasonalities. It'll struggle with price-driven demand shifts.

---

## Phase 7: Model 3 — LSTM (Deep Learning)

This is the hardest part. Do it right.

### Data Prep
1. Scale all features to [0, 1] using `MinMaxScaler`. Fit scaler on training data only.
2. Create sequences: for each time step t, the input is `[features at t-lookback, ..., features at t-1]` and the target is `sales at t`. Use `lookback = 28`.
3. Shape: `(num_samples, lookback, num_features)`.

### Architecture (PyTorch)

```python
import torch
import torch.nn as nn

class SalesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc2(self.dropout(self.relu(self.fc1(last_hidden))))
        return out
```

### Training
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-5
- Scheduler: `ReduceLROnPlateau(patience=5, factor=0.5)`
- Loss: MSELoss (or Huber loss if you have outliers — `nn.SmoothL1Loss()`)
- Batch size: 256
- Epochs: 50–100, early stopping on validation loss with patience=10
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

### 28-Day Forecast Strategy
Two options:
- **Recursive:** Predict day 1, feed that prediction back as input, predict day 2, etc. Simple but error compounds.
- **Direct (better):** Train 28 separate models, one for each horizon. Or train one model with a horizon embedding. More work, less error drift.

Use recursive for v1, note in writeup that direct multi-step is the production approach.

---

## Phase 8: Model 4 — LightGBM (Add This — Likely Best Model)

Tree-based models consistently beat LSTMs on tabular time series. Add LightGBM to your comparison.

```python
import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 127,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbosity': -1
}

model = lgb.train(params, train_data, valid_sets=[valid_data],
                  num_boost_round=1000, callbacks=[lgb.early_stopping(50)])
```

Feed it all the features from Phase 2 directly — no sequence formatting needed. LightGBM eats lag features for breakfast.

**This will likely be your best model.** The M5 competition was won by tree-based ensembles, not deep learning.

---

## Phase 9: Evaluation

On the 28-day test set, compute for each model:

- **RMSE** (Root Mean Squared Error) — penalizes big misses
- **MAE** (Mean Absolute Error) — average magnitude of error
- **MAPE** (Mean Absolute Percentage Error) — interpretable for business people, but undefined when actual = 0, so use sMAPE or WRMSSE instead
- **WRMSSE** (Weighted Root Mean Squared Scaled Error) — this is the actual M5 competition metric. Use it. Formula is in the M5 competition guide.

### Visualizations
1. Table comparing all four models on all metrics.
2. Time series plot: actual vs predicted for each model, overlaid, for 3–4 representative series (one smooth, one noisy, one with holiday spike).
3. Error distribution histogram per model.
4. Feature importance plot for LightGBM (built-in) and SHAP values.
5. Residual plots: residuals over time. Look for systematic patterns (means your model is missing something).

---

## Phase 10: Business Impact Section

Frame it like this:

> "Across the test period, our best model (LightGBM) achieves a MAPE of X%. The SARIMA baseline has a MAPE of Y%."

Then quantify:

### Inventory Cost Model
- Assume holding cost = 25% of item cost per year (industry standard).
- Overstock cost = holding cost × excess units × days held.
- Stockout cost = lost margin × missed units. Assume margin = 30% of sell price.
- Calculate total overstock and stockout costs under each model's forecast vs actual.
- Show the dollar savings of the best model vs the baseline.

### Example Framing
"Reducing forecast error from 15% (SARIMA) to 8% (LightGBM) across 10 stores would reduce annual excess inventory costs by an estimated $X and prevent $Y in lost sales from stockouts, for a combined savings of $Z."

Use the actual sell_prices from M5 to make this calculation real, not hypothetical.

---

## Phase 11: Deliverable Structure

Jupyter notebook or set of `.py` scripts. Sections:

1. **Introduction** — problem statement, dataset descriptions, why you merged two sources
2. **EDA** — distributions, seasonality decomposition, correlation heatmap, missing data analysis
3. **Feature Engineering** — full list with rationale
4. **Modeling** — each model in its own section with code, tuning process, and individual results
5. **Comparison** — head-to-head table + plots
6. **Business Impact** — dollar-value analysis
7. **Conclusions** — which model wins, why, what you'd do next (e.g., ensemble, hierarchical forecasting, adding external data like Google Trends)

---

## Key Things That Will Maximize Accuracy

1. **LightGBM with heavy feature engineering** will almost certainly beat LSTM and the statistical models. The M5 leaderboard proved this.
2. **Lag features are everything.** Sales 7 days ago is the single most predictive feature for sales today. Don't skip rolling stats.
3. **Aggregate first, then optionally go granular.** Dept-store level models are more stable. Item-level is noisy.
4. **The dataset merge is your differentiator.** Nobody else combining these two datasets has CPI, unemployment, and temperature alongside price and SNAP data. That's your edge for the writeup.
5. **WRMSSE as your primary metric** shows you know the domain, not just generic ML.
6. **Ensemble (optional but powerful):** Weighted average of LightGBM + Prophet predictions. Use validation set performance to set weights. Often beats any individual model by 2–5%.