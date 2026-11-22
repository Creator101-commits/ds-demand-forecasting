"""Load M5 competition files into panels and metadata (protocol v2).

Files (Kaggle competition ``m5-forecasting-accuracy``):
- ``sales_train_evaluation.csv``: d_1..d_1941 for 30,490 series. This is
  the only sales file; the organizers' private-window truth
  (d_1942..d_1969) comes from a separate verified file via
  :func:`load_test_truth`.
- ``calendar.csv``: d_1..d_1969 with real dates, week ids, events, SNAP.
- ``sell_prices.csv``: weekly sell price per store x item.

The sales panel covers d_1..d_1941 while calendar and prices span
d_1..d_1969, so ``price_days`` has one column per calendar day. Unit
sales are float32; dollar-sales weights are computed in float64 from
the weekly CSV prices via :func:`series_dollar_sales_window`.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
# The organizers' files carry no ``id`` column; series identity is the
# five structural columns.
SERIES_KEY = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
CAL_COLUMNS = [
    "d",
    "date",
    "wm_yr_wk",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]
PRICE_COLS = ["store_id", "item_id", "wm_yr_wk", "sell_price"]

COMPETITION_FILES = (
    "sales_train_evaluation.csv",
    "calendar.csv",
    "sell_prices.csv",
)

# Protocol m5-benchmark-v2 day boundaries (1-based d_N, inclusive).
# Train d_1..d_1913, validation d_1914..d_1941 (public leaderboard
# window, origin d_1913), test d_1942..d_1969 (private leaderboard
# window, origin d_1941) scored against the organizers' truth file.
TRAIN_END_DAY = 1913
VAL_END_DAY = 1941
TEST_START_DAY = 1942
TEST_END_DAY = 1969
PANEL_END_DAY = 1941  # last day present in sales_train_evaluation.csv
HORIZON = 28
WEIGHT_WINDOW = 28  # last 28 days ending at the forecast origin

# Smoke mode shifts the whole protocol back 28 days so pseudo-test
# d_1914..d_1941 comes from the panel and never touches official truth.
SMOKE_TRAIN_END_DAY = 1885
SMOKE_VAL_END_DAY = 1913

TEST_TRUTH_SHA256 = (
    "73773250dd7f7464f7a1bf3a54874dc7fc8cbb49885cd0bf1efe09bf48fd930d"
)
TEST_TRUTH_COLUMNS = SERIES_KEY + [f"d_{d}" for d in range(1942, 1970)]


@dataclass(frozen=True)
class Bounds:
    """Frozen window boundaries in 1-based inclusive day numbers."""

    train_end_day: int
    val_end_day: int

    @property
    def test_start_day(self) -> int:
        return self.val_end_day + 1

    @property
    def test_end_day(self) -> int:
        return self.val_end_day + HORIZON


FULL_BOUNDS = Bounds(TRAIN_END_DAY, VAL_END_DAY)
SMOKE_BOUNDS = Bounds(SMOKE_TRAIN_END_DAY, SMOKE_VAL_END_DAY)


@dataclass
class M5Data:
    """Loaded competition data with consistent day indexing.

    ``panel`` columns are days d_1..d_{PANEL_END_DAY}; ``price_days``,
    ``calendar`` and ``price_weekly`` span d_1..d_1969. Panel column j
    and calendar row j both refer to day d_{j+1}.
    """

    ids: pd.DataFrame  # ID_COLS, one row per series, panel row order
    panel: np.ndarray  # (n_series, PANEL_END_DAY) float32 unit sales
    price_days: np.ndarray  # (n_series, 1969) float32 sell price per day
    price_weekly: np.ndarray  # (n_series, n_weeks) float64 weekly price
    week_cols: list[int]  # wm_yr_wk values, sorted, aligned to price_weekly
    day_week_pos: np.ndarray  # (1969,) int32 week position per calendar day
    calendar: pd.DataFrame
    prices: pd.DataFrame
    day_cols: list[str] = field(default_factory=list)  # panel day cols

    @property
    def n_series(self) -> int:
        return int(self.panel.shape[0])

    @property
    def n_days(self) -> int:
        return int(self.panel.shape[1])

    @property
    def n_calendar_days(self) -> int:
        return int(self.calendar.shape[0])


def _day_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        (c for c in frame.columns if c.startswith("d_")),
        key=lambda c: int(c.split("_")[1]),
    )


def _day_num(day_col: str) -> int:
    return int(day_col.split("_")[1])


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_m5(directory: Path) -> M5Data:
    """Load the competition CSVs into one panel covering d_1..d_1941."""
    sales_path = directory / "sales_train_evaluation.csv"
    sales = pd.read_csv(sales_path)
    day_cols = _day_columns(sales)
    if not day_cols or _day_num(day_cols[-1]) != PANEL_END_DAY:
        raise ValueError(
            f"sales_train_evaluation.csv must cover d_1..d_{PANEL_END_DAY}"
        )
    ids = sales[ID_COLS].reset_index(drop=True)
    panel = sales[day_cols].to_numpy(dtype=np.float32)

    calendar = pd.read_csv(directory / "calendar.csv", usecols=CAL_COLUMNS)
    cal_days = calendar["d"].tolist()
    if len(cal_days) != TEST_END_DAY or cal_days[0] != "d_1":
        raise ValueError(
            f"calendar.csv must cover d_1..d_{TEST_END_DAY}, got {len(cal_days)} rows"
        )
    prices = pd.read_csv(directory / "sell_prices.csv", usecols=PRICE_COLS)

    price_days, price_weekly, week_cols, day_week_pos = _price_matrices(
        ids, prices, calendar, cal_days
    )
    return M5Data(
        ids=ids,
        panel=panel,
        price_days=price_days,
        price_weekly=price_weekly,
        week_cols=week_cols,
        day_week_pos=day_week_pos,
        calendar=calendar,
        prices=prices,
        day_cols=day_cols,
    )


def _price_matrices(
    ids: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
    cal_days: list[str],
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
    """Build price-per-day (float32), weekly price (float64), week map."""
    n_series = len(ids)

    series_idx = pd.Series(np.arange(n_series), name="series_idx")
    key = ids[["store_id", "item_id"]].copy()
    key["series_idx"] = series_idx.to_numpy()
    merged = prices.merge(key, on=["store_id", "item_id"], how="inner")
    week_cols = sorted(int(w) for w in prices["wm_yr_wk"].unique())
    week_pos = {w: i for i, w in enumerate(week_cols)}
    weekly = (
        merged.pivot_table(
            index="series_idx",
            columns="wm_yr_wk",
            values="sell_price",
            aggfunc="first",
        )
        .reindex(index=np.arange(n_series), columns=week_cols)
        .to_numpy(dtype=np.float64)
    )
    # (series, week) with no price row means the product was not sold
    # that week; its unit sales are zero there, so 0 is safe.
    weekly = np.nan_to_num(weekly, nan=0.0)

    cal_wk = calendar.set_index("d")["wm_yr_wk"]
    day_weeks = np.array([int(cal_wk[d]) for d in cal_days])
    day_week_pos = np.array([week_pos[w] for w in day_weeks], dtype=np.int32)

    price_days = weekly[:, day_week_pos].astype(np.float32)
    return price_days, weekly, week_cols, day_week_pos


def series_dollar_sales_window(
    data: M5Data, day_start_idx: int, day_end_idx: int
) -> np.ndarray:
    """Per-series dollar sales over calendar days [start, end] inclusive.

    Indices are 0-based calendar-day positions (day d_{i+1}); the window
    must lie inside the sales panel. Computed in float64 from the CSV
    weekly prices, matching how the organizers' weight files were built.
    """
    if day_end_idx >= data.panel.shape[1]:
        raise ValueError("dollar-sales window extends beyond the sales panel")
    weeks, inverse = np.unique(
        data.day_week_pos[day_start_idx : day_end_idx + 1], return_inverse=True
    )
    day_idx = np.arange(day_start_idx, day_end_idx + 1)
    out = np.zeros(data.n_series, dtype=np.float64)
    for i, w in enumerate(weeks):
        cols = day_idx[inverse == i]
        sales_w = data.panel[:, cols].astype(np.float64).sum(axis=1)
        out += sales_w * data.price_weekly[:, int(w)]
    return out


def day_index(day_col: str) -> int:
    """0-based position of ``d_N`` in day ordering."""
    return int(day_col.split("_")[1]) - 1


def window_indices(bounds: Bounds) -> dict[str, tuple[int, int]]:
    """0-based inclusive (start, end) panel indices for train/validation."""
    t, v = bounds.train_end_day, bounds.val_end_day
    return {
        "train": (0, t - 1),
        "validation": (t, v - 1),
    }


def origin_indices(bounds: Bounds) -> dict[str, int]:
    """0-based forecast origins: last observed day before each window."""
    t, v = bounds.train_end_day, bounds.val_end_day
    return {
        "validation": t - 1,  # day d_T predicts d_{T+1}..d_{T+28}
        "test": v - 1,  # day d_V predicts d_{V+1}..d_{V+28}
    }


def load_test_truth(
    path: Path, ids: pd.DataFrame
) -> tuple[np.ndarray, float]:
    """Load the organizers' private-window truth (d_1942..d_1969).

    Verifies the file sha256, column layout, and a 1:1 join on the five
    structural key columns with no missing keys. Returns the (n_series,
    28) float32 panel in ``ids`` row order plus the epoch at which the
    file was first read; callers must persist their plan before calling
    this so ``first_test_read_epoch`` postdates ``card_write_epoch``.
    """
    epoch = time.time()
    actual = sha256_file(path)
    if actual != TEST_TRUTH_SHA256:
        raise ValueError(
            f"test truth sha256 mismatch: {actual} != {TEST_TRUTH_SHA256}"
        )
    truth = pd.read_csv(path)
    if list(truth.columns) != TEST_TRUTH_COLUMNS:
        raise ValueError("test truth columns do not match d_1942..d_1969 layout")
    day_cols = [c for c in truth.columns if c.startswith("d_")]
    merged = ids[SERIES_KEY].merge(
        truth,
        on=SERIES_KEY,
        how="left",
        validate="1:1",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("test truth is missing series keys")
    if merged[day_cols].isna().any().any():
        raise ValueError("test truth contains missing values")
    panel = merged[day_cols].to_numpy(dtype=np.float32)
    return panel, epoch
