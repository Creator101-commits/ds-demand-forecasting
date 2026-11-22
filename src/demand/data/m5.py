"""Load M5 competition files into panels and metadata.

Files (Kaggle competition ``m5-forecasting-accuracy``):
- ``sales_train_validation.csv``: d_1..d_1913 for 30,490 series.
- ``sales_train_evaluation.csv``: d_1914..d_1941 (evaluation ground truth).
- ``calendar.csv``: day -> wm_yr_wk, weekday, month, events, SNAP flags.
- ``sell_prices.csv``: weekly sell price per store x item.

The two sales files are concatenated so the panel covers d_1..d_1941.
Unit sales are kept as a float32 (n_series, n_days) matrix; a matching
float32 price-per-day matrix supports dollar-sales weighting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
# The two sales files suffix ids differently (_validation/_evaluation),
# so series identity is the five structural columns, not the id string.
SERIES_KEY = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
CAL_COLUMNS = [
    "d",
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

# Frozen protocol day boundaries (1-based d_N, inclusive).
TRAIN_END_DAY = 1885  # train: d_1..d_1885
VAL_END_DAY = 1913  # validation: d_1886..d_1913
TEST_END_DAY = 1941  # test: d_1914..d_1941
HORIZON = 28
WEIGHT_WINDOW = 28  # last 28 days before each forecast origin


@dataclass
class M5Data:
    """Loaded competition data with consistent day indexing."""

    ids: pd.DataFrame  # ID_COLS, one row per series, panel row order
    panel: np.ndarray  # (n_series, n_days) float32 unit sales
    price_days: np.ndarray  # (n_series, n_days) float32 sell price per day
    calendar: pd.DataFrame
    prices: pd.DataFrame
    day_cols: list[str] = field(default_factory=list)

    @property
    def n_series(self) -> int:
        return int(self.panel.shape[0])

    @property
    def n_days(self) -> int:
        return int(self.panel.shape[1])


def _day_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        (c for c in frame.columns if c.startswith("d_")),
        key=lambda c: int(c.split("_")[1]),
    )


def _day_num(day_col: str) -> int:
    return int(day_col.split("_")[1])


def load_m5(directory: Path, *, boundary_day: int = VAL_END_DAY) -> M5Data:
    """Load the competition CSVs into one panel.

    ``sales_train_validation.csv`` covers d_1..d_1913 and
    ``sales_train_evaluation.csv`` covers d_1..d_1941, so the evaluation
    file's first 1,913 day columns duplicate the validation file. The
    panel takes days through ``boundary_day`` from the validation file
    and the later days from the evaluation file, joined on series id.
    ``boundary_day`` defaults to the frozen protocol boundary.
    """
    val = pd.read_csv(directory / "sales_train_validation.csv")
    ev = pd.read_csv(directory / "sales_train_evaluation.csv")
    val_days = _day_columns(val)
    ev_days = _day_columns(ev)
    train_days = [d for d in val_days if _day_num(d) <= boundary_day]
    test_days = [d for d in ev_days if _day_num(d) > boundary_day]
    ev_max = max(_day_num(d) for d in ev_days)
    if len(test_days) != ev_max - boundary_day:
        raise ValueError(
            f"expected {ev_max - boundary_day} eval days after "
            f"d_{boundary_day}, found {len(test_days)}"
        )
    day_cols = train_days + test_days
    if len(day_cols) != ev_max:
        raise ValueError(
            f"panel has {len(day_cols)} day columns, expected d_1..d_{ev_max}"
        )

    wide = val[ID_COLS + train_days].merge(
        ev[SERIES_KEY + test_days],
        left_on=SERIES_KEY,
        right_on=SERIES_KEY,
        how="inner",
        validate="1:1",
    )
    ids = wide[ID_COLS].reset_index(drop=True)
    panel = wide[day_cols].to_numpy(dtype=np.float32)

    # Spot-check that the evaluation file's overlapping days agree with
    # the validation file on a sample of series.
    overlap = [d for d in ev_days if _day_num(d) <= boundary_day]
    if overlap:
        val_idx = val.set_index(SERIES_KEY)
        ev_idx = ev.set_index(SERIES_KEY)
        sample_keys = list(val_idx.index[:500])
        val_sub = val_idx.loc[sample_keys, overlap]
        ev_sub = ev_idx.loc[sample_keys, overlap]
        if not np.array_equal(
            ev_sub.to_numpy(dtype=np.float32),
            val_sub.to_numpy(dtype=np.float32),
        ):
            raise ValueError(
                "sales_train_evaluation.csv disagrees with "
                "sales_train_validation.csv on overlapping days"
            )

    calendar = pd.read_csv(directory / "calendar.csv", usecols=CAL_COLUMNS)
    prices = pd.read_csv(directory / "sell_prices.csv", usecols=PRICE_COLS)
    prices["sell_price"] = prices["sell_price"].astype(np.float32)

    price_days = _price_day_matrix(ids, prices, calendar, day_cols)
    return M5Data(
        ids=ids,
        panel=panel,
        price_days=price_days,
        calendar=calendar,
        prices=prices,
        day_cols=day_cols,
    )


def _price_day_matrix(
    ids: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
    day_cols: list[str],
) -> np.ndarray:
    """(n_series, n_days) sell price per day via weekly price lookup."""
    n_series = len(ids)
    n_days = len(day_cols)
    out = np.zeros((n_series, n_days), dtype=np.float32)

    cal_wk = calendar.set_index("d")["wm_yr_wk"]
    day_weeks = np.array([cal_wk[d] for d in day_cols])

    # Pivot weekly prices to (series, week) then index by each day's week.
    series_idx = pd.Series(np.arange(n_series), name="series_idx")
    key = ids[["store_id", "item_id"]].copy()
    key["series_idx"] = series_idx.to_numpy()
    merged = prices.merge(key, on=["store_id", "item_id"], how="inner")
    week_cols = np.sort(prices["wm_yr_wk"].unique())
    week_pos = {w: i for i, w in enumerate(week_cols)}
    wide = (
        merged.pivot_table(
            index="series_idx",
            columns="wm_yr_wk",
            values="sell_price",
            aggfunc="first",
        )
        .reindex(index=np.arange(n_series), columns=week_cols)
        .to_numpy(dtype=np.float32)
    )
    day_week_pos = np.array([week_pos[w] for w in day_weeks])
    out[:, :] = wide[:, day_week_pos]
    # Missing (series, week) means the product was not sold that week;
    # units are zero there anyway, so NaN -> 0 is safe for weighting.
    return np.nan_to_num(out, nan=0.0)


def day_index(day_col: str) -> int:
    """0-based position of ``d_N`` in the day_cols ordering."""
    return int(day_col.split("_")[1]) - 1


def split_indices() -> dict[str, tuple[int, int]]:
    """Frozen day windows as (start, end) 0-based inclusive indices."""
    return {
        "train": (0, TRAIN_END_DAY - 1),
        "validation": (TRAIN_END_DAY, VAL_END_DAY - 1),
        "test": (VAL_END_DAY, TEST_END_DAY - 1),
    }


def series_dollar_sales(
    data: M5Data, day_start: int, day_end: int
) -> np.ndarray:
    """Per-series dollar sales over days [day_start, day_end] inclusive."""
    sl = slice(day_start, day_end + 1)
    return np.asarray(
        (data.panel[:, sl] * data.price_days[:, sl]).sum(axis=1),
        dtype=np.float64,
    )
