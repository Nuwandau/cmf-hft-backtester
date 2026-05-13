import numpy as np

from cmf_backtester.data.loaders import MarketDataArrays
from cmf_backtester.market.microprice import MicropriceEstimator


def test_microprice_estimator_fits_finite_adjustments() -> None:
    arrays = MarketDataArrays(
        timestamps=np.arange(8, dtype=np.int64),
        best_bid_ticks=np.array([99, 100, 99, 100, 99, 100, 99, 100], dtype=np.int64),
        best_ask_ticks=np.array([101, 102, 101, 102, 101, 102, 101, 102], dtype=np.int64),
        bid_size=np.ones(8),
        ask_size=np.ones(8),
        mid_ticks=np.array([100, 101, 100, 101, 100, 101, 100, 101], dtype=np.float64),
        mid_half_ticks=np.array([200, 202, 200, 202, 200, 202, 200, 202], dtype=np.int64),
        spread_ticks=np.full(8, 2, dtype=np.int64),
        imbalance=np.array([0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8]),
        split=np.array(["train"] * 8),
        date=np.array(["2024-01-01"] * 8),
        tick_size=0.01,
    )
    estimator = MicropriceEstimator(n_imbalance_buckets=2, max_spread_state_ticks=2).fit(arrays)
    assert estimator.adjustment_ticks.shape == (6,)
    assert np.all(np.isfinite(estimator.adjustment_ticks))


def test_microprice_filters_large_snapshot_jumps() -> None:
    arrays = MarketDataArrays(
        timestamps=np.arange(5, dtype=np.int64),
        best_bid_ticks=np.array([99, 100, 130, 101, 102], dtype=np.int64),
        best_ask_ticks=np.array([101, 102, 132, 103, 104], dtype=np.int64),
        bid_size=np.ones(5),
        ask_size=np.ones(5),
        mid_ticks=np.array([100, 101, 131, 102, 103], dtype=np.float64),
        mid_half_ticks=np.array([200, 202, 262, 204, 206], dtype=np.int64),
        spread_ticks=np.full(5, 2, dtype=np.int64),
        imbalance=np.array([0.2, 0.8, 0.2, 0.8, 0.2]),
        split=np.array(["train"] * 5),
        date=np.array(["2024-01-01"] * 5),
        tick_size=0.01,
    )
    estimator = MicropriceEstimator(
        n_imbalance_buckets=2,
        max_spread_state_ticks=2,
        max_mid_move_ticks=1.0,
    ).fit(arrays)

    assert estimator.diagnostics is not None
    assert estimator.diagnostics.n_filtered_transitions == 2
