from cmf_backtester.market.features import (
    build_state_id,
    compute_imbalance,
    compute_mid,
    compute_spread,
    imbalance_bucket,
    spread_state,
)


def test_top_of_book_features() -> None:
    assert compute_mid(100.0, 102.0) == 101.0
    assert compute_spread(100.0, 102.0) == 2.0
    assert compute_imbalance(3.0, 1.0) == 0.75
    assert compute_imbalance(0.0, 0.0) == 0.5


def test_state_mapping() -> None:
    assert imbalance_bucket(0.01, 10) == 0
    assert imbalance_bucket(1.0, 10) == 9
    assert spread_state(1, 10) == 0
    assert spread_state(11, 10) == 10
    assert build_state_id(1, 0.1, 10, 10) == 0
    assert build_state_id(2, 0.95, 10, 10) == 19

