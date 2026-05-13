import polars as pl

from cmf_backtester.data.preprocessing import preprocess_lob_l1


def test_preprocess_parquet_preserves_event_id_order(tmp_path) -> None:
    raw_path = tmp_path / "lob.parquet"
    out_path = tmp_path / "lob_l1.parquet"
    pl.DataFrame(
        {
            "event_id": [2, 1],
            "local_timestamp": [100, 100],
            "asks[0].price": [1.02, 1.01],
            "asks[0].amount": [10.0, 20.0],
            "bids[0].price": [1.00, 0.99],
            "bids[0].amount": [30.0, 40.0],
        }
    ).write_parquet(raw_path)

    preprocess_lob_l1(
        raw_path,
        out_path,
        tick_size=0.01,
        train_dates=[],
        validation_dates=[],
        test_dates=[],
    )

    processed = pl.read_parquet(out_path)
    assert processed["event_id"].to_list() == [1, 2]
    assert processed["local_timestamp"].to_list() == [100, 100]
