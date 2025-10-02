from pathlib import Path

import pandas as pd
import pytest

from assignment2_25739083.data.io import save_dataframe, load_csv


def test_save_dataframe_writes_csv(tmp_path: Path):
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "x": [1]})
    out = save_dataframe(df, base_filename="test_artifact", processed_dir=tmp_path, write_parquet=False)
    assert "csv" in out
    assert out["csv"].exists()
    # Parquet intentionally skipped
    assert "parquet" not in out

def test_save_dataframe_parquet_best_effort(tmp_path: Path, monkeypatch):
    df = pd.DataFrame({"a": [1, 2]})
    # Force Parquet failure to exercise warning path
    def boom(*args, **kwargs):
        raise RuntimeError("Forced parquet error")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)

    out = save_dataframe(df, base_filename="with_parquet", processed_dir=tmp_path, write_parquet=True)
    # CSV still written
    assert out["csv"].exists()
    # Parquet not present due to error
    assert "parquet" not in out

def test_save_dataframe_noop_on_empty_df(tmp_path: Path, capsys):
    df = pd.DataFrame()
    out = save_dataframe(df, base_filename="empty", processed_dir=tmp_path)
    assert out == {}
    # nothing created
    assert not any(tmp_path.iterdir())

def test_load_csv_roundtrip(tmp_path: Path):
    p = tmp_path / "round.csv"
    df_in = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "y": [1, 2]})
    df_in.to_csv(p, index=False)

    df_out = load_csv(p, parse_dates=["date"])
    assert pd.api.types.is_datetime64_any_dtype(df_out["date"])
    assert df_out.equals(df_in)