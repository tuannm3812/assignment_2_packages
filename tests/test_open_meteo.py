import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from assignment2_25739083.data.open_meteo import (
    requests_session_with_retries,
    daily_json_to_df,
    fetch_daily_year,
    fetch_and_process_years,
    fetch_and_process_partial_year,
)

# ---------- Fixtures ----------

@pytest.fixture
def sample_payload():
    # Minimal Open-Meteo-like daily payload
    return {
        "daily": {
            "time": ["2024-01-01", "2024-01-02"],
            "temperature_2m_max": [30.1, 28.3],
            "precipitation_sum": [0.0, 5.2],
            "weathercode": [0, 61],
        }
    }

@pytest.fixture
def dummy_session(monkeypatch):
    """A requests-like Session whose .get() we can control per-test."""
    class DummyResp:
        def __init__(self, payload, status=200):
            self._payload = payload
            self.status_code = status
        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")
        def json(self):
            return self._payload

    class DummySession:
        def __init__(self):
            self._next_payload = None
            self._next_status = 200
        def set_next(self, payload, status=200):
            self._next_payload = payload
            self._next_status = status
        def get(self, *_, **__):
            return DummyResp(self._next_payload, self._next_status)

    s = DummySession()
    return s

# ---------- Unit tests ----------

def test_requests_session_with_retries_config():
    s = requests_session_with_retries(total=7, backoff=0.25)
    # Verify adapters are mounted and retries configured
    http_adapter = s.adapters["http://"]
    https_adapter = s.adapters["https://"]
    assert http_adapter.max_retries.total == 7
    assert https_adapter.max_retries.total == 7

def test_daily_json_to_df_ok(sample_payload):
    df = daily_json_to_df(sample_payload)
    assert not df.empty
    assert list(df.columns) == ["date", "temperature_2m_max", "precipitation_sum", "weathercode"]
    assert pd.api.types.is_datetime64_ns_dtype(df["date"])

def test_daily_json_to_df_empty_when_missing_daily():
    df = daily_json_to_df({})
    assert df.empty

def test_fetch_daily_year_calls_session_and_returns_json(dummy_session, sample_payload):
    dummy_session.set_next(sample_payload, status=200)
    out = fetch_daily_year(
        dummy_session,
        2024,
        lat=-33.8688,
        lon=151.2093,
        timezone="Australia/Sydney",
        daily_vars=["temperature_2m_max", "precipitation_sum"],
        base_url="https://example.test/forecast",
        end_date=date(2024, 12, 31),
        timeout_s=5,
    )
    assert out == sample_payload

def test_fetch_and_process_years_saves_raw_and_concatenates(tmp_path: Path, dummy_session, sample_payload):
    raw_dir = tmp_path / "raw"
    # Simulate same payload for both years
    dummy_session.set_next(sample_payload)

    df = fetch_and_process_years(
        dummy_session,
        start_year=2024,
        end_year=2025,
        lat=-33.86,
        lon=151.21,
        timezone="Australia/Sydney",
        daily_vars=["temperature_2m_max", "precipitation_sum"],
        raw_dir=raw_dir,
        base_url="https://example.test/forecast",
        sleep_seconds=0.0,
    )
    # Raw files exist
    assert (raw_dir / "open_meteo_daily_2024.json").exists()
    assert (raw_dir / "open_meteo_daily_2025.json").exists()
    # DataFrame shape / annotations
    assert not df.empty
    assert {"latitude", "longitude", "timezone", "year"}.issubset(df.columns)
    # Sorted by date
    assert df["date"].is_monotonic_increasing

def test_fetch_and_process_partial_year(tmp_path: Path, dummy_session, sample_payload):
    raw_dir = tmp_path / "raw"
    dummy_session.set_next(sample_payload)

    df = fetch_and_process_partial_year(
        dummy_session,
        2025,
        lat=-33.86,
        lon=151.21,
        timezone="Australia/Sydney",
        daily_vars=["temperature_2m_max", "precipitation_sum"],
        raw_dir=raw_dir,
        base_url="https://example.test/forecast",
        until=date(2025, 1, 31),
        sleep_seconds=0.0,
    )
    assert not df.empty
    assert (raw_dir / "open_meteo_daily_2025_partial.json").exists()
    assert {"latitude", "longitude", "timezone", "year"}.issubset(df.columns)