from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from time import sleep
from typing import Iterable, Mapping, Sequence

import logging
import pandas as pd
import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


def requests_session_with_retries(
    total: int = 5,
    backoff: float = 0.5,
    status_forcelist: Iterable[int] = (429, 500, 502, 503, 504),
    allowed_methods: Iterable[str] = ("GET",),
) -> Session:
    """
    Construct a `requests.Session` configured with idempotent retries.

    Parameters
    ----------
    total : int
        Maximum total retry attempts (applied to read/connect).
    backoff : float
        Exponential backoff factor in seconds.
    status_forcelist : Iterable[int]
        HTTP status codes that trigger a retry.
    allowed_methods : Iterable[str]
        HTTP methods eligible for retry.

    Returns
    -------
    requests.Session
        Session pre-mounted with retrying adapters for HTTP/HTTPS.
    """
    retry = Retry(
        total=total,
        read=total,
        connect=total,
        backoff_factor=backoff,
        status_forcelist=tuple(status_forcelist),
        allowed_methods=frozenset(allowed_methods),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def fetch_daily_year(
    session: Session,
    year: int,
    *,
    lat: float,
    lon: float,
    timezone: str,
    daily_vars: Sequence[str],
    base_url: str = "https://api.open-meteo.com/v1/forecast",
    end_date: date | None = None,
    timeout_s: int = 60,
) -> dict:
    """
    Retrieve one year's worth of daily weather from Open-Meteo.

    Parameters
    ----------
    session : requests.Session
        A configured HTTP session (e.g., from `requests_session_with_retries`).
    year : int
        Calendar year to fetch (e.g., 2024).
    lat, lon : float
        Geographic coordinates (degrees).
    timezone : str
        IANA timezone name (e.g., "Australia/Sydney").
    daily_vars : Sequence[str]
        Daily variables to request (e.g., ["temperature_2m_max", "precipitation_sum"]).
    base_url : str
        Open-Meteo daily forecast endpoint.
    end_date : date | None
        Last date to include; defaults to 31 Dec of `year`.
    timeout_s : int
        HTTP request timeout in seconds.

    Returns
    -------
    dict
        The parsed JSON payload.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
        "start_date": f"{year}-01-01",
        "end_date": (end_date or date(year, 12, 31)).strftime("%Y-%m-%d"),
        "timeformat": "iso8601",
        "windspeed_unit": "kmh",
        "precipitation_unit": "mm",
        "daily": ",".join([*daily_vars, "weathercode"]),  # include for QC/EDA
    }
    logger.info("Requesting Open-Meteo daily data: year=%s params=%s", year, params)
    resp = session.get(base_url, params=params, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def daily_json_to_df(payload: Mapping[str, object]) -> pd.DataFrame:
    """
    Convert an Open-Meteo daily payload to a tidy DataFrame.

    The returned frame includes a `date` column of dtype `datetime64[ns]` at day precision.

    Parameters
    ----------
    payload : Mapping[str, object]
        JSON payload from `fetch_daily_year`.

    Returns
    -------
    pandas.DataFrame
        Tidy table with one row per date; empty DataFrame if no data present.
    """
    daily = payload.get("daily", {})
    if not isinstance(daily, Mapping) or not daily:
        logger.warning("Payload contains no 'daily' block; returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.DataFrame(daily)
    if "time" not in df.columns:
        logger.warning("'daily' block lacks 'time' column; returning empty DataFrame.")
        return pd.DataFrame()

    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=False).dt.normalize()
    return df


def fetch_and_process_years(
    session: Session,
    start_year: int,
    end_year: int,
    *,
    lat: float,
    lon: float,
    timezone: str,
    daily_vars: Sequence[str],
    raw_dir: Path,
    base_url: str = "https://api.open-meteo.com/v1/forecast",
    sleep_seconds: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch multiple years of daily data, persist raw JSON per year, and return a concatenated DataFrame.

    Parameters
    ----------
    session : requests.Session
        HTTP session with retries.
    start_year, end_year : int
        Inclusive range of years to download.
    lat, lon, timezone, daily_vars, base_url
        See `fetch_daily_year`.
    raw_dir : pathlib.Path
        Directory to receive the raw JSON files.
    sleep_seconds : float
        Politeness delay between requests.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame sorted by date, annotated with `latitude`, `longitude`, `timezone`, and `year`.
        Empty DataFrame if nothing was fetched.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for yr in range(start_year, end_year + 1):
        try:
            payload = fetch_daily_year(
                session, yr, lat=lat, lon=lon, timezone=timezone, daily_vars=daily_vars, base_url=base_url
            )
            raw_path = raw_dir / f"open_meteo_daily_{yr}.json"
            raw_path.write_text(pd.io.json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            df_yr = daily_json_to_df(payload)
            if df_yr.empty:
                logger.warning("No daily data returned for %d; skipping.", yr)
            else:
                df_yr["year"] = yr
                frames.append(df_yr)
        except Exception as exc:  # keep loop resilient
            logger.exception("Failed to fetch/process year %d: %s", yr, exc)
        finally:
            sleep(sleep_seconds)

    if not frames:
        logger.warning("No data collected for %d-%d; returning empty DataFrame.", start_year, end_year)
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True).sort_values("date")
    df["latitude"] = lat
    df["longitude"] = lon
    df["timezone"] = timezone

    dupes = df["date"].duplicated().sum()
    if dupes:
        logger.warning("Found %d duplicate dates; keeping first occurrence.", dupes)
        df = df.drop_duplicates(subset=["date"], keep="first")

    return df.reset_index(drop=True)


def fetch_and_process_partial_year(
    session: Session,
    year: int,
    *,
    lat: float,
    lon: float,
    timezone: str,
    daily_vars: Sequence[str],
    raw_dir: Path,
    base_url: str = "https://api.open-meteo.com/v1/forecast",
    until: date | None = None,
    sleep_seconds: float = 0.5,
) -> pd.DataFrame:
    """
    Fetch a partial year's daily data up to `until` (default: yesterday), save raw JSON, and return a clean frame.

    Parameters
    ----------
    session : requests.Session
        HTTP session with retries.
    year : int
        Year of interest (must be the current year if `until` is None).
    lat, lon, timezone, daily_vars, base_url
        See `fetch_daily_year`.
    raw_dir : pathlib.Path
        Directory to receive the raw JSON file.
    until : date | None
        Last date to include; if None, defaults to `date.today() - timedelta(days=1)`.
    sleep_seconds : float
        Politeness delay after the request.

    Returns
    -------
    pandas.DataFrame
        Daily observations for the requested partial year, annotated with location fields.
        Empty DataFrame on error.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    end_date = until or (date.today() - timedelta(days=1))
    logger.info("Fetching partial year=%d up to %s", year, end_date.isoformat())
    try:
        payload = fetch_daily_year(
            session,
            year,
            lat=lat,
            lon=lon,
            timezone=timezone,
            daily_vars=daily_vars,
            base_url=base_url,
            end_date=end_date,
        )
        raw_path = raw_dir / f"open_meteo_daily_{year}_partial.json"
        raw_path.write_text(pd.io.json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        df = daily_json_to_df(payload)
        if df.empty:
            logger.warning("No daily data returned for partial year=%d; returning empty DataFrame.", year)
            return pd.DataFrame()

        df["year"] = year
        df["latitude"] = lat
        df["longitude"] = lon
        df["timezone"] = timezone

        dupes = df["date"].duplicated().sum()
        if dupes:
            logger.warning("Found %d duplicate dates in partial year; dropping duplicates.", dupes)
            df = df.drop_duplicates(subset=["date"], keep="first")

        return df.reset_index(drop=True)
    except Exception as exc:
        logger.exception("Failed to fetch partial year %d: %s", year, exc)
        return pd.DataFrame()
    finally:
        sleep(sleep_seconds)