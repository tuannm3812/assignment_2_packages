from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def save_dataframe(
    df: pd.DataFrame,
    base_filename: str,
    processed_dir: Path,
    *,
    write_parquet: bool = True,
    parquet_engine: Optional[str] = None,
) -> dict[str, Path]:
    """
    Persist a non-empty DataFrame to CSV (and optionally Parquet).

    Parameters
    ----------
    df : pandas.DataFrame
        Table to persist; if empty, the function is a no-op.
    base_filename : str
        Stem for output files (without extension), e.g., "sydney_daily_2024".
    processed_dir : pathlib.Path
        Directory under which files will be written.
    write_parquet : bool
        If True, write a Parquet copy alongside CSV.
    parquet_engine : Optional[str]
        Parquet engine hint (e.g., "pyarrow"); if None, let pandas decide.

    Returns
    -------
    dict[str, Path]
        Mapping of artifact names to file paths that were written.
    """
    artifacts: dict[str, Path] = {}
    if df.empty:
        logger.warning("DataFrame is empty; skipping save for '%s'.", base_filename)
        return artifacts

    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_path = processed_dir / f"{base_filename}.csv"
    df.to_csv(csv_path, index=False)
    artifacts["csv"] = csv_path

    if write_parquet:
        try:
            pq_path = processed_dir / f"{base_filename}.parquet"
            df.to_parquet(pq_path, index=False, engine=parquet_engine)
            artifacts["parquet"] = pq_path
        except Exception as exc:
            logger.warning("Parquet write failed for '%s' (%s). CSV was written.", base_filename, exc)

    logger.info("Saved artifacts for '%s': %s", base_filename, {k: str(v) for k, v in artifacts.items()})
    return artifacts


def load_csv(
    path: str | Path,
    *,
    parse_dates: Optional[Iterable[str]] = None,
    dtype: Optional[dict] = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame with common conveniences.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to the CSV file.
    parse_dates : Optional[Iterable[str]]
        Columns to parse as datetimes (e.g., ["date"]).
    dtype : Optional[dict]
        Optional dtype mapping for columns.
    **read_csv_kwargs
        Additional `pandas.read_csv` keyword arguments.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=list(parse_dates) if parse_dates else None, dtype=dtype, **read_csv_kwargs)
    return df