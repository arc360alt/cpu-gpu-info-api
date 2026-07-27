"""
CPU Info API - Data Fetcher and Processor

Fetches CPU information from Wikipedia sources for AMD and Intel, processes
the data, and outputs to JSON format.

Unlike the GPU pipeline (update.py), "List of AMD processors" is a
prose/history overview page with no real spec tables, so AMD data is
assembled from several per-family pages (Ryzen, FX, Athlon, ...) while Intel
is a single page with everything in it. Because of that, this script also
avoids pandas' `pd.read_html(match=...)` filter (its lxml backend silently
drops the compiled regex's IGNORECASE flag - see update.py's clean_html
history) and instead filters parsed tables in plain Python.
"""
import argparse
import json
import logging
import re
import shutil
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from config import (
    CPU_VENDOR_CONFIGS,
    REFERENCES_AT_END,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_MIN_WAIT,
    RETRY_MAX_WAIT,
    DEFAULT_CPU_OUTPUT_FILE,
    JSON_INDENT,
    CREATE_BACKUP,
    MIN_TABLE_ROWS,
    MIN_TABLE_COLS,
    MIN_EXPECTED_CPUS,
    LOG_FORMAT,
    LOG_LEVEL,
)
from update import clean_html, normalize_columns, fetch_url_with_retry
from validators import validate_dataframe, validate_output, ValidationError

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = LOG_LEVEL, log_to_file: bool = True) -> None:
    """Configure logging for the application."""
    handlers = [logging.StreamHandler()]

    if log_to_file:
        handlers.append(logging.FileHandler("cpu_info_api.log", mode='w', encoding='utf-8'))

    logging.basicConfig(
        level=getattr(logging, log_level),
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )

    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


# ---------------------------------------------
#  DataFrame Processing Utilities
# ---------------------------------------------

def is_cpu_spec_table(df: pd.DataFrame) -> bool:
    """
    Heuristic check for whether a parsed table actually holds CPU specs
    (as opposed to navboxes, infoboxes, or unrelated link tables).
    """
    header_text = " ".join(str(c) for c in df.columns).lower()
    return bool(re.search(r"cores?|clock|frequency|ghz", header_text))


def merge_split_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unify the many "brand + specific model" column layouts seen across
    AMD/Intel tables into a single canonical "Model" column.

    Two patterns show up repeatedly:
    - A merged header cell (e.g. "Branding and Model") spanning two
      sub-columns that pandas disambiguates as "X" and "X.1"; the two
      sub-columns hold the family (e.g. "Ryzen 7") and specific model
      (e.g. "1800X") respectively.
    - Two separately-named columns, e.g. Intel's "Processor Family"
      ("Core Ultra 9") and "Model" ("285K").
    """
    cols = list(df.columns)

    # Pattern 1: pandas-deduplicated "X" / "X.1" pair from a merged header.
    dup_variants: Dict[str, List[str]] = {}
    for c in cols:
        m = re.match(r"^(.+)\.(\d+)$", c)
        if m and m.group(1) in cols:
            dup_variants.setdefault(m.group(1), []).append(c)

    for base, variants in dup_variants.items():
        if re.search(r"model|brand", base, re.I):
            group = [base] + sorted(variants)
            df["Model"] = (
                df[group].fillna("").astype(str)
                .agg(" ".join, axis=1)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
            df.loc[df["Model"] == "", "Model"] = pd.NA
            return df.drop(columns=group)

    # Pattern 2: separate family + model columns.
    family_col = next(
        (c for c in df.columns if re.search(r"processor family|processor branding|^branding$", c, re.I)),
        None,
    )
    model_col = next((c for c in df.columns if c.strip().lower() in ("model", "sku")), None)
    if family_col and model_col:
        df["Model"] = (
            (df[family_col].fillna("").astype(str) + " " + df[model_col].fillna("").astype(str))
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df.loc[df["Model"] == "", "Model"] = pd.NA
        return df

    # Pattern 3: a single already-canonical model column under another name
    # (matched as a prefix since some tables tag it with extra context, e.g.
    # "Model Number standard power" on dual standard/low-power tables).
    for c in df.columns:
        if re.match(r"model|sku|chip", c.strip(), re.I):
            if c != "Model":
                df = df.rename(columns={c: "Model"})
            break

    return df


def process_cpu_dataframe(df: pd.DataFrame, vendor: str) -> pd.DataFrame:
    """Process a raw CPU DataFrame from Wikipedia tables."""
    logger.debug(f"{vendor}: Processing DataFrame with shape {df.shape}")

    df.columns = normalize_columns(df.columns)
    df.columns = [re.sub(r"(?:\[[A-Za-z0-9]+\])+", "", c) for c in df.columns]
    df.columns = [c.replace("- ", "").replace("/ ", "/").strip() for c in df.columns]

    df = merge_split_model_columns(df)
    df["Vendor"] = vendor

    # Launch / Release Date extraction
    launch_cols = [c for c in df.columns if re.search(r"launch|release[d]? date", c, re.I)]
    if launch_cols:
        col = launch_cols[0]
        df[col] = (
            df[col].astype(str)
            .str.replace(REFERENCES_AT_END, "", regex=True)
            .str.extract(r"([A-Za-z]+\s*\d{1,2},?\s*(?:19|20)\d{2}|(?:19|20)\d{2})", expand=False)
        )
        df["Launch"] = pd.to_datetime(df[col], errors="coerce")
        logger.debug(f"{vendor}: Extracted {df['Launch'].notna().sum()} launch dates")
    else:
        df["Launch"] = pd.NaT
        logger.debug(f"{vendor}: No launch date column found")

    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]

    logger.debug(f"{vendor}: Processed to shape {df.shape}")
    return df


def remove_bracketed_references(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Remove citation brackets [1], [2], etc. from specified columns."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(r"\[\d+\]", "", regex=True).str.strip()
    return df


# ---------------------------------------------
#  Network & Fetching
# ---------------------------------------------

def fetch_cpu_tables(vendor: str, url: str) -> List[pd.DataFrame]:
    """
    Fetch and parse CPU spec tables from a single Wikipedia page.

    Deliberately does not use pd.read_html's `match=` filter - lxml's
    backend ignores the compiled regex's re.IGNORECASE flag, and CPU
    tables' column names vary wildly across pages/eras. Instead, every
    table is parsed and then filtered in plain Python.
    """
    logger.info(f"Fetching {vendor} data from {url}...")

    try:
        html = fetch_url_with_retry(url)
        html = clean_html(html)

        dfs = pd.read_html(StringIO(html))
        spec_tables = [df for df in dfs if is_cpu_spec_table(df)]

        logger.info(f"{vendor}: {url} -> {len(spec_tables)}/{len(dfs)} tables look like CPU specs")
        return spec_tables

    except requests.RequestException as e:
        logger.error(f"{vendor}: Network error fetching {url}: {e}")
        raise
    except Exception as e:
        logger.error(f"{vendor}: Error parsing tables from {url}: {e}")
        raise


# ---------------------------------------------
#  Assembly & Export
# ---------------------------------------------

def record_key(row: Dict[str, str]) -> str:
    """Generate a unique key for a CPU record, preferring Model."""
    vendor = row.get("Vendor", "UNKNOWN").strip()
    model = str(row.get("Model", "UnknownModel")).strip()
    model = re.sub(r"[^A-Za-z0-9]+", "_", model) or "UnknownModel"
    return f"{vendor}_{model}"


def create_backup(output_path: Path) -> None:
    """Create a backup of existing output file."""
    if output_path.exists():
        backup_path = output_path.with_suffix('.json.backup')
        shutil.copy2(output_path, backup_path)
        logger.info(f"Created backup: {backup_path}")


# ---------------------------------------------
#  Pipeline
# ---------------------------------------------

def main(output_file: str = DEFAULT_CPU_OUTPUT_FILE, dry_run: bool = False) -> Dict[str, Dict]:
    """Main pipeline: fetch, process, validate, and export CPU data."""
    logger.info("=" * 60)
    logger.info("CPU Info API - Starting data fetch and processing")
    logger.info("=" * 60)

    output_path = Path(output_file)
    frames: List[pd.DataFrame] = []
    failed_sources: List[str] = []

    for vendor, info in CPU_VENDOR_CONFIGS.items():
        for url in info["urls"]:
            try:
                vendor_dfs = fetch_cpu_tables(vendor, url)

                for idx, raw_df in enumerate(vendor_dfs):
                    if raw_df.shape[0] < MIN_TABLE_ROWS or raw_df.shape[1] < MIN_TABLE_COLS:
                        logger.warning(f"{vendor} table {idx} ({url}): Too small ({raw_df.shape}), skipping")
                        continue

                    processed_df = process_cpu_dataframe(raw_df, vendor)
                    is_valid, warnings = validate_dataframe(processed_df, vendor)

                    if is_valid:
                        frames.append(processed_df)
                    else:
                        logger.warning(f"{vendor} table {idx} ({url}): Validation failed, skipping")

            except Exception as e:
                logger.error(f"{vendor}: Failed to fetch/process {url}: {e}", exc_info=True)
                failed_sources.append(url)
                continue

    if not frames:
        raise RuntimeError(
            "No CPU tables parsed successfully. "
            "All sources failed or wiki markup may have changed."
        )

    if failed_sources:
        logger.warning(f"Failed to process sources: {', '.join(failed_sources)}")

    logger.info(f"Combining {len(frames)} tables...")
    df = pd.concat(frames, ignore_index=True, sort=False)

    df = remove_bracketed_references(df, ["Model", "Code name", "Codename"])

    # Some source tables include blank divider/section-break rows (e.g.
    # separating model families) that carry no spec data at all - not even
    # a part number. There's nothing to name these with, so drop them
    # rather than emit placeholder "UnknownModel" records.
    if "Model" in df.columns:
        before = len(df)
        df["Model"] = df["Model"].replace(r"^\s*$", pd.NA, regex=True)
        df = df.dropna(subset=["Model"])
        dropped = before - len(df)
        if dropped:
            logger.info(f"Dropped {dropped} row(s) with no Model/name value")
    else:
        logger.warning("No 'Model' column present after combining tables; dropping all rows")
        df = df.iloc[0:0]

    # Some source rows resolve to a "Model" but have every other spec field
    # blank - table-parsing artifacts (a stray footnote row, a roadmap
    # placeholder like "Q1 2026", a template leftover like "-N/a"). A record
    # with no actual specs isn't a usable CPU entry, so drop it.
    spec_cols = [
        c for c in df.columns
        if c not in ("Vendor", "Model", "Launch")
        and not re.search(r"processor family|processor branding|^branding$", c, re.I)
    ]
    if spec_cols:
        before = len(df)
        df = df[df[spec_cols].notna().any(axis=1)]
        dropped = before - len(df)
        if dropped:
            logger.info(f"Dropped {dropped} row(s) with a Model but no spec data")

    logger.info("Converting to JSON format...")
    result: Dict[str, Dict[str, str]] = {}

    for record in df.to_dict(orient="records"):
        compact = {k: v for k, v in record.items() if pd.notna(v)}
        key = record_key(compact)

        if key in result:
            i = 2
            while f"{key}_{i}" in result:
                i += 1
            original_key = key
            key = f"{key}_{i}"
            logger.debug(f"Duplicate key '{original_key}' renamed to '{key}'")

        result[key] = compact

    logger.info("Validating output...")
    try:
        validate_output(result, output_path, min_expected=MIN_EXPECTED_CPUS)
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        raise

    if not dry_run:
        if CREATE_BACKUP:
            create_backup(output_path)

        logger.info(f"Writing output to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(result, fp, indent=JSON_INDENT, ensure_ascii=False, default=str)

        logger.info(f"✅ Successfully saved {len(result)} CPUs to {output_path}")
    else:
        logger.info(f"✅ Dry run: {len(result)} CPUs validated (not written to file)")

    vendor_counts = {}
    for record in result.values():
        vendor = record.get("Vendor", "Unknown")
        vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1

    logger.info("Summary by vendor:")
    for vendor, count in sorted(vendor_counts.items()):
        logger.info(f"  {vendor}: {count} CPUs")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch and process CPU information from Wikipedia"
    )
    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_CPU_OUTPUT_FILE,
        help=f"Output JSON file path (default: {DEFAULT_CPU_OUTPUT_FILE})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data without writing output file"
    )
    parser.add_argument(
        "--log-level",
        default=LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help=f"Logging level (default: {LOG_LEVEL})"
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable logging to file"
    )

    args = parser.parse_args()

    setup_logging(log_level=args.log_level, log_to_file=not args.no_log_file)

    try:
        main(output_file=args.output, dry_run=args.dry_run)
        sys.exit(0)
    except ValidationError as e:
        logger.error(f"❌ Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
