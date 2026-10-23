#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 0 - RAW to Parquet conversion, per station.

This script performs no scientific analysis. It reads each daily RAW file
once (all stations together), splits the observations by station, adds
UTC/local time columns, station metadata (longitude/latitude) and
data-quality flag columns, and writes one Parquet file per station.

It does not compute completeness, CCDF, percentiles, exceedance frequencies,
or decide which days are valid. Every RAW row belonging to a processed
station is kept - the quality-flag columns let later scripts decide what to
filter; nothing is removed here.

RAW provenance: the first build of a year records what it was built from
(results/global/{year}/0_raw_source_{year}.json): the RAW folder, the DoY
range, the name/size/mtime of each day's file, and the configuration Step 0
writes into the Parquet files (stations, UTC offsets, coordinates, QC
thresholds). A later run for that year must match that record exactly;
otherwise - or if the year already has data but no record - Step 0 stops
before writing anything, so Parquet files are never replaced under results
derived from them. Nothing is ever deleted.

RAW format: every line must have the official 12 whitespace-separated
columns (section [B]). A file with another number of columns on any line
stops Step 0 with RawFormatError, naming the file and the first such line,
before any result file is written - a shifted column must never become a
silently wrong value.
"""

import gzip
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import project_paths


# ======================================================================
# [A] STUDY CONFIGURATION - EDIT ONLY THIS SECTION
# ======================================================================

YEAR = 2024
DATA_DIR = Path("datos_Estudio")
RESULTS_DIR = project_paths.RESULTS_ROOT  # <project>/results, whatever the CWD
DOY_START = 1
DOY_END = 366  # 2024 is a leap year

# Station selection - console defaults. In web mode, web_server.py temporarily
# replaces SELECTED_STATIONS and STATION_UTC_OFFSETS with app/stations.json
# (this script never reads that file), so the duplication is intentional.
USE_SELECTED_STATIONS = True
SELECTED_STATIONS = ["UNSA", "KOUG", "WHIT", "YELL"]

# Fixed UTC offsets, used ONLY to derive hour_local/month_local. Local time is
# a fixed-offset clock time, not a civil time zone (no daylight saving, so it
# is consistent across the year) and not magnetic local time.
#   UNSA (Salta, Argentina): UTC-3 year-round, no DST since 2009.
#   KOUG (Kourou, French Guiana): UTC-3 year-round, no DST.
#   WHIT (Whitehorse, Yukon): UTC-7 year-round since Yukon adopted permanent
#       Pacific Daylight Time in Nov 2020 (no more seasonal clock changes).
#   YELL (Yellowknife, NWT): the civil clock DOES observe DST (MST/MDT).
#       -7 is used here as a fixed standard-time reference, not the civil
#       clock, for the same reason as above.
STATION_UTC_OFFSETS = {
    "UNSA": -3,
    "KOUG": -3,
    "WHIT": -7,
    "YELL": -7,
}

# Station metadata (longitude/latitude), from sta.coor.
STATION_METADATA = {
    "KOUG": {"longitude_deg": -52.639750693, "latitude_deg": 5.064519822},
    "WHIT": {"longitude_deg": -135.222116127, "latitude_deg": 60.586191916},
    "UNSA": {"longitude_deg": -65.407643063, "latitude_deg": -24.581580151},
    "YELL": {"longitude_deg": -114.480707966, "latitude_deg": 62.322894746},
}

# Data-quality flag thresholds (reference: prior TFG memoria and the gAGE
# ITM 2020 paper for this same dataset). These do NOT remove any row here -
# they are stored as boolean columns so later analysis scripts can decide
# whether to filter on them.
QC_MIN_ELEV_DEG = 30.0
QC_REQUIRED_NDAT = 60
QC_MAX_ROTI = 100.0

PARQUET_ENGINE = "pyarrow"
PARQUET_COMPRESSION = "snappy"


def station_parquet_path(station: str) -> Path:
    """Per-station, per-year output path: results/{STATION}/{YEAR}/0_parquet/..."""
    station_dir = RESULTS_DIR / station / str(YEAR) / "0_parquet"
    station_dir.mkdir(parents=True, exist_ok=True)
    return station_dir / f"0_{station}_{YEAR}_observations.parquet"


def resolve_global_paths(year: int) -> dict:
    """
    Per-year global (multi-station) output paths: results/global/{year}/...
    `year` is passed explicitly instead of being read from the YEAR global,
    so the paths stay correct when web_server.py runs main() with temporarily
    patched module globals.
    """
    global_dir = RESULTS_DIR / "global" / str(year)
    global_dir.mkdir(parents=True, exist_ok=True)
    return {
        "global_dir": global_dir,
        "inventory_csv": global_dir / f"0_raw_file_inventory_{year}.csv",
        "station_summary_csv": global_dir / f"0_station_summary_{year}.csv",
    }


# ======================================================================
# [B] RAW COLUMN DEFINITION - DO NOT TOUCH
# ======================================================================
# Official rotiS4.YYYYDDD format: exactly these 12 whitespace-separated
# columns, no header (confirmed by the dataset's author):
#    1 second of the day (GPS time)      7 ROTI from LGF [TECU/min]
#    2 receiver name                     8 Ndat(ROTI L1) - also S4 L1/L2, ROTI L2
#    3 satellite (1-32 GPS, >40 Galileo) 9 Ndat(ROTI LGF)
#    4 elevation [0,90] deg             10 S4 from L1
#    5 azimuth (-180,180] deg           11 S4 from L2
#    6 ROTI from L1 [TECU/min]          12 ROTI from L2 [TECU/min]
# The column count is checked in read_raw_day(): a line with fewer fields
# would be read with NaN in the missing columns, one with more would shift
# the data, so either stops Step 0 (RawFormatError).

RAW_COLS = [
    "01_sec_of_day",
    "02_station",
    "03_satellite",
    "04_elev_deg",
    "05_az_deg",
    "06_roti_l1",
    "07_roti_lgf",
    "08_ndat_roti_l1",
    "09_ndat_roti_lgf",
    "10_s4_l1",
    "11_s4_l2",
    "12_roti_l2",
]
RAW_N_COLUMNS = len(RAW_COLS)


# ======================================================================
# [C] READING FUNCTIONS
# ======================================================================

# Preferred daily RAW name: RAW_PREFERRED_PREFIX + YYYYDDD, e.g.
# "rotiS4.2024001". Accepted variants, in priority order: all of them hold
# the same 12-column text; ".gz" ones are read compressed by pd.read_csv
# itself (compression is inferred from the name, no temporary file).
# Appended to the base name as strings on purpose: Path.with_suffix() would
# treat ".2024001" in "rotiS4.2024001" as the suffix and replace it.
RAW_PREFERRED_PREFIX = "rotiS4."
RAW_CANDIDATE_SUFFIXES = ("", ".gz", ".tmp", ".tmp.gz")


class AmbiguousRawFileError(ValueError):
    """More than one generically named RAW file found for the same day."""


class RawFormatError(ValueError):
    """A RAW file has a line without the official RAW_N_COLUMNS columns."""


def is_generic_raw_name(name: str, day_id: str) -> bool:
    """
    True if `name` holds `day_id` (YYYYDDD) as a whole number - no digit
    right before or after it, so "2025120" does not match "20251201" - and
    is in one of the accepted formats: no extension, .gz, .tmp or .tmp.gz.
    The rest of the name (prefix, separators) does not matter.
    """
    core = name[:-3] if name.endswith(".gz") else name
    core = core[:-4] if core.endswith(".tmp") else core
    extension = Path(core).suffix
    if extension and not extension[1:].isdigit():
        return False  # .txt, .csv, .zip... - not a RAW format; ".2025120" is the date, not an extension
    return re.search(rf"(?<!\d){day_id}(?!\d)", core) is not None


def resolve_raw_file(data_dir: Path, year: int, doy: int) -> tuple[Path, list[Path]]:
    """
    Daily RAW file to read for `year`/`doy`, plus every candidate found.

    1. Preferred name, RAW_PREFERRED_PREFIX + YYYYDDD: the first existing
       variant in RAW_CANDIDATE_SUFFIXES order (several -> main() warns).
    2. Only when none exists: any other file whose name holds YYYYDDD in an
       accepted format (is_generic_raw_name), e.g. "roti.2025120.gz". If
       more than one does, nothing is chosen: AmbiguousRawFileError.
    When nothing matches, the preferred plain name is returned and the day
    is reported as "missing_file".
    """
    day_id = f"{year}{doy:03d}"
    base = data_dir / f"{RAW_PREFERRED_PREFIX}{day_id}"
    found = [
        data_dir / f"{base.name}{suffix}"
        for suffix in RAW_CANDIDATE_SUFFIXES
        if (data_dir / f"{base.name}{suffix}").exists()
    ]
    if found:
        return found[0], found

    generic = sorted(
        (p for p in data_dir.iterdir() if p.is_file() and is_generic_raw_name(p.name, day_id)),
        key=lambda p: p.name,
    ) if data_dir.is_dir() else []
    if len(generic) > 1:
        raise AmbiguousRawFileError(
            f"DoY {doy:03d}: several RAW files contain {day_id} and none is "
            f"{base.name}[.gz|.tmp|.tmp.gz]: {', '.join(p.name for p in generic)}"
        )
    return (generic[0] if generic else base), generic


def find_bad_raw_lines(file_path: Path) -> tuple[int, int, int] | None:
    """
    First line of a RAW file whose number of whitespace-separated fields is
    not RAW_N_COLUMNS, as (line number, number of fields, number of such
    lines in the file), or None if every line has RAW_N_COLUMNS. Blank lines
    are skipped, as pd.read_csv() does. ".gz" files are read compressed.
    """
    opener = gzip.open if file_path.name.lower().endswith(".gz") else open
    first, n_bad = None, 0
    with opener(file_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            n_fields = len(line.split())
            if n_fields in (0, RAW_N_COLUMNS):
                continue
            n_bad += 1
            if first is None:
                first = (line_number, n_fields)
    return None if first is None else (first[0], first[1], n_bad)


def raw_format_error(
    file_path: Path, bad: tuple[int, int, int],
    consequence: str = "Step 0 stopped before writing any result file.",
) -> RawFormatError:
    """RawFormatError for find_bad_raw_lines()'s result: file, first bad line, count."""
    line_number, n_fields, n_bad = bad
    return RawFormatError(
        f"RAW file {file_path.name} does not have the expected format: "
        f"{RAW_N_COLUMNS} whitespace-separated columns per line, no header. "
        f"Line {line_number} has {n_fields} column(s); {n_bad} line(s) of this file "
        f"differ. File: {file_path}. {consequence}".rstrip()
    )


# Detect-from-RAW (web Station configuration): how many daily files are read.
RAW_SAMPLE_FILES = 12


def sample_raw_receivers(
    data_dir: Path, year: int, doy_start: int, doy_end: int, max_files: int = RAW_SAMPLE_FILES,
) -> dict:
    """
    Receiver names (column 2) found in a sample of the daily RAW files Step
    0 would read for year/DoY range (resolve_raw_file()) - never the whole
    year, which is tens of millions of lines.

    Up to max_files files, spread evenly over the days that have one, the
    first and the last always included: a station that starts or stops
    during the year is still likely to be seen, but one present only on
    days between the sampled files can be missed. Each sampled file is read
    line by line keeping only the second field (no DataFrame), and every
    line must have RAW_N_COLUMNS fields (RawFormatError otherwise).

    Returns {"files_found": n, "files_scanned": [names], "receivers":
    {name: rows}}. Raises FileNotFoundError if the range has no RAW file
    and AmbiguousRawFileError as resolve_raw_file(). Reads only.
    """
    found = []
    for doy in range(doy_start, doy_end + 1):
        path, _ = resolve_raw_file(data_dir, year, doy)
        if path.exists():
            found.append(path)
    if not found:
        raise FileNotFoundError(f"No RAW file for {year}, DoY {doy_start}-{doy_end}, in {data_dir}.")
    if len(found) <= max_files:
        sample = found
    else:
        picks = sorted({round(i * (len(found) - 1) / (max_files - 1)) for i in range(max_files)})
        sample = [found[i] for i in picks]

    receivers: dict[str, int] = {}
    for path in sample:
        opener = gzip.open if path.name.lower().endswith(".gz") else open
        first_bad, n_bad = None, 0
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != RAW_N_COLUMNS:
                    n_bad += 1
                    first_bad = first_bad or (line_number, len(fields))
                    continue
                receivers[fields[1]] = receivers.get(fields[1], 0) + 1
        if first_bad is not None:
            raise raw_format_error(path, (*first_bad, n_bad), consequence="")
    return {
        "files_found": len(found),
        "files_scanned": [p.name for p in sample],
        "receivers": receivers,
    }


def read_raw_day(file_path: Path) -> tuple[pd.DataFrame | None, str]:
    """
    Read one daily RAW file (all stations mixed together).

    Returns (dataframe, note). dataframe is None when the file could not be
    used; note explains why ("ok" on success).

    Raises RawFormatError if any line does not have RAW_N_COLUMNS columns.
    pd.read_csv() would otherwise accept such a file: a first line with
    more fields turns the extra leading columns into the index (shifting
    every column), a later one raises ParserError, and a shorter line is
    padded with NaN. The file is only scanned line by line
    (find_bad_raw_lines()) when one of these signs appears, so a valid file
    is read once; NaN written as a value in a 12-column line stays allowed.
    """
    if not file_path.exists():
        return None, "missing_file"

    if file_path.stat().st_size == 0:
        return None, "empty_file"

    try:
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            names=RAW_COLS,
        )
    except pd.errors.EmptyDataError:
        return None, "empty_file"
    except pd.errors.ParserError as exc:
        bad = find_bad_raw_lines(file_path)
        if bad is not None:
            raise raw_format_error(file_path, bad) from None
        return None, f"read_error:{type(exc).__name__}"
    except Exception as exc:
        return None, f"read_error:{type(exc).__name__}"

    if df.empty:
        return None, "empty_file"

    if not isinstance(df.index, pd.RangeIndex) or df.isna().to_numpy().any():
        bad = find_bad_raw_lines(file_path)
        if bad is not None:
            raise raw_format_error(file_path, bad)

    return df, "ok"


def cast_raw_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast RAW integer columns to smaller dtypes to reduce Parquet size.

    ROTI/S4/elevation/azimuth stay float64 - precision matters more there
    than the space saved by downcasting them.
    """
    df["01_sec_of_day"] = df["01_sec_of_day"].astype(np.int32)
    df["03_satellite"] = df["03_satellite"].astype(np.int16)
    df["08_ndat_roti_l1"] = df["08_ndat_roti_l1"].astype(np.int16)
    df["09_ndat_roti_lgf"] = df["09_ndat_roti_lgf"].astype(np.int16)
    return df


# ======================================================================
# [D] ENRICHMENT FUNCTIONS
# ======================================================================

def add_utc_time_columns(df: pd.DataFrame, doy: int, month_utc: int) -> pd.DataFrame:
    """Add year/doy/month/hour columns in UTC (constant for the whole day)."""
    df["13_year"] = np.int16(YEAR)
    df["14_doy_utc"] = np.int16(doy)
    df["15_month_utc"] = np.int8(month_utc)
    df["16_hour_utc"] = (
        df["01_sec_of_day"].to_numpy(dtype=np.int64) // 3600
    ).astype(np.int8)
    return df


def add_local_time_columns(
    df: pd.DataFrame,
    doy: int,
    warned_stations: set,
) -> pd.DataFrame:
    """
    Add utc_offset_hours / hour_local / month_local per station, using the
    fixed offsets in STATION_UTC_OFFSETS.

    Loops over the handful of distinct stations present in this day's file;
    the time arithmetic itself is vectorised per station. Stations without a
    configured offset get NaN here and a one-time warning, without stopping
    the rest of the processing.
    """
    utc_midnight = pd.Timestamp(f"{YEAR}-01-01") + pd.Timedelta(days=doy - 1)

    df["17_utc_offset_hours"] = np.nan
    df["18_hour_local"] = np.nan
    df["19_month_local"] = np.nan

    for station in df["02_station"].unique():
        offset = STATION_UTC_OFFSETS.get(station)
        station_mask = df["02_station"] == station

        if offset is None:
            if station not in warned_stations:
                print(
                    f"[WARNING] No UTC offset configured for station "
                    f"'{station}'. hour_local/month_local will be NaN for it."
                )
                warned_stations.add(station)
            continue

        seconds = df.loc[station_mask, "01_sec_of_day"].to_numpy(dtype=np.int64)
        local_time = (
            utc_midnight
            + pd.to_timedelta(seconds, unit="s")
            + pd.Timedelta(hours=offset)
        )

        df.loc[station_mask, "17_utc_offset_hours"] = offset
        df.loc[station_mask, "18_hour_local"] = local_time.hour
        df.loc[station_mask, "19_month_local"] = local_time.month

    return df


def add_station_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add longitude_deg/latitude_deg from STATION_METADATA. Stations that
    are not in the dictionary get NaN instead of stopping the script."""
    df["20_longitude_deg"] = df["02_station"].map(
        lambda s: STATION_METADATA.get(s, {}).get("longitude_deg", np.nan)
    )
    df["21_latitude_deg"] = df["02_station"].map(
        lambda s: STATION_METADATA.get(s, {}).get("latitude_deg", np.nan)
    )
    return df


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add data-quality flag columns. Nothing is removed here - these are
    informational booleans for later analysis scripts to use if they choose.

    Criteria come from the reference methodology for this dataset (prior
    TFG memoria and the gAGE ITM 2020 paper):
      - elevation >= 30 deg (avoids multipath at low elevation)
      - Ndat(ROTI L1) == 60 (full 60-second averaging window)
      - ROTI L1 <= 100 (values above that are considered unrealistic)
    """
    df["22_qc_elev_ok"] = df["04_elev_deg"] >= QC_MIN_ELEV_DEG
    df["23_qc_ndat_ok"] = df["08_ndat_roti_l1"] == QC_REQUIRED_NDAT
    df["24_qc_roti_plausible"] = df["06_roti_l1"] <= QC_MAX_ROTI
    df["25_qc_all_ok"] = (
        df["22_qc_elev_ok"]
        & df["23_qc_ndat_ok"]
        & df["24_qc_roti_plausible"]
    )
    return df


# ======================================================================
# [E] VALIDATION
# ======================================================================

def validate_raw_seconds(df: pd.DataFrame, doy: int) -> None:
    """
    Validate sec_of_day right after a RAW file is read, so a corrupt file
    fails fast (with the exact DoY) instead of only being caught after a
    full year of processing has already run.
    """
    if not df["01_sec_of_day"].between(0, 86399).all():
        raise ValueError(
            f"DoY {doy:03d}: sec_of_day values outside [0, 86399]."
        )


def validate_written_parquet(station: str, path: Path, expected_rows: int) -> None:
    """
    Verify the Parquet file actually written to disk has the expected row
    count, by reading it back from disk - not just trusting the in-memory
    DataFrame that was passed to to_parquet().
    """
    written_rows = pq.read_metadata(path).num_rows
    if written_rows != expected_rows:
        raise ValueError(
            f"{station}: written Parquet row count ({written_rows:,}) does "
            f"not match the expected row count ({expected_rows:,})."
        )


def validate_station_dataset(station: str, df: pd.DataFrame) -> None:
    """Check the internal consistency of one station's assembled dataset."""
    if df.empty:
        raise ValueError(f"Assembled dataset for station {station} is empty.")

    if not df["01_sec_of_day"].between(0, 86399).all():
        raise ValueError(f"{station}: sec_of_day values outside [0, 86399].")

    if not df["16_hour_utc"].between(0, 23).all():
        raise ValueError(f"{station}: hour_utc values outside [0, 23].")

    local_hours = df["18_hour_local"].dropna()
    if not local_hours.between(0, 23).all():
        raise ValueError(f"{station}: hour_local values outside [0, 23].")


def validate_totals(df_summary: pd.DataFrame, df_inventory: pd.DataFrame) -> None:
    """Cross-check row totals between the per-station outputs and the
    RAW-file inventory."""
    total_summary = int(df_summary["n_rows"].sum())
    total_inventory = int(df_inventory["n_rows_total"].sum())

    if USE_SELECTED_STATIONS:
        if total_summary > total_inventory:
            raise ValueError(
                "Summary row total exceeds total RAW rows read - "
                f"summary={total_summary}, inventory={total_inventory}."
            )
    else:
        if total_summary != total_inventory:
            raise ValueError(
                "Automatic mode: summary row total must match the "
                f"inventory total exactly - summary={total_summary}, "
                f"inventory={total_inventory}."
            )


# ======================================================================
# [E.2] RAW PROVENANCE - which RAW input a year's study was built from
# ======================================================================
RAW_SOURCE_FORMAT = "gnss-step0-raw-source"
# 2: the Step 0 configuration (step0_config_fingerprint) is part of the
# identity. A version 1 record does not hold it and is not accepted.
RAW_SOURCE_VERSION = 2


class RawSourceConflictError(RuntimeError):
    """Step 0 would rebuild a year whose existing data come from another (or an unverifiable) input or configuration."""


def raw_source_manifest_path(year: int) -> Path:
    """Provenance record of a year, next to its RAW inventory (no folder is created)."""
    return RESULTS_DIR / "global" / str(year) / f"0_raw_source_{year}.json"


def raw_source_fingerprint(data_dir: Path, year: int, doy_start: int, doy_end: int,
                           raw_files: dict) -> dict:
    """
    Identity of a run's RAW input, from the files resolve_raw_file() chose
    (raw_files: doy -> (path, candidates)): the RAW folder as an absolute,
    resolved path, the DoY range, and for each DoY the file name with its
    size and modification time (st_mtime_ns), or "missing": true. No
    content hash - the RAW files can add up to several GB.
    """
    files = []
    for doy in range(doy_start, doy_end + 1):
        file_path, _ = raw_files[doy]
        entry = {"doy": doy, "file": file_path.name}
        if file_path.exists():
            stat = file_path.stat()
            entry.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        else:
            entry["missing"] = True
        files.append(entry)
    return {
        "format": RAW_SOURCE_FORMAT,
        "version": RAW_SOURCE_VERSION,
        "year": year,
        "doy_start": doy_start,
        "doy_end": doy_end,
        "raw_dir": str(Path(data_dir).resolve()),
        "files": files,
    }


def step0_config_fingerprint() -> dict:
    """
    The configuration that changes what Step 0 writes into the Parquet
    files, as part of its identity: the station selection, each station's
    UTC offset (columns 17-19, local time - used by Steps 4-6) and
    coordinates (columns 20-21), and the QC thresholds (columns 22-25). In
    web mode the selection and the offsets come from stations.json, whose
    only fields Step 0 receives are code and utc_offset_hours; the order of
    the stations does not matter. In automatic mode (every station found in
    the RAW files) every configured station is included.
    """
    if USE_SELECTED_STATIONS:
        selection, stations = "selected", sorted(SELECTED_STATIONS)
    else:
        selection, stations = "all_in_raw", sorted(set(STATION_UTC_OFFSETS) | set(STATION_METADATA))
    return {
        "station_selection": selection,
        "station_config": {
            station: {
                "utc_offset_hours": STATION_UTC_OFFSETS.get(station),
                "longitude_deg": STATION_METADATA.get(station, {}).get("longitude_deg"),
                "latitude_deg": STATION_METADATA.get(station, {}).get("latitude_deg"),
            }
            for station in stations
        },
        "qc_thresholds": {
            "min_elev_deg": QC_MIN_ELEV_DEG,
            "required_ndat": QC_REQUIRED_NDAT,
            "max_roti": QC_MAX_ROTI,
        },
    }


def _describe_file(entry: dict) -> str:
    if entry.get("missing"):
        return f"{entry['file']} missing"
    return f"{entry['file']} ({entry['size']:,} bytes, modified {entry['mtime_ns']})"


def raw_source_differences(recorded: dict, current: dict) -> list[str]:
    """
    What differs between a recorded and a current fingerprint, as short
    sentences ([] = same RAW input). Folders are compared with
    os.path.normcase(), as Windows paths are case-insensitive.
    """
    differences = []
    if recorded["year"] != current["year"]:
        differences.append(f"year {current['year']} (record of {recorded['year']})")
    if os.path.normcase(recorded["raw_dir"]) != os.path.normcase(current["raw_dir"]):
        differences.append(f"RAW folder {current['raw_dir']} (built from {recorded['raw_dir']})")
    if (recorded["doy_start"], recorded["doy_end"]) != (current["doy_start"], current["doy_end"]):
        differences.append(
            f"DoY range {current['doy_start']}-{current['doy_end']} "
            f"(built from {recorded['doy_start']}-{recorded['doy_end']})"
        )
    if not differences:
        changed = [
            f"DoY {new['doy']:03d}: {_describe_file(new)}, was {_describe_file(old)}"
            for old, new in zip(recorded["files"], current["files"]) if old != new
        ]
        if changed:
            more = f" (and {len(changed) - 3} more)" if len(changed) > 3 else ""
            differences.append(f"{len(changed)} RAW file(s) changed: " + "; ".join(changed[:3]) + more)
    return differences


def config_differences(recorded: dict, current: dict) -> list[str]:
    """What differs in the Step 0 configuration (step0_config_fingerprint), as short sentences."""
    labels = {"utc_offset_hours": "UTC offset", "longitude_deg": "longitude", "latitude_deg": "latitude"}
    differences = []
    if recorded["station_selection"] != current["station_selection"]:
        differences.append(
            f"station selection {current['station_selection']} (built with {recorded['station_selection']})"
        )
    old, new = recorded["station_config"], current["station_config"]
    added, removed = sorted(set(new) - set(old)), sorted(set(old) - set(new))
    if added:
        differences.append("stations added: " + ", ".join(added))
    if removed:
        differences.append("stations removed: " + ", ".join(removed))
    for station in sorted(set(old) & set(new)):
        for key, label in labels.items():
            if old[station].get(key) != new[station].get(key):
                differences.append(f"{station} {label}: {old[station].get(key)} -> {new[station].get(key)}")
    for key, value in current["qc_thresholds"].items():
        if recorded["qc_thresholds"].get(key) != value:
            differences.append(f"QC threshold {key}: {recorded['qc_thresholds'].get(key)} -> {value}")
    return differences


def existing_year_files(year: int) -> list[Path]:
    """Files already in results for `year`: <STATION>/{year}/... and global/{year}/..."""
    return sorted(p for p in RESULTS_DIR.glob(f"*/{year}/**/*") if p.is_file())


def check_raw_source(fingerprint: dict) -> None:
    """
    Pre-flight check of a whole Step 0 run, before anything is written.
    Allowed: the year has no data yet, or its provenance record describes
    exactly this RAW input and Step 0 configuration. Otherwise
    RawSourceConflictError - the run would replace Parquet files under
    completeness and results built from other (or unverifiable) input.
    Nothing is deleted.
    """
    year = fingerprint["year"]
    record_path = raw_source_manifest_path(year)
    advice = (
        f"Step 0 stopped before writing anything: the existing {year} Parquet files and every "
        f"result built from them are unchanged. To build {year} from this input, first move "
        f"or clear the existing {year} study - the {Path('<STATION>', str(year))} folders and "
        f"{Path('global', str(year))} in {RESULTS_DIR}."
    )
    if record_path.exists():
        try:
            recorded = json.loads(record_path.read_text(encoding="utf-8"))
            if (recorded.get("format"), recorded.get("version")) != (RAW_SOURCE_FORMAT, RAW_SOURCE_VERSION):
                raise ValueError(f"unsupported format {recorded.get('format')!r} version {recorded.get('version')!r}")
            raw_changes = raw_source_differences(recorded, fingerprint)
            config_changes = config_differences(recorded, fingerprint)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise RawSourceConflictError(
                f"The RAW provenance record {record_path} cannot be read "
                f"({type(exc).__name__}: {exc}), so the source of the existing {year} data "
                f"cannot be verified. {advice}"
            ) from None
        if raw_changes:
            raise RawSourceConflictError(
                f"The existing {year} data were built from a different RAW input or DoY range: "
                + "; ".join(raw_changes + config_changes) + f". {advice}"
            )
        if config_changes:
            raise RawSourceConflictError(
                f"The RAW input is unchanged, but the configuration Step 0 writes into the Parquet "
                f"files changed since the {year} data were built: " + "; ".join(config_changes)
                + f". {advice}"
            )
        return

    existing = existing_year_files(year)
    if existing:
        examples = ", ".join(str(p.relative_to(RESULTS_DIR)) for p in existing[:3])
        raise RawSourceConflictError(
            f"{year} data already exist in {RESULTS_DIR} ({len(existing)} files, e.g. {examples}) "
            f"but were created before RAW provenance tracking, so their RAW source cannot be "
            f"verified. Steps 1-6 can still use them. {advice}"
        )


def write_raw_source_manifest(fingerprint: dict) -> Path:
    """
    Writes the provenance record of a first build (an identical re-run keeps
    the existing one). Written through a temporary file, so a reader never
    sees a partial JSON.
    """
    path = raw_source_manifest_path(fingerprint["year"])
    if path.exists():
        return path
    record = {
        **fingerprint,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # informative
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


# ======================================================================
# [F] MAIN PROGRAM
# ======================================================================

def main() -> None:
    # Locate every day's RAW file before reading any (and before creating
    # any output folder): an ambiguous generic name stops the run here,
    # listing every such day, instead of picking one of the files.
    raw_files, ambiguous = {}, []
    for doy in range(DOY_START, DOY_END + 1):
        try:
            raw_files[doy] = resolve_raw_file(DATA_DIR, YEAR, doy)
        except AmbiguousRawFileError as exc:
            ambiguous.append(str(exc))
    if ambiguous:
        raise AmbiguousRawFileError(
            "Ambiguous RAW files - leave a single file per day:\n" + "\n".join(ambiguous)
        )

    # Provenance pre-flight for the whole run, before any folder or file is
    # created: stops here if this year already holds data from another (or
    # an unverifiable) RAW input or Step 0 configuration.
    fingerprint = {
        **raw_source_fingerprint(DATA_DIR, YEAR, DOY_START, DOY_END, raw_files),
        **step0_config_fingerprint(),
    }
    check_raw_source(fingerprint)

    global_paths = resolve_global_paths(YEAR)

    print("Step 0 - RAW to Parquet conversion")
    print("===================================")
    print(f"Year: {YEAR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {RESULTS_DIR}/<STATION>/{YEAR}/0_parquet/")
    print(f"Global output directory: {global_paths['global_dir']}")
    if USE_SELECTED_STATIONS:
        print(f"Station mode: manual -> {SELECTED_STATIONS}")
    else:
        print("Station mode: automatic (all stations found in RAW files)")

    station_frames: dict = {}
    inventory_rows = []
    warned_stations: set = set()

    for doy in range(DOY_START, DOY_END + 1):
        file_path, candidates = raw_files[doy]
        if len(candidates) > 1:  # only the preferred name can get here with several
            ignored = ", ".join(p.name for p in candidates[1:])
            print(f"WARNING: DoY {doy:03d}: several RAW files found - "
                  f"using {file_path.name}, ignoring {ignored}")
        raw_df, note = read_raw_day(file_path)

        inventory_rows.append({
            "doy": doy,
            "raw_file": file_path.name,
            "file_exists": file_path.exists(),
            "read_ok": raw_df is not None,
            "n_rows_total": 0 if raw_df is None else int(len(raw_df)),
            "stations_found": (
                ""
                if raw_df is None
                else ",".join(sorted(raw_df["02_station"].unique()))
            ),
            "note": note,
        })

        if raw_df is None:
            continue

        raw_df = cast_raw_dtypes(raw_df)
        validate_raw_seconds(raw_df, doy)

        if USE_SELECTED_STATIONS:
            day_df = raw_df[raw_df["02_station"].isin(SELECTED_STATIONS)].copy()
        else:
            day_df = raw_df.copy()

        if day_df.empty:
            continue

        month_utc = (
            pd.Timestamp(f"{YEAR}-01-01") + pd.Timedelta(days=doy - 1)
        ).month

        day_df = add_utc_time_columns(day_df, doy, month_utc)
        day_df = add_local_time_columns(day_df, doy, warned_stations)
        day_df = add_station_metadata_columns(day_df)
        day_df = add_quality_flags(day_df)

        for station, group in day_df.groupby("02_station"):
            station_frames.setdefault(station, []).append(group)

        if doy % 50 == 0:
            print(f"Processed DoY {doy:03d}/{DOY_END}...")

    if USE_SELECTED_STATIONS:
        missing_stations = sorted(set(SELECTED_STATIONS) - set(station_frames))
        if missing_stations:
            print(
                f"[WARNING] Selected stations with zero rows in the whole "
                f"year: {missing_stations}"
            )

    if not station_frames:
        raise RuntimeError(
            f"No data found for any station in {DATA_DIR} for DOY "
            f"{DOY_START}-{DOY_END}. Check the RAW directory and DoY range."
        )

    # Every RAW file has been read and checked: record the input before the
    # first Parquet is written, so an interrupted run can only be resumed
    # from this same RAW input.
    write_raw_source_manifest(fingerprint)

    summary_rows = []

    for station in sorted(station_frames):
        df_station = pd.concat(station_frames[station], ignore_index=True)
        validate_station_dataset(station, df_station)

        out_path = station_parquet_path(station)
        df_station.to_parquet(
            out_path,
            index=False,
            engine=PARQUET_ENGINE,
            compression=PARQUET_COMPRESSION,
        )
        validate_written_parquet(station, out_path, expected_rows=len(df_station))

        offset = STATION_UTC_OFFSETS.get(station, np.nan)
        meta = STATION_METADATA.get(station, {})

        summary_rows.append({
            "station": station,
            "n_rows": int(len(df_station)),
            "first_doy": int(df_station["14_doy_utc"].min()),
            "last_doy": int(df_station["14_doy_utc"].max()),
            "n_doys_with_data": int(df_station["14_doy_utc"].nunique()),
            "utc_offset_hours": offset,
            "longitude_deg": meta.get("longitude_deg", np.nan),
            "latitude_deg": meta.get("latitude_deg", np.nan),
            # Relative to the project root, so the CSV holds no machine-specific path.
            "parquet_file": str(out_path.relative_to(project_paths.PROJECT_ROOT)),
        })

        print(
            f"{station}: {len(df_station):,} rows, "
            f"{df_station['14_doy_utc'].nunique()} days -> {out_path}"
        )

    df_inventory = (
        pd.DataFrame(inventory_rows).sort_values("doy").reset_index(drop=True)
    )
    df_inventory.to_csv(global_paths["inventory_csv"], index=False)

    df_summary = (
        pd.DataFrame(summary_rows).sort_values("station").reset_index(drop=True)
    )
    validate_totals(df_summary, df_inventory)
    df_summary.to_csv(global_paths["station_summary_csv"], index=False)

    print("\nValidation summary")
    print("------------------")
    print(f"Stations processed: {sorted(station_frames)}")
    print(f"Total rows across output Parquet files: {int(df_summary['n_rows'].sum()):,}")
    print(f"Total rows read from RAW (all stations): {int(df_inventory['n_rows_total'].sum()):,}")

    print("\nOutput files")
    print("------------")
    print(f"RAW file inventory: {global_paths['inventory_csv']}")
    print(f"Station summary: {global_paths['station_summary_csv']}")
    print(f"RAW provenance record: {raw_source_manifest_path(YEAR)}")
    print(
        f"Per-station Parquet files: {RESULTS_DIR}/<STATION>/{YEAR}/0_parquet/"
        f"0_<STATION>_{YEAR}_observations.parquet"
    )
    print("\nStep 0 completed successfully.")


if __name__ == "__main__":
    main()
