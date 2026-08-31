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
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ======================================================================
# [A] STUDY CONFIGURATION - EDIT ONLY THIS SECTION
# ======================================================================

YEAR = 2024
DATA_DIR = Path("datos_Estudio")
RESULTS_DIR = Path("results")
DOY_START = 1
DOY_END = 366  # 2024 is a leap year

# Station selection.
USE_SELECTED_STATIONS = True
SELECTED_STATIONS = ["UNSA", "KOUG", "WHIT", "YELL"]

# Fixed manual UTC offsets, used ONLY to derive hour_local/month_local.
# These are NOT dynamic civil timezones - no daylight-saving changes are
# applied, on purpose, so "local time" stays physically consistent across
# the whole year for every station.
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
    `year` is a required, explicit argument - never read from the YEAR
    global - so this returns correct paths even when called after a caller
    has patched and then restored module-level YEAR (e.g. web_server.py's
    _patched_globals, used to run this script's unmodified main() from the
    web UI). Same "computed from the arguments received, not the module
    globals" rule as every resolve_paths() in Steps 1-6 (CLAUDE.md section
    11). Previously GLOBAL_DIR/OUTPUT_INVENTORY_CSV/OUTPUT_STATION_SUMMARY_CSV
    were plain module-level constants, computed once from the literal
    YEAR=2024 default at import time - correct for a single console run,
    silently stale for any second main() call in the same process with a
    different YEAR. Technical fix only (CLAUDE.md section 14): identical
    paths for the existing YEAR=2024 console case, correct paths for any
    other year.
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
# NOTE: this assumes every RAW file has exactly these 12 whitespace
# separated columns, which holds for 100% of the files in datos_Estudio at
# the time this script was written. A future file with fewer columns would
# fail to parse and be logged as a read_error in the inventory rather than
# silently misaligning data.

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


# ======================================================================
# [C] READING FUNCTIONS
# ======================================================================

def read_raw_day(file_path: Path) -> tuple[pd.DataFrame | None, str]:
    """
    Read one daily RAW file (all stations mixed together).

    Returns (dataframe, note). dataframe is None when the file could not be
    used; note explains why ("ok" on success).
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
    except Exception as exc:
        return None, f"read_error:{type(exc).__name__}"

    if df.empty:
        return None, "empty_file"

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
# [F] MAIN PROGRAM
# ======================================================================

def main() -> None:
    file_prefix = f"rotiS4.{YEAR}"
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
        file_path = DATA_DIR / f"{file_prefix}{doy:03d}"
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
            "parquet_file": str(out_path),
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
    print(
        f"Per-station Parquet files: {RESULTS_DIR}/<STATION>/{YEAR}/0_parquet/"
        f"0_<STATION>_{YEAR}_observations.parquet"
    )
    print("\nStep 0 completed successfully.")


if __name__ == "__main__":
    main()
