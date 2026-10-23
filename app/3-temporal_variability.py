#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3 - Annual thresholds and daily exceedance frequency.

Inputs: the Step 0 Parquet and the Step 1 completeness CSV. Outputs under
results/{STATION}/{YEAR}/3_temporal_variability/{INDEX}/<dir_tag>/. Same
structure as Steps 1-2: globals in section [A] are console defaults,
run_temporal_analysis() is a pure parametrised calculation (no prints, no
file writes), main() is the console entry point.

Days and observations always come from the same Ndat criterion: the valid
days of that criterion's Step 1 completeness (1_completeness/<dir_tag>/)
and, within them, the observations that satisfy it ("all": every
observation).

The annual p90/p99 thresholds are station-specific and specific to the Ndat
criterion: each station's are computed from that same population. Steps 4-6
read the annual-thresholds CSV written here and reuse them unchanged. An
exceedance is a value >= threshold.

The analysed index (ROTI L1, S4 L1, ...) and its Ndat column, labels and
unit come from index_config.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import completeness_paths
import index_config as indices
import ndat_filter
import plot_style
import project_paths


# ======================================================================
# [A] STUDY CONFIGURATION - EDIT ONLY THIS SECTION
# ======================================================================

STATION = "YELL"
YEAR = 2024
DOY_START = 1
DOY_END = 366

# Index to analyse: any key of index_config.INDEX_CONFIG (e.g. "roti_l1", "s4_l1").
VALUE_COL = indices.DEFAULT_INDEX

# Ndat criterion (a canonical ndat_filter mode, e.g. "eq60"). It selects both the
# Step 1 completeness whose valid days are used and the observations; Step 1
# must have been run with the same criterion. "all" keeps every observation.
NDAT_MODE = "all"

TH_COV = 0.75

PERCENTILE_HIGH = 90
PERCENTILE_EXTREME = 99

SHOW_PLOTS = False
PLOT_DPI = 250


# ======================================================================
# [C] PATH RESOLUTION
# ======================================================================
def resolve_paths(
    station: str,
    year: int,
    doy_start: int,
    doy_end: int,
    th_cov: float,
    value_col: str,
    index_config: dict,
    ndat_mode: str = "all",
) -> dict:
    """
    Input/output paths, computed from the arguments received (not from
    the module globals). Outputs go under
    results/{STATION}/{YEAR}/3_temporal_variability/{VALUE_COL}/<dir_tag>/,
    with the criterion's file_tag in the file prefix and the run tag (DoY
    range, th_cov) in every file name, as in 2-ccdf.py. Steps 4-6 read the
    thresholds through the same names, so they only find those of their
    own criterion, DoY range and th_cov.

    coverage_csv is the Step 1 completeness of the same criterion
    (completeness_paths). Raises ValueError if ndat_mode is unknown.
    """
    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)

    parquet_path = (
        project_paths.RESULTS_ROOT / station / str(year) / "0_parquet"
        / f"0_{station}_{year}_observations.parquet"
    )
    coverage_csv = completeness_paths.resolve_completeness_paths(
        station, year, doy_start, doy_end, th_cov, ndat_mode, index=value_col
    )["output_csv"]

    index_dir = project_paths.RESULTS_ROOT / station / str(year) / "3_temporal_variability" / value_col
    ndat_dir = index_dir / ndat_config["dir_tag"]
    prefix = f"3-{station}_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"

    run_tag = completeness_paths.run_tag(doy_start, doy_end, th_cov)

    return {
        "parquet_path": parquet_path,
        "coverage_csv": coverage_csv,
        "index_dir": index_dir,
        "ndat_dir": ndat_dir,
        "output_thresholds_csv": ndat_dir / f"{prefix}_{run_tag}_annual_thresholds.csv",
        "output_daily_csv": ndat_dir / f"{prefix}_{run_tag}_daily_frequency.csv",
        "output_daily_plot": ndat_dir / f"{prefix}_{run_tag}_daily_frequency.png",
        "prefix": prefix,
    }


# ======================================================================
# [D] INPUT LOADING
# ======================================================================
def load_valid_days(coverage_csv: Path, doy_start: int, doy_end: int) -> pd.DataFrame:
    """Read the Step 1 CSV and return valid days inside the selected range."""
    df_cov = pd.read_csv(coverage_csv)
    required = {"DoY", "coverage", "status"}
    missing = required - set(df_cov.columns)

    if missing:
        raise ValueError(f"Coverage CSV is missing columns: {sorted(missing)}")

    df_cov = df_cov.copy()
    df_cov["DoY"] = pd.to_numeric(df_cov["DoY"], errors="coerce")
    df_cov["coverage"] = pd.to_numeric(df_cov["coverage"], errors="coerce")
    df_cov = df_cov.dropna(subset=["DoY", "coverage"])
    df_cov["DoY"] = df_cov["DoY"].astype(int)

    if df_cov["DoY"].duplicated().any():
        raise ValueError("Coverage CSV contains duplicated DoY values.")

    df_valid = df_cov[
        (df_cov["status"] == "valid")
        & df_cov["DoY"].between(doy_start, doy_end)
    ][["DoY", "coverage"]]

    df_valid = df_valid.sort_values("DoY").reset_index(drop=True)

    if df_valid.empty:
        raise ValueError("No valid days were found in the selected range.")

    return df_valid


# ======================================================================
# [E] INTERNAL CONSISTENCY CHECKS (no prints - raise on inconsistency)
# ======================================================================
def validate_results(
    df_valid: pd.DataFrame,
    annual_values: np.ndarray,
    threshold_high: float,
    threshold_extreme: float,
    df_daily: pd.DataFrame,
    percentile_high: int,
    percentile_extreme: int,
) -> None:
    """Check the internal consistency of the calculated outputs."""
    if not np.isfinite([threshold_high, threshold_extreme]).all():
        raise ValueError("Annual thresholds are not finite.")

    if threshold_high > threshold_extreme:
        raise ValueError(f"Annual p{percentile_high} is greater than annual p{percentile_extreme}.")

    if int(df_daily["n_values"].sum()) != int(annual_values.size):
        raise ValueError("Daily and annual sample totals do not match.")

    n_high_col = f"n_at_or_above_p{percentile_high}"
    n_extreme_col = f"n_at_or_above_p{percentile_extreme}"
    freq_high_col = f"frequency_p{percentile_high}_pct"
    freq_extreme_col = f"frequency_p{percentile_extreme}_pct"

    invalid_counts = (
        (df_daily[n_extreme_col] > df_daily[n_high_col])
        | (df_daily[n_high_col] > df_daily["n_values"])
    )
    if invalid_counts.any():
        raise ValueError("Invalid daily threshold-exceedance counts detected.")

    for column in [freq_high_col, freq_extreme_col]:
        if not df_daily[column].between(0.0, 100.0).all():
            raise ValueError(f"Values outside [0, 100] detected in {column}.")


# ======================================================================
# [F] SCIENTIFIC CALCULATION (no prints, no output files)
# ======================================================================
def run_temporal_analysis(
    station: str = STATION,
    year: int = YEAR,
    doy_start: int = DOY_START,
    doy_end: int = DOY_END,
    th_cov: float = TH_COV,
    value_col: str = VALUE_COL,
    percentile_high: int = PERCENTILE_HIGH,
    percentile_extreme: int = PERCENTILE_EXTREME,
    ndat_mode: str = "all",
) -> dict:
    """
    Annual thresholds (p_high, p_extreme) and daily exceedance frequency
    for one station, year and DoY range. Pure calculation: every input is
    an argument, nothing is printed or written.

    Population: the valid days of the Step 1 completeness of ndat_mode and,
    within them, the rows that satisfy ndat_filter.apply_ndat_filter()
    ("all": every row). The annual thresholds and both terms of the daily
    frequency come from this same population, so different criteria are
    never mixed. An exceedance is a value >= threshold.

    Raises FileNotFoundError if the Step 0 Parquet or the Step 1
    completeness of this criterion is missing (no fallback to another
    criterion), and ValueError if value_col is not scientifically
    supported or ndat_mode is unknown.

    Returns a dict with:
      - "index_config": index_config.INDEX_CONFIG entry of value_col;
      - "ndat_config": ndat_filter.validate_ndat_mode(ndat_mode);
      - "annual_values": 1D array with every observation used;
      - "threshold_high", "threshold_extreme": annual thresholds;
      - "df_thresholds": one-row summary table;
      - "df_daily": daily exceedance frequency;
      - "df_valid", "skipped_doys": Step 1 valid days, and those of them
        without usable values.
    """
    index_config = indices.validate_index_supported(value_col)
    parquet_value_col = index_config["parquet_column"]
    ndat_column = index_config["ndat_column"]
    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
    indices.validate_ndat_for_index(value_col, ndat_config)

    paths = resolve_paths(station, year, doy_start, doy_end, th_cov, value_col, index_config, ndat_mode)

    if not paths["parquet_path"].exists():
        raise FileNotFoundError(
            f"Step 0 Parquet not found for station {station}: {paths['parquet_path']}\n"
            "Run 0-toParquet.py first."
        )

    coverage_csv = completeness_paths.require_completeness_csv(
        station, year, doy_start, doy_end, th_cov, ndat_mode, index=value_col
    )
    df_valid = load_valid_days(coverage_csv, doy_start, doy_end)

    df_station = pd.read_parquet(
        paths["parquet_path"],
        columns=["14_doy_utc", parquet_value_col, ndat_column]
    )

    df_work = df_station[df_station["14_doy_utc"].isin(df_valid["DoY"])].copy()
    df_work = ndat_filter.apply_ndat_filter(df_work, ndat_mode, ndat_column)

    numeric_values = pd.to_numeric(df_work[parquet_value_col], errors="coerce")
    finite_mask = numeric_values.notna() & np.isfinite(numeric_values.to_numpy(dtype=float))
    df_work = df_work.loc[finite_mask].copy()
    df_work[parquet_value_col] = numeric_values.loc[df_work.index]

    if df_work.empty:
        raise ValueError("No usable observations found for the selected valid days.")

    annual_values = df_work[parquet_value_col].to_numpy(dtype=float)
    n_days_processed = int(df_work["14_doy_utc"].nunique())

    thresholds = np.percentile(annual_values, [percentile_high, percentile_extreme])
    threshold_high = float(thresholds[0])
    threshold_extreme = float(thresholds[1])

    n_values = int(annual_values.size)
    n_high = int(np.count_nonzero(annual_values >= threshold_high))
    n_extreme = int(np.count_nonzero(annual_values >= threshold_extreme))

    df_thresholds = pd.DataFrame([{
        "station": station,
        "year": year,
        "value_column": value_col,
        "unit": index_config["unit"],
        "coverage_threshold": th_cov,
        "n_valid_days_from_coverage": len(df_valid),
        "n_days_processed": n_days_processed,
        "n_values": n_values,
        f"p{percentile_high}_annual": threshold_high,
        f"p{percentile_extreme}_annual": threshold_extreme,
        f"n_at_or_above_p{percentile_high}": n_high,
        f"frequency_p{percentile_high}_pct": 100.0 * n_high / n_values,
        f"n_at_or_above_p{percentile_extreme}": n_extreme,
        f"frequency_p{percentile_extreme}_pct": 100.0 * n_extreme / n_values,
    }])

    # ------------------------------------------------------------------
    # Daily exceedance frequency: one groupby over the observations used
    # for the annual thresholds.
    # ------------------------------------------------------------------
    df_work["above_high"] = df_work[parquet_value_col] >= threshold_high
    df_work["above_extreme"] = df_work[parquet_value_col] >= threshold_extreme

    daily = (
        df_work.groupby("14_doy_utc")
        .agg(
            n_values=(parquet_value_col, "size"),
            n_high=("above_high", "sum"),
            n_extreme=("above_extreme", "sum"),
        )
        .reset_index()
        .rename(columns={"14_doy_utc": "doy"})
    )

    daily["n_values"] = daily["n_values"].astype(int)
    daily["n_high"] = daily["n_high"].astype(int)
    daily["n_extreme"] = daily["n_extreme"].astype(int)

    daily = daily.merge(
        df_valid.rename(columns={"DoY": "doy"}),
        on="doy",
        how="left",
    )

    daily[f"frequency_p{percentile_high}_pct"] = 100.0 * daily["n_high"] / daily["n_values"]
    daily[f"frequency_p{percentile_extreme}_pct"] = 100.0 * daily["n_extreme"] / daily["n_values"]

    daily = daily.rename(columns={
        "n_high": f"n_at_or_above_p{percentile_high}",
        "n_extreme": f"n_at_or_above_p{percentile_extreme}",
    })

    daily["month"] = daily["doy"].apply(lambda d: doy_to_month(year, int(d)))
    daily["date"] = daily["doy"].apply(lambda d: doy_to_date(year, int(d)).date().isoformat())

    df_daily = daily[
        [
            "doy", "date", "month", "coverage", "n_values",
            f"n_at_or_above_p{percentile_high}", f"frequency_p{percentile_high}_pct",
            f"n_at_or_above_p{percentile_extreme}", f"frequency_p{percentile_extreme}_pct",
        ]
    ].sort_values("doy").reset_index(drop=True)

    validate_results(
        df_valid, annual_values, threshold_high, threshold_extreme,
        df_daily, percentile_high, percentile_extreme,
    )

    skipped_doys = sorted(set(df_valid["DoY"]) - set(df_daily["doy"]))

    return {
        "index_config": index_config,
        "ndat_config": ndat_config,
        "annual_values": annual_values,
        "threshold_high": threshold_high,
        "threshold_extreme": threshold_extreme,
        "df_thresholds": df_thresholds,
        "df_daily": df_daily,
        "df_valid": df_valid,
        "skipped_doys": skipped_doys,
    }


def doy_to_date(year: int, doy: int):
    """Convert year and day of year to a datetime object."""
    from datetime import datetime
    return datetime.strptime(f"{year}-{doy:03d}", "%Y-%j")


def doy_to_month(year: int, doy: int) -> int:
    """Convert year and day of year to a calendar month (1..12)."""
    return doy_to_date(year, doy).month


# ======================================================================
# [G] PLOT
# ======================================================================
@plot_style.with_font_sizes(sizes=plot_style.font_sizes_plus(7))
def save_daily_frequency_plot(
    df_daily: pd.DataFrame,
    threshold_high: float,
    threshold_extreme: float,
    station: str,
    year: int,
    index_label: str,
    index_unit: str,
    doy_start: int,
    doy_end: int,
    percentile_high: int,
    percentile_extreme: int,
    out_png: Path,
    plot_dpi: int = PLOT_DPI,
    show_plot: bool = SHOW_PLOTS,
) -> None:
    """Save the daily threshold-exceedance frequency plot."""
    freq_high_col = f"frequency_p{percentile_high}_pct"
    freq_extreme_col = f"frequency_p{percentile_extreme}_pct"

    # Reindexing creates gaps for invalid or unavailable days.
    full_doy = pd.Index(range(doy_start, doy_end + 1), name="doy")
    df_plot = df_daily.set_index("doy").reindex(full_doy)

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(
        df_plot.index,
        df_plot[freq_high_col],
        linewidth=1.2,
        label=(
            f"At or above annual p{percentile_high} "
            f"(T{percentile_high}={indices.value_with_unit(f'{threshold_high:.3f}', index_unit)})"
        )
    )

    ax.plot(
        df_plot.index,
        df_plot[freq_extreme_col],
        linewidth=1.2,
        label=(
            f"At or above annual p{percentile_extreme} "
            f"(T{percentile_extreme}={indices.value_with_unit(f'{threshold_extreme:.3f}', index_unit)})"
        )
    )

    ax.set_xlim(doy_start, doy_end)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Day of year (DoY)")
    ax.set_ylabel("Exceedance frequency [% of observations]")
    ax.set_title(
        f"Daily frequency of annual {index_label} threshold exceedances "
        f"— {station} — {year}\n"
        "Calculated from valid receiver-satellite observations"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=plot_dpi)

    if show_plot:
        plt.show()

    plt.close(fig)


# ======================================================================
# [H] MAIN (console mode): calculation + CSV/PNG outputs + console report
# ======================================================================
def main() -> None:
    print("Step 3 — Annual thresholds and daily exceedance frequency")
    print("===========================================================")
    print(f"Station: {STATION}")
    print(f"Year: {YEAR}")

    results = run_temporal_analysis(
        station=STATION,
        year=YEAR,
        doy_start=DOY_START,
        doy_end=DOY_END,
        th_cov=TH_COV,
        value_col=VALUE_COL,
        percentile_high=PERCENTILE_HIGH,
        percentile_extreme=PERCENTILE_EXTREME,
        ndat_mode=NDAT_MODE,
    )

    index_config = results["index_config"]
    index_label = index_config["label"]
    index_unit = index_config["unit"]
    annual_values = results["annual_values"]
    threshold_high = results["threshold_high"]
    threshold_extreme = results["threshold_extreme"]
    df_thresholds = results["df_thresholds"]
    df_daily = results["df_daily"]
    df_valid = results["df_valid"]
    skipped_doys = results["skipped_doys"]

    print("\nValidation summary")
    print("------------------")
    print(f"Valid days selected from Step 1: {len(df_valid)}")
    print(f"Days successfully processed: {len(df_daily)}")
    print(f"Valid days skipped: {len(skipped_doys)}")
    print(f"Annual sample count: {annual_values.size:,}")
    print(f"Index: {index_label}")
    print(f"Annual p{PERCENTILE_HIGH}: {indices.value_with_unit(f'{threshold_high:.6f}', index_unit)}")
    print(f"Annual p{PERCENTILE_EXTREME}: {indices.value_with_unit(f'{threshold_extreme:.6f}', index_unit)}")

    annual_freq_high = 100.0 * np.mean(annual_values >= threshold_high)
    annual_freq_extreme = 100.0 * np.mean(annual_values >= threshold_extreme)
    print(f"Observed annual frequency at or above p{PERCENTILE_HIGH}: {annual_freq_high:.6f}%")
    print(f"Observed annual frequency at or above p{PERCENTILE_EXTREME}: {annual_freq_extreme:.6f}%")
    print(
        "Note: repeated values at a percentile threshold can make the observed "
        "frequency slightly higher than 10% or 1%."
    )

    for doy in skipped_doys:
        print(f"[WARNING] DoY {doy:03d} skipped: no_usable_values")

    paths = resolve_paths(STATION, YEAR, DOY_START, DOY_END, TH_COV, VALUE_COL, index_config, NDAT_MODE)
    paths["ndat_dir"].mkdir(parents=True, exist_ok=True)

    df_thresholds.to_csv(paths["output_thresholds_csv"], index=False)
    df_daily.to_csv(paths["output_daily_csv"], index=False)

    save_daily_frequency_plot(
        df_daily, threshold_high, threshold_extreme,
        STATION, YEAR, index_label, index_unit,
        DOY_START, DOY_END, PERCENTILE_HIGH, PERCENTILE_EXTREME,
        paths["output_daily_plot"],
        plot_dpi=PLOT_DPI, show_plot=SHOW_PLOTS,
    )

    print("\nOutput files")
    print("------------")
    print(f"Annual thresholds CSV: {paths['output_thresholds_csv']}")
    print(f"Daily frequency CSV: {paths['output_daily_csv']}")
    print(f"Daily frequency plot: {paths['output_daily_plot']}")
    print("\nStep 3 completed successfully.")


if __name__ == "__main__":
    main()
