#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3.2 - Cross-station comparison of Step 3 daily variability.

Reads the annual_thresholds.csv and daily_frequency.csv that
3-temporal_variability.py already saves for each station and builds:
  - a per-station comparison table (annual p90/p99, n_values, and the day
    of max p90/p99 exceedance frequency for that station, with that day's
    own n_values/coverage so a maximum is never read without knowing how
    much data supports it);
  - the top-N days by p99 exceedance frequency for each available
    station (N configurable, default 10);
  - which calendar dates appear in the top-N of two or more stations at
    once ("coincidences") - a precise, reproducible rule, not a
    subjective "looks similar";
  - one combined plot overlaying every station's daily p99 frequency on
    the same axis, so a day that stands out for several stations is
    visible directly.

Does not recompute anything and never touches the Step 0 Parquet or the
Step 1 completeness CSV - it only reads what 3-temporal_variability.py
has already produced per station. Same philosophy as
2.2-ccdf_comparison.py: a station whose Step 3 has not been run yet is
skipped, not fatal.

IMPORTANT scientific caveat (documented here because it is easy to
misread the coincidences table otherwise): each station's p90/p99
threshold is its OWN annual threshold. A day appearing in two stations'
top-N lists at once means both stations show elevated activity RELATIVE
TO THEIR OWN yearly distribution - it does NOT mean both stations had
the same absolute ROTI L1 magnitude that day. This script does not
attempt to compare absolute magnitude between stations; that would need
a different analysis (see Step 2's monthly percentiles/CCDF for that).

This step is explicitly descriptive, not interpretive (CLAUDE.md
section 1): it detects and reports candidate days, it does not attempt
to explain them physically. Candidates worth investigating further
(against Kp/Dst/F10.7/known storms/literature) belong in hallazgos.md as
"candidates to investigate", written by hand after reviewing the actual
output - not generated automatically by this script.

Scientific scope: only ROTI L1 (roti_l1) is implemented/validated - see
INDEX_CONFIG / validate_index_supported() (CLAUDE.md section 3-bis).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import ndat_filter


# ======================================================================
# [A] STUDY CONFIGURATION - EDIT ONLY THIS SECTION
# ======================================================================

STATIONS = ["UNSA", "KOUG", "WHIT", "YELL"]
YEAR = 2024
DOY_START = 1
DOY_END = 366
TH_COV = 0.75

# Index to compare. Only "roti_l1" is scientifically implemented/validated.
VALUE_COL = "roti_l1"

# Ndat criterion to compare across stations. See ndat_filter.py - eq60 is
# the TFM official default.
NDAT_MODE = ndat_filter.NDAT_DEFAULT_MODE

# How many top days (by frequency_p99_pct) to keep per station, and to
# check for cross-station coincidences.
TOP_N_DAYS = 10

# Fixed color per station (identity, not position in a list - same
# motivation as month_color()/STATION_COLORS in 2-ccdf.py). Independent
# copy, not imported - each script here is self-contained by convention.
STATION_COLORS = {"UNSA": "tab:orange", "KOUG": "tab:blue", "WHIT": "tab:green", "YELL": "tab:red"}

SHOW_PLOTS = False
PLOT_DPI = 250


# ======================================================================
# [B] INDICES: PARQUET COLUMN <-> FILE TAG <-> SCIENTIFIC LABEL <-> UNIT
# ======================================================================
# Same copy already used by every other script in this project
# (established convention: each script is self-contained).

INDEX_CONFIG = {
    "roti_l1": {
        "parquet_column": "06_roti_l1",
        "file_tag": "ROTIL1",
        "label": "ROTI L1",
        "unit": "TECU/min",
        "scientifically_supported": True,
    },
    "roti_lgf": {
        "parquet_column": "07_roti_lgf",
        "file_tag": "ROTILGF",
        "label": "ROTI LGF",
        "unit": "TECU/min",
        "scientifically_supported": False,
    },
    "roti_l2": {
        "parquet_column": "12_roti_l2",
        "file_tag": "ROTIL2",
        "label": "ROTI L2",
        "unit": "TECU/min",
        "scientifically_supported": False,
    },
    "s4_l1": {
        "parquet_column": "10_s4_l1",
        "file_tag": "S4L1",
        "label": "S4 L1",
        "unit": "",  # S4 is dimensionless
        "scientifically_supported": False,
    },
    "s4_l2": {
        "parquet_column": "11_s4_l2",
        "file_tag": "S4L2",
        "label": "S4 L2",
        "unit": "",  # S4 is dimensionless
        "scientifically_supported": False,
    },
}


def validate_index_supported(value_col: str) -> dict:
    """Same check as 3-temporal_variability.py - value_col must have a validated analysis."""
    if value_col not in INDEX_CONFIG:
        raise ValueError(
            f"Index '{value_col}' is not defined in INDEX_CONFIG. "
            f"Known indices: {sorted(INDEX_CONFIG)}"
        )

    config = INDEX_CONFIG[value_col]

    if not config["scientifically_supported"]:
        raise ValueError(
            f"Index '{value_col}' is available in the Parquet but its "
            "Step 3 scientific analysis has not yet been implemented/validated."
        )

    return config


# ======================================================================
# [C] PATH RESOLUTION
# ======================================================================
def resolve_station_thresholds_csv(
    station: str,
    year: int,
    doy_start: int,
    doy_end: int,
    value_col: str,
    index_config: dict,
    ndat_mode: str | None = None,
) -> Path:
    """
    Mirrors 3-temporal_variability.py's own resolve_paths() naming exactly,
    including its ndat_mode branch (None = legacy path/prefix, a real mode
    adds ndat_config["dir_tag"]/["file_tag"]) - see that function's own
    docstring for what each branch does. value_col is now an explicit
    parameter (previously this function read the module-level VALUE_COL
    directly, a pre-existing deviation from CLAUDE.md section 11 unrelated
    to Ndat, fixed here since this signature is already being touched).
    """
    index_dir = Path("results") / station / str(year) / "3_temporal_variability" / value_col

    if ndat_mode is None:
        ndat_dir = index_dir
        prefix = f"3-{station}_{index_config['file_tag']}_{year}"
    else:
        ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
        ndat_dir = index_dir / ndat_config["dir_tag"]
        prefix = f"3-{station}_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"

    doy_tag = f"DOY{doy_start}_{doy_end}"
    return ndat_dir / f"{prefix}_{doy_tag}_annual_thresholds.csv"


def resolve_station_daily_csv(
    station: str,
    year: int,
    doy_start: int,
    doy_end: int,
    value_col: str,
    index_config: dict,
    ndat_mode: str | None = None,
) -> Path:
    """
    Mirrors 3-temporal_variability.py's own resolve_paths() naming exactly
    - same logic as resolve_station_thresholds_csv() above, only the
    output filename differs.
    """
    index_dir = Path("results") / station / str(year) / "3_temporal_variability" / value_col

    if ndat_mode is None:
        ndat_dir = index_dir
        prefix = f"3-{station}_{index_config['file_tag']}_{year}"
    else:
        ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
        ndat_dir = index_dir / ndat_config["dir_tag"]
        prefix = f"3-{station}_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"

    doy_tag = f"DOY{doy_start}_{doy_end}"
    return ndat_dir / f"{prefix}_{doy_tag}_daily_frequency.csv"


def resolve_comparison_paths(
    year: int,
    index_config: dict,
    value_col: str = VALUE_COL,
    ndat_mode: str | None = None,
) -> dict:
    """
    Output paths under results/global/{YEAR}/ - never inside a single
    station's own folder.

    value_col/ndat_mode are new, trailing, defaulted parameters -
    web_server.py's existing call, resolve_comparison_paths(year,
    index_config) (2 positional arguments), keeps working completely
    unchanged. Deliberately placed after index_config, not before like
    2.2-ccdf_comparison.py's equivalent (year, value_col, index_config,
    ndat_mode=None) - that order would break the existing positional call
    here.

    ndat_mode=None (default) reproduces exactly today's flat path (no
    index or Ndat directory level - this global path was never nested by
    index before Ndat existed). A real mode adds *both* a value_col level
    and an ndat_config["dir_tag"] level at once, atomically, only in this
    branch - same mechanism already validated in
    2.2-ccdf_comparison.py::resolve_comparison_paths().
    """
    base_dir = Path("results") / "global" / str(year) / "3_temporal_comparison"

    if ndat_mode is None:
        comparison_dir = base_dir
        prefix = f"3-ALL_{index_config['file_tag']}_{year}"
    else:
        ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
        comparison_dir = base_dir / value_col / ndat_config["dir_tag"]
        prefix = f"3-ALL_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"

    return {
        "comparison_dir": comparison_dir,
        "output_table_csv": comparison_dir / f"{prefix}_annual_summary_by_station.csv",
        "output_top_days_csv": comparison_dir / f"{prefix}_top{TOP_N_DAYS}_days.csv",
        "output_coincidences_csv": comparison_dir / f"{prefix}_coincidences.csv",
        "output_daily_comparison_png": comparison_dir / f"{prefix}_daily_p99_frequency_comparison.png",
    }


# ======================================================================
# [D] CROSS-STATION COMPARISON (pure - no prints, no file I/O)
# ======================================================================
_COINCIDENCE_COLUMNS = [
    "date", "n_stations", "stations",
    "frequency_p99_pct_by_station", "rank_by_station",
]


def _format_by_station(row: pd.Series, value_col: str, fmt: str) -> str:
    """
    "STATION: value | STATION: value", sorted alphabetically by station -
    same order as the "stations" column. Works for any number/combination
    of stations (2, 3, 4...), never hardcoded to specific station names.
    """
    pairs = sorted(zip(row["station"], row[value_col]))
    return " | ".join(f"{station}: {fmt.format(value)}" for station, value in pairs)


def _find_coincidences(top_days: pd.DataFrame) -> pd.DataFrame:
    """
    Dates present in the top-N list of 2+ stations - see module docstring
    for what this does and does not mean scientifically.

    frequency_p99_pct_by_station / rank_by_station are a per-station
    summary formatted from columns top_days already carries (rank,
    frequency_p99_pct) - not a new computation, not a new file read. The
    full-precision, unformatted values remain available in top_days /
    top_days.csv; these two columns exist only so the coincidences table
    (web and CSV) does not require cross-referencing top_days.csv by hand
    to see how "elevated" each involved station actually was that day.
    """
    if top_days.empty:
        return pd.DataFrame(columns=_COINCIDENCE_COLUMNS)

    grouped = (
        top_days
        .groupby("date")[["station", "rank", "frequency_p99_pct"]]
        .agg(list)
        .reset_index()
    )
    grouped["n_stations"] = grouped["station"].apply(len)
    grouped = grouped[grouped["n_stations"] >= 2].copy()

    grouped["stations"] = grouped["station"].apply(lambda s: ", ".join(sorted(s)))
    grouped["frequency_p99_pct_by_station"] = grouped.apply(
        lambda r: _format_by_station(r, "frequency_p99_pct", "{:.2f}"), axis=1
    )
    grouped["rank_by_station"] = grouped.apply(
        lambda r: _format_by_station(r, "rank", "{}"), axis=1
    )

    grouped = grouped.drop(columns=["station", "rank", "frequency_p99_pct"]).sort_values(
        ["n_stations", "date"], ascending=[False, True]
    ).reset_index(drop=True)
    return grouped[_COINCIDENCE_COLUMNS]


def run_temporal_comparison(
    stations: list[str] = STATIONS,
    year: int = YEAR,
    doy_start: int = DOY_START,
    doy_end: int = DOY_END,
    value_col: str = VALUE_COL,
    top_n: int = TOP_N_DAYS,
    ndat_mode: str | None = None,
) -> dict:
    """
    Reads each station's already-saved Step 3 CSVs and builds the
    comparison table, top-N days, and coincidences described in the
    module docstring.

    Pure function: no prints, no file writes. Raises ValueError if
    value_col is not scientifically supported. Raises FileNotFoundError
    only if none of the requested stations have Step 3 output yet - a
    partial set is a normal, expected result, not an error.

    ndat_mode=None (default) is the legacy behaviour: reads each
    station's flat, un-tagged CSVs, unchanged from before this script knew
    about Ndat - so a caller that does not pass this argument (today,
    web_server.py) is unaffected. With a real mode, each station's CSVs
    are looked up under that exact Ndat directory/tag
    (resolve_station_thresholds_csv/resolve_station_daily_csv) - there is
    no fallback to another mode or to the legacy path if they are
    missing: a station missing its Ndat-specific output is reported in
    missing_stations with the Ndat-tagged path that was actually checked,
    exactly like any other missing station. All stations in one call
    share the same single ndat_mode value (like year/value_col already
    do), so mixing Ndat modes across stations within one comparison is
    structurally impossible.
    """
    index_config = validate_index_supported(value_col)
    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode) if ndat_mode is not None else None

    table_rows = []
    top_days_frames = []
    daily_by_station: dict[str, pd.DataFrame] = {}
    missing_stations: dict[str, str] = {}

    for station in stations:
        thresholds_csv = resolve_station_thresholds_csv(
            station, year, doy_start, doy_end, value_col, index_config, ndat_mode
        )
        daily_csv = resolve_station_daily_csv(
            station, year, doy_start, doy_end, value_col, index_config, ndat_mode
        )

        if not thresholds_csv.exists() or not daily_csv.exists():
            missing_path = thresholds_csv if not thresholds_csv.exists() else daily_csv
            missing_stations[station] = f"Step 3 output not found: {missing_path}"
            continue

        df_thresholds = pd.read_csv(thresholds_csv)
        df_daily = pd.read_csv(daily_csv)
        daily_by_station[station] = df_daily

        max_p90_row = df_daily.loc[df_daily["frequency_p90_pct"].idxmax()]
        max_p99_row = df_daily.loc[df_daily["frequency_p99_pct"].idxmax()]
        thresholds_row = df_thresholds.iloc[0]

        table_rows.append({
            "station": station,
            "n_values": int(thresholds_row["n_values"]),
            "p90_annual": float(thresholds_row["p90_annual"]),
            "p99_annual": float(thresholds_row["p99_annual"]),
            "frequency_p90_pct": float(thresholds_row["frequency_p90_pct"]),
            "frequency_p99_pct": float(thresholds_row["frequency_p99_pct"]),
            "max_p90_freq_doy": int(max_p90_row["doy"]),
            "max_p90_freq_date": max_p90_row["date"],
            "max_p90_freq_pct": float(max_p90_row["frequency_p90_pct"]),
            "max_p90_freq_n_values": int(max_p90_row["n_values"]),
            "max_p90_freq_coverage": float(max_p90_row["coverage"]),
            "max_p99_freq_doy": int(max_p99_row["doy"]),
            "max_p99_freq_date": max_p99_row["date"],
            "max_p99_freq_pct": float(max_p99_row["frequency_p99_pct"]),
            "max_p99_freq_n_values": int(max_p99_row["n_values"]),
            "max_p99_freq_coverage": float(max_p99_row["coverage"]),
        })

        top = df_daily.sort_values("frequency_p99_pct", ascending=False, kind="stable").head(top_n).copy()
        top.insert(0, "station", station)
        top.insert(1, "rank", range(1, len(top) + 1))
        top_days_frames.append(top[[
            "station", "rank", "doy", "date", "frequency_p99_pct",
            "frequency_p90_pct", "n_values", "coverage",
        ]])

    if not table_rows:
        ndat_note = f", ndat_mode='{ndat_mode}'" if ndat_mode is not None else ""
        raise FileNotFoundError(
            f"No station has Step 3 output for value_col='{value_col}'{ndat_note}, "
            f"year={year}, DOY{doy_start}_{doy_end}. Run 3-temporal_variability.py "
            "for at least one station first."
        )

    table = pd.DataFrame(table_rows)
    top_days = pd.concat(top_days_frames, ignore_index=True)
    coincidences = _find_coincidences(top_days)

    return {
        "index_config": index_config,
        "ndat_config": ndat_config,
        "available_stations": sorted(table["station"]),
        "missing_stations": missing_stations,
        "table": table,
        "top_days": top_days,
        "coincidences": coincidences,
        "daily_by_station": daily_by_station,
        "top_n": top_n,
    }


# ======================================================================
# [E] OUTPUT - CSV TABLES
# ======================================================================
def save_comparison_outputs(
    table: pd.DataFrame,
    top_days: pd.DataFrame,
    coincidences: pd.DataFrame,
    paths: dict,
) -> None:
    paths["comparison_dir"].mkdir(parents=True, exist_ok=True)
    table.to_csv(paths["output_table_csv"], index=False)
    top_days.to_csv(paths["output_top_days_csv"], index=False)
    coincidences.to_csv(paths["output_coincidences_csv"], index=False)


# ======================================================================
# [F] OUTPUT - COMBINED PLOT (PNG)
# ======================================================================
def save_daily_comparison_plot(
    stations: list[str],
    daily_by_station: dict,
    index_config: dict,
    output_png: Path,
    doy_start: int = DOY_START,
    doy_end: int = DOY_END,
    plot_dpi: int = PLOT_DPI,
    show_plot: bool = SHOW_PLOTS,
) -> None:
    """
    Overlays every station's daily p99 exceedance frequency on one shared
    axis - color fixed per station (STATION_COLORS), so a day that stands
    out for several stations at once is visible directly, not only
    inferable from the coincidences table.
    """
    output_png.parent.mkdir(parents=True, exist_ok=True)
    label = index_config["label"]

    fig, ax = plt.subplots(figsize=(13, 6))
    full_doy = pd.Index(range(doy_start, doy_end + 1), name="doy")

    for station in stations:
        df_plot = daily_by_station[station].set_index("doy").reindex(full_doy)
        ax.plot(
            df_plot.index, df_plot["frequency_p99_pct"],
            linewidth=1.0, color=STATION_COLORS.get(station), label=station,
        )

    ax.set_xlim(doy_start, doy_end)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Day of year (DoY)")
    ax.set_ylabel("Frequency at/above own annual p99 [% of observations]")
    ax.set_title(f"Daily p99 exceedance frequency - station comparison — {label}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=plot_dpi)

    if show_plot:
        plt.show()
    plt.close(fig)


# ======================================================================
# [G] MAIN - CONSOLE MODE
# ======================================================================
def main():
    result = run_temporal_comparison(
        stations=STATIONS, year=YEAR, doy_start=DOY_START, doy_end=DOY_END,
        value_col=VALUE_COL, top_n=TOP_N_DAYS, ndat_mode=NDAT_MODE,
    )

    index_config = result["index_config"]
    available = result["available_stations"]
    missing = result["missing_stations"]
    table = result["table"]
    top_days = result["top_days"]
    coincidences = result["coincidences"]

    title = "Step 3.2 - Cross-station comparison of Step 3 daily variability"
    print(title)
    print("=" * len(title))
    print(f"Index: {index_config['label']} ({VALUE_COL})")
    print(f"Year: {YEAR}")
    print(f"Top N days per station: {TOP_N_DAYS}")
    print()
    print(f"Stations included ({len(available)}): {', '.join(available)}")
    if missing:
        print(f"Stations missing ({len(missing)}):")
        for station, reason in missing.items():
            print(f"  - {station}: {reason}")
    else:
        print("Stations missing: none")

    print()
    print("Annual summary by station")
    print("-" * 26)
    print(table.to_string(index=False))

    print()
    print(f"Cross-station coincidences (date in top-{TOP_N_DAYS} of 2+ stations)")
    print("-" * 60)
    if coincidences.empty:
        print("None found.")
    else:
        print(coincidences.to_string(index=False))
        print(
            "\nReminder: a coincidence means both stations were elevated "
            "relative to their OWN annual threshold, not that they had the "
            "same absolute ROTI L1 magnitude (see module docstring)."
        )

    paths = resolve_comparison_paths(YEAR, index_config, value_col=VALUE_COL, ndat_mode=NDAT_MODE)
    save_comparison_outputs(table, top_days, coincidences, paths)
    save_daily_comparison_plot(
        available, result["daily_by_station"], index_config,
        paths["output_daily_comparison_png"],
        doy_start=DOY_START, doy_end=DOY_END,
        plot_dpi=PLOT_DPI, show_plot=SHOW_PLOTS,
    )

    print()
    print("Output files")
    print("-" * 12)
    print(f"Annual summary CSV: {paths['output_table_csv']}")
    print(f"Top days CSV: {paths['output_top_days_csv']}")
    print(f"Coincidences CSV: {paths['output_coincidences_csv']}")
    print(f"Daily comparison plot: {paths['output_daily_comparison_png']}")
    print()
    print("Step 3.2 completed successfully.")


if __name__ == "__main__":
    main()
