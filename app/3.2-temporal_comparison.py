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

Nothing is recomputed, and neither the Step 0 Parquet nor the Step 1 CSV
is read - only what 3-temporal_variability.py has saved per station.

One comparison uses a single Ndat criterion (ndat_mode), DoY range and
coverage threshold (th_cov) for every station: only the Step 3 outputs with
that criterion and run tag (completeness_paths.run_tag) are read, and the
comparison is written to its own <dir_tag> folder with the same run tag
(plus top_n in the names of the outputs that depend on it). As in
2.2-ccdf_comparison.py, a station without that exact Step 3 output is
skipped and reported, never replaced by the output of another criterion,
DoY range or threshold.

IMPORTANT scientific caveat: each station's p90/p99 threshold is its OWN
annual threshold. A day in the top-N lists of two stations means both
show elevated activity RELATIVE TO THEIR OWN yearly distribution - NOT
that both had the same absolute magnitude of the index that day. Absolute
magnitudes are compared in Step 2 (monthly percentiles/CCDF).

The output is descriptive: candidate days are detected and reported, not
interpreted physically.

The analysed index (ROTI L1, S4 L1, ...) and its Ndat column, labels and
unit come from index_config.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import completeness_paths
import index_config as indices
import ndat_filter
import plot_style
import project_paths


# ======================================================================
# [A] STUDY CONFIGURATION - EDIT ONLY THIS SECTION
# ======================================================================

# CONSOLE DEFAULT - used only when this script is run directly. In web mode,
# web_server.py passes the station list from app/stations.json instead
# (this script never reads that file). Not accidental duplication: keep it.
STATIONS = ["UNSA", "KOUG", "WHIT", "YELL"]
YEAR = 2024
DOY_START = 1
DOY_END = 366
TH_COV = 0.75

# Index to compare: any key of index_config.INDEX_CONFIG (e.g. "roti_l1", "s4_l1").
VALUE_COL = indices.DEFAULT_INDEX

# Ndat criterion of the comparison (a canonical ndat_filter mode): the
# Step 3 outputs of this criterion are read for every station. "all" = ALL_NDAT.
NDAT_MODE = "all"

# How many top days (by frequency_p99_pct) to keep per station, and to
# check for cross-station coincidences.
TOP_N_DAYS = 10

# Fixed colour per station (by name, not by list position), same as in
# 2-ccdf.py; stations without an entry get a fallback colour (plot_style).
STATION_COLORS = {"UNSA": "tab:orange", "KOUG": "tab:blue", "WHIT": "tab:green", "YELL": "tab:red"}

SHOW_PLOTS = False
PLOT_DPI = 250


# ======================================================================
# [C] PATH RESOLUTION
# ======================================================================
def resolve_station_thresholds_csv(
    station: str,
    year: int,
    doy_start: int,
    doy_end: int,
    th_cov: float,
    value_col: str,
    index_config: dict,
    ndat_mode: str = "all",
) -> Path:
    """
    Path to the annual_thresholds.csv saved by 3-temporal_variability.py
    for one station, Ndat criterion, DoY range and th_cov. Mirrors the
    naming of its resolve_paths() (dir_tag subfolder, file_tag prefix tag
    and run tag). Raises ValueError if ndat_mode is unknown.
    """
    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
    index_dir = project_paths.RESULTS_ROOT / station / str(year) / "3_temporal_variability" / value_col
    ndat_dir = index_dir / ndat_config["dir_tag"]
    prefix = f"3-{station}_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"

    run_tag = completeness_paths.run_tag(doy_start, doy_end, th_cov)
    return ndat_dir / f"{prefix}_{run_tag}_annual_thresholds.csv"


def resolve_station_daily_csv(
    station: str,
    year: int,
    doy_start: int,
    doy_end: int,
    th_cov: float,
    value_col: str,
    index_config: dict,
    ndat_mode: str = "all",
) -> Path:
    """
    Path to the daily_frequency.csv saved by 3-temporal_variability.py;
    same logic as resolve_station_thresholds_csv().
    """
    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
    index_dir = project_paths.RESULTS_ROOT / station / str(year) / "3_temporal_variability" / value_col
    ndat_dir = index_dir / ndat_config["dir_tag"]
    prefix = f"3-{station}_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"

    run_tag = completeness_paths.run_tag(doy_start, doy_end, th_cov)
    return ndat_dir / f"{prefix}_{run_tag}_daily_frequency.csv"


def resolve_comparison_paths(
    year: int,
    doy_start: int,
    doy_end: int,
    th_cov: float,
    value_col: str,
    index_config: dict,
    ndat_mode: str = "all",
    top_n: int = TOP_N_DAYS,
) -> dict:
    """
    Output paths under
    results/global/{YEAR}/3_temporal_comparison/{VALUE_COL}/<dir_tag>/ -
    never inside a station folder - with the run tag (DoY range, th_cov)
    in every file name and top_n in the names of the outputs that depend
    on it (top days, coincidences). Raises ValueError if ndat_mode is
    unknown.
    """
    base_dir = project_paths.RESULTS_ROOT / "global" / str(year) / "3_temporal_comparison"

    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
    comparison_dir = base_dir / value_col / ndat_config["dir_tag"]
    prefix = (
        f"3-ALL_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"
        f"_{completeness_paths.run_tag(doy_start, doy_end, th_cov)}"
    )

    return {
        "comparison_dir": comparison_dir,
        "output_table_csv": comparison_dir / f"{prefix}_annual_summary_by_station.csv",
        "output_top_days_csv": comparison_dir / f"{prefix}_top{top_n}_days.csv",
        "output_coincidences_csv": comparison_dir / f"{prefix}_top{top_n}_coincidences.csv",
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

    frequency_p99_pct_by_station / rank_by_station are formatted from the
    rank and frequency_p99_pct columns of top_days (full precision stays
    in top_days.csv), so the coincidences table shows how elevated each
    station was without cross-referencing top_days.csv.
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
    if grouped.empty:
        return pd.DataFrame(columns=_COINCIDENCE_COLUMNS)

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
    th_cov: float = TH_COV,
    value_col: str = VALUE_COL,
    top_n: int = TOP_N_DAYS,
    ndat_mode: str = "all",
) -> dict:
    """
    Reads each station's already-saved Step 3 CSVs and builds the
    comparison table, top-N days, and coincidences described in the
    module docstring.

    Pure function: no prints, no file writes. Raises ValueError if
    value_col is not scientifically supported or ndat_mode is unknown,
    and FileNotFoundError only if no requested station has Step 3 output
    for ndat_mode and th_cov (a partial set is a normal result).

    Every station is looked up for exactly the given ndat_mode, DoY range
    and th_cov. There is no fallback to another criterion or threshold: a
    station without that output is reported in missing_stations with the
    criterion, th_cov and the path that was checked. All stations share one
    ndat_mode and th_cov, so they are never mixed within a comparison.
    """
    index_config = indices.validate_index_supported(value_col)
    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
    criterion = f"Ndat criterion '{ndat_mode}' ({ndat_config['label']}) and th_cov={th_cov}"

    table_rows = []
    top_days_frames = []
    daily_by_station: dict[str, pd.DataFrame] = {}
    missing_stations: dict[str, str] = {}

    for station in stations:
        thresholds_csv = resolve_station_thresholds_csv(
            station, year, doy_start, doy_end, th_cov, value_col, index_config, ndat_mode
        )
        daily_csv = resolve_station_daily_csv(
            station, year, doy_start, doy_end, th_cov, value_col, index_config, ndat_mode
        )

        if not thresholds_csv.exists() or not daily_csv.exists():
            missing_path = thresholds_csv if not thresholds_csv.exists() else daily_csv
            missing_stations[station] = f"Step 3 output for {criterion} not found: {missing_path}"
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
        raise FileNotFoundError(
            f"No station has Step 3 output for value_col='{value_col}', {criterion}, "
            f"year={year}, DOY{doy_start}_{doy_end}. Run 3-temporal_variability.py "
            "with the same Ndat criterion and coverage threshold for at least one "
            "station first."
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
@plot_style.with_font_sizes(sizes=plot_style.font_sizes_plus(4))
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
            linewidth=1.0, color=plot_style.station_color(station, STATION_COLORS, stations), label=station,
        )

    ax.set_xlim(doy_start, doy_end)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Day of year (DoY)")
    # Two lines: at this font size the one-line label is taller than the axes.
    ax.set_ylabel("Frequency at/above own annual p99\n[% of observations]")
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
        stations=STATIONS, year=YEAR, doy_start=DOY_START, doy_end=DOY_END, th_cov=TH_COV,
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
            f"same absolute {index_config['label']} magnitude (see module docstring)."
        )

    paths = resolve_comparison_paths(
        YEAR, DOY_START, DOY_END, TH_COV, VALUE_COL, index_config, NDAT_MODE, top_n=TOP_N_DAYS
    )
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
