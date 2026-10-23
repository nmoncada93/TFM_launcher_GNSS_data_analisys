#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 5.2 - Cross-station comparison of Step 5 daypart exceedance frequencies.

Reads the daypart_frequency.csv saved by 5-daypart_variability.py for each
station and merges them into one comparison table (CSV) and one combined
plot (PNG). Nothing is recomputed and the Step 0 Parquet is not read.

One comparison uses a single Ndat criterion (ndat_mode), DoY range and
coverage threshold (th_cov) for every station: only the Step 5 outputs with
that criterion and run tag (completeness_paths.run_tag) are read, and the
comparison is written to its own <dir_tag> folder with the same run tag. A
station without that exact Step 5 output is skipped and reported, never
replaced by the output of another criterion, DoY range or threshold.

Daypart-layout check: before merging, every compared station must have the
same daypart_order/daypart/hour_range triplets. A month number always
means the same month, but a daypart name depends on the DAYPARTS
configuration of 5-daypart_variability.py, so merging by name alone could
compare different time windows under the same label. A mismatch is a hard
error (ValueError): a "Night" that means 18-06 for one station and 00-06
for another cannot be compared.

th_cov check: as a further check, each compared station must also have the
Step 1 completeness of the same criterion and th_cov (completeness_paths);
a missing one is reported in th_cov_warnings.

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
# Step 5 outputs of this criterion are read for every station. "all" = ALL_NDAT.
# Only used by main(): the web interface passes the selected mode explicitly.
NDAT_MODE = "all"

# Fixed colour per station (by name, not by list position): with the
# default colour cycle, colours would shift whenever a station is missing.
# Stations without an entry get a fallback colour (plot_style).
STATION_COLORS = {"UNSA": "tab:orange", "KOUG": "tab:blue", "WHIT": "tab:green", "YELL": "tab:red"}

SHOW_PLOTS = False
PLOT_DPI = 250


# ======================================================================
# [C] PATH RESOLUTION
# ======================================================================
def resolve_station_daypart_csv(
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
    Path to the daypart_frequency.csv saved by 5-daypart_variability.py
    for one station, Ndat criterion, DoY range and th_cov. Mirrors the
    naming of its resolve_paths() (dir_tag subfolder, file_tag prefix tag
    and run tag). Raises ValueError if ndat_mode is unknown.
    """
    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
    index_dir = project_paths.RESULTS_ROOT / station / str(year) / "5_daypart_variability" / value_col
    ndat_dir = index_dir / ndat_config["dir_tag"]
    prefix = f"5-{station}_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"

    run_tag = completeness_paths.run_tag(doy_start, doy_end, th_cov)
    return ndat_dir / f"{prefix}_{run_tag}_daypart_frequency.csv"


def resolve_station_coverage_csv(
    station: str, year: int, doy_start: int, doy_end: int, th_cov: float, ndat_mode: str = "all",
    value_col: str = VALUE_COL,
) -> Path:
    """
    Path to the Step 1 completeness CSV of one station for this index, Ndat
    criterion and th_cov (completeness_paths). Used only for the th_cov
    cross-check described in the module docstring.
    """
    return completeness_paths.resolve_completeness_paths(
        station, year, doy_start, doy_end, th_cov, ndat_mode, index=value_col
    )["output_csv"]


def resolve_comparison_paths(
    year: int,
    doy_start: int,
    doy_end: int,
    th_cov: float,
    value_col: str,
    index_config: dict,
    ndat_mode: str = "all",
) -> dict:
    """
    Output paths of the comparison, under
    results/global/{YEAR}/5_daypart_comparison/{VALUE_COL}/<dir_tag>/ (never
    inside a station folder), with the run tag (DoY range, th_cov) in the
    file names. Raises ValueError if ndat_mode is unknown.
    """
    base_dir = project_paths.RESULTS_ROOT / "global" / str(year) / "5_daypart_comparison"

    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
    comparison_dir = base_dir / value_col / ndat_config["dir_tag"]
    prefix = (
        f"5-ALL_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"
        f"_{completeness_paths.run_tag(doy_start, doy_end, th_cov)}"
    )

    return {
        "comparison_dir": comparison_dir,
        "output_csv": comparison_dir / f"{prefix}_daypart_frequency_by_station.csv",
        "output_png": comparison_dir / f"{prefix}_daypart_comparison.png",
    }


# ======================================================================
# [D] CROSS-STATION COMPARISON (pure - no prints, no file I/O)
# ======================================================================
def run_daypart_comparison(
    stations: list[str] = STATIONS,
    year: int = YEAR,
    doy_start: int = DOY_START,
    doy_end: int = DOY_END,
    value_col: str = VALUE_COL,
    th_cov: float = TH_COV,
    ndat_mode: str = "all",
) -> dict:
    """
    Reads the daypart_frequency.csv already saved by
    5-daypart_variability.py for each station and merges n_values/
    frequency_p90/frequency_p99 into one wide table: one row per
    daypart, one column group per available station.

    Pure function: no prints, no file writes. Raises ValueError if
    value_col is not scientifically supported, if ndat_mode is unknown,
    or if the compared stations do not share the same daypart definition
    (a hard error, see module docstring). Raises FileNotFoundError only if
    no requested station has Step 5 output for ndat_mode.

    Every station is looked up for exactly the given ndat_mode, DoY range
    and th_cov. There is no fallback to another criterion or threshold: a
    station without that output is reported in missing_stations with the
    criterion, th_cov and the path that was checked. All stations share one
    ndat_mode and th_cov, so they are never mixed within a comparison.
    """
    index_config = indices.validate_index_supported(value_col)
    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
    criterion = f"Ndat criterion '{ndat_mode}' ({ndat_config['label']}) and th_cov={th_cov}"

    available_frames: dict[str, pd.DataFrame] = {}
    missing_stations: dict[str, str] = {}

    for station in stations:
        csv_path = resolve_station_daypart_csv(
            station, year, doy_start, doy_end, th_cov, value_col, index_config, ndat_mode
        )
        if not csv_path.exists():
            missing_stations[station] = f"Step 5 output for {criterion} not found: {csv_path}"
            continue
        available_frames[station] = pd.read_csv(csv_path, usecols=[
            "daypart_order", "daypart", "hour_range",
            "n_values", "frequency_p90_pct", "frequency_p99_pct",
        ])

    if not available_frames:
        raise FileNotFoundError(
            f"No station has Step 5 output for value_col='{value_col}', {criterion}, "
            f"year={year}, DOY{doy_start}_{doy_end}. Run "
            "5-daypart_variability.py with the same Ndat criterion and coverage "
            "threshold for at least one station first."
        )

    # Daypart-definition consistency check - hard error, see module docstring.
    reference_station = next(iter(available_frames))
    reference_layout = (
        available_frames[reference_station][["daypart_order", "daypart", "hour_range"]]
        .sort_values("daypart_order")
        .reset_index(drop=True)
    )
    for station, df_station in available_frames.items():
        layout = (
            df_station[["daypart_order", "daypart", "hour_range"]]
            .sort_values("daypart_order")
            .reset_index(drop=True)
        )
        if not layout.equals(reference_layout):
            raise ValueError(
                f"Daypart definition mismatch: station '{station}' does not use "
                f"the same daypart boundaries as '{reference_station}'. Comparing "
                "them would silently mix incompatible time windows under the same "
                f"label.\n{station}:\n{layout}\n{reference_station}:\n{reference_layout}"
            )

    # Further th_cov check - see module docstring.
    th_cov_warnings: dict[str, str] = {}
    for station in available_frames:
        coverage_csv = resolve_station_coverage_csv(
            station, year, doy_start, doy_end, th_cov, ndat_mode, value_col
        )
        if not coverage_csv.exists():
            th_cov_warnings[station] = (
                f"No Step 1 completeness for {criterion} found "
                f"({coverage_csv}). This station's Step 5 output may have "
                "been generated with a different coverage threshold than "
                "the other stations being compared."
            )

    table = reference_layout.copy()
    for station, df_station in available_frames.items():
        renamed = df_station[[
            "daypart_order", "n_values", "frequency_p90_pct", "frequency_p99_pct",
        ]].rename(columns={
            "n_values": f"n_values_{station}",
            "frequency_p90_pct": f"freq_p90_{station}",
            "frequency_p99_pct": f"freq_p99_{station}",
        })
        table = table.merge(renamed, on="daypart_order", how="left")

    return {
        "index_config": index_config,
        "ndat_config": ndat_config,
        "available_stations": sorted(available_frames),
        "missing_stations": missing_stations,
        "th_cov_warnings": th_cov_warnings,
        "table": table,
    }


# ======================================================================
# [E] OUTPUT - COMPARISON TABLE (CSV)
# ======================================================================
def save_comparison_table(table: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)


# ======================================================================
# [F] OUTPUT - COMBINED PLOT (PNG)
# ======================================================================
@plot_style.with_font_sizes
def save_comparison_plot(
    table: pd.DataFrame,
    available_stations: list[str],
    index_config: dict,
    output_png: Path,
    plot_dpi: int = PLOT_DPI,
    show_plot: bool = SHOW_PLOTS,
) -> None:
    """Grouped bar chart: one group per daypart, one bar per station."""
    output_png.parent.mkdir(parents=True, exist_ok=True)

    label = index_config["label"]
    dayparts = table["daypart"].tolist()
    x = np.arange(len(dayparts))
    n_stations = len(available_stations)
    width = 0.8 / max(n_stations, 1)

    fig, (ax_p90, ax_p99) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for i, station in enumerate(available_stations):
        offset = (i - (n_stations - 1) / 2) * width
        color = plot_style.station_color(station, STATION_COLORS, available_stations)
        ax_p90.bar(x + offset, table[f"freq_p90_{station}"], width, color=color, label=station)
        ax_p99.bar(x + offset, table[f"freq_p99_{station}"], width, color=color, label=station)

    ax_p90.set_title(f"Daypart {label} threshold exceedance - station comparison")
    ax_p90.set_ylabel("p90 exceedance frequency [%]")
    ax_p90.set_xticks(x)
    ax_p90.set_xticklabels(dayparts)
    ax_p90.legend()
    ax_p90.grid(True, axis="y", alpha=0.3)

    ax_p99.set_ylabel("p99 exceedance frequency [%]")
    ax_p99.set_xlabel("Daypart")
    ax_p99.set_xticks(x)
    ax_p99.set_xticklabels(dayparts)
    ax_p99.legend()
    ax_p99.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_png, dpi=plot_dpi)

    if show_plot:
        plt.show()
    plt.close(fig)


# ======================================================================
# [G] MAIN - CONSOLE MODE
# ======================================================================
def main():
    result = run_daypart_comparison(
        stations=STATIONS,
        year=YEAR,
        doy_start=DOY_START,
        doy_end=DOY_END,
        value_col=VALUE_COL,
        th_cov=TH_COV,
        ndat_mode=NDAT_MODE,
    )

    index_config = result["index_config"]
    available = result["available_stations"]
    missing = result["missing_stations"]
    th_cov_warnings = result["th_cov_warnings"]
    table = result["table"]

    title = "Step 5.2 - Cross-station comparison of Step 5 daypart frequencies"
    print(title)
    print("=" * len(title))
    print(f"Index: {index_config['label']} ({VALUE_COL})")
    print(f"Year: {YEAR}")
    print(f"DOY range: {DOY_START}-{DOY_END}")
    print(f"th_cov (expected): {TH_COV}")
    print()
    print(f"Stations included ({len(available)}): {', '.join(available)}")
    if missing:
        print(f"Stations missing ({len(missing)}):")
        for station, reason in missing.items():
            print(f"  - {station}: {reason}")
    else:
        print("Stations missing: none")

    if th_cov_warnings:
        print(f"th_cov warnings ({len(th_cov_warnings)}):")
        for station, reason in th_cov_warnings.items():
            print(f"  - {station}: {reason}")
    else:
        print("th_cov warnings: none - all compared stations have a matching "
              "Step 1 coverage CSV")

    print()
    print("Daypart definition (shared by all compared stations)")
    print("-" * 54)
    print(table[["daypart_order", "daypart", "hour_range"]].to_string(index=False))

    print()
    print("Daypart comparison table")
    print("-" * 24)
    print(table.to_string(index=False))

    paths = resolve_comparison_paths(YEAR, DOY_START, DOY_END, TH_COV, VALUE_COL, index_config, NDAT_MODE)
    save_comparison_table(table, paths["output_csv"])
    save_comparison_plot(
        table, available, index_config, paths["output_png"], PLOT_DPI, SHOW_PLOTS,
    )

    print()
    print("Output files")
    print("-" * 12)
    print(f"Comparison table CSV: {paths['output_csv']}")
    print(f"Comparison plot: {paths['output_png']}")
    print()
    print("Step 5.2 completed successfully.")


if __name__ == "__main__":
    main()
