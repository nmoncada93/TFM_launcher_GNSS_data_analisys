#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2.2 - Cross-station comparison of Step 2 monthly percentiles.

Reads the monthly_percentiles.csv saved by 2-ccdf.py for each station and
merges them into one comparison table (CSV) and one combined plot (PNG).
Nothing is recomputed and the Step 0 Parquet is not read.

One comparison uses a single Ndat criterion (ndat_mode), DoY range and
coverage threshold (th_cov) for every station: only the Step 2 outputs with
that criterion and run tag (completeness_paths.run_tag) are read, and the
comparison is written to its own <dir_tag> folder with the same run tag. A
station without that exact Step 2 output is skipped and reported, never
replaced by the output of another criterion, DoY range or threshold.

n_days/n_values are carried over per station-month, so low-support months
(few valid days behind a percentile) are visible in the table and flagged
in the plot.

th_cov check: as a further check, each compared station must also have the
Step 1 completeness of the same criterion and th_cov (completeness_paths);
a missing one is reported in th_cov_warnings.

The analysed index (ROTI L1, S4 L1, ...) and its Ndat column, labels and
unit come from index_config.py.
"""

import calendar
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
# Step 2 outputs of this criterion are read for every station. "all" = ALL_NDAT.
NDAT_MODE = "all"

# Same threshold 2-ccdf.py already uses for its own "few valid days"
# warning (MIN_VALID_DAYS_PER_MONTH) - reused here, not re-invented, so
# a month flagged as low-support means the same thing in both scripts.
MIN_VALID_DAYS_PER_MONTH = 10

# Color fixed per station (same colors as 2-ccdf.py / 3.2 / 4.2 / 5.2);
# stations without an entry get a distinct fallback color (plot_style).
STATION_COLORS = {"UNSA": "tab:orange", "KOUG": "tab:blue", "WHIT": "tab:green", "YELL": "tab:red"}

SHOW_PLOTS = False
PLOT_DPI = 250


# ======================================================================
# [C] PATH RESOLUTION
# ======================================================================
def resolve_station_pcts_csv(
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
    Path to the monthly_percentiles.csv saved by 2-ccdf.py for one
    station, Ndat criterion, DoY range and th_cov. Mirrors the naming of
    2-ccdf.py's resolve_paths() (dir_tag subfolder, file_tag prefix tag and
    run tag). Raises ValueError if ndat_mode is unknown.
    """
    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
    index_dir = project_paths.RESULTS_ROOT / station / str(year) / "2_ccdf" / value_col
    ndat_dir = index_dir / ndat_config["dir_tag"]
    prefix = f"2-{station}_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"

    run_tag = completeness_paths.run_tag(doy_start, doy_end, th_cov)
    return ndat_dir / f"{prefix}_{run_tag}_monthly_percentiles.csv"


def resolve_station_coverage_csv(
    station: str, year: int, doy_start: int, doy_end: int, th_cov: float, ndat_mode: str = "all",
    value_col: str = VALUE_COL,
) -> Path:
    """
    Path to the Step 1 completeness CSV of one station for this index, Ndat
    criterion and th_cov (completeness_paths). Used only for the th_cov
    cross-check described in the module docstring - not read for its
    contents anywhere in this script.
    """
    return completeness_paths.resolve_completeness_paths(
        station, year, doy_start, doy_end, th_cov, ndat_mode, index=value_col
    )["output_csv"]


def resolve_comparison_paths(
    year: int, doy_start: int, doy_end: int, th_cov: float,
    value_col: str, index_config: dict, ndat_mode: str = "all",
) -> dict:
    """
    Output paths of the comparison, under
    results/global/{YEAR}/2_ccdf_comparison/{VALUE_COL}/<dir_tag>/ (never
    inside a station folder), with the run tag (DoY range, th_cov) in the
    file names. Raises ValueError if ndat_mode is unknown.
    """
    base_dir = project_paths.RESULTS_ROOT / "global" / str(year) / "2_ccdf_comparison"

    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
    comparison_dir = base_dir / value_col / ndat_config["dir_tag"]
    prefix = (
        f"2-ALL_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"
        f"_{completeness_paths.run_tag(doy_start, doy_end, th_cov)}"
    )

    return {
        "comparison_dir": comparison_dir,
        "output_csv": comparison_dir / f"{prefix}_monthly_percentiles_by_station.csv",
        "output_png": comparison_dir / f"{prefix}_monthly_percentile_comparison.png",
    }


# ======================================================================
# [D] CROSS-STATION COMPARISON (pure - no prints, no file I/O)
# ======================================================================
def run_ccdf_comparison(
    stations: list[str] = STATIONS,
    year: int = YEAR,
    doy_start: int = DOY_START,
    doy_end: int = DOY_END,
    value_col: str = VALUE_COL,
    th_cov: float = TH_COV,
    ndat_mode: str = "all",
) -> dict:
    """
    Reads the monthly_percentiles.csv already saved by 2-ccdf.py for each
    station and merges n_days/n_values/p90/p99 into one wide table: one
    row per month, one column group per available station.

    Pure function: no prints, no file writes. Raises ValueError if
    value_col is not scientifically supported or ndat_mode is unknown,
    and FileNotFoundError only if no requested station has Step 2 output
    for ndat_mode (a partial set is a normal result).

    Every station is looked up for exactly the given ndat_mode, DoY range
    and th_cov. There is no fallback to another criterion or threshold: a
    station without that output is reported in missing_stations with the
    criterion, th_cov and the path that was checked.
    """
    index_config = indices.validate_index_supported(value_col)
    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
    criterion = f"Ndat criterion '{ndat_mode}' ({ndat_config['label']}) and th_cov={th_cov}"

    available_frames: dict[str, pd.DataFrame] = {}
    missing_stations: dict[str, str] = {}

    for station in stations:
        csv_path = resolve_station_pcts_csv(
            station, year, doy_start, doy_end, th_cov, value_col, index_config, ndat_mode
        )
        if not csv_path.exists():
            missing_stations[station] = f"Step 2 output for {criterion} not found: {csv_path}"
            continue
        available_frames[station] = pd.read_csv(
            csv_path, usecols=["month", "n_days", "n_values", "p90", "p99"]
        )

    if not available_frames:
        raise FileNotFoundError(
            f"No station has Step 2 output for value_col='{value_col}', {criterion}, "
            f"year={year}, DOY{doy_start}_{doy_end}. Run 2-ccdf.py with the same "
            "Ndat criterion and coverage threshold for at least one station first."
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
                f"({coverage_csv}). This station's Step 2 output may have "
                "been generated with a different coverage threshold than "
                "the other stations being compared."
            )

    table = pd.DataFrame({"month": range(1, 13)})
    for station, df_station in available_frames.items():
        renamed = df_station.rename(columns={
            "n_days": f"n_days_{station}",
            "n_values": f"n_values_{station}",
            "p90": f"p90_{station}",
            "p99": f"p99_{station}",
        })
        table = table.merge(renamed, on="month", how="left")

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
    min_valid_days_per_month: int = MIN_VALID_DAYS_PER_MONTH,
    plot_dpi: int = PLOT_DPI,
    show_plot: bool = SHOW_PLOTS,
) -> None:
    """
    Low-support months (n_days below min_valid_days_per_month for that
    station) get an extra red marker plus an "n=N" label on top of the
    normal line point, on both panels - a low n_days should be visible
    at a glance, not something you only find by opening the CSV.
    """
    output_png.parent.mkdir(parents=True, exist_ok=True)

    month_labels = [calendar.month_abbr[m] for m in table["month"]]
    label = index_config["label"]
    unit = index_config["unit"]
    y_suffix = f" ({unit})" if unit else ""

    fig, (ax_p90, ax_p99) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for station in available_stations:
        low_support = table[f"n_days_{station}"] < min_valid_days_per_month

        for ax, pct in ((ax_p90, "p90"), (ax_p99, "p99")):
            ax.plot(
                month_labels, table[f"{pct}_{station}"], marker="o", label=station,
                color=plot_style.station_color(station, STATION_COLORS, available_stations),
            )
            if low_support.any():
                ax.scatter(
                    [m for m, low in zip(month_labels, low_support) if low],
                    table.loc[low_support, f"{pct}_{station}"],
                    marker="x", color="red", s=110, linewidths=2, zorder=5,
                )
                for m, low, n_days, value in zip(
                    month_labels, low_support, table[f"n_days_{station}"],
                    table[f"{pct}_{station}"],
                ):
                    if low:
                        ax.annotate(
                            f"n={int(n_days)}", (m, value),
                            textcoords="offset points", xytext=(0, 8),
                            fontsize=plot_style.FONT_SIZES["text"], color="red", ha="center",
                        )

    ax_p90.set_title(f"Monthly {label} percentiles - station comparison")
    ax_p90.set_ylabel(f"p90 {label}{y_suffix}")
    ax_p90.legend()
    ax_p90.grid(True, alpha=0.3)

    ax_p99.set_ylabel(f"p99 {label}{y_suffix}")
    ax_p99.set_xlabel("Month")
    ax_p99.legend()
    ax_p99.grid(True, alpha=0.3)

    fig.text(
        0.5, 0.01,
        f"x = month with fewer than {min_valid_days_per_month} valid days "
        "for that station (low statistical support)",
        ha="center", fontsize=plot_style.FONT_SIZES["text"], color="red",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(output_png, dpi=plot_dpi)

    if show_plot:
        plt.show()
    plt.close(fig)


# ======================================================================
# [G] MAIN - CONSOLE MODE
# ======================================================================
def main():
    result = run_ccdf_comparison(
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

    title = "Step 2.2 - Cross-station comparison of Step 2 monthly percentiles"
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

    low_support_notes = []
    for station in available:
        low = table[table[f"n_days_{station}"] < MIN_VALID_DAYS_PER_MONTH]
        for _, row in low.iterrows():
            month_name = calendar.month_abbr[int(row["month"])]
            low_support_notes.append(
                f"{station} {month_name} (n_days={int(row[f'n_days_{station}'])})"
            )
    print()
    if low_support_notes:
        print(f"Low-support months (< {MIN_VALID_DAYS_PER_MONTH} valid days): "
              + ", ".join(low_support_notes))
    else:
        print(f"Low-support months (< {MIN_VALID_DAYS_PER_MONTH} valid days): none")

    print()
    print("Monthly percentile comparison table")
    print("-" * 36)
    print(table.to_string(index=False))

    paths = resolve_comparison_paths(YEAR, DOY_START, DOY_END, TH_COV, VALUE_COL, index_config, NDAT_MODE)
    save_comparison_table(table, paths["output_csv"])
    save_comparison_plot(
        table, available, index_config, paths["output_png"],
        MIN_VALID_DAYS_PER_MONTH, PLOT_DPI, SHOW_PLOTS,
    )

    print()
    print("Output files")
    print("-" * 12)
    print(f"Comparison table CSV: {paths['output_csv']}")
    print(f"Comparison plot: {paths['output_png']}")
    print()
    print("Step 2.2 completed successfully.")


if __name__ == "__main__":
    main()
