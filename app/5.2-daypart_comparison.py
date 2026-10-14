#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 5.2 - Cross-station comparison of Step 5 daypart exceedance frequencies.

Reads the daypart_frequency.csv saved by 5-daypart_variability.py for each
station and merges them into one comparison table (CSV) and one combined
plot (PNG). Nothing is recomputed and the Step 0 Parquet is not read. A
station without Step 5 output is skipped and reported, not fatal.

Daypart-layout check: before merging, every compared station must have the
same daypart_order/daypart/hour_range triplets. A month number always
means the same month, but a daypart name depends on the DAYPARTS
configuration of 5-daypart_variability.py, so merging by name alone could
compare different time windows under the same label. A mismatch is a hard
error (ValueError), unlike the th_cov check: a value computed with a
slightly different coverage threshold is still comparable, but a "Night"
that means 18-06 for one station and 00-06 for another is not.

th_cov check: Step 5 output names do not encode th_cov, so this script can
only check that each compared station has a Step 1 coverage CSV for the
same th_cov. It is a best-effort check: it confirms that such a Step 1
file exists, not that Step 5 was generated from it.

Scientific scope: only ROTI L1 (roti_l1) is implemented/validated - see
INDEX_CONFIG / validate_index_supported().
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# Index to compare. Only "roti_l1" is scientifically implemented/validated.
VALUE_COL = "roti_l1"

# Ndat criterion to compare across stations (see ndat_filter.py); "eq60"
# is the official criterion of the study. Only used by main(): the web
# interface passes the selected mode explicitly. With this default, the
# untagged outputs of 5-daypart_variability.py's console default
# (NDAT_MODE = None) are not read.
NDAT_MODE = ndat_filter.NDAT_DEFAULT_MODE

# Fixed colour per station (by name, not by list position): with the
# default colour cycle, colours would shift whenever a station is missing.
# Stations without an entry get a fallback colour (plot_style).
STATION_COLORS = {"UNSA": "tab:orange", "KOUG": "tab:blue", "WHIT": "tab:green", "YELL": "tab:red"}

SHOW_PLOTS = False
PLOT_DPI = 250


# ======================================================================
# [B] INDICES: PARQUET COLUMN <-> FILE TAG <-> SCIENTIFIC LABEL <-> UNIT
# ======================================================================
# Same index definitions as in Steps 2-6 (each script is self-contained).

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
    """
    Same check as every other step: value_col must have a validated
    scientific analysis, not just exist as a Parquet column.
    """
    if value_col not in INDEX_CONFIG:
        raise ValueError(
            f"Index '{value_col}' is not defined in INDEX_CONFIG. "
            f"Known indices: {sorted(INDEX_CONFIG)}"
        )

    config = INDEX_CONFIG[value_col]

    if not config["scientifically_supported"]:
        raise ValueError(
            f"Index '{value_col}' is available in the Parquet but its "
            "Step 5 scientific analysis has not yet been "
            "implemented/validated."
        )

    return config


# ======================================================================
# [C] PATH RESOLUTION
# ======================================================================
def resolve_station_daypart_csv(
    station: str,
    year: int,
    doy_start: int,
    doy_end: int,
    value_col: str,
    index_config: dict,
    ndat_mode: str | None = None,
) -> Path:
    """
    Path to the daypart_frequency.csv saved by 5-daypart_variability.py
    for one station. Mirrors the naming of its resolve_paths(), including
    ndat_mode (None: untagged; a mode: dir_tag subfolder and file_tag
    prefix tag).
    """
    index_dir = project_paths.RESULTS_ROOT / station / str(year) / "5_daypart_variability" / value_col

    if ndat_mode is None:
        ndat_dir = index_dir
        prefix = f"5-{station}_{index_config['file_tag']}_{year}"
    else:
        ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
        ndat_dir = index_dir / ndat_config["dir_tag"]
        prefix = f"5-{station}_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"

    doy_tag = f"DOY{doy_start}_{doy_end}"
    return ndat_dir / f"{prefix}_{doy_tag}_daypart_frequency.csv"


def resolve_station_coverage_csv(
    station: str, year: int, doy_start: int, doy_end: int, th_cov: float
) -> Path:
    """
    Path to the Step 1 coverage CSV for one station - mirrors
    1-completitud.py's own naming exactly. Used only for the th_cov
    cross-check described in the module docstring.
    """
    return (
        project_paths.RESULTS_ROOT / station / str(year) / "1_completeness"
        / f"coverage_{station}_{year}_DOY{doy_start}_{doy_end}_coverageTH{th_cov}.csv"
    )


def resolve_comparison_paths(
    year: int,
    value_col: str,
    index_config: dict,
    ndat_mode: str | None = None,
) -> dict:
    """
    Output paths of the comparison, under results/global/{YEAR}/ (never
    inside a station folder). ndat_mode=None: directly in
    5_daypart_comparison/; a mode adds the value_col and
    ndat_config["dir_tag"] subfolders.
    """
    base_dir = project_paths.RESULTS_ROOT / "global" / str(year) / "5_daypart_comparison"

    if ndat_mode is None:
        comparison_dir = base_dir
        prefix = f"5-ALL_{index_config['file_tag']}_{year}"
    else:
        ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
        comparison_dir = base_dir / value_col / ndat_config["dir_tag"]
        prefix = f"5-ALL_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"

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
    ndat_mode: str | None = None,
) -> dict:
    """
    Reads the daypart_frequency.csv already saved by
    5-daypart_variability.py for each station and merges n_values/
    frequency_p90/frequency_p99 into one wide table: one row per
    daypart, one column group per available station.

    Pure function: no prints, no file writes. Raises ValueError if
    value_col is not scientifically supported, or if the compared
    stations do not share the same daypart definition (a hard error, see
    module docstring). Raises FileNotFoundError only if no requested
    station has Step 5 output.

    Each station's CSV is looked up for exactly the given ndat_mode
    (None: untagged path). There is no fallback to another mode: a
    station without that output is reported in missing_stations with the
    path that was checked. All stations share one ndat_mode, so Ndat
    criteria are never mixed within a comparison.
    """
    index_config = validate_index_supported(value_col)
    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode) if ndat_mode is not None else None

    available_frames: dict[str, pd.DataFrame] = {}
    missing_stations: dict[str, str] = {}

    for station in stations:
        csv_path = resolve_station_daypart_csv(
            station, year, doy_start, doy_end, value_col, index_config, ndat_mode
        )
        if not csv_path.exists():
            missing_stations[station] = f"Step 5 output not found: {csv_path}"
            continue
        available_frames[station] = pd.read_csv(csv_path, usecols=[
            "daypart_order", "daypart", "hour_range",
            "n_values", "frequency_p90_pct", "frequency_p99_pct",
        ])

    if not available_frames:
        ndat_note = f", ndat_mode='{ndat_mode}'" if ndat_mode is not None else ""
        raise FileNotFoundError(
            f"No station has Step 5 output for value_col='{value_col}'{ndat_note}, "
            f"year={year}, DOY{doy_start}_{doy_end}. Run "
            "5-daypart_variability.py for at least one station first."
        )

    # Daypart-definition consistency check - hard error, see module
    # docstring for why this cannot be a soft warning like th_cov.
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

    # Best-effort th_cov cross-check - see module docstring for exactly
    # what this does and does not guarantee.
    th_cov_warnings: dict[str, str] = {}
    for station in available_frames:
        coverage_csv = resolve_station_coverage_csv(
            station, year, doy_start, doy_end, th_cov
        )
        if not coverage_csv.exists():
            th_cov_warnings[station] = (
                f"No Step 1 coverage CSV found for th_cov={th_cov} "
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

    paths = resolve_comparison_paths(YEAR, VALUE_COL, index_config, NDAT_MODE)
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
