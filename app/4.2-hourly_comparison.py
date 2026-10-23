#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 4.2 - Cross-station comparison of Step 4 hourly variability.

Reads the hourly_frequency.csv that 4-hourly_variability.py already saves
for each station and builds:
  - a per-station comparison table (local hour(s) of max p90/p99
    exceedance frequency, that frequency, and the n_values support behind
    it - an hour tied for the maximum is never collapsed to a single
    value, see _hours_at_max());
  - two combined plots, one per percentile, overlaying every station's
    full 24-hour frequency curve on the same axis.

Nothing is recomputed; only what 4-hourly_variability.py has saved per
station is read.

One comparison uses a single Ndat criterion (ndat_mode), DoY range and
coverage threshold (th_cov) for every station: only the Step 4 outputs with
that criterion and run tag (completeness_paths.run_tag) are read, and the
comparison is written to its own <dir_tag> folder with the same run tag. As
in 2.2/3.2, a station without that exact Step 4 output is skipped and
reported, never replaced by the output of another criterion, DoY range or
threshold.

IMPORTANT scientific caveat: "local hour" is UTC plus each station's own
fixed offset (not magnetic local time), so it is not a simultaneous
instant across stations - hour 22 at UNSA and hour 22 at KOUG are
different moments. This script compares each station's diurnal pattern
in its own local time, not synchronized events; for those, see Step
3.2's DoY-based candidates.

Local hour is also cyclic: 23h and 00h are consecutive, not the two ends
of a linear scale. Activity near both edges of the 0-23 plot may be one
continuous nocturnal structure, not two separate peaks.

The output is descriptive: the cross-station hourly pattern is reported,
not interpreted physically.

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
# Step 4 outputs of this criterion are read for every station. "all" = ALL_NDAT.
# Only used by main(): the web interface passes the selected mode explicitly.
NDAT_MODE = "all"

# Fixed colour per station (by name, not by list position), same as in
# 2-ccdf.py; stations without an entry get a fallback colour (plot_style).
STATION_COLORS = {"UNSA": "tab:orange", "KOUG": "tab:blue", "WHIT": "tab:green", "YELL": "tab:red"}

SHOW_PLOTS = False
PLOT_DPI = 250


# ======================================================================
# [C] PATH RESOLUTION
# ======================================================================
def resolve_station_hourly_csv(
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
    Path to the hourly_frequency.csv saved by 4-hourly_variability.py for
    one station, Ndat criterion, DoY range and th_cov. Mirrors the naming
    of its resolve_paths() (dir_tag subfolder, file_tag prefix tag and run
    tag). Raises ValueError if ndat_mode is unknown.
    """
    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
    index_dir = project_paths.RESULTS_ROOT / station / str(year) / "4_hourly_variability" / value_col
    ndat_dir = index_dir / ndat_config["dir_tag"]
    prefix = f"4-{station}_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"

    run_tag = completeness_paths.run_tag(doy_start, doy_end, th_cov)
    return ndat_dir / f"{prefix}_{run_tag}_hourly_frequency.csv"


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
    Output paths under
    results/global/{YEAR}/4_hourly_comparison/{VALUE_COL}/<dir_tag>/ -
    never inside a station folder - with the run tag (DoY range, th_cov)
    in the file names. Raises ValueError if ndat_mode is unknown.
    """
    base_dir = project_paths.RESULTS_ROOT / "global" / str(year) / "4_hourly_comparison"

    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
    comparison_dir = base_dir / value_col / ndat_config["dir_tag"]
    prefix = (
        f"4-ALL_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"
        f"_{completeness_paths.run_tag(doy_start, doy_end, th_cov)}"
    )

    return {
        "comparison_dir": comparison_dir,
        "output_table_csv": comparison_dir / f"{prefix}_hourly_summary_by_station.csv",
        "output_p90_comparison_png": comparison_dir / f"{prefix}_hourly_p90_comparison.png",
        "output_p99_comparison_png": comparison_dir / f"{prefix}_hourly_p99_comparison.png",
    }


# ======================================================================
# [D] CROSS-STATION COMPARISON (pure - no prints, no file I/O)
# ======================================================================
def _hours_at_max(df_hourly: pd.DataFrame, freq_col: str) -> tuple[list[int], float, int]:
    """
    All hours tied for the maximum of freq_col - idxmax() alone would
    return only the first one, hiding a genuine tie.
    """
    max_value = df_hourly[freq_col].max()
    tied = df_hourly[df_hourly[freq_col] == max_value]
    return tied["hour_local"].tolist(), float(max_value), int(tied["n_values"].iloc[0])


def run_hourly_comparison(
    stations: list[str] = STATIONS,
    year: int = YEAR,
    doy_start: int = DOY_START,
    doy_end: int = DOY_END,
    th_cov: float = TH_COV,
    value_col: str = VALUE_COL,
    ndat_mode: str = "all",
) -> dict:
    """
    Reads each station's already-saved Step 4 hourly CSV and builds the
    comparison table and the per-station hourly curves used for the
    combined plots.

    Pure function: no prints, no file writes. Raises ValueError if
    value_col is not scientifically supported or ndat_mode is unknown,
    and FileNotFoundError only if no requested station has Step 4 output
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
    hourly_by_station: dict[str, pd.DataFrame] = {}
    missing_stations: dict[str, str] = {}

    for station in stations:
        hourly_csv = resolve_station_hourly_csv(
            station, year, doy_start, doy_end, th_cov, value_col, index_config, ndat_mode
        )

        if not hourly_csv.exists():
            missing_stations[station] = f"Step 4 output for {criterion} not found: {hourly_csv}"
            continue

        df_hourly = pd.read_csv(hourly_csv)
        hourly_by_station[station] = df_hourly

        p90_hours, p90_freq, p90_n = _hours_at_max(df_hourly, "frequency_p90_pct")
        p99_hours, p99_freq, p99_n = _hours_at_max(df_hourly, "frequency_p99_pct")

        table_rows.append({
            "station": station,
            "max_p90_hours": "/".join(f"{h:02d}" for h in sorted(p90_hours)),
            "max_p90_freq_pct": p90_freq,
            "max_p90_n_values": p90_n,
            "max_p99_hours": "/".join(f"{h:02d}" for h in sorted(p99_hours)),
            "max_p99_freq_pct": p99_freq,
            "max_p99_n_values": p99_n,
        })

    if not table_rows:
        raise FileNotFoundError(
            f"No station has Step 4 output for value_col='{value_col}', {criterion}, "
            f"year={year}, DOY{doy_start}_{doy_end}. Run 4-hourly_variability.py "
            "with the same Ndat criterion and coverage threshold for at least one "
            "station first."
        )

    table = pd.DataFrame(table_rows)

    return {
        "index_config": index_config,
        "ndat_config": ndat_config,
        "available_stations": sorted(table["station"]),
        "missing_stations": missing_stations,
        "table": table,
        "hourly_by_station": hourly_by_station,
    }


# ======================================================================
# [E] OUTPUT - CSV TABLE
# ======================================================================
def save_comparison_table(table: pd.DataFrame, paths: dict) -> None:
    paths["comparison_dir"].mkdir(parents=True, exist_ok=True)
    table.to_csv(paths["output_table_csv"], index=False)


# ======================================================================
# [F] OUTPUT - COMBINED PLOT (PNG), one call per percentile
# ======================================================================
@plot_style.with_font_sizes(sizes=plot_style.font_sizes_plus(2))
def save_hourly_comparison_plot(
    stations: list[str],
    hourly_by_station: dict,
    index_config: dict,
    percentile: int,
    output_png: Path,
    plot_dpi: int = PLOT_DPI,
    show_plot: bool = SHOW_PLOTS,
) -> None:
    """
    Overlays every station's full 24-hour exceedance-frequency curve for
    one percentile (90 or 99) on a shared axis - color fixed per station
    (STATION_COLORS).

    "Local hour" uses each station's own fixed UTC offset - stations are
    compared by diurnal pattern, not by simultaneous real time. The hour axis is also cyclic (23h and
    00h are consecutive), so activity near both edges of the plot may be
    one continuous nocturnal structure rather than two separate peaks.
    """
    output_png.parent.mkdir(parents=True, exist_ok=True)
    label = index_config["label"]
    freq_col = f"frequency_p{percentile}_pct"

    fig, ax = plt.subplots(figsize=(11, 6))

    for station in stations:
        df_plot = hourly_by_station[station].sort_values("hour_local")
        ax.plot(
            df_plot["hour_local"], df_plot[freq_col],
            marker="o", linewidth=1.5, color=plot_style.station_color(station, STATION_COLORS, stations), label=station,
        )

    ax.set_xticks(range(24))
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Local hour (each station's own fixed UTC offset)")
    ax.set_ylabel(f"Frequency at/above own annual p{percentile} [% of observations]")
    ax.set_title(f"Hourly p{percentile} exceedance frequency — station comparison — {label}")
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
    result = run_hourly_comparison(
        stations=STATIONS, year=YEAR, doy_start=DOY_START, doy_end=DOY_END, th_cov=TH_COV,
        value_col=VALUE_COL, ndat_mode=NDAT_MODE,
    )

    index_config = result["index_config"]
    available = result["available_stations"]
    missing = result["missing_stations"]
    table = result["table"]

    title = "Step 4.2 - Cross-station comparison of Step 4 hourly variability"
    print(title)
    print("=" * len(title))
    print(f"Index: {index_config['label']} ({VALUE_COL})")
    print(f"Year: {YEAR}")
    print()
    print(f"Stations included ({len(available)}): {', '.join(available)}")
    if missing:
        print(f"Stations missing ({len(missing)}):")
        for station, reason in missing.items():
            print(f"  - {station}: {reason}")
    else:
        print("Stations missing: none")

    print()
    print("Hourly summary by station")
    print("-" * 26)
    print(table.to_string(index=False))
    print(
        "\nReminder: 'local hour' is each station's own fixed UTC offset, not "
        "a simultaneous instant across stations (see module docstring)."
    )

    paths = resolve_comparison_paths(YEAR, DOY_START, DOY_END, TH_COV, VALUE_COL, index_config, NDAT_MODE)
    save_comparison_table(table, paths)
    save_hourly_comparison_plot(
        available, result["hourly_by_station"], index_config, 90,
        paths["output_p90_comparison_png"], plot_dpi=PLOT_DPI, show_plot=SHOW_PLOTS,
    )
    save_hourly_comparison_plot(
        available, result["hourly_by_station"], index_config, 99,
        paths["output_p99_comparison_png"], plot_dpi=PLOT_DPI, show_plot=SHOW_PLOTS,
    )

    print()
    print("Output files")
    print("-" * 12)
    print(f"Hourly summary CSV: {paths['output_table_csv']}")
    print(f"p90 comparison plot: {paths['output_p90_comparison_png']}")
    print(f"p99 comparison plot: {paths['output_p99_comparison_png']}")
    print()
    print("Step 4.2 completed successfully.")


if __name__ == "__main__":
    main()
