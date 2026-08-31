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

Does not recompute anything and never touches the Step 0 Parquet, the
Step 1 completeness CSV or the Step 3 thresholds - it only reads what
4-hourly_variability.py has already produced per station. Same
philosophy as 2.2-ccdf_comparison.py / 3.2-temporal_comparison.py: a
station whose Step 4 has not been run yet is skipped, not fatal.

IMPORTANT scientific caveat (documented here because it is easy to
misread the comparison otherwise): "local hour" is each station's own
fixed UTC offset (CLAUDE.md section 6), not a simultaneous instant
across stations - hour 22 at UNSA and hour 22 at KOUG are not the same
moment in real time. This script compares diurnal/solar-local patterns
per station, not synchronized events; for that, see Step 3.2's
DoY-based candidates instead.

Local hour is also a cyclic variable: 23h and 00h are consecutive, not
the two ends of a linear scale. A station whose activity sits near both
edges of the 0-23 plot may have one continuous nocturnal structure, not
two separate peaks (see hallazgos.md, "Alcance y limitaciones").

This step is explicitly descriptive, not interpretive (CLAUDE.md
section 1): it reports the cross-station hourly pattern, it does not
attempt to explain it physically.

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
# the TFM official default. Safe here (unlike 4-hourly_variability.py's
# NDAT_MODE=None): run_hourly_comparison() is a pure function and
# web_server.py calls it without passing ndat_mode, so this default only
# affects console/main() runs, never the web (same situation as 2.2/3.2).
NDAT_MODE = ndat_filter.NDAT_DEFAULT_MODE

# Fixed color per station (identity, not position in a list - same
# motivation as STATION_COLORS in 2-ccdf.py / 3.2-temporal_comparison.py).
# Independent copy, not imported - each script here is self-contained by
# convention.
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
    """Same check as 4-hourly_variability.py - value_col must have a validated analysis."""
    if value_col not in INDEX_CONFIG:
        raise ValueError(
            f"Index '{value_col}' is not defined in INDEX_CONFIG. "
            f"Known indices: {sorted(INDEX_CONFIG)}"
        )

    config = INDEX_CONFIG[value_col]

    if not config["scientifically_supported"]:
        raise ValueError(
            f"Index '{value_col}' is available in the Parquet but its "
            "Step 4 scientific analysis has not yet been implemented/validated."
        )

    return config


# ======================================================================
# [C] PATH RESOLUTION
# ======================================================================
def resolve_station_hourly_csv(
    station: str,
    year: int,
    doy_start: int,
    doy_end: int,
    value_col: str,
    index_config: dict,
    ndat_mode: str | None = None,
) -> Path:
    """
    Mirrors 4-hourly_variability.py's own resolve_paths() naming exactly,
    including its ndat_mode branch (None = legacy path/prefix, a real mode
    adds ndat_config["dir_tag"]/["file_tag"]). value_col is now an explicit
    parameter (previously this function read the module-level VALUE_COL
    directly, a pre-existing deviation from CLAUDE.md section 11 unrelated
    to Ndat, already fixed the same way in 3.2-temporal_comparison.py).
    """
    index_dir = Path("results") / station / str(year) / "4_hourly_variability" / value_col

    if ndat_mode is None:
        ndat_dir = index_dir
        prefix = f"4-{station}_{index_config['file_tag']}_{year}"
    else:
        ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
        ndat_dir = index_dir / ndat_config["dir_tag"]
        prefix = f"4-{station}_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"

    doy_tag = f"DOY{doy_start}_{doy_end}"
    return ndat_dir / f"{prefix}_{doy_tag}_hourly_frequency.csv"


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
    2.2/3.2's equivalent, for the same reason already documented there:
    that order would break the existing positional call here.

    ndat_mode=None (default) reproduces exactly today's flat path (no
    index or Ndat directory level). A real mode adds *both* a value_col
    level and an ndat_config["dir_tag"] level at once, atomically, only in
    this branch - same mechanism already validated in
    2.2/3.2's resolve_comparison_paths().
    """
    base_dir = Path("results") / "global" / str(year) / "4_hourly_comparison"

    if ndat_mode is None:
        comparison_dir = base_dir
        prefix = f"4-ALL_{index_config['file_tag']}_{year}"
    else:
        ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
        comparison_dir = base_dir / value_col / ndat_config["dir_tag"]
        prefix = f"4-ALL_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"

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
    silently return only the first one, hiding a genuine tie. Own copy,
    not imported from web_server.py - same logic, kept independent per
    this project's "no cross-script imports between numbered pipeline
    scripts" convention.
    """
    max_value = df_hourly[freq_col].max()
    tied = df_hourly[df_hourly[freq_col] == max_value]
    return tied["hour_local"].tolist(), float(max_value), int(tied["n_values"].iloc[0])


def run_hourly_comparison(
    stations: list[str] = STATIONS,
    year: int = YEAR,
    doy_start: int = DOY_START,
    doy_end: int = DOY_END,
    value_col: str = VALUE_COL,
    ndat_mode: str | None = None,
) -> dict:
    """
    Reads each station's already-saved Step 4 hourly CSV and builds the
    comparison table and the per-station hourly curves used for the
    combined plots.

    Pure function: no prints, no file writes. Raises ValueError if
    value_col is not scientifically supported. Raises FileNotFoundError
    only if none of the requested stations have Step 4 output yet - a
    partial set is a normal, expected result, not an error.

    ndat_mode=None (default) is the legacy behaviour: reads each
    station's flat, un-tagged CSV, unchanged from before this script knew
    about Ndat - so a caller that does not pass this argument (today,
    web_server.py) is unaffected. With a real mode, each station's CSV is
    looked up under that exact Ndat directory/tag
    (resolve_station_hourly_csv) - there is no fallback to another mode
    or to the legacy path if it is missing: a station missing its
    Ndat-specific output is reported in missing_stations with the
    Ndat-tagged path that was actually checked, exactly like any other
    missing station. All stations in one call share the same single
    ndat_mode value (like year/value_col already do), so mixing Ndat
    modes across stations within one comparison is structurally
    impossible.
    """
    index_config = validate_index_supported(value_col)
    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode) if ndat_mode is not None else None

    table_rows = []
    hourly_by_station: dict[str, pd.DataFrame] = {}
    missing_stations: dict[str, str] = {}

    for station in stations:
        hourly_csv = resolve_station_hourly_csv(
            station, year, doy_start, doy_end, value_col, index_config, ndat_mode
        )

        if not hourly_csv.exists():
            missing_stations[station] = f"Step 4 output not found: {hourly_csv}"
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
        ndat_note = f", ndat_mode='{ndat_mode}'" if ndat_mode is not None else ""
        raise FileNotFoundError(
            f"No station has Step 4 output for value_col='{value_col}'{ndat_note}, "
            f"year={year}, DOY{doy_start}_{doy_end}. Run 4-hourly_variability.py "
            "for at least one station first."
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

    "Local hour" is each station's own fixed UTC offset (CLAUDE.md
    section 6) - stations are compared by diurnal/solar-local pattern,
    not by simultaneous real time. The hour axis is also cyclic (23h and
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
            marker="o", linewidth=1.5, color=STATION_COLORS.get(station), label=station,
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
        stations=STATIONS, year=YEAR, doy_start=DOY_START, doy_end=DOY_END, value_col=VALUE_COL,
        ndat_mode=NDAT_MODE,
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

    paths = resolve_comparison_paths(YEAR, index_config, value_col=VALUE_COL, ndat_mode=NDAT_MODE)
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
