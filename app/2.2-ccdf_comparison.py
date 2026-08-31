#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2.2 - Cross-station comparison of Step 2 monthly percentiles.

Reads the monthly_percentiles.csv that 2-ccdf.py already saves for each
station and merges them into one comparison table and one combined plot.
Does not recompute anything and never touches the Step 0 Parquet - it
only reads what 2-ccdf.py has already produced per station. All four
stations were run with the same default parameters in this project, so
re-deriving the numbers here would only add cost, not correctness.

A station whose Step 2 has not been run yet is skipped, not fatal -
comparing a partial set of stations is an expected, normal case, not an
error.

Also carries over n_days/n_values per station-month (not just p90/p99),
so a low-support month (few valid days behind that percentile) is
visible in the table and flagged in the plot instead of requiring a
manual check - this project already has one such case (UNSA, August,
n_days=1). And cross-checks that every compared station has a Step 1
coverage CSV matching this script's own th_cov: Step 2's own filename
does not encode th_cov (unlike Step 1's), so nothing before this
prevented silently comparing stations run with different coverage
thresholds. This cross-check is a best-effort sanity check, not a
guarantee - it confirms a matching Step 1 file exists, not that Step 2
itself was generated with that same threshold. A fully reliable check
would need th_cov stored inside Step 2's own output, which is a change
to 2-ccdf.py, not to this script - left for later, not needed now.

Scientific scope: only ROTI L1 (roti_l1) is implemented/validated - see
INDEX_CONFIG / validate_index_supported() (CLAUDE.md section 3-bis).

Purpose: a fast, agile cross-station view (one CSV + one PNG) meant to
be read directly or handed to an AI assistant for interpretation help -
not the final LaTeX-formatted output. That is a separate, later step
built on top of these same files, not a redesign of them.
"""

import calendar
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

# Same threshold 2-ccdf.py already uses for its own "few valid days"
# warning (MIN_VALID_DAYS_PER_MONTH) - reused here, not re-invented, so
# a month flagged as low-support means the same thing in both scripts.
MIN_VALID_DAYS_PER_MONTH = 10

SHOW_PLOTS = False
PLOT_DPI = 250


# ======================================================================
# [B] INDICES: PARQUET COLUMN <-> FILE TAG <-> SCIENTIFIC LABEL <-> UNIT
# ======================================================================
# Same copy already used by 1-completitud.py through 6-month_hour_heatmaps.py
# (established convention: each script is self-contained, no shared
# config module yet).

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
    Same check as 2-ccdf.py: value_col must have a validated Step 2
    scientific analysis, not just exist as a Parquet column. Receives
    value_col explicitly so it works the same from main() or a future
    caller with a different index.
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
            "Step 2 scientific analysis has not yet been "
            "implemented/validated."
        )

    return config


# ======================================================================
# [C] PATH RESOLUTION
# ======================================================================
def resolve_station_pcts_csv(
    station: str,
    year: int,
    doy_start: int,
    doy_end: int,
    value_col: str,
    index_config: dict,
    ndat_mode: str | None = None,
) -> Path:
    """
    Path to the monthly_percentiles.csv that 2-ccdf.py already saves for
    one station. Mirrors 2-ccdf.py's own resolve_paths() naming exactly -
    not re-derived independently, so a naming change there is the only
    place this would need to follow.

    ndat_mode=None (default) reproduces exactly the path/prefix from
    before this script knew about Ndat - no existing caller that does not
    pass this argument (today, web_server.py) sees any path change. A
    real mode ("eq60"/"ge30"/"all"/"lt30") adds one more directory level
    (ndat_config["dir_tag"]) and tags the filename too
    (ndat_config["file_tag"]) - same mechanism 2-ccdf.py's resolve_paths()
    already uses.
    """
    index_dir = Path("results") / station / str(year) / "2_ccdf" / value_col

    if ndat_mode is None:
        ndat_dir = index_dir
        prefix = f"2-{station}_{index_config['file_tag']}_{year}"
    else:
        ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
        ndat_dir = index_dir / ndat_config["dir_tag"]
        prefix = f"2-{station}_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"

    doy_tag = f"DOY{doy_start}_{doy_end}"
    return ndat_dir / f"{prefix}_{doy_tag}_monthly_percentiles.csv"


def resolve_station_coverage_csv(
    station: str, year: int, doy_start: int, doy_end: int, th_cov: float
) -> Path:
    """
    Path to the Step 1 coverage CSV for one station - mirrors
    1-completitud.py's own naming exactly. Used only to cross-check that
    a station's Step 2 output has a matching Step 1 run at this same
    th_cov (see module docstring for what this check does and does not
    guarantee) - not read for its contents anywhere in this script.
    """
    return (
        Path("results") / station / str(year) / "1_completeness"
        / f"coverage_{station}_{year}_DOY{doy_start}_{doy_end}_coverageTH{th_cov}.csv"
    )


def resolve_comparison_paths(
    year: int, value_col: str, index_config: dict, ndat_mode: str | None = None,
) -> dict:
    """
    Output paths for the comparison itself, under results/global/{YEAR}/ -
    same convention 0-toParquet.py already uses for cross-station files,
    never inside a single station's own folder.

    ndat_mode=None (default) reproduces exactly today's path (flat, no
    index or Ndat directory level - this global path never nested by
    index before Ndat existed). A real mode adds *both* a value_col level
    and an ndat_config["dir_tag"] level at once, atomically, only in this
    branch - the legacy branch above is untouched, so this is additive,
    not a restructuring of the existing path.
    """
    base_dir = Path("results") / "global" / str(year) / "2_ccdf_comparison"

    if ndat_mode is None:
        comparison_dir = base_dir
        prefix = f"2-ALL_{index_config['file_tag']}_{year}"
    else:
        ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
        comparison_dir = base_dir / value_col / ndat_config["dir_tag"]
        prefix = f"2-ALL_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"

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
    ndat_mode: str | None = None,
) -> dict:
    """
    Reads the monthly_percentiles.csv already saved by 2-ccdf.py for each
    station and merges n_days/n_values/p90/p99 into one wide table: one
    row per month, one column group per available station.

    Pure function: no prints, no file writes, so it behaves the same
    from main() (console mode) or from a future caller. Raises
    ValueError if value_col is not scientifically supported (see
    validate_index_supported). Raises FileNotFoundError only if none of
    the requested stations have Step 2 output yet - a partial set is a
    normal, expected result, not an error.

    ndat_mode=None (default) is the legacy behaviour: reads each
    station's flat, un-tagged CSV, unchanged from before this script knew
    about Ndat - so a caller that does not pass this argument (today,
    web_server.py) is unaffected. With a real mode, each station's CSV is
    looked up under that exact Ndat directory/tag (resolve_station_pcts_csv)
    - there is no fallback to another mode or to the legacy path if it is
    missing: a station missing its Ndat-specific output is reported in
    missing_stations with the Ndat-tagged path that was actually checked,
    exactly like any other missing station.
    """
    index_config = validate_index_supported(value_col)
    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode) if ndat_mode is not None else None

    available_frames: dict[str, pd.DataFrame] = {}
    missing_stations: dict[str, str] = {}

    for station in stations:
        csv_path = resolve_station_pcts_csv(
            station, year, doy_start, doy_end, value_col, index_config, ndat_mode
        )
        if not csv_path.exists():
            missing_stations[station] = f"Step 2 output not found: {csv_path}"
            continue
        available_frames[station] = pd.read_csv(
            csv_path, usecols=["month", "n_days", "n_values", "p90", "p99"]
        )

    if not available_frames:
        ndat_note = f", ndat_mode='{ndat_mode}'" if ndat_mode is not None else ""
        raise FileNotFoundError(
            f"No station has Step 2 output for value_col='{value_col}'{ndat_note}, "
            f"year={year}, DOY{doy_start}_{doy_end}. Run 2-ccdf.py for at "
            "least one station first."
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
                month_labels, table[f"{pct}_{station}"], marker="o", label=station
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
                            fontsize=7, color="red", ha="center",
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
        ha="center", fontsize=8, color="red",
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

    paths = resolve_comparison_paths(YEAR, VALUE_COL, index_config, NDAT_MODE)
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
