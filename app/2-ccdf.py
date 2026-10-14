#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2 - Monthly CCDF and percentiles.

Inputs: the Step 0 Parquet and the Step 1 coverage CSV. Outputs under
results/{STATION}/{YEAR}/2_ccdf/{INDEX}/. Same structure as 1-completitud.py:
globals in section [A] are console defaults, run_ccdf_analysis() is a pure
parametrised calculation (no prints, no file writes), main() is the console
entry point, and plots are saved without depending on plt.show().

Only the Step 1 valid days are used; within them the Ndat criterion selects
observations. The Step 1 coverage threshold is not re-applied after Ndat.

Scientific scope: only ROTI L1 (roti_l1) is implemented/validated - see
INDEX_CONFIG / validate_index_supported().
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

import ndat_filter
import plot_style
import project_paths

# ======================================================================
# [A] STUDY CONFIGURATION
# ======================================================================

# [A.1] Study case
STATION = "YELL"
YEAR = 2024
DOY_START = 1
DOY_END = 366  # 2024 is a leap year

# [A.2] Index to analyse. Only "roti_l1" is scientifically
# implemented/validated in Step 2 - see INDEX_CONFIG.
VALUE_COL = "roti_l1"

# [A.2b] Ndat criterion for the ROTI L1 observations (see ndat_filter.py);
# "eq60" is the official criterion of the study.
NDAT_MODE = ndat_filter.NDAT_DEFAULT_MODE

# [A.3] Step 1 completeness threshold (used to locate its CSV)
TH_COV = 0.75

# [A.4] Percentiles to extract
PCTS = [90, 95, 99]

# [A.5] Analysis range
# "all": use every valid day of the year.
ANALYSIS_MODE = "all"

# With ANALYSIS_MODE = "months", only the valid days of these months are used
MONTHS_TO_ANALYZE = [3, 7, 11]

# [A.6] Months drawn together in the same comparison figure
MONTHS_TO_COMPARE = [3, 7, 11]

# [A.7] Plot settings - console defaults. The plot functions receive them
# as arguments rather than reading them from here (see [E]).
USE_LOG_Y = True
X_LIM = (0, 20)
Y_LIM = (1e-5, 1)

SAVE_INDIVIDUAL_CCDF = True
SAVE_COMPARISON_CCDF = True
SHOW_PLOTS = False

# [A.8] Months with fewer valid days than this are flagged in the plot and
# excluded from the min/max of the summary (low statistical support)
MIN_VALID_DAYS_PER_MONTH = 10

# [A.9] Cross-station CCDF comparison: fixed colour per station (by name,
# not by list position - same reason as month_color(), see [E]) and one
# linestyle per month (cycles beyond 4 months).
STATION_COLORS = {"UNSA": "tab:orange", "KOUG": "tab:blue", "WHIT": "tab:green", "YELL": "tab:red"}
MONTH_LINESTYLES = ["-", "--", ":", "-."]


# ======================================================================
# [B] INDICES: PARQUET COLUMN <-> FILE TAG <-> SCIENTIFIC LABEL
# ======================================================================
# Single definition of every index available in the Step 0 Parquet, so a
# label can never disagree with the data (e.g. a "ROTI L1" plot showing S4).
#
# "scientifically_supported" = False: the column exists in the Parquet but
# its Step 2 analysis is not implemented/validated. Enabling it is a
# methodological change, not a technical one.
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
    Checks that value_col has an implemented and validated Step 2
    analysis - not just that the column exists in the Parquet. Returns its
    INDEX_CONFIG entry.
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
def resolve_paths(
    station: str,
    year: int,
    doy_start: int,
    doy_end: int,
    th_cov: float,
    value_col: str,
    index_config: dict,
    ndat_mode: str | None = None,
) -> dict:
    """
    Input/output paths, computed from the arguments received (not from
    the module globals). Outputs go under
    results/{STATION}/{YEAR}/2_ccdf/{VALUE_COL}/.

    ndat_mode=None: no Ndat subfolder and no Ndat tag in the file prefix.
    A mode ("eq60"/"ge30"/"all"/"lt30") adds the ndat_config["dir_tag"]
    subfolder and the ndat_config["file_tag"] prefix tag, so results of
    different criteria never overwrite each other.
    """
    parquet_path = (
        project_paths.RESULTS_ROOT / station / str(year) / "0_parquet"
        / f"0_{station}_{year}_observations.parquet"
    )
    coverage_csv = (
        project_paths.RESULTS_ROOT / station / str(year) / "1_completeness"
        / f"coverage_{station}_{year}_DOY{doy_start}_{doy_end}_coverageTH{th_cov}.csv"
    )

    index_dir = project_paths.RESULTS_ROOT / station / str(year) / "2_ccdf" / value_col

    if ndat_mode is None:
        ndat_dir = index_dir
        prefix = f"2-{station}_{index_config['file_tag']}_{year}"
    else:
        ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
        ndat_dir = index_dir / ndat_config["dir_tag"]
        prefix = f"2-{station}_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"

    ccdf_dir = ndat_dir / "ccdf"
    ccdf_comparisons_dir = ndat_dir / "ccdf_comparisons"
    percentile_dir = ndat_dir / "monthly_percentile_variability"
    doy_tag = f"DOY{doy_start}_{doy_end}"

    return {
        "parquet_path": parquet_path,
        "coverage_csv": coverage_csv,
        "index_dir": index_dir,
        "ndat_dir": ndat_dir,
        "ccdf_dir": ccdf_dir,
        "ccdf_comparisons_dir": ccdf_comparisons_dir,
        "percentile_dir": percentile_dir,
        "output_pcts_csv": ndat_dir / f"{prefix}_{doy_tag}_monthly_percentiles.csv",
        "output_monthly_percentiles_png": (
            percentile_dir / f"{prefix}_{doy_tag}_monthly_percentile_variability.png"
        ),
        "prefix": prefix,
    }


def individual_ccdf_png(paths: dict, doy_start: int, doy_end: int, month: int) -> Path:
    """
    File name of a single-month CCDF. Includes DOY{start}_{end} so runs
    with different DoY ranges do not overwrite each other. Shared by
    main() and the web server.
    """
    doy_tag = f"DOY{doy_start}_{doy_end}"
    return paths["ccdf_dir"] / f"{paths['prefix']}_{doy_tag}_M{month:02d}_ccdf_logY.png"


def multi_ccdf_png(paths: dict, doy_start: int, doy_end: int, months: list[int]) -> Path:
    """
    File name of the multi-month CCDF comparison. Stored in
    ccdf_comparisons_dir, apart from the 12 single-month CCDFs, because
    each month combination produces a new file.
    """
    doy_tag = f"DOY{doy_start}_{doy_end}"
    months_tag = "_".join(f"M{m:02d}" for m in sorted(months))
    return paths["ccdf_comparisons_dir"] / f"{paths['prefix']}_{doy_tag}_{months_tag}_ccdf_comparison_logY.png"


def resolve_cross_station_ccdf_path(
    year: int, doy_start: int, doy_end: int,
    stations: list[str], months: list[int],
    index_config: dict,
    ndat_mode: str | None = None,
) -> Path:
    """
    Cross-station CCDF comparison output path - global, not scoped to one
    station: results/global/{year}/2_ccdf_cross_station/, separate from
    Step 2.2's 2_ccdf_comparison/ (full CCDF curves, not percentile tables).

    ndat_mode is handled as in resolve_paths() (None: no Ndat subfolder
    or tag).
    """
    doy_tag = f"DOY{doy_start}_{doy_end}"
    stations_tag = "_".join(sorted(stations))
    months_tag = "_".join(f"M{m:02d}" for m in sorted(months))
    base_dir = project_paths.RESULTS_ROOT / "global" / str(year) / "2_ccdf_cross_station"

    if ndat_mode is None:
        out_dir = base_dir
        prefix = f"2-{stations_tag}_{index_config['file_tag']}_{year}"
    else:
        ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
        out_dir = base_dir / ndat_config["dir_tag"]
        prefix = f"2-{stations_tag}_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"

    filename = f"{prefix}_{doy_tag}_{months_tag}_ccdf_cross_station_logY.png"
    return out_dir / filename


# ======================================================================
# [D] HELPER FUNCTIONS
# ======================================================================

def doy_to_month(year: int, doy: int) -> int:
    """Year + DoY -> calendar month (1..12)."""
    dt = datetime.strptime(f"{year}-{doy:03d}", "%Y-%j")
    return dt.month


def month_label(month: int) -> str:
    """Short month name for legends and labels."""
    labels = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    return labels.get(month, f"M{month:02d}")


def accumulate_values_by_month(
    df_station: pd.DataFrame,
    valid_doys: list[int],
    year: int,
    parquet_value_col: str,
) -> tuple[dict, dict]:
    """
    Groups the values of parquet_value_col by month for the DoYs in
    valid_doys, in a single vectorised pass. The month of each DoY comes
    from doy_to_month().

    Returns (month_values, n_days_by_month):
    - month_values[m]: 1D array with every non-NaN value of that month;
    - n_days_by_month[m]: number of distinct DoYs that contributed at
      least one non-NaN value that month.
    """
    doy_to_month_map = {doy: doy_to_month(year, doy) for doy in valid_doys}

    df_valid = df_station[df_station["14_doy_utc"].isin(valid_doys)].copy()

    numeric_values = pd.to_numeric(df_valid[parquet_value_col], errors="coerce")
    df_valid = df_valid.loc[numeric_values.notna()].copy()
    df_valid[parquet_value_col] = numeric_values.loc[df_valid.index]

    df_valid["month"] = df_valid["14_doy_utc"].map(doy_to_month_map)

    month_values = {}
    n_days_by_month = {}

    for m in range(1, 13):
        group = df_valid[df_valid["month"] == m]

        if group.empty:
            month_values[m] = np.array([], dtype=float)
            n_days_by_month[m] = 0
            continue

        month_values[m] = group[parquet_value_col].to_numpy(dtype=float)
        n_days_by_month[m] = int(group["14_doy_utc"].nunique())

    return month_values, n_days_by_month


def compute_ccdf(values: np.ndarray):
    """
    Empirical CCDF:
        CCDF(x_i) ≈ P(X >= x_i) = (n - i + 1) / n

    With this definition the CCDF never reaches exactly zero, which suits
    a logarithmic y-axis.
    """
    x = np.sort(values)
    n = x.size

    if n == 0:
        return None, None

    i = np.arange(1, n + 1)
    ccdf = (n - i + 1) / n

    return x, ccdf


def format_n(n: int) -> str:
    """Sample count with "." as thousands separator."""
    return f"{n:,}".replace(",", ".")


# ======================================================================
# [E] PLOT FUNCTIONS (use_log_y/x_lim/y_lim/show_plot are arguments, with
# the console globals as defaults)
# ======================================================================

def month_color(month: int):
    """
    Fixed colour per calendar month (1-12), independent of the other
    months in the figure, so a month keeps its colour across comparisons.
    Sampled from a cyclic colormap (twilight) to follow the seasonal cycle.
    """
    cmap = plt.get_cmap("twilight")
    return cmap((month - 1) / 12)


@plot_style.with_font_sizes
def save_ccdf_plot(
    month: int,
    x: np.ndarray,
    ccdf: np.ndarray,
    out_png: Path,
    station: str,
    year: int,
    index_label: str,
    index_unit: str,
    use_log_y: bool = USE_LOG_Y,
    x_lim: tuple = X_LIM,
    y_lim: tuple = Y_LIM,
    show_plot: bool = SHOW_PLOTS
):
    """
    Saves the CCDF of a single month, with reference lines at
    CCDF = 0.1 (p90) and 0.01 (p99).
    """
    plt.figure(figsize=(8, 5))

    plt.step(
        x,
        ccdf,
        where="post",
        linewidth=1.4,
        label=f"{month_label(month)}"
    )

    if use_log_y:
        plt.yscale("log")

    plt.xlim(x_lim)
    plt.ylim(y_lim)

    # Percentile reference lines
    plt.axhline(0.1, linestyle="--", linewidth=0.9, alpha=0.7, label="CCDF=0.1 (p90)")
    plt.axhline(0.01, linestyle="--", linewidth=0.9, alpha=0.7, label="CCDF=0.01 (p99)")

    plt.xlabel(f"{index_label} [{index_unit}]")
    plt.ylabel(f"CCDF = P({index_label} ≥ x)")
    plt.title(f"CCDF {index_label} — {station} — {year} — {month_label(month)}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_png, dpi=200)

    if show_plot:
        plt.show()

    plt.close()


@plot_style.with_font_sizes(sizes=plot_style.font_sizes_plus(4))
def save_multi_ccdf_plot(
    ccdf_by_month: dict,
    pcts_by_month: dict,
    months_to_compare: list[int],
    out_png: Path,
    station: str,
    year: int,
    index_label: str,
    index_unit: str,
    use_log_y: bool = USE_LOG_Y,
    x_lim: tuple = X_LIM,
    y_lim: tuple = Y_LIM,
    show_plot: bool = SHOW_PLOTS
):
    """
    Saves several monthly CCDFs overlaid in one figure, with each month's
    p90/p99 marked on its own curve and reference lines at CCDF = 0.1 and
    0.01.
    """
    plt.figure(figsize=(9, 6))

    for month in months_to_compare:
        if month not in ccdf_by_month:
            print(f"[AVISO] No hay CCDF disponible para Month {month:02d}. Se omite.")
            continue

        x, ccdf, n_values = ccdf_by_month[month]

        label = f"{month_label(month)} (n={format_n(n_values)})"
        color = month_color(month)

        plt.step(
            x,
            ccdf,
            where="post",
            linewidth=1.5,
            color=color,
            label=label
        )

        # p90/p99 markers in the colour of their own month's curve.
        if month in pcts_by_month:
            p90 = pcts_by_month[month].get("p90", np.nan)
            p99 = pcts_by_month[month].get("p99", np.nan)

            if not np.isnan(p90):
                plt.scatter(p90, 0.1, s=25, color=color)
                plt.vlines(p90, ymin=y_lim[0], ymax=0.1, linestyle=":", linewidth=0.9, color=color)

            if not np.isnan(p99):
                plt.scatter(p99, 0.01, s=25, color=color)
                plt.vlines(p99, ymin=y_lim[0], ymax=0.01, linestyle=":", linewidth=0.9, color=color)

    if use_log_y:
        plt.yscale("log")

    plt.xlim(x_lim)
    plt.ylim(y_lim)

    # Percentile reference lines
    plt.axhline(0.1, linestyle="--", linewidth=1.0, alpha=0.7)
    plt.axhline(0.01, linestyle="--", linewidth=1.0, alpha=0.7)

    plt.text(x_lim[0] + 0.2, 0.105, "CCDF = 0.1 (p90)", va="bottom")
    plt.text(x_lim[0] + 0.2, 0.0105, "CCDF = 0.01 (p99)", va="bottom")

    plt.xlabel(f"{index_label} [{index_unit}]")
    plt.ylabel(f"CCDF = P({index_label} ≥ x)")
    plt.title(
        f"Monthly CCDF comparison — {index_label} — {station} — {year}\n"
        f"Months: {', '.join(month_label(m) for m in months_to_compare)}"
    )

    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_png, dpi=250)

    if show_plot:
        plt.show()

    plt.close()


@plot_style.with_font_sizes
def save_cross_station_ccdf_plot(
    ccdf_by_station_month: dict,
    stations: list[str],
    months: list[int],
    out_png: Path,
    year: int,
    index_label: str,
    index_unit: str,
    use_log_y: bool = USE_LOG_Y,
    x_lim: tuple = X_LIM,
    y_lim: tuple = Y_LIM,
    show_plot: bool = SHOW_PLOTS,
):
    """
    Overlays CCDF curves across stations and months on one shared axis -
    complementary to save_multi_ccdf_plot (fixed station, several months):
    here several stations are compared for the same month(s).

    ccdf_by_station_month: {station: {month: (x_sorted, ccdf, n_values)}}.
    Colour is fixed per station (STATION_COLORS), linestyle per month
    (MONTH_LINESTYLES, cycling beyond 4 months). The sample size is given
    as n= in the legend; there is no separate low-support marker.
    """
    plt.figure(figsize=(9, 6))

    for station in stations:
        station_data = ccdf_by_station_month.get(station, {})
        color = plot_style.station_color(station, STATION_COLORS, stations)

        for i, month in enumerate(months):
            if month not in station_data:
                continue
            x, ccdf, n_values = station_data[month]
            linestyle = MONTH_LINESTYLES[i % len(MONTH_LINESTYLES)]
            label = f"{station} {month_label(month)} (n={format_n(n_values)})"

            plt.step(
                x, ccdf, where="post",
                linewidth=1.5, color=color, linestyle=linestyle,
                label=label,
            )

    if use_log_y:
        plt.yscale("log")

    plt.xlim(x_lim)
    plt.ylim(y_lim)

    # Grey reference lines, so they cannot share a colour with a station.
    plt.axhline(0.1, linestyle="--", linewidth=1.0, alpha=0.7, color="gray")
    plt.axhline(0.01, linestyle="--", linewidth=1.0, alpha=0.7, color="gray")
    plt.text(x_lim[0] + 0.2, 0.105, "CCDF = 0.1 (p90)", va="bottom")
    plt.text(x_lim[0] + 0.2, 0.0105, "CCDF = 0.01 (p99)", va="bottom")

    plt.xlabel(f"{index_label} [{index_unit}]")
    plt.ylabel(f"CCDF = P({index_label} ≥ x)")
    plt.title(
        f"Cross-station CCDF comparison — {index_label} — {year}\n"
        f"Stations: {', '.join(stations)} | Months: {', '.join(month_label(m) for m in months)}"
    )

    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=plot_style.FONT_SIZES["legend"])
    plt.tight_layout()

    plt.savefig(out_png, dpi=250)

    if show_plot:
        plt.show()

    plt.close()


@plot_style.with_font_sizes(sizes=plot_style.font_sizes_plus(3))
def save_monthly_percentiles_plot(
    df_pcts: pd.DataFrame,
    out_png: Path,
    station: str,
    year: int,
    index_label: str,
    index_unit: str,
    min_valid_days: int = 10,
    show_plot: bool = SHOW_PLOTS
):
    """
    Plots the monthly p90 and p99 values over the year.

    Months with fewer than min_valid_days valid days stay in the figure
    but are annotated with their n_days, so they are not read as
    representative of the whole month.
    """

    required_columns = {
        "month",
        "n_days",
        "p90",
        "p99"
    }

    missing_columns = required_columns - set(df_pcts.columns)

    if missing_columns:
        raise ValueError(
            "No se puede generar la gráfica mensual. "
            f"Faltan las columnas: {sorted(missing_columns)}"
        )

    # Sorted copy; the caller's DataFrame is not modified
    df_plot = (
        df_pcts.copy()
        .sort_values("month")
        .reset_index(drop=True)
    )

    # Drop only months without data
    df_plot = df_plot[
        (df_plot["n_days"] > 0)
        & df_plot["p90"].notna()
        & df_plot["p99"].notna()
    ]

    if df_plot.empty:
        print(
            "[AVISO] No hay meses con datos suficientes "
            "para generar la gráfica de percentiles."
        )
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Monthly percentile series
    ax.plot(
        df_plot["month"],
        df_plot["p90"],
        marker="o",
        linewidth=1.8,
        label="p90"
    )

    ax.plot(
        df_plot["month"],
        df_plot["p99"],
        marker="o",
        linewidth=1.8,
        label="p99"
    )

    # Annotate months with few valid days
    low_coverage_months = df_plot[
        df_plot["n_days"] < min_valid_days
    ]

    for _, row in low_coverage_months.iterrows():
        annotation_y = max(row["p90"], row["p99"])

        ax.annotate(
            f"n_days={int(row['n_days'])}",
            xy=(row["month"], annotation_y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=plt.rcParams["font.size"],
            # light backing so a neighbouring line segment never runs through the label
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.0),
        )

    months = list(range(1, 13))
    month_names = [month_label(month) for month in months]

    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(bottom=0)

    ax.set_xlabel("Month")
    ax.set_ylabel(f"{index_label} percentile value [{index_unit}]")

    ax.set_title(
        f"Monthly variability of {index_label} percentiles — "
        f"{station} — {year}\n"
        f"p90 and p99 calculated from valid days"
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=250)

    if show_plot:
        plt.show()

    plt.close(fig)

    print(f"Gráfica mensual de percentiles guardada en {out_png}")

    if not low_coverage_months.empty:
        months_warning = [
            f"{month_label(int(row['month']))} "
            f"(n_days={int(row['n_days'])})"
            for _, row in low_coverage_months.iterrows()
        ]

        print(
            "[AVISO] Meses con menos de "
            f"{min_valid_days} días válidos: "
            + ", ".join(months_warning)
        )


# ======================================================================
# [F] SCIENTIFIC CALCULATION (no prints, no output files)
# ======================================================================
def run_ccdf_analysis(
    station: str = STATION,
    year: int = YEAR,
    doy_start: int = DOY_START,
    doy_end: int = DOY_END,
    th_cov: float = TH_COV,
    value_col: str = VALUE_COL,
    pcts: list[int] = PCTS,
    analysis_mode: str = ANALYSIS_MODE,
    months_to_analyze: list[int] = MONTHS_TO_ANALYZE,
    ndat_mode: str | None = None,
) -> dict:
    """
    Monthly percentiles and CCDFs of value_col for one station, year and
    DoY range. Pure calculation: every input is an argument, nothing is
    printed or written.

    Days: the Step 1 valid days (optionally restricted to
    months_to_analyze). Observations: with a ndat_mode, only the rows of
    those days that satisfy ndat_filter.apply_ndat_filter(); the valid
    days themselves do not depend on Ndat. ndat_mode=None applies no Ndat
    filter.

    Raises FileNotFoundError if the Step 0 Parquet or the Step 1 coverage
    CSV is missing, and ValueError if value_col is not scientifically
    supported (see validate_index_supported).

    Returns a dict with:
      - "index_config": INDEX_CONFIG entry of value_col;
      - "ndat_config": NDAT_MODES entry of ndat_mode, or None;
      - "valid_doys": DoYs used in the analysis;
      - "df_pcts": DataFrame with month, n_days, n_values, p{pct}...;
      - "ccdf_by_month": {month: (x_sorted, ccdf, n_values)};
      - "pcts_by_month": {month: {"p90": ..., "p99": ...}}.
    """
    index_config = validate_index_supported(value_col)
    parquet_value_col = index_config["parquet_column"]
    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode) if ndat_mode is not None else None

    paths = resolve_paths(station, year, doy_start, doy_end, th_cov, value_col, index_config, ndat_mode)

    if not paths["parquet_path"].exists():
        raise FileNotFoundError(
            f"Step 0 Parquet not found for station {station}: {paths['parquet_path']}\n"
            "Run 0-toParquet.py first."
        )

    if not paths["coverage_csv"].exists():
        raise FileNotFoundError(
            f"Step 1 coverage CSV not found: {paths['coverage_csv']}\n"
            "Run 1-completitud.py first."
        )

    df_station = pd.read_parquet(
        paths["parquet_path"],
        columns=["14_doy_utc", parquet_value_col, ndat_filter.NDAT_PARQUET_COLUMN]
    )
    if ndat_mode is not None:
        df_station = ndat_filter.apply_ndat_filter(df_station, ndat_mode)

    df_cov = pd.read_csv(paths["coverage_csv"])

    if "DoY" not in df_cov.columns or "status" not in df_cov.columns:
        raise ValueError("El CSV de completitud debe contener columnas: DoY, status")

    valid_doys = (
        df_cov[df_cov["status"] == "valid"]["DoY"]
        .astype(int)
        .tolist()
    )
    valid_doys = [d for d in valid_doys if doy_start <= d <= doy_end]

    if analysis_mode == "months":
        valid_doys = [
            d for d in valid_doys
            if doy_to_month(year, d) in months_to_analyze
        ]

    month_values, n_days_by_month = accumulate_values_by_month(
        df_station, valid_doys, year, parquet_value_col
    )

    rows = []
    ccdf_by_month = {}
    pcts_by_month = {}

    for m in range(1, 13):
        all_vals = month_values[m]
        n_days = n_days_by_month[m]

        if all_vals.size == 0:
            rows.append({
                "month": m,
                "n_days": 0,
                "n_values": 0,
                **{f"p{p}": np.nan for p in pcts}
            })
            continue

        n = int(all_vals.size)
        pvals = np.percentile(all_vals, pcts)

        row = {"month": m, "n_days": n_days, "n_values": n}
        pcts_by_month[m] = {}

        for p, v in zip(pcts, pvals):
            row[f"p{p}"] = float(v)
            pcts_by_month[m][f"p{p}"] = float(v)

        rows.append(row)

        x, ccdf = compute_ccdf(all_vals)
        if x is not None:
            ccdf_by_month[m] = (x, ccdf, n)

    df_pcts = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)

    return {
        "index_config": index_config,
        "ndat_config": ndat_config,
        "valid_doys": valid_doys,
        "df_pcts": df_pcts,
        "ccdf_by_month": ccdf_by_month,
        "pcts_by_month": pcts_by_month,
    }


# ======================================================================
# [G] SUMMARY (derived from df_pcts, no prints, no I/O)
# ======================================================================
def compute_ccdf_summary(
    df_pcts: pd.DataFrame,
    min_valid_days: int = MIN_VALID_DAYS_PER_MONTH,
) -> dict:
    """
    Comparison statistics derived from the monthly percentile table of
    run_ccdf_analysis(); no percentile is recomputed. Pure function.

    For p90 and p99 separately, the min/max month is taken only among
    months with n_days >= min_valid_days, because an extreme based on a
    few days is not reliable. The CSV and the plot still show all 12
    months; the exclusion applies only to this summary.

    If the raw min/max month (over all months) is excluded for low
    support, it is reported in {col}_min_excluded / {col}_max_excluded
    (month, value, n_days; None otherwise).

    If no month reaches min_valid_days, the raw values are used and
    {col}_all_months_low_support is set to True.
    """
    summary = {}
    reliable = df_pcts[df_pcts["n_days"] >= min_valid_days]

    for col in ("p90", "p99"):
        raw_min_row = df_pcts.loc[df_pcts[col].idxmin()]
        raw_max_row = df_pcts.loc[df_pcts[col].idxmax()]

        if reliable.empty:
            min_row, max_row = raw_min_row, raw_max_row
            summary[f"{col}_all_months_low_support"] = True
        else:
            min_row = reliable.loc[reliable[col].idxmin()]
            max_row = reliable.loc[reliable[col].idxmax()]
            summary[f"{col}_all_months_low_support"] = False

        summary[f"{col}_min_month"] = month_label(int(min_row["month"]))
        summary[f"{col}_min_value"] = float(min_row[col])
        summary[f"{col}_min_n_days"] = int(min_row["n_days"])

        summary[f"{col}_max_month"] = month_label(int(max_row["month"]))
        summary[f"{col}_max_value"] = float(max_row[col])
        summary[f"{col}_max_n_days"] = int(max_row["n_days"])

        summary[f"{col}_range"] = float(max_row[col] - min_row[col])

        summary[f"{col}_min_excluded"] = None
        if not reliable.empty and int(raw_min_row["month"]) != int(min_row["month"]):
            summary[f"{col}_min_excluded"] = {
                "month": month_label(int(raw_min_row["month"])),
                "value": float(raw_min_row[col]),
                "n_days": int(raw_min_row["n_days"]),
            }

        summary[f"{col}_max_excluded"] = None
        if not reliable.empty and int(raw_max_row["month"]) != int(max_row["month"]):
            summary[f"{col}_max_excluded"] = {
                "month": month_label(int(raw_max_row["month"])),
                "value": float(raw_max_row[col]),
                "n_days": int(raw_max_row["n_days"]),
            }

    return summary


# ======================================================================
# [H] MAIN (console mode): calculation + CSV/PNG outputs + console report
# ======================================================================
def main() -> None:
    results = run_ccdf_analysis(
        station=STATION,
        year=YEAR,
        doy_start=DOY_START,
        doy_end=DOY_END,
        th_cov=TH_COV,
        value_col=VALUE_COL,
        pcts=PCTS,
        analysis_mode=ANALYSIS_MODE,
        months_to_analyze=MONTHS_TO_ANALYZE,
        ndat_mode=NDAT_MODE,
    )

    index_config = results["index_config"]
    index_label = index_config["label"]
    index_unit = index_config["unit"]
    valid_doys = results["valid_doys"]
    df_pcts = results["df_pcts"]
    ccdf_by_month = results["ccdf_by_month"]
    pcts_by_month = results["pcts_by_month"]

    print(f"\nDías válidos para el análisis (según Paso 1 + filtro Paso 2): {len(valid_doys)}")
    if ANALYSIS_MODE == "months":
        print(f"Meses analizados (MONTHS_TO_ANALYZE): {MONTHS_TO_ANALYZE}")
    else:
        print("Meses analizados: todos los meses disponibles del año")

    print("\nValores acumulados por mes (n_days = días usados):")
    for _, row in df_pcts.iterrows():
        print(f"  Month {int(row['month']):02d}: {int(row['n_days'])}")

    paths = resolve_paths(STATION, YEAR, DOY_START, DOY_END, TH_COV, VALUE_COL, index_config, NDAT_MODE)
    paths["index_dir"].mkdir(parents=True, exist_ok=True)
    paths["ccdf_dir"].mkdir(parents=True, exist_ok=True)
    paths["ccdf_comparisons_dir"].mkdir(parents=True, exist_ok=True)
    paths["percentile_dir"].mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Single-month CCDFs
    # ------------------------------------------------------------------
    if SAVE_INDIVIDUAL_CCDF:
        for m in sorted(ccdf_by_month):
            x, ccdf, _ = ccdf_by_month[m]
            out_png = individual_ccdf_png(paths, DOY_START, DOY_END, m)
            save_ccdf_plot(
                m, x, ccdf, out_png, STATION, YEAR, index_label, index_unit,
                use_log_y=USE_LOG_Y, x_lim=X_LIM, y_lim=Y_LIM, show_plot=SHOW_PLOTS
            )
            print(f"CCDF individual guardada en {out_png}")

    # ------------------------------------------------------------------
    # Multi-month CCDF comparison
    # ------------------------------------------------------------------
    if SAVE_COMPARISON_CCDF:
        out_multi_png = multi_ccdf_png(paths, DOY_START, DOY_END, MONTHS_TO_COMPARE)

        save_multi_ccdf_plot(
            ccdf_by_month=ccdf_by_month,
            pcts_by_month=pcts_by_month,
            months_to_compare=MONTHS_TO_COMPARE,
            out_png=out_multi_png,
            station=STATION,
            year=YEAR,
            index_label=index_label,
            index_unit=index_unit,
            use_log_y=USE_LOG_Y, x_lim=X_LIM, y_lim=Y_LIM,
            show_plot=SHOW_PLOTS
        )

        print(f"\nCCDF comparativa guardada en {out_multi_png}")

    # ------------------------------------------------------------------
    # Monthly percentile table
    # ------------------------------------------------------------------
    print("\nTabla mensual de percentiles:")
    print(df_pcts)

    df_pcts.to_csv(paths["output_pcts_csv"], index=False)
    print(f"\nPercentiles guardados en {paths['output_pcts_csv']}")

    # ------------------------------------------------------------------
    # Monthly p90/p99 variability plot
    # ------------------------------------------------------------------
    save_monthly_percentiles_plot(
        df_pcts=df_pcts,
        out_png=paths["output_monthly_percentiles_png"],
        station=STATION,
        year=YEAR,
        index_label=index_label,
        index_unit=index_unit,
        min_valid_days=MIN_VALID_DAYS_PER_MONTH,
        show_plot=SHOW_PLOTS
    )


if __name__ == "__main__":
    main()
