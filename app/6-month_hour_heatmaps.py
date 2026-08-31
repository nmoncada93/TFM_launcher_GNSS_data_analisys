#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 6 - Month-hour heatmaps of threshold exceedances.

Purpose
-------
This script analyses when annual threshold exceedances occur as a
function of month and local hour.

Reads from the Step 0 Parquet, the Step 1 coverage CSV and the Step 3
annual-thresholds CSV, following the same philosophy as Steps 1-5
(CLAUDE.md section 11). Filters to the Step 1 valid days itself, same
reason as Step 5: the Step 0 Parquet contains all 366 days, not only the
valid ones, unlike the Step 4 "prepared" Parquet this script used to read
(which no longer exists - see Steps 3-4).

Also drops the "fall back to month_utc if month_local is missing" branch
and the candidate-path lookups: both existed to tolerate uncertainty from
earlier, inconsistent output-naming iterations. The Step 0 Parquet always
has 19_month_local and deterministic paths, so neither case can happen
anymore.

Scientific scope: only ROTI L1 (roti_l1) is implemented/validated - see
INDEX_CONFIG / validate_index_supported() (CLAUDE.md section 3-bis).

Main outputs
------------
1) CSV table with month-hour exceedance frequencies.
2) Heatmap of p90 exceedance frequency by month and local hour.
3) Heatmap of p99 exceedance frequency by month and local hour.
4) Optional support heatmap showing the number of observations per cell.

The month-hour calculation and the plotting code are otherwise unchanged
from the validated version.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ndat_filter


# ======================================================================
# [A] STUDY CONFIGURATION
# ======================================================================

# [A.1] Study case
STATION = "YELL"

# Used only by get_cross_station_maxima() (cross-station summary view, not a
# Step 6.2 comparator - Step 6 has none). STATION above remains the single-
# station default for main()/console mode, unchanged. No shared "stations"
# module exists in this project (verified: 2.2/3.2/4.2/5.2 each keep their
# own independent copy of this exact list) - this mirrors that established
# convention rather than introducing a new shared dependency.
STATIONS = ["UNSA", "KOUG", "WHIT", "YELL"]

YEAR = 2024
DOY_START = 1
DOY_END = 366

# Índice a analizar. Solo "roti_l1" está científicamente
# implementado/validado - ver INDEX_CONFIG.
VALUE_COL = "roti_l1"

# Criterio Ndat. IMPORTANTE: a diferencia de 2-ccdf.py / 3-temporal_variability.py,
# aquí el default NO es ndat_filter.NDAT_DEFAULT_MODE ("eq60"). main() no tiene
# función pura equivalente (CLAUDE.md sección 11, "adapt only") y web_server.py
# llama a main() directamente vía _patched_globals sin parchear NDAT_MODE - si el
# default fuera "eq60", la web empezaría a exigir/usar Ndat=60 sin que nadie lo
# pidiera. NDAT_MODE=None aquí es una solución TEMPORAL de compatibilidad con la
# web legacy, no el default científico del TFM - los outputs oficiales se generan
# explícitamente con ndat_mode="eq60" (vía _patched_globals o edición manual de
# esta variable). Revisar/eliminar esta excepción cuando la web se migre a un
# selector de Ndat. Mismo criterio ya aplicado en 4-hourly_variability.py /
# 5-daypart_variability.py.
NDAT_MODE = None

TH_COV = 0.75

# How many top cells (by frequency_p90_pct / frequency_p99_pct) to save and
# print - same role as TOP_N_DAYS in 3.2-temporal_comparison.py.
TOP_N_CELLS = 10

# [A.2] Month-hour grouping columns
# Default: month_local × hour_local, para ser consistente con los Pasos
# 4 y 5. Ambas columnas vienen siempre del Parquet del Paso 0.
MONTH_COL = "month_local"
HOUR_COL = "hour_local"
DAY_COUNT_COL = "doy_utc"

# Nombres reales de esas mismas columnas en el Parquet del Paso 0,
# mapeados a los nombres cortos de arriba al cargar los datos.
DAY_COUNT_COL_RAW = "14_doy_utc"
MONTH_COL_RAW = "19_month_local"
HOUR_COL_RAW = "18_hour_local"

# [A.3] Plot settings
SHOW_PLOTS = False
PLOT_DPI = 250
SAVE_SUPPORT_HEATMAP = True

MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# ======================================================================
# [B] ÍNDICES: PARQUET <-> NOMBRE DE FICHERO <-> ETIQUETA CIENTÍFICA
# ======================================================================
# Misma fuente única de verdad que en 2-ccdf.py / 3-temporal_variability.py
# / 4-hourly_variability.py / 5-daypart_variability.py (CLAUDE.md §3-bis).
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
        "unit": "",  # S4 es adimensional
        "scientifically_supported": False,
    },
    "s4_l2": {
        "parquet_column": "11_s4_l2",
        "file_tag": "S4L2",
        "label": "S4 L2",
        "unit": "",  # S4 es adimensional
        "scientifically_supported": False,
    },
}


def validate_index_supported(value_col: str) -> dict:
    """
    Comprueba que value_col tiene un análisis científico implementado y
    validado - no solo que la columna exista en el Parquet.
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
            "scientific analysis has not yet been implemented/validated."
        )

    return config


# ======================================================================
# [C] RESOLUCIÓN DE RUTAS
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
    Resuelve rutas de entrada/salida a partir de los parámetros recibidos.

    ndat_mode=None (default) reproduce exactamente la ruta/prefix de antes de
    que este script conociera Ndat, tanto para lo que escribe Step 6 como para
    dónde busca el Step 3 del que depende - ningún caller existente que no
    pase este argumento (hoy, web_server.py y main() con su default) ve ningún
    cambio de ruta. Un modo real ("eq60"/"ge30"/"all"/"lt30") se aplica a la
    vez a ambas resoluciones (lectura de Step 3 y escritura de Step 6) con el
    mismo ndat_config - mismo mecanismo ya validado en
    3-temporal_variability.py / 4-hourly_variability.py /
    5-daypart_variability.py::resolve_paths().
    """
    parquet_path = (
        Path("results") / station / str(year) / "0_parquet"
        / f"0_{station}_{year}_observations.parquet"
    )
    coverage_csv = (
        Path("results") / station / str(year) / "1_completeness"
        / f"coverage_{station}_{year}_DOY{doy_start}_{doy_end}_coverageTH{th_cov}.csv"
    )

    doy_tag = f"DOY{doy_start}_{doy_end}"
    step3_index_dir = Path("results") / station / str(year) / "3_temporal_variability" / value_col
    index_dir = Path("results") / station / str(year) / "6_month_hour_heatmaps" / value_col

    if ndat_mode is None:
        step3_dir = step3_index_dir
        step3_prefix = f"3-{station}_{index_config['file_tag']}_{year}"
        out_dir = index_dir
        prefix = f"6-{station}_{index_config['file_tag']}_{year}"
    else:
        ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
        step3_dir = step3_index_dir / ndat_config["dir_tag"]
        step3_prefix = f"3-{station}_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"
        out_dir = index_dir / ndat_config["dir_tag"]
        prefix = f"6-{station}_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"

    thresholds_csv = step3_dir / f"{step3_prefix}_{doy_tag}_annual_thresholds.csv"

    return {
        "parquet_path": parquet_path,
        "coverage_csv": coverage_csv,
        "thresholds_csv": thresholds_csv,
        "index_dir": index_dir,
        "ndat_dir": out_dir,
        "output_month_hour_csv": out_dir / f"{prefix}_{doy_tag}_month_hour_frequency.csv",
        "output_heatmap_p90": out_dir / f"{prefix}_{doy_tag}_heatmap_p90.png",
        "output_heatmap_p99": out_dir / f"{prefix}_{doy_tag}_heatmap_p99.png",
        "output_heatmap_nvalues": out_dir / f"{prefix}_{doy_tag}_heatmap_nvalues.png",
        "output_top_cells_p90_csv": out_dir / f"{prefix}_{doy_tag}_top{TOP_N_CELLS}_cells_p90.csv",
        "output_top_cells_p99_csv": out_dir / f"{prefix}_{doy_tag}_top{TOP_N_CELLS}_cells_p99.csv",
        "prefix": prefix,
    }


# ======================================================================
# [D] DATA LOADING
# ======================================================================

def load_valid_days(coverage_csv: Path, doy_start: int, doy_end: int) -> pd.DataFrame:
    """Load valid days selected in Step 1."""
    if not coverage_csv.exists():
        raise FileNotFoundError(f"Coverage CSV not found: {coverage_csv}")

    df = pd.read_csv(coverage_csv)
    required = {"DoY", "coverage", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Coverage CSV is missing columns: {sorted(missing)}")

    df["DoY"] = pd.to_numeric(df["DoY"], errors="coerce")
    df = df.dropna(subset=["DoY"])
    df["DoY"] = df["DoY"].astype(int)

    df = df[
        (df["status"] == "valid")
        & df["DoY"].between(doy_start, doy_end)
    ]

    df = df.sort_values("DoY").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid days were found in the selected range.")
    return df


def load_thresholds(thresholds_csv: Path) -> tuple[float, float, dict]:
    """Loads annual p90 and p99 thresholds from Step 3."""
    if not thresholds_csv.exists():
        raise FileNotFoundError(f"Thresholds CSV not found: {thresholds_csv}")

    df = pd.read_csv(thresholds_csv)

    required_columns = {"p90_annual", "p99_annual"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Thresholds CSV is missing columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Thresholds CSV is empty.")

    row = df.iloc[0].to_dict()

    p90_annual = float(row["p90_annual"])
    p99_annual = float(row["p99_annual"])

    if not np.isfinite(p90_annual) or not np.isfinite(p99_annual):
        raise ValueError("Annual thresholds must be finite numeric values.")

    if p90_annual > p99_annual:
        raise ValueError("Annual p90 threshold cannot exceed annual p99.")

    return p90_annual, p99_annual, row


def load_prepared_dataset(
    parquet_path: Path,
    parquet_value_col: str,
    value_col: str,
    valid_doys,
    ndat_mode: str | None = None,
) -> pd.DataFrame:
    """
    Loads doy_utc/month_local/hour_local/value_col from the Step 0
    Parquet, filtered to the valid days from Step 1.

    Unlike the previous version (which read an already-filtered "prepared"
    Parquet from Step 4, with a fallback to month_utc if month_local was
    missing), the Step 0 Parquet always has month_local and contains all
    366 days - so the valid-day filter now happens here explicitly, and
    the fallback branch is no longer needed.

    ndat_mode=None (default) is the legacy behaviour: no row is discarded
    by Ndat. With a real mode, observations are restricted via
    ndat_filter.apply_ndat_filter() right after the Step 1 valid-day
    intersection and before anything else (same point already used in
    3-temporal_variability.py / 4-hourly_variability.py /
    5-daypart_variability.py). The Ndat column is dropped unconditionally
    right after (not only inside the `if`) so it never reaches the
    returned DataFrame in any mode, including None - same discipline as
    4-hourly_variability.py, needed here because this function (unlike
    Step 5's) does not end in a narrow column selection.
    """
    df = pd.read_parquet(
        parquet_path,
        columns=[DAY_COUNT_COL_RAW, MONTH_COL_RAW, HOUR_COL_RAW, parquet_value_col, ndat_filter.NDAT_PARQUET_COLUMN],
    )

    df = df[df[DAY_COUNT_COL_RAW].isin(valid_doys)].copy()

    if ndat_mode is not None:
        df = ndat_filter.apply_ndat_filter(df, ndat_mode)
    df = df.drop(columns=[ndat_filter.NDAT_PARQUET_COLUMN])

    df = df.rename(columns={
        DAY_COUNT_COL_RAW: DAY_COUNT_COL,
        MONTH_COL_RAW: MONTH_COL,
        HOUR_COL_RAW: HOUR_COL,
        parquet_value_col: value_col,
    })

    for column in [DAY_COUNT_COL, MONTH_COL, HOUR_COL, value_col]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[DAY_COUNT_COL, MONTH_COL, HOUR_COL, value_col]).copy()

    df[DAY_COUNT_COL] = df[DAY_COUNT_COL].astype(int)
    df[MONTH_COL] = df[MONTH_COL].astype(int)
    df[HOUR_COL] = df[HOUR_COL].astype(int)

    if df.empty:
        raise ValueError("No usable observations found for the selected valid days.")

    invalid_months = df[~df[MONTH_COL].between(1, 12)]
    if not invalid_months.empty:
        raise ValueError(f"Prepared dataset contains invalid {MONTH_COL} values.")

    invalid_hours = df[~df[HOUR_COL].between(0, 23)]
    if not invalid_hours.empty:
        raise ValueError(f"Prepared dataset contains invalid {HOUR_COL} values.")

    return df


# ======================================================================
# [E] COMPUTATION (lógica sin cambios; value_col ahora es un parámetro
# en vez de leer VALUE_COL directamente)
# ======================================================================

def build_complete_month_hour_grid() -> pd.DataFrame:
    """Builds a complete 12 × 24 month-hour grid."""
    return pd.MultiIndex.from_product(
        [range(1, 13), range(0, 24)],
        names=[MONTH_COL, HOUR_COL],
    ).to_frame(index=False)


def compute_month_hour_frequency(
    df: pd.DataFrame,
    value_col: str,
    p90_annual: float,
    p99_annual: float,
) -> pd.DataFrame:
    """Computes threshold exceedance frequencies by month and local hour."""
    df_work = df.copy()

    df_work["above_p90"] = df_work[value_col] >= p90_annual
    df_work["above_p99"] = df_work[value_col] >= p99_annual

    grouped = (
        df_work
        .groupby([MONTH_COL, HOUR_COL], as_index=False)
        .agg(
            n_days=(DAY_COUNT_COL, "nunique"),
            n_values=(value_col, "size"),
            n_at_or_above_p90=("above_p90", "sum"),
            n_at_or_above_p99=("above_p99", "sum"),
        )
    )

    grouped["n_days"] = grouped["n_days"].astype(int)
    grouped["n_values"] = grouped["n_values"].astype(int)
    grouped["n_at_or_above_p90"] = grouped["n_at_or_above_p90"].astype(int)
    grouped["n_at_or_above_p99"] = grouped["n_at_or_above_p99"].astype(int)

    grid = build_complete_month_hour_grid()

    result = (
        grid
        .merge(grouped, on=[MONTH_COL, HOUR_COL], how="left")
        .sort_values([MONTH_COL, HOUR_COL])
        .reset_index(drop=True)
    )

    count_columns = ["n_days", "n_values", "n_at_or_above_p90", "n_at_or_above_p99"]
    result[count_columns] = result[count_columns].fillna(0).astype(int)

    # Frequencies are set to NaN when there are no observations in the cell.
    # This is preferable to reporting 0 %, because an empty cell is not
    # evidence of low activity.
    result["frequency_p90_pct"] = np.where(
        result["n_values"] > 0,
        100.0 * result["n_at_or_above_p90"] / result["n_values"],
        np.nan,
    )

    result["frequency_p99_pct"] = np.where(
        result["n_values"] > 0,
        100.0 * result["n_at_or_above_p99"] / result["n_values"],
        np.nan,
    )

    return result


# ======================================================================
# [F] VALIDATION (lógica sin cambios)
# ======================================================================

def validate_month_hour_results(
    df: pd.DataFrame,
    df_month_hour: pd.DataFrame,
    threshold_row: dict,
) -> None:
    """Validates month-hour aggregation against the prepared dataset."""
    expected_rows = 12 * 24

    if len(df_month_hour) != expected_rows:
        raise ValueError(
            f"Month-hour table should contain {expected_rows} rows, "
            f"but it contains {len(df_month_hour)} rows."
        )

    total_prepared = int(len(df))
    total_month_hour = int(df_month_hour["n_values"].sum())

    if total_prepared != total_month_hour:
        raise ValueError(
            "Month-hour sample total does not match prepared dataset rows. "
            f"Prepared={total_prepared}, month-hour={total_month_hour}"
        )

    if not (df_month_hour["n_at_or_above_p99"] <= df_month_hour["n_at_or_above_p90"]).all():
        raise ValueError("p99 exceedance count cannot exceed p90 exceedance count.")

    if not (df_month_hour["n_at_or_above_p90"] <= df_month_hour["n_values"]).all():
        raise ValueError("p90 exceedance count cannot exceed total samples.")

    for column in ["frequency_p90_pct", "frequency_p99_pct"]:
        non_null = df_month_hour[column].dropna()
        if not non_null.between(0, 100).all():
            raise ValueError(f"{column} contains values outside the [0, 100] range.")

    month_values = set(df_month_hour[MONTH_COL].unique())
    hour_values = set(df_month_hour[HOUR_COL].unique())

    if month_values != set(range(1, 13)):
        raise ValueError(f"Month-hour table does not contain all months 1-12: {month_values}")

    if hour_values != set(range(0, 24)):
        raise ValueError(f"Month-hour table does not contain all hours 0-23: {hour_values}")

    if "n_values" in threshold_row:
        expected_n_values = int(threshold_row["n_values"])
        if expected_n_values != total_prepared:
            raise ValueError(
                "Prepared dataset row count does not match Step 3 annual total. "
                f"Prepared={total_prepared}, Step3={expected_n_values}"
            )

    if "n_at_or_above_p90" in threshold_row:
        expected_p90 = int(threshold_row["n_at_or_above_p90"])
        observed_p90 = int(df_month_hour["n_at_or_above_p90"].sum())
        if expected_p90 != observed_p90:
            raise ValueError(
                "p90 exceedance total does not match Step 3. "
                f"Observed={observed_p90}, Step3={expected_p90}"
            )

    if "n_at_or_above_p99" in threshold_row:
        expected_p99 = int(threshold_row["n_at_or_above_p99"])
        observed_p99 = int(df_month_hour["n_at_or_above_p99"].sum())
        if expected_p99 != observed_p99:
            raise ValueError(
                "p99 exceedance total does not match Step 3. "
                f"Observed={observed_p99}, Step3={expected_p99}"
            )


# ======================================================================
# [G] PLOTTING HELPERS (lógica de dibujo sin cambios)
# ======================================================================

def make_heatmap_matrix(df_month_hour: pd.DataFrame, value_column: str) -> np.ndarray:
    """
    Converts the month-hour table into a 12 × 24 matrix.
    Rows represent months 1-12. Columns represent local hours 0-23.
    """
    pivot = (
        df_month_hour
        .pivot(index=MONTH_COL, columns=HOUR_COL, values=value_column)
        .reindex(index=range(1, 13), columns=range(0, 24))
    )
    return pivot.to_numpy(dtype=float)


def cells_at_max(matrix: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    """
    All (month, hour) cells tied for the maximum of a month-hour matrix -
    np.argwhere(...)[0] alone would silently keep only the first one,
    hiding a genuine tie (e.g. two cells both at exactly 100% because each
    has a single observation that happens to exceed the threshold). Same
    role as _hours_at_max()/_dayparts_at_max() elsewhere in this project,
    adapted to a (month, hour) matrix instead of a DataFrame column.
    """
    max_value = float(np.nanmax(matrix))
    positions = np.argwhere(matrix == max_value)
    cells = [(int(row_idx + 1), int(col_idx)) for row_idx, col_idx in positions]
    return cells, max_value


def save_month_hour_heatmap(
    df_month_hour: pd.DataFrame,
    value_column: str,
    out_png: Path,
    title: str,
    colorbar_label: str,
    value_format: str | None = None,
    plot_dpi: int = PLOT_DPI,
    show_plot: bool = SHOW_PLOTS,
) -> None:
    """
    Saves a month-hour heatmap. The default matplotlib color map is used
    intentionally, so the plot remains simple and consistent without
    hard-coded colors.
    """
    matrix = make_heatmap_matrix(df_month_hour=df_month_hour, value_column=value_column)

    fig, ax = plt.subplots(figsize=(12, 6.9))

    image = ax.imshow(matrix, aspect="auto", origin="upper")

    ax.set_title(title, pad=12)
    ax.set_xlabel("Local hour")
    ax.set_ylabel("Month")

    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([str(h) for h in range(0, 24)])

    ax.set_yticks(range(0, 12))
    ax.set_yticklabels(MONTH_LABELS)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)

    finite_values = matrix[np.isfinite(matrix)]

    if finite_values.size > 0:
        tied_cells, max_value = cells_at_max(matrix)
        formatted_max = f"{max_value:.3f}" if value_format is None else value_format.format(max_value)

        if len(tied_cells) == 1:
            month, hour = tied_cells[0]
            location_text = f"month {month}, local hour {hour:02d}:00"
        elif len(tied_cells) <= 5:
            cells_text = ", ".join(f"{m}/{h:02d}:00" for m, h in tied_cells)
            location_text = f"{len(tied_cells)} tied cells (month/hour): {cells_text}"
        else:
            location_text = f"{len(tied_cells)} tied cells (see top-cells CSV for the full list)"

        # Figure-level caption below the x-axis label, not above the
        # title (where it used to collide with 2-line titles) - figure
        # coordinates so its position does not depend on how many lines
        # the title has. Purely a layout change: same max-selection logic
        # (cells_at_max), same data, same colorbar scale.
        fig.text(
            0.5, 0.01,
            f"Maximum: {formatted_max} at {location_text}",
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout(rect=(0.0, 0.055, 1.0, 1.0))
    fig.savefig(out_png, dpi=plot_dpi)

    if show_plot:
        plt.show()

    plt.close(fig)


# ======================================================================
# [H] SUMMARY HELPERS (lógica sin cambios)
# ======================================================================

def get_top_cells(df_month_hour: pd.DataFrame, value_column: str, top_n: int = TOP_N_CELLS) -> pd.DataFrame:
    """
    Top N month-hour cells by value_column, with an explicit rank column -
    same sort/selection logic print_top_cells() already used (unchanged
    methodology), just made reusable so the result can be saved to CSV,
    not only printed. kind="stable" makes tie-breaking deterministic
    (preserves month/hour order for equal values), same choice already
    made for the equivalent sort in 3.2-temporal_comparison.py.
    """
    top = (
        df_month_hour
        .dropna(subset=[value_column])
        .sort_values(value_column, ascending=False, kind="stable")
        .head(top_n)
        .reset_index(drop=True)
    )
    top.insert(0, "rank", range(1, len(top) + 1))
    return top


def print_top_cells(df_top: pd.DataFrame, label: str, top_n: int) -> None:
    """Prints an already-computed top-N cells table (see get_top_cells())."""
    print(f"\nTop {top_n} cells for {label}")
    print("-" * (len(f"Top {top_n} cells for {label}")))

    columns_to_show = [
        "rank", MONTH_COL, HOUR_COL, "n_days", "n_values",
        "n_at_or_above_p90", "n_at_or_above_p99",
        "frequency_p90_pct", "frequency_p99_pct",
    ]

    print(df_top[columns_to_show].to_string(index=False))


def get_cross_station_maxima(
    stations: list[str],
    year: int,
    doy_start: int,
    doy_end: int,
    th_cov: float,
    value_col: str,
    index_config: dict,
    ndat_mode: str | None = None,
) -> dict:
    """
    Rank-1 row of the Top-10 p90/p99 CSVs main() already saved for each
    station (get_top_cells()) - reads existing output only, never touches
    the Parquet, never recomputes anything. Same "a missing station is not
    fatal" convention as 2.2/3.2/4.2/5.2's comparators: a station without
    Step 6 output for this exact year/DOY/th_cov/ndat_mode combination (or
    whose CSV, for any reason, does not contain a rank=1 row) is reported
    in missing_stations instead of raising.

    The returned table preserves the order of `stations` as given (not
    resorted alphabetically) - callers that want UNSA/KOUG/WHIT/YELL order
    simply pass the list in that order.
    """
    rows: list[dict] = []
    missing_stations: dict[str, str] = {}

    for station in stations:
        paths = resolve_paths(
            station, year, doy_start, doy_end, th_cov, value_col, index_config, ndat_mode
        )
        p90_csv = paths["output_top_cells_p90_csv"]
        p99_csv = paths["output_top_cells_p99_csv"]

        if not p90_csv.exists() or not p99_csv.exists():
            missing_csv = p90_csv if not p90_csv.exists() else p99_csv
            missing_stations[station] = f"Step 6 output not found: {missing_csv}"
            continue

        p90_rank1 = pd.read_csv(p90_csv).loc[lambda df: df["rank"] == 1]
        p99_rank1 = pd.read_csv(p99_csv).loc[lambda df: df["rank"] == 1]

        if p90_rank1.empty or p99_rank1.empty:
            missing_stations[station] = (
                f"Step 6 Top-10 CSV for {station} has no rank=1 row "
                f"(p90_csv={p90_csv}, p99_csv={p99_csv})"
            )
            continue

        max_p90 = p90_rank1.iloc[0]
        max_p99 = p99_rank1.iloc[0]

        rows.append({
            "station": station,
            "max_p90_month_local": int(max_p90["month_local"]),
            "max_p90_hour_local": int(max_p90["hour_local"]),
            "max_p90_freq_pct": float(max_p90["frequency_p90_pct"]),
            "max_p90_n_days": int(max_p90["n_days"]),
            "max_p90_n_values": int(max_p90["n_values"]),
            "max_p99_month_local": int(max_p99["month_local"]),
            "max_p99_hour_local": int(max_p99["hour_local"]),
            "max_p99_freq_pct": float(max_p99["frequency_p99_pct"]),
            "max_p99_n_days": int(max_p99["n_days"]),
            "max_p99_n_values": int(max_p99["n_values"]),
        })

    if not rows:
        ndat_note = f", ndat_mode='{ndat_mode}'" if ndat_mode is not None else ""
        raise FileNotFoundError(
            f"No station has usable Step 6 output for value_col='{value_col}'{ndat_note}, "
            f"year={year}, DOY{doy_start}_{doy_end}. Run 6-month_hour_heatmaps.py "
            "for at least one station first."
        )

    table = pd.DataFrame(rows)  # order of `stations` preserved, not resorted

    return {
        "available_stations": sorted(r["station"] for r in rows),
        "missing_stations": missing_stations,
        "table": table,
    }


# ======================================================================
# [I] MAIN PROGRAM
# ======================================================================

def main() -> None:
    index_config = validate_index_supported(VALUE_COL)
    parquet_value_col = index_config["parquet_column"]
    index_label = index_config["label"]
    index_unit = index_config["unit"]

    paths = resolve_paths(STATION, YEAR, DOY_START, DOY_END, TH_COV, VALUE_COL, index_config, NDAT_MODE)
    paths["ndat_dir"].mkdir(parents=True, exist_ok=True)

    print("Step 6 — Month-hour heatmaps of threshold exceedances")
    print("=======================================================")
    print(f"Station: {STATION}")
    print(f"Year: {YEAR}")
    print(f"Grouping: {MONTH_COL} × {HOUR_COL}")
    print(f"Step 0 Parquet: {paths['parquet_path']}")
    print(f"Coverage CSV: {paths['coverage_csv']}")
    print(f"Annual thresholds CSV: {paths['thresholds_csv']}")

    if not paths["parquet_path"].exists():
        raise FileNotFoundError(
            f"Step 0 Parquet not found for station {STATION}: {paths['parquet_path']}\n"
            "Run 0-toParquet.py first."
        )

    valid_days = load_valid_days(paths["coverage_csv"], DOY_START, DOY_END)
    p90_annual, p99_annual, threshold_row = load_thresholds(paths["thresholds_csv"])

    df_prepared = load_prepared_dataset(
        paths["parquet_path"], parquet_value_col, VALUE_COL, set(valid_days["DoY"]), NDAT_MODE
    )

    df_month_hour = compute_month_hour_frequency(
        df=df_prepared, value_col=VALUE_COL, p90_annual=p90_annual, p99_annual=p99_annual,
    )

    validate_month_hour_results(
        df=df_prepared, df_month_hour=df_month_hour, threshold_row=threshold_row,
    )

    df_month_hour.to_csv(paths["output_month_hour_csv"], index=False)

    save_month_hour_heatmap(
        df_month_hour=df_month_hour,
        value_column="frequency_p90_pct",
        out_png=paths["output_heatmap_p90"],
        title=(
            f"Month-hour {index_label} p90 exceedance frequency — {STATION} — {YEAR}\n"
            f"{index_label} ≥ T90 = {p90_annual:.3f} {index_unit}"
        ),
        colorbar_label="p90 exceedance frequency [%]",
        value_format="{:.2f}%",
    )

    save_month_hour_heatmap(
        df_month_hour=df_month_hour,
        value_column="frequency_p99_pct",
        out_png=paths["output_heatmap_p99"],
        title=(
            f"Month-hour {index_label} p99 exceedance frequency — {STATION} — {YEAR}\n"
            f"{index_label} ≥ T99 = {p99_annual:.3f} {index_unit}"
        ),
        colorbar_label="p99 exceedance frequency [%]",
        value_format="{:.2f}%",
    )

    if SAVE_SUPPORT_HEATMAP:
        save_month_hour_heatmap(
            df_month_hour=df_month_hour,
            value_column="n_values",
            out_png=paths["output_heatmap_nvalues"],
            title=(
                f"Month-hour observation support — {STATION} — {YEAR}\n"
                "Number of observations per cell"
            ),
            colorbar_label="Number of observations",
            value_format="{:.0f}",
        )

    print("\nValidation summary")
    print("------------------")
    print(f"Valid days selected from Step 1: {len(valid_days)}")
    print(f"Prepared dataset rows: {len(df_prepared):,}")
    print(f"Month-hour sample total: {int(df_month_hour['n_values'].sum()):,}")
    print(f"p90 exceedance total: {int(df_month_hour['n_at_or_above_p90'].sum()):,}")
    print(f"p99 exceedance total: {int(df_month_hour['n_at_or_above_p99'].sum()):,}")
    print("Month-hour totals match the validated Step 3 annual totals.")
    print("Month-hour table contains 12 × 24 = 288 rows.")

    top_p90 = get_top_cells(df_month_hour, "frequency_p90_pct", TOP_N_CELLS)
    top_p99 = get_top_cells(df_month_hour, "frequency_p99_pct", TOP_N_CELLS)
    top_p90.to_csv(paths["output_top_cells_p90_csv"], index=False)
    top_p99.to_csv(paths["output_top_cells_p99_csv"], index=False)

    print_top_cells(top_p90, "p90 exceedance frequency", TOP_N_CELLS)
    print_top_cells(top_p99, "p99 exceedance frequency", TOP_N_CELLS)

    print("\nOutput files")
    print("------------")
    print(f"Month-hour frequency CSV: {paths['output_month_hour_csv']}")
    print(f"p90 heatmap: {paths['output_heatmap_p90']}")
    print(f"p99 heatmap: {paths['output_heatmap_p99']}")
    if SAVE_SUPPORT_HEATMAP:
        print(f"Observation-support heatmap: {paths['output_heatmap_nvalues']}")
    print(f"Top {TOP_N_CELLS} cells CSV (p90): {paths['output_top_cells_p90_csv']}")
    print(f"Top {TOP_N_CELLS} cells CSV (p99): {paths['output_top_cells_p99_csv']}")

    print("\nStep 6 completed successfully.")


if __name__ == "__main__":
    main()
