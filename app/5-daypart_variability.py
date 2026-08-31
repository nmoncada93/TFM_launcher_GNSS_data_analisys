#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 5 - Daypart variability of threshold exceedances.

Reads from the Step 0 Parquet, the Step 1 coverage CSV and the Step 3
annual-thresholds CSV, following the same philosophy as Steps 1-4
(CLAUDE.md section 11): global config as defaults, main() for console
mode, import-safe.

Reads row-level hour_local/value data directly from the Step 0 Parquet
instead of the Step 4 "prepared" Parquet (which no longer exists - see
Steps 3-4). Because the Step 0 Parquet contains all 366 days, not only
the valid ones, this script now filters to the Step 1 valid days itself
before computing anything - the previous version relied on that filtering
having already happened upstream (in Step 3/4's now-removed prepared
Parquet), so it never needed to do it here.

Scientific scope: only ROTI L1 (roti_l1) is implemented/validated - see
INDEX_CONFIG / validate_index_supported() (CLAUDE.md section 3-bis).

The daypart-frequency calculation and the plotting code are otherwise
unchanged from the validated version.
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
STATION = "WHIT"
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
# selector de Ndat. Mismo criterio ya aplicado en 4-hourly_variability.py.
NDAT_MODE = None

TH_COV = 0.75

PERCENTILE_HIGH = 90
PERCENTILE_EXTREME = 99

# [A.2] Daypart definition
# Intervals use local integer hours and follow the convention:
#   start_hour <= hour < end_hour
#
# Intervals that cross midnight are supported (end_hour < start_hour).
# end_hour is also allowed to be 24 (meaning "up to but not including
# midnight"), which does not need the cross-midnight case at all.
#
# The default configuration covers the full local day exactly once, as
# 4 six-hour dayparts (requested by the thesis director, replacing the
# previous 3 uneven dayparts - see pipeline-migration memory).
DAYPARTS = [
    {
        "name": "Night",
        "start_hour": 0,
        "end_hour": 6,
    },
    {
        "name": "Morning",
        "start_hour": 6,
        "end_hour": 12,
    },
    {
        "name": "Afternoon",
        "start_hour": 12,
        "end_hour": 18,
    },
    {
        "name": "Evening",
        "start_hour": 18,
        "end_hour": 24,
    },
]

# [A.3] Plot settings
SHOW_PLOTS = False
PLOT_DPI = 250


# ======================================================================
# [B] ÍNDICES: PARQUET <-> NOMBRE DE FICHERO <-> ETIQUETA CIENTÍFICA
# ======================================================================
# Misma fuente única de verdad que en 2-ccdf.py / 3-temporal_variability.py
# / 4-hourly_variability.py (CLAUDE.md sección 3-bis).
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
    que este script conociera Ndat, tanto para lo que escribe Step 5 como para
    dónde busca el Step 3 del que depende - ningún caller existente que no
    pase este argumento (hoy, web_server.py y main() con su default) ve ningún
    cambio de ruta. Un modo real ("eq60"/"ge30"/"all"/"lt30") se aplica a la
    vez a ambas resoluciones (lectura de Step 3 y escritura de Step 5) con el
    mismo ndat_config - mismo mecanismo ya validado en
    3-temporal_variability.py / 4-hourly_variability.py::resolve_paths().
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
    index_dir = Path("results") / station / str(year) / "5_daypart_variability" / value_col

    if ndat_mode is None:
        step3_dir = step3_index_dir
        step3_prefix = f"3-{station}_{index_config['file_tag']}_{year}"
        out_dir = index_dir
        prefix = f"5-{station}_{index_config['file_tag']}_{year}"
    else:
        ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
        step3_dir = step3_index_dir / ndat_config["dir_tag"]
        step3_prefix = f"3-{station}_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"
        out_dir = index_dir / ndat_config["dir_tag"]
        prefix = f"5-{station}_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"

    thresholds_csv = step3_dir / f"{step3_prefix}_{doy_tag}_annual_thresholds.csv"

    return {
        "parquet_path": parquet_path,
        "coverage_csv": coverage_csv,
        "thresholds_csv": thresholds_csv,
        "index_dir": index_dir,
        "ndat_dir": out_dir,
        "output_daypart_csv": out_dir / f"{prefix}_{doy_tag}_daypart_frequency.csv",
        "output_daypart_plot": out_dir / f"{prefix}_{doy_tag}_daypart_frequency.png",
        "prefix": prefix,
    }


# ======================================================================
# [D] DAYPART HELPERS (lógica sin cambios)
# ======================================================================

def hours_in_daypart(start_hour: int, end_hour: int) -> list[int]:
    """
    Returns the local integer hours included in a daypart.

    The interval convention is:
        start_hour <= hour < end_hour

    If end_hour is smaller than start_hour, the interval crosses midnight.
    """
    if start_hour == end_hour:
        raise ValueError(
            "A daypart cannot have the same start_hour and end_hour, "
            "because that would be ambiguous."
        )

    if not (0 <= start_hour <= 23):
        raise ValueError(f"Invalid start_hour: {start_hour}")

    if not (0 <= end_hour <= 24):
        raise ValueError(f"Invalid end_hour: {end_hour}")

    if start_hour < end_hour:
        return list(range(start_hour, end_hour))

    # Cross-midnight interval
    return list(range(start_hour, 24)) + list(range(0, end_hour))


def validate_dayparts(dayparts: list[dict]) -> dict[int, str]:
    """
    Validates the daypart configuration and returns a mapping:
        hour_local -> daypart_name

    The current implementation requires that the 24 local hours are covered
    exactly once. This avoids double counting and missing time intervals.
    """
    if not dayparts:
        raise ValueError("DAYPARTS cannot be empty.")

    hour_to_daypart = {}
    names = []

    for item in dayparts:
        name = item.get("name")
        start_hour = item.get("start_hour")
        end_hour = item.get("end_hour")

        if not name:
            raise ValueError("Each daypart must have a non-empty name.")

        if name in names:
            raise ValueError(f"Duplicated daypart name: {name}")

        names.append(name)

        hours = hours_in_daypart(
            start_hour=int(start_hour),
            end_hour=int(end_hour),
        )

        for hour in hours:
            if hour in hour_to_daypart:
                previous = hour_to_daypart[hour]
                raise ValueError(
                    f"Hour {hour} is assigned to more than one daypart: "
                    f"{previous} and {name}."
                )

            hour_to_daypart[hour] = name

    missing_hours = sorted(set(range(24)) - set(hour_to_daypart.keys()))

    if missing_hours:
        raise ValueError(
            "The daypart configuration does not cover all 24 hours. "
            f"Missing local hours: {missing_hours}"
        )

    return hour_to_daypart


def daypart_hour_range_label(start_hour: int, end_hour: int) -> str:
    """Returns a compact label for a daypart hour interval."""
    return f"{start_hour:02d}:00-{end_hour:02d}:00"


def daypart_hours_label(start_hour: int, end_hour: int) -> str:
    """Returns the explicit list of local hours included in a daypart."""
    hours = hours_in_daypart(start_hour, end_hour)
    return ",".join(f"{h:02d}" for h in hours)


def build_daypart_metadata(dayparts: list[dict]) -> pd.DataFrame:
    """Builds an ordered metadata table for the configured dayparts."""
    rows = []

    for order, item in enumerate(dayparts, start=1):
        start_hour = int(item["start_hour"])
        end_hour = int(item["end_hour"])

        rows.append({
            "daypart_order": order,
            "daypart": item["name"],
            "hour_range": daypart_hour_range_label(start_hour, end_hour),
            "hours_local": daypart_hours_label(start_hour, end_hour),
        })

    return pd.DataFrame(rows)


# ======================================================================
# [E] DATA LOADING
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


def load_thresholds(thresholds_csv: Path, percentile_high: int, percentile_extreme: int) -> tuple[float, float, dict]:
    """Loads annual p_high and p_extreme thresholds from Step 3."""
    if not thresholds_csv.exists():
        raise FileNotFoundError(f"Thresholds CSV not found: {thresholds_csv}")

    df = pd.read_csv(thresholds_csv)

    p_high_col = f"p{percentile_high}_annual"
    p_extreme_col = f"p{percentile_extreme}_annual"
    required_columns = {p_high_col, p_extreme_col}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Thresholds CSV is missing columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Thresholds CSV is empty.")

    row = df.iloc[0].to_dict()

    p_high = float(row[p_high_col])
    p_extreme = float(row[p_extreme_col])

    if not np.isfinite(p_high) or not np.isfinite(p_extreme):
        raise ValueError("Annual thresholds must be finite numeric values.")

    if p_high > p_extreme:
        raise ValueError(f"Annual p{percentile_high} threshold cannot exceed annual p{percentile_extreme}.")

    return p_high, p_extreme, row


def load_prepared_dataset(
    parquet_path: Path,
    parquet_value_col: str,
    value_col: str,
    valid_doys,
    ndat_mode: str | None = None,
) -> pd.DataFrame:
    """
    Loads row-level hour_local/value_col data from the Step 0 Parquet,
    filtered to the valid days from Step 1.

    Unlike the previous version (which read an already-filtered "prepared"
    Parquet from Step 4), the Step 0 Parquet contains all 366 days, so the
    valid-day filter now happens here explicitly - skipping it would
    silently include low-completeness days in the daypart frequencies.

    ndat_mode=None (default) is the legacy behaviour: no row is discarded
    by Ndat. With a real mode, observations are restricted via
    ndat_filter.apply_ndat_filter() right after the Step 1 valid-day
    intersection and before anything else (same point already used in
    3-temporal_variability.py / 4-hourly_variability.py). No explicit drop
    of the Ndat column is needed afterwards - unlike Step 4's `prepared`,
    this function already ends with a narrow `return
    df[["hour_local", value_col]]`, which excludes the Ndat column from
    what reaches compute_daypart_frequency()/validate_daypart_results() in
    every mode, including None.
    """
    df = pd.read_parquet(
        parquet_path,
        columns=["14_doy_utc", "18_hour_local", parquet_value_col, ndat_filter.NDAT_PARQUET_COLUMN],
    )

    df = df[df["14_doy_utc"].isin(valid_doys)].copy()

    if ndat_mode is not None:
        df = ndat_filter.apply_ndat_filter(df, ndat_mode)

    df = df.rename(columns={
        "18_hour_local": "hour_local",
        parquet_value_col: value_col,
    })

    df["hour_local"] = pd.to_numeric(df["hour_local"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    df = df.dropna(subset=["hour_local", value_col]).copy()
    df["hour_local"] = df["hour_local"].astype(int)

    invalid_hours = df[~df["hour_local"].between(0, 23)]
    if not invalid_hours.empty:
        raise ValueError("Prepared dataset contains invalid local-hour values.")

    if df.empty:
        raise ValueError("No usable observations found for the selected valid days.")

    return df[["hour_local", value_col]]


# ======================================================================
# [F] COMPUTATION (lógica sin cambios, parametrizada en vez de leer
# VALUE_COL/PERCENTILE_HIGH/PERCENTILE_EXTREME directamente)
# ======================================================================

def compute_daypart_frequency(
    df: pd.DataFrame,
    value_col: str,
    p_high: float,
    p_extreme: float,
    percentile_high: int,
    percentile_extreme: int,
    dayparts: list[dict],
) -> pd.DataFrame:
    """Computes exceedance frequencies by configured local-time daypart."""
    hour_to_daypart = validate_dayparts(dayparts)
    daypart_metadata = build_daypart_metadata(dayparts)

    df_work = df.copy()

    df_work["daypart"] = df_work["hour_local"].map(hour_to_daypart)

    if df_work["daypart"].isna().any():
        raise ValueError("Some observations could not be assigned to a daypart.")

    n_high_col = f"n_at_or_above_p{percentile_high}"
    n_extreme_col = f"n_at_or_above_p{percentile_extreme}"
    freq_high_col = f"frequency_p{percentile_high}_pct"
    freq_extreme_col = f"frequency_p{percentile_extreme}_pct"

    df_work["above_high"] = df_work[value_col] >= p_high
    df_work["above_extreme"] = df_work[value_col] >= p_extreme

    grouped = (
        df_work
        .groupby("daypart", as_index=False)
        .agg(
            n_values=(value_col, "size"),
            **{n_high_col: ("above_high", "sum")},
            **{n_extreme_col: ("above_extreme", "sum")},
        )
    )

    grouped[n_high_col] = grouped[n_high_col].astype(int)
    grouped[n_extreme_col] = grouped[n_extreme_col].astype(int)

    grouped[freq_high_col] = 100.0 * grouped[n_high_col] / grouped["n_values"]
    grouped[freq_extreme_col] = 100.0 * grouped[n_extreme_col] / grouped["n_values"]

    result = (
        daypart_metadata
        .merge(grouped, on="daypart", how="left")
        .sort_values("daypart_order")
        .reset_index(drop=True)
    )

    count_columns = ["n_values", n_high_col, n_extreme_col]
    result[count_columns] = result[count_columns].fillna(0).astype(int)

    frequency_columns = [freq_high_col, freq_extreme_col]
    result[frequency_columns] = result[frequency_columns].fillna(0.0)

    return result


# ======================================================================
# [G] VALIDATION
# ======================================================================

def validate_daypart_results(
    df: pd.DataFrame,
    df_daypart: pd.DataFrame,
    threshold_row: dict,
    percentile_high: int,
    percentile_extreme: int,
) -> None:
    """Validates daypart aggregation against the prepared dataset."""
    n_high_col = f"n_at_or_above_p{percentile_high}"
    n_extreme_col = f"n_at_or_above_p{percentile_extreme}"
    freq_high_col = f"frequency_p{percentile_high}_pct"
    freq_extreme_col = f"frequency_p{percentile_extreme}_pct"

    total_prepared = int(len(df))
    total_daypart = int(df_daypart["n_values"].sum())

    if total_prepared != total_daypart:
        raise ValueError(
            "Daypart sample total does not match prepared dataset rows. "
            f"Prepared={total_prepared}, daypart={total_daypart}"
        )

    if not (df_daypart[n_extreme_col] <= df_daypart[n_high_col]).all():
        raise ValueError("Extreme exceedance count cannot exceed high exceedance count.")

    if not (df_daypart[n_high_col] <= df_daypart["n_values"]).all():
        raise ValueError("High exceedance count cannot exceed total samples.")

    for column in [freq_high_col, freq_extreme_col]:
        if not df_daypart[column].between(0, 100).all():
            raise ValueError(f"{column} contains values outside the [0, 100] range.")

    if "n_values" in threshold_row:
        expected_n_values = int(threshold_row["n_values"])
        if expected_n_values != total_prepared:
            raise ValueError(
                "Prepared dataset row count does not match Step 3 annual total. "
                f"Prepared={total_prepared}, Step3={expected_n_values}"
            )

    if f"n_at_or_above_p{percentile_high}" in threshold_row:
        expected_high = int(threshold_row[n_high_col])
        observed_high = int(df_daypart[n_high_col].sum())
        if expected_high != observed_high:
            raise ValueError(
                "High exceedance total does not match Step 3. "
                f"Observed={observed_high}, Step3={expected_high}"
            )

    if f"n_at_or_above_p{percentile_extreme}" in threshold_row:
        expected_extreme = int(threshold_row[n_extreme_col])
        observed_extreme = int(df_daypart[n_extreme_col].sum())
        if expected_extreme != observed_extreme:
            raise ValueError(
                "Extreme exceedance total does not match Step 3. "
                f"Observed={observed_extreme}, Step3={expected_extreme}"
            )


# ======================================================================
# [H] PLOT
# ======================================================================

def save_daypart_plot(
    df_daypart: pd.DataFrame,
    out_png: Path,
    p_high: float,
    p_extreme: float,
    station: str,
    year: int,
    index_label: str,
    index_unit: str,
    percentile_high: int,
    percentile_extreme: int,
    plot_dpi: int = PLOT_DPI,
    show_plot: bool = SHOW_PLOTS,
) -> None:
    """Saves a grouped bar plot of exceedance frequency by daypart."""
    freq_high_col = f"frequency_p{percentile_high}_pct"
    freq_extreme_col = f"frequency_p{percentile_extreme}_pct"

    x = np.arange(len(df_daypart))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars_high = ax.bar(
        x - width / 2,
        df_daypart[freq_high_col],
        width,
        label=f"{index_label} ≥ T{percentile_high} ({p_high:.3f} {index_unit})",
    )

    bars_extreme = ax.bar(
        x + width / 2,
        df_daypart[freq_extreme_col],
        width,
        label=f"{index_label} ≥ T{percentile_extreme} ({p_extreme:.3f} {index_unit})",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df_daypart["daypart"])
    ax.set_xlabel("Local-time daypart")
    ax.set_ylabel("Threshold exceedance frequency [%]")

    ax.set_title(
        f"Daypart {index_label} threshold exceedance — {station} — {year}\n"
        "Frequency computed from receiver-satellite observations"
    )

    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    # Annotate bars with frequency values.
    for bars in [bars_high, bars_extreme]:
        for bar in bars:
            height = bar.get_height()

            ax.annotate(
                f"{height:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    max_frequency = float(
        max(
            df_daypart[freq_high_col].max(),
            df_daypart[freq_extreme_col].max(),
        )
    )

    ax.set_ylim(0, max_frequency * 1.20 if max_frequency > 0 else 1)

    fig.tight_layout()
    fig.savefig(out_png, dpi=plot_dpi)

    if show_plot:
        plt.show()

    plt.close(fig)


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

    print("Step 5 — Daypart variability of threshold exceedances")
    print("=======================================================")
    print(f"Station: {STATION}")
    print(f"Year: {YEAR}")
    print(f"Step 0 Parquet: {paths['parquet_path']}")
    print(f"Coverage CSV: {paths['coverage_csv']}")
    print(f"Annual thresholds CSV: {paths['thresholds_csv']}")

    if not paths["parquet_path"].exists():
        raise FileNotFoundError(
            f"Step 0 Parquet not found for station {STATION}: {paths['parquet_path']}\n"
            "Run 0-toParquet.py first."
        )

    print("\nConfigured dayparts")
    print("-------------------")
    for item in DAYPARTS:
        print(f"{item['name']}: {int(item['start_hour']):02d}:00-{int(item['end_hour']):02d}:00")

    valid_days = load_valid_days(paths["coverage_csv"], DOY_START, DOY_END)
    p_high, p_extreme, threshold_row = load_thresholds(
        paths["thresholds_csv"], PERCENTILE_HIGH, PERCENTILE_EXTREME
    )

    df_prepared = load_prepared_dataset(
        paths["parquet_path"], parquet_value_col, VALUE_COL, set(valid_days["DoY"]), NDAT_MODE
    )

    df_daypart = compute_daypart_frequency(
        df=df_prepared,
        value_col=VALUE_COL,
        p_high=p_high,
        p_extreme=p_extreme,
        percentile_high=PERCENTILE_HIGH,
        percentile_extreme=PERCENTILE_EXTREME,
        dayparts=DAYPARTS,
    )

    validate_daypart_results(
        df=df_prepared,
        df_daypart=df_daypart,
        threshold_row=threshold_row,
        percentile_high=PERCENTILE_HIGH,
        percentile_extreme=PERCENTILE_EXTREME,
    )

    df_daypart.to_csv(paths["output_daypart_csv"], index=False)

    save_daypart_plot(
        df_daypart=df_daypart,
        out_png=paths["output_daypart_plot"],
        p_high=p_high,
        p_extreme=p_extreme,
        station=STATION,
        year=YEAR,
        index_label=index_label,
        index_unit=index_unit,
        percentile_high=PERCENTILE_HIGH,
        percentile_extreme=PERCENTILE_EXTREME,
        plot_dpi=PLOT_DPI,
        show_plot=SHOW_PLOTS,
    )

    n_high_col = f"n_at_or_above_p{PERCENTILE_HIGH}"
    n_extreme_col = f"n_at_or_above_p{PERCENTILE_EXTREME}"

    print("\nValidation summary")
    print("------------------")
    print(f"Valid days selected from Step 1: {len(valid_days)}")
    print(f"Prepared dataset rows: {len(df_prepared):,}")
    print(f"Daypart sample total: {int(df_daypart['n_values'].sum()):,}")
    print(f"p{PERCENTILE_HIGH} exceedance total: {int(df_daypart[n_high_col].sum()):,}")
    print(f"p{PERCENTILE_EXTREME} exceedance total: {int(df_daypart[n_extreme_col].sum()):,}")
    print("Daypart totals match the validated Step 3 annual totals.")

    print("\nDaypart frequency table")
    print("-----------------------")
    print(df_daypart)

    print("\nOutput files")
    print("------------")
    print(f"Daypart frequency CSV: {paths['output_daypart_csv']}")
    print(f"Daypart frequency plot: {paths['output_daypart_plot']}")

    print("\nStep 5 completed successfully.")


if __name__ == "__main__":
    main()
