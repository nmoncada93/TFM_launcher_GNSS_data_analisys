#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 4 - Hourly variability (local hour) of threshold exceedances.

Reads from the Step 0 Parquet, the Step 1 coverage CSV and the Step 3
annual-thresholds CSV, following the same philosophy as Steps 1-3
(CLAUDE.md section 11): global config as defaults, main() for console
mode, import-safe.

No longer derives hour_utc/hour_local by hand from sec_of_day + a
hardcoded UTC offset. The Step 0 Parquet already carries 16_hour_utc and
18_hour_local, computed per row with each station's correct fixed offset
(CLAUDE.md section 6) - re-deriving them here with a single hardcoded
UTC_OFFSET_HOURS (as the previous version did) only worked for whichever
station that constant happened to match, and would silently miscompute
local hours for any other station. This step also no longer writes its
own "prepared" Parquet - it was a subset of what the Step 0 Parquet
already provides (same reasoning as Step 3, see CLAUDE.md section 2).

Scientific scope: only ROTI L1 (roti_l1) is implemented/validated - see
INDEX_CONFIG / validate_index_supported() (CLAUDE.md section 3-bis).

The hourly-frequency calculation and the plotting code are otherwise
unchanged from the validated version.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ndat_filter


# ======================================================================
# [A] STUDY CONFIGURATION - EDIT ONLY THIS SECTION
# ======================================================================

STATION = "YELL"
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
# selector de Ndat.
NDAT_MODE = None

TH_COV = 0.75

PERCENTILE_HIGH = 90
PERCENTILE_EXTREME = 99

SHOW_PLOTS = False
PLOT_DPI = 250


# ======================================================================
# [B] ÍNDICES: PARQUET <-> NOMBRE DE FICHERO <-> ETIQUETA CIENTÍFICA
# ======================================================================
# Misma fuente única de verdad que en 2-ccdf.py / 3-temporal_variability.py
# (CLAUDE.md sección 3-bis).
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
    validado - no solo que la columna exista en el Parquet. Recibe
    value_col como argumento explícito (no lee VALUE_COL global).
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
    que este script conociera Ndat, tanto para lo que escribe Step 4 como para
    dónde busca el Step 3 del que depende - ningún caller existente que no pase
    este argumento (hoy, web_server.py y main() con su default) ve ningún
    cambio de ruta. Un modo real ("eq60"/"ge30"/"all"/"lt30") se aplica a la
    vez a ambas resoluciones (lectura de Step 3 y escritura de Step 4) con el
    mismo ndat_config - mismo mecanismo ya validado en
    3-temporal_variability.py::resolve_paths(), de modo que es estructuralmente
    imposible leer un threshold de un modo y escribir/filtrar con otro.
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
    index_dir = Path("results") / station / str(year) / "4_hourly_variability" / value_col

    if ndat_mode is None:
        step3_dir = step3_index_dir
        step3_prefix = f"3-{station}_{index_config['file_tag']}_{year}"
        out_dir = index_dir
        prefix = f"4-{station}_{index_config['file_tag']}_{year}"
    else:
        ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
        step3_dir = step3_index_dir / ndat_config["dir_tag"]
        step3_prefix = f"3-{station}_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"
        out_dir = index_dir / ndat_config["dir_tag"]
        prefix = f"4-{station}_{index_config['file_tag']}_{ndat_config['file_tag']}_{year}"

    thresholds_csv = step3_dir / f"{step3_prefix}_{doy_tag}_annual_thresholds.csv"
    reference_daily_csv = step3_dir / f"{step3_prefix}_{doy_tag}_daily_frequency.csv"

    return {
        "parquet_path": parquet_path,
        "coverage_csv": coverage_csv,
        "thresholds_csv": thresholds_csv,
        "reference_daily_csv": reference_daily_csv,
        "index_dir": index_dir,
        "ndat_dir": out_dir,
        "output_hourly_csv": out_dir / f"{prefix}_{doy_tag}_hourly_frequency.csv",
        "output_hourly_plot": out_dir / f"{prefix}_{doy_tag}_hourly_frequency.png",
        "prefix": prefix,
    }


# ======================================================================
# [D] INPUT FUNCTIONS
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
    df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
    df = df.dropna(subset=["DoY", "coverage"])
    df["DoY"] = df["DoY"].astype(int)

    if df["DoY"].duplicated().any():
        raise ValueError("Coverage CSV contains duplicated DoY values.")

    df = df[
        (df["status"] == "valid")
        & df["DoY"].between(doy_start, doy_end)
    ][["DoY", "coverage"]]

    df = df.sort_values("DoY").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid days were found in the selected range.")
    return df


def load_thresholds(thresholds_csv: Path, station: str, year: int,
                     percentile_high: int, percentile_extreme: int) -> tuple[float, float, int]:
    """Load annual p_high and p_extreme thresholds generated by Step 3."""
    if not thresholds_csv.exists():
        raise FileNotFoundError(f"Threshold CSV not found: {thresholds_csv}")

    df = pd.read_csv(thresholds_csv)
    p_high_col = f"p{percentile_high}_annual"
    p_extreme_col = f"p{percentile_extreme}_annual"
    required = {"station", "year", "n_values", p_high_col, p_extreme_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Threshold CSV is missing columns: {sorted(missing)}")
    if len(df) != 1:
        raise ValueError("Threshold CSV must contain exactly one row.")

    row = df.iloc[0]
    if str(row["station"]) != station or int(row["year"]) != year:
        raise ValueError("Threshold CSV station or year does not match.")

    p_high = float(row[p_high_col])
    p_extreme = float(row[p_extreme_col])
    expected_n = int(row["n_values"])

    if not np.isfinite([p_high, p_extreme]).all() or p_high > p_extreme:
        raise ValueError("Invalid annual thresholds.")
    return p_high, p_extreme, expected_n


def load_hourly_dataset(
    df_valid: pd.DataFrame,
    parquet_path: Path,
    parquet_value_col: str,
    value_col: str,
    ndat_mode: str | None = None,
):
    """
    Load the Step 0 Parquet, already containing UTC/local hour columns,
    filtered to the valid days from Step 1.

    Unlike the previous version, this does not derive hour_utc/hour_local
    from sec_of_day - Step 0 already computed them per row using each
    station's correct fixed UTC offset. Columns are renamed to the short
    names used by the rest of this script (doy_utc, month_utc, hour_utc,
    hour_local, month_local, <value_col>) so the calculation/validation/
    plot functions below did not need to change. The Ndat migration does
    not touch this local-hour logic at all - it only removes rows.

    ndat_mode=None (default) is the legacy behaviour: no row is discarded
    by Ndat. With a real mode, observations are restricted via
    ndat_filter.apply_ndat_filter() right after the Step 1 valid-day
    intersection (same point already used in 3-temporal_variability.py).
    The Ndat column is always dropped from `prepared` afterwards -
    unconditionally, in every mode including None - so it never reaches
    calculate_hourly_frequency()/validate_results() and can never affect
    the generic NaN check there, even when ndat_mode is None.
    """
    df = pd.read_parquet(
        parquet_path,
        columns=[
            "14_doy_utc", "15_month_utc", "16_hour_utc",
            "17_utc_offset_hours", "18_hour_local", "19_month_local",
            parquet_value_col, ndat_filter.NDAT_PARQUET_COLUMN,
        ]
    )

    numeric_values = pd.to_numeric(df[parquet_value_col], errors="coerce")
    finite_mask = numeric_values.notna() & np.isfinite(numeric_values.to_numpy(dtype=float))
    df = df.loc[finite_mask].copy()
    df[parquet_value_col] = numeric_values.loc[df.index]

    df = df.rename(columns={
        "14_doy_utc": "doy_utc",
        "15_month_utc": "month_utc",
        "16_hour_utc": "hour_utc",
        "18_hour_local": "hour_local",
        "19_month_local": "month_local",
        parquet_value_col: value_col,
    })

    utc_offsets = df["17_utc_offset_hours"].unique()
    if len(utc_offsets) != 1:
        raise ValueError(
            f"Expected a single UTC offset for station data, found: {utc_offsets}"
        )
    utc_offset_hours = int(utc_offsets[0])

    prepared = df[df["doy_utc"].isin(df_valid["DoY"])].drop(columns=["17_utc_offset_hours"]).copy()

    if ndat_mode is not None:
        prepared = ndat_filter.apply_ndat_filter(prepared, ndat_mode)

    prepared = prepared.drop(columns=[ndat_filter.NDAT_PARQUET_COLUMN])

    if prepared.empty:
        raise ValueError("No observations found for the selected valid days.")

    daily_counts = (
        prepared
        .groupby("doy_utc")
        .agg(month_utc=("month_utc", "first"), n_values=(value_col, "size"))
        .reset_index()
    )

    summary = (
        daily_counts
        .merge(
            df_valid.rename(columns={"DoY": "doy_utc"})[["doy_utc", "coverage"]],
            on="doy_utc",
            how="left"
        )
        .sort_values("doy_utc")
        .reset_index(drop=True)
    )[["doy_utc", "month_utc", "coverage", "n_values"]]

    if summary["coverage"].isna().any():
        raise ValueError(
            "Some days in the Step 0 Parquet have no matching coverage "
            "entry from Step 1."
        )

    processed_doys = {int(d) for d in summary["doy_utc"]}
    skipped_days = [
        (int(doy), "missing_from_step0_parquet")
        for doy in df_valid["DoY"]
        if int(doy) not in processed_doys
    ]

    return prepared, summary, skipped_days, utc_offset_hours


# ======================================================================
# [E] HOURLY ANALYSIS (lógica sin cambios)
# ======================================================================

def calculate_hourly_frequency(
    df: pd.DataFrame,
    value_col: str,
    p_high: float,
    p_extreme: float,
    percentile_high: int,
    percentile_extreme: int,
) -> pd.DataFrame:
    """Calculate p_high and p_extreme exceedance frequencies by local hour."""
    work = df[["hour_local", value_col]].copy()
    work["above_high"] = work[value_col] >= p_high
    work["above_extreme"] = work[value_col] >= p_extreme

    result = work.groupby("hour_local", observed=True).agg(
        n_values=(value_col, "size"),
        n_at_or_above_high=("above_high", "sum"),
        n_at_or_above_extreme=("above_extreme", "sum")
    ).reindex(range(24), fill_value=0)

    result.index.name = "hour_local"
    result = result.reset_index()
    result[f"n_at_or_above_p{percentile_high}"] = result["n_at_or_above_high"]
    result[f"n_at_or_above_p{percentile_extreme}"] = result["n_at_or_above_extreme"]
    result[f"frequency_p{percentile_high}_pct"] = np.where(
        result["n_values"] > 0,
        100.0 * result["n_at_or_above_high"] / result["n_values"],
        np.nan
    )
    result[f"frequency_p{percentile_extreme}_pct"] = np.where(
        result["n_values"] > 0,
        100.0 * result["n_at_or_above_extreme"] / result["n_values"],
        np.nan
    )
    return result.drop(columns=["n_at_or_above_high", "n_at_or_above_extreme"])


# ======================================================================
# [F] VALIDATION
# ======================================================================

def validate_results(
    df_valid: pd.DataFrame,
    prepared: pd.DataFrame,
    summary: pd.DataFrame,
    hourly: pd.DataFrame,
    skipped_days,
    expected_n_values: int,
    reference_daily_csv: Path,
    percentile_high: int,
    percentile_extreme: int,
):
    """Check consistency with the validated Step 3 outputs."""
    if len(prepared) != expected_n_values:
        raise ValueError(
            f"Prepared rows ({len(prepared):,}) do not match Step 3 "
            f"samples ({expected_n_values:,})."
        )

    if len(summary) + len(skipped_days) != len(df_valid):
        raise ValueError("Processed and skipped day totals are inconsistent.")

    if prepared.isna().any().any():
        raise ValueError("Prepared dataset contains missing values.")
    if not prepared["hour_utc"].between(0, 23).all():
        raise ValueError("Invalid UTC hours in prepared dataset.")
    if not prepared["hour_local"].between(0, 23).all():
        raise ValueError("Invalid local hours in prepared dataset.")

    n_high_col = f"n_at_or_above_p{percentile_high}"
    n_extreme_col = f"n_at_or_above_p{percentile_extreme}"
    freq_high_col = f"frequency_p{percentile_high}_pct"
    freq_extreme_col = f"frequency_p{percentile_extreme}_pct"

    if int(hourly["n_values"].sum()) != len(prepared):
        raise ValueError("Hourly and prepared sample totals do not match.")
    if not (hourly[n_extreme_col] <= hourly[n_high_col]).all():
        raise ValueError("Hourly extreme counts exceed hourly high counts.")

    for column in [freq_high_col, freq_extreme_col]:
        if not hourly[column].dropna().between(0, 100).all():
            raise ValueError(f"Values outside [0, 100] detected in {column}.")

    if not reference_daily_csv.exists():
        raise FileNotFoundError(f"Reference daily CSV not found: {reference_daily_csv}")

    reference = pd.read_csv(reference_daily_csv)[["doy", "n_values"]]
    current = summary[["doy_utc", "n_values"]].rename(columns={"doy_utc": "doy"})
    check = reference.merge(current, on="doy", how="outer", suffixes=("_ref", "_new"))

    mismatch = check[
        check["n_values_ref"].isna()
        | check["n_values_new"].isna()
        | (check["n_values_ref"] != check["n_values_new"])
    ]
    if not mismatch.empty:
        raise ValueError("Daily sample counts do not match the Step 3 CSV.")


# ======================================================================
# [G] PLOT
# ======================================================================

def save_hourly_plot(
    hourly: pd.DataFrame,
    p_high: float,
    p_extreme: float,
    station: str,
    year: int,
    index_label: str,
    index_unit: str,
    local_time_label: str,
    percentile_high: int,
    percentile_extreme: int,
    out_png: Path,
    plot_dpi: int = PLOT_DPI,
    show_plot: bool = SHOW_PLOTS,
) -> None:
    """Save the local-hour threshold-exceedance plot."""
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        hourly["hour_local"], hourly[f"frequency_p{percentile_high}_pct"],
        marker="o", linewidth=1.5,
        label=f"At or above annual p{percentile_high} (T{percentile_high}={p_high:.3f} {index_unit})"
    )
    ax.plot(
        hourly["hour_local"], hourly[f"frequency_p{percentile_extreme}_pct"],
        marker="o", linewidth=1.5,
        label=f"At or above annual p{percentile_extreme} (T{percentile_extreme}={p_extreme:.3f} {index_unit})"
    )

    ax.set_xticks(range(24))
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"Local hour [{local_time_label}]")
    ax.set_ylabel("Exceedance frequency [% of observations]")
    ax.set_title(
        f"Hourly frequency of annual {index_label} threshold exceedances "
        f"— {station} — {year}\n"
        "Calculated from valid receiver-satellite observations"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=plot_dpi)

    if show_plot:
        plt.show()
    plt.close(fig)


# ======================================================================
# [H] MAIN EXECUTION
# ======================================================================

def main():
    index_config = validate_index_supported(VALUE_COL)
    parquet_value_col = index_config["parquet_column"]
    index_label = index_config["label"]
    index_unit = index_config["unit"]

    paths = resolve_paths(STATION, YEAR, DOY_START, DOY_END, TH_COV, VALUE_COL, index_config, NDAT_MODE)
    paths["ndat_dir"].mkdir(parents=True, exist_ok=True)

    print("Step 4 — Hourly variability (local hour) of threshold exceedances")
    print("====================================================================")
    print(f"Station: {STATION}")
    print(f"Year: {YEAR}")
    print(f"Coverage CSV: {paths['coverage_csv']}")
    print(f"Annual thresholds CSV: {paths['thresholds_csv']}")
    print(f"Step 0 Parquet: {paths['parquet_path']}")

    if not paths["parquet_path"].exists():
        raise FileNotFoundError(
            f"Step 0 Parquet not found for station {STATION}: {paths['parquet_path']}\n"
            "Run 0-toParquet.py first."
        )

    valid_days = load_valid_days(paths["coverage_csv"], DOY_START, DOY_END)
    p_high, p_extreme, expected_n_values = load_thresholds(
        paths["thresholds_csv"], STATION, YEAR, PERCENTILE_HIGH, PERCENTILE_EXTREME
    )
    prepared, summary, skipped_days, utc_offset_hours = load_hourly_dataset(
        valid_days, paths["parquet_path"], parquet_value_col, VALUE_COL, NDAT_MODE
    )
    local_time_label = f"UTC{utc_offset_hours:+03d}:00"

    hourly = calculate_hourly_frequency(
        prepared, VALUE_COL, p_high, p_extreme, PERCENTILE_HIGH, PERCENTILE_EXTREME
    )

    validate_results(
        valid_days, prepared, summary, hourly, skipped_days, expected_n_values,
        paths["reference_daily_csv"], PERCENTILE_HIGH, PERCENTILE_EXTREME,
    )

    print("\nValidation summary")
    print("------------------")
    print(f"Valid days selected from Step 1: {len(valid_days)}")
    print(f"Days successfully prepared: {len(summary)}")
    print(f"Valid days skipped: {len(skipped_days)}")
    print(f"Prepared dataset rows: {len(prepared):,}")
    print(f"Local-time conversion: {local_time_label} (from Step 0 Parquet, not hardcoded)")
    print("Daily sample counts match the validated Step 3 CSV.")
    print("Hourly sample totals match the prepared dataset.")

    for doy, reason in skipped_days:
        print(f"[WARNING] DoY {doy:03d} skipped: {reason}")

    hourly.to_csv(paths["output_hourly_csv"], index=False)

    save_hourly_plot(
        hourly, p_high, p_extreme, STATION, YEAR, index_label, index_unit,
        local_time_label, PERCENTILE_HIGH, PERCENTILE_EXTREME,
        paths["output_hourly_plot"], plot_dpi=PLOT_DPI, show_plot=SHOW_PLOTS,
    )

    print("\nOutput files")
    print("------------")
    print(f"Hourly frequency CSV: {paths['output_hourly_csv']}")
    print(f"Hourly frequency plot: {paths['output_hourly_plot']}")
    print("\nStep 4 completed successfully.")


if __name__ == "__main__":
    main()
