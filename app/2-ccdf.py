#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2 - Monthly CCDF and percentiles.

Reads from the Step 0 Parquet and the results/{STATION}/{YEAR}/{STEP}/{INDEX}/
layout, following the same philosophy as 1-completitud.py (CLAUDE.md
section 11): global config as defaults, a pure parametrised
run_ccdf_analysis() with no printing/file I/O, main() for console mode,
import-safe, plots savable without depending on plt.show().

Scientific scope: only ROTI L1 (roti_l1) is implemented/validated for
Step 2 - see INDEX_CONFIG / validate_index_supported() (CLAUDE.md
section 3-bis).

Neither the CCDF/percentile calculation nor the plotting code changed in
this pass - only how they are organised and parametrised.
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

import ndat_filter

# ======================================================================
# [A] CONFIG DEL ESTUDIO (TOCAR SOLO AQUÍ)
# ======================================================================

# [A.1] Caso de estudio
STATION = "YELL"
YEAR = 2024
DOY_START = 1
DOY_END = 366  # 2024 es bisiesto

# [A.2] Índice a analizar. Solo "roti_l1" está científicamente
# implementado/validado en el Step 2 - ver INDEX_CONFIG.
VALUE_COL = "roti_l1"

# [A.2b] Criterio Ndat para las observaciones ROTI L1. Ver ndat_filter.py -
# eq60 es el default oficial del TFM.
NDAT_MODE = ndat_filter.NDAT_DEFAULT_MODE

# [A.3] Umbral de completitud usado en el Paso 1 (para localizar su CSV)
TH_COV = 0.75

# [A.4] Percentiles a extraer
PCTS = [90, 95, 99]

# [A.5] Rango temporal de análisis
# "all":
# el script usa todos los días válidos y luego compara solo los meses indicados.
ANALYSIS_MODE = "all"

# Si ANALYSIS_MODE = "months", solo se usarán estos meses
MONTHS_TO_ANALYZE = [3, 7, 11]

# [A.6] Meses que se quieren representar juntos en una misma gráfica
# Sugerencia del profesor: Marzo, Julio y Noviembre
MONTHS_TO_COMPARE = [3, 7, 11]

# [A.7] Configuración de las gráficas - valores por defecto del modo
# consola. Las funciones de plot los reciben como argumentos, no los leen
# directamente de aquí (ver [E]).
USE_LOG_Y = True
X_LIM = (0, 20)
Y_LIM = (1e-5, 1)

SAVE_INDIVIDUAL_CCDF = True
SAVE_COMPARISON_CCDF = True
SHOW_PLOTS = False

# [A.8] Meses con menos días válidos que este valor se marcan en la gráfica
MIN_VALID_DAYS_PER_MONTH = 10

# [A.9] Cross-station CCDF comparison: color fijo por estación (no por
# posición en una lista - misma motivación que month_color(), ver [E]) y
# linestyle por mes (cicla si se piden más de 4 meses a la vez).
STATION_COLORS = {"UNSA": "tab:orange", "KOUG": "tab:blue", "WHIT": "tab:green", "YELL": "tab:red"}
MONTH_LINESTYLES = ["-", "--", ":", "-."]


# ======================================================================
# [B] ÍNDICES: PARQUET <-> NOMBRE DE FICHERO <-> ETIQUETA CIENTÍFICA
# ======================================================================
# Fuente única de verdad para cada índice disponible en el Parquet del
# Paso 0. Evita hardcodes contradictorios (p.ej. una gráfica etiquetada
# "ROTI L1" mostrando en realidad datos de S4).
#
# "scientifically_supported" = False significa que la columna existe en
# el Parquet (Step 0 la conserva para extensibilidad futura) pero su
# metodología de análisis en el Step 2 todavía no está implementada ni
# validada - habilitarla es un cambio metodológico, no técnico
# (CLAUDE.md sección 14).
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
    Comprueba que value_col tiene un análisis científico de Step 2
    implementado y validado - no solo que la columna exista en el
    Parquet. Devuelve su entrada de INDEX_CONFIG si es válido.

    Recibe value_col como argumento explícito (no lee VALUE_COL global),
    para que sea igual de correcta llamada desde main() o desde una
    futura app con otro índice.
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
    Resuelve rutas de entrada/salida a partir de los parámetros recibidos,
    no de los globales del módulo, siguiendo el mismo patrón que
    1-completitud.py.

    Salidas bajo results/{STATION}/{YEAR}/2_ccdf/{VALUE_COL}/ - Steps 0 y
    1 no cambian (no dependen del índice).

    ndat_mode=None (default) reproduce exactamente la ruta/prefix de antes
    de que este script conociera Ndat - ningún caller existente que no
    pase este argumento (hoy, web_server.py) ve ningún cambio de ruta.
    Pasar un modo real ("eq60"/"ge30"/"all"/"lt30") añade un nivel más
    (ndat_config["dir_tag"]) y etiqueta también el prefix del fichero
    (ndat_config["file_tag"]) - mismo mecanismo que ya usa el índice.
    """
    parquet_path = (
        Path("results") / station / str(year) / "0_parquet"
        / f"0_{station}_{year}_observations.parquet"
    )
    coverage_csv = (
        Path("results") / station / str(year) / "1_completeness"
        / f"coverage_{station}_{year}_DOY{doy_start}_{doy_end}_coverageTH{th_cov}.csv"
    )

    index_dir = Path("results") / station / str(year) / "2_ccdf" / value_col

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
    Nombre de fichero de una CCDF individual (un mes). Incluye DOY{start}_{end}
    - antes no lo llevaba, a diferencia de output_pcts_csv /
    output_monthly_percentiles_png, así que dos corridas con distinto rango de
    DoY se pisaban en silencio. Única fuente de verdad usada por main() y por
    la web, para que no puedan desincronizarse.
    """
    doy_tag = f"DOY{doy_start}_{doy_end}"
    return paths["ccdf_dir"] / f"{paths['prefix']}_{doy_tag}_M{month:02d}_ccdf_logY.png"


def multi_ccdf_png(paths: dict, doy_start: int, doy_end: int, months: list[int]) -> Path:
    """
    Nombre de fichero de la CCDF comparativa de varios meses. Vive en
    ccdf_comparisons_dir, no en ccdf_dir - cada combinación de meses que se
    pruebe genera un fichero nuevo, así que se mantiene separada de los 12
    individuales "canónicos" (mismo patrón que monthly_percentile_variability/).
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
    station (CLAUDE.md section 3-bis). Lives under results/global/{year}/
    2_ccdf_cross_station/, sibling to Step 2.2's 2_ccdf_comparison/ - kept
    separate because it is a different kind of output (full CCDF curves,
    not a percentile table/plot).

    ndat_mode=None (default) reproduces exactly the path/prefix from
    before this function knew about Ndat - same dir_tag/file_tag mechanism
    already used by resolve_paths() in this same script.
    """
    doy_tag = f"DOY{doy_start}_{doy_end}"
    stations_tag = "_".join(sorted(stations))
    months_tag = "_".join(f"M{m:02d}" for m in sorted(months))
    base_dir = Path("results") / "global" / str(year) / "2_ccdf_cross_station"

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
# [D] FUNCIONES AUXILIARES (lógica de cálculo sin cambios)
# ======================================================================

def doy_to_month(year: int, doy: int) -> int:
    """Convierte Year + DoY -> mes (1..12)."""
    dt = datetime.strptime(f"{year}-{doy:03d}", "%Y-%j")
    return dt.month


def month_label(month: int) -> str:
    """Etiqueta simple para leyenda."""
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
    Agrupa los valores de parquet_value_col por mes, para los DoY en
    valid_doys, en una sola pasada vectorizada sobre el Parquet ya
    cargado - en vez de filtrar el DataFrame completo una vez por cada
    DoY (hasta 366 veces).

    El mes de cada DoY se obtiene con doy_to_month() (la misma función
    que usaba la versión anterior), no de una columna del Parquet, para
    no introducir una segunda fuente de la misma información.

    Devuelve (month_values, n_days_by_month):
    - month_values[m]: array 1D con todos los valores válidos (no-NaN)
      de ese mes, equivalente a concatenar los arrays por día de la
      versión anterior.
    - n_days_by_month[m]: número de DoY distintos que aportaron al menos
      un valor válido ese mes - equivalente a contar cuántos días
      añadieron un array no vacío en la versión anterior.
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
    Calcula la CCDF empírica.

    Se usa:
        CCDF(x_i) ≈ P(X >= x_i) = (n - i + 1) / n

    Esta definición es más adecuada para escala logarítmica porque la CCDF
    no llega exactamente a cero.
    """
    x = np.sort(values)
    n = x.size

    if n == 0:
        return None, None

    i = np.arange(1, n + 1)
    ccdf = (n - i + 1) / n

    return x, ccdf


def format_n(n: int) -> str:
    """Formato simple para número de muestras."""
    return f"{n:,}".replace(",", ".")


# ======================================================================
# [E] FUNCIONES DE PLOT (lógica de dibujo sin cambios; use_log_y/x_lim/
# y_lim/show_plot llegan como argumentos con los globales como valor por
# defecto, en vez de leerse directamente dentro de la función)
# ======================================================================

def month_color(month: int):
    """
    Color fijo por mes calendario (1-12), constante sin importar qué otros
    meses se estén comparando en la misma figura - así el mismo mes se lee
    con el mismo color en dos comparaciones distintas. Muestreado de un
    colormap cíclico (twilight) para que además el color siga la
    progresión estacional del año.
    """
    cmap = plt.get_cmap("twilight")
    return cmap((month - 1) / 12)


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
    Guarda la CCDF individual de un mes.

    Mejoras aplicadas:
    - eje Y logarítmico
    - eje X acotado
    - líneas de referencia para p90 y p99
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

    # Líneas de referencia para percentiles
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
    Guarda una figura con varias CCDF superpuestas.

    Esta es la gráfica más importante para enviar al profesor:
    - varias curvas en la misma figura
    - eje Y logarítmico
    - eje X limitado a [0, 20]
    - líneas horizontales para p90 y p99
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

        # Marcar p90 y p99 sobre la curva - mismo color que su propia linea,
        # para que el marcador se pueda atribuir a su mes sin ambigüedad
        # (antes venían del ciclo de color automático, no coincidían con
        # la línea de su propio mes).
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

    # Referencias de percentil en CCDF
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
    Color is fixed per station (STATION_COLORS - by identity, not by
    position, same motivation as month_color()). Linestyle is fixed per
    month (MONTH_LINESTYLES, cycles past 4 months). No low-support marker,
    same as save_multi_ccdf_plot - the n= in the legend already carries
    that for CCDF-curve views (the red-X marking belongs to the percentile
    summary views instead).
    """
    plt.figure(figsize=(9, 6))

    for station in stations:
        station_data = ccdf_by_station_month.get(station, {})
        color = STATION_COLORS.get(station)

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

    # Líneas de referencia neutras (gris) - con color de estación variable
    # y numero de curvas variable, no deben competir por un color del ciclo
    # automático que podría coincidir con alguna estación.
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
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(out_png, dpi=250)

    if show_plot:
        plt.show()

    plt.close()


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
    Genera la gráfica de evolución mensual de los percentiles p90 y p99.

    Eje X:
        Meses del año.

    Eje Y:
        Valor mensual del percentil.

    Los meses con pocos días válidos se mantienen en la figura, pero se
    identifican mediante una anotación para evitar interpretarlos como
    representativos de todo el mes.
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

    # Copia ordenada para no modificar el DataFrame original
    df_plot = (
        df_pcts.copy()
        .sort_values("month")
        .reset_index(drop=True)
    )

    # Excluir únicamente meses completamente vacíos
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

    # Evolución mensual de los percentiles
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

    # Marcar los meses con pocos días válidos
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
            fontsize=8
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
# [F] CÁLCULO CIENTÍFICO DE ALTO NIVEL (sin prints ni I/O de salida)
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
    Calcula percentiles mensuales y CCDF de value_col para una estación,
    año y rango de DoY.

    Orquesta exactamente el mismo cálculo que hacía antes el código suelto
    a nivel de módulo (misma selección de días válidos del Step 1, mismo
    agrupado por mes vía accumulate_values_by_month(), misma fórmula de
    percentiles y CCDF) - no es un cálculo científico nuevo, solo queda
    encapsulado en una función parametrizada y reutilizable en vez de
    código de módulo.

    No imprime nada ni guarda ningún fichero - función de cálculo pura,
    reutilizable igual desde main() (consola) o desde una futura
    aplicación. Todos los parámetros de entrada llegan como argumentos;
    no se lee STATION/YEAR/VALUE_COL/etc. directamente del módulo dentro
    de esta función.

    Lanza FileNotFoundError si falta el Parquet del Step 0 o el CSV de
    completitud del Step 1, y ValueError si value_col no está
    científicamente soportado (ver validate_index_supported).

    ndat_mode=None (default) es el comportamiento legacy: ninguna fila se
    descarta por Ndat, idéntico a antes de que este script conociera
    ndat_filter - así cualquier caller que todavía no pase este argumento
    (hoy, web_server.py) no ve ningún cambio. Con un modo real
    ("eq60"/"ge30"/"all"/"lt30"), las observaciones se restringen vía
    ndat_filter.apply_ndat_filter() antes de agrupar por mes - los días
    válidos siguen siendo exactamente los de Step 1 (valid_doys no
    depende de Ndat), el filtro solo actúa sobre las filas dentro de esos
    días.

    Devuelve un dict con:
      - "index_config": entrada de INDEX_CONFIG para value_col;
      - "ndat_config": entrada de NDAT_MODES para ndat_mode, o None si
        ndat_mode is None;
      - "valid_doys": lista de DoY usados en el análisis;
      - "df_pcts": DataFrame con month, n_days, n_values, p{pct}...;
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
# [G] RESUMEN COMPARATIVO (derivado de df_pcts, sin prints ni I/O)
# ======================================================================
def compute_ccdf_summary(
    df_pcts: pd.DataFrame,
    min_valid_days: int = MIN_VALID_DAYS_PER_MONTH,
) -> dict:
    """
    Deriva estadísticas de comparación a partir de la tabla mensual de
    percentiles que ya devuelve run_ccdf_analysis() - no recalcula ningún
    percentil, solo resume df_pcts (mismo criterio que
    compute_completeness_summary() en 1-completitud.py).

    Para p90 y p99 por separado, el mínimo/máximo se calcula solo sobre
    meses con n_days >= min_valid_days ("fiables") - un mínimo/máximo
    basado en un mes de 1 día no es tan fiable como uno basado en 30, y
    dejarlo entrar en el resumen puede llevar a una lectura errónea
    (decisión metodológica explícita, ver conversación/CLAUDE.md §14, no
    un cambio silencioso). Esto NO afecta a los datos de origen: la
    gráfica (save_monthly_percentiles_plot) y el CSV siguen mostrando los
    12 meses tal cual, con su propia anotación de baja muestra - la
    exclusión ocurre solo dentro de este resumen derivado.

    Si el mes con el valor crudo mínimo/máximo (sobre todos los meses)
    queda excluido por baja muestra, se registra en
    {col}_min_excluded / {col}_max_excluded (mes, valor y n_days de ese
    mes excluido; None si no hubo exclusión) - así el dato no desaparece,
    solo se explica por qué no es el que encabeza el resumen.

    Caso límite: si ningún mes del año llega al umbral, se usan los
    valores crudos igualmente (mejor eso que fallar o devolver un
    resumen vacío) y se marca {col}_all_months_low_support = True.

    Pura: sin prints ni escritura de ficheros.
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
# [H] MAIN (modo consola): orquesta el cálculo + guarda CSV/PNG/consola
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
    # Guardar CCDF individuales
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
    # Guardar gráfica comparativa de varios meses
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
    # Guardar tabla de percentiles
    # ------------------------------------------------------------------
    print("\nTabla mensual de percentiles:")
    print(df_pcts)

    df_pcts.to_csv(paths["output_pcts_csv"], index=False)
    print(f"\nPercentiles guardados en {paths['output_pcts_csv']}")

    # ------------------------------------------------------------------
    # Gráfica de variabilidad mensual de p90 y p99
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
