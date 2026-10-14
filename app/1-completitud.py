#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1 - Daily data completeness.

Usable two ways:
  1) Standalone from the console: `python 1-completitud.py`. Edit the
     globals in section [A] and run.
  2) Imported (e.g. by web_server.py): call
     run_completeness(station=..., year=..., th_cov=...) directly. The
     globals below are only the console defaults; the calculation itself
     does not depend on them.

A day is valid when its temporal coverage (unique minutes / 1440) reaches
TH_COV. The status is computed without any Ndat criterion; Steps 2-6 then
apply Ndat to the observations of the valid days only.

Optional DIAGNOSTIC (section [D.2], off by default): how much a Ndat
criterion would reduce each day's coverage. Informative only - the valid/
invalid status is always the one computed WITHOUT Ndat, and the official
CSV/PNG and the days used by Steps 2-6 never change.
"""

import calendar
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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

# [A.2] Daily completeness threshold
# coverage = n_unique_minutes / 1440
TH_COV = 0.75

# [A.3] Console mode only: also open the plot in a window (the PNG is always saved).
SHOW_PLOT = True

# [A.4] Optional DIAGNOSTIC: impact of a Ndat criterion on completeness
# (section [D.2]). Informative only - never changes the official outputs
# or the valid/invalid status.
SHOW_NDAT_IMPACT = False
NDAT_MODE = ndat_filter.NDAT_DEFAULT_MODE  # only read when SHOW_NDAT_IMPACT is True
# Width of the band just below th_cov used to group the valid days by their
# post-Ndat coverage (a fraction, like coverage: 0.05 = 5 percentage points).
NDAT_COVERAGE_BAND = 0.05

PLOT_TITLE = f"Daily data coverage — Station {STATION} ({YEAR})"


# ======================================================================
# [B] PARQUET COLUMNS USED (DO NOT TOUCH)
# ======================================================================
# Completeness only needs these columns from the Step 0 parquet:
#   - 01_sec_of_day
#   - 14_doy_utc
# The parquet is already single-station, no need to filter by station.
PARQUET_COLUMNS = ["01_sec_of_day", "14_doy_utc"]


# ======================================================================
# [C] PATH RESOLUTION
# ======================================================================
def resolve_paths(station: str, year: int, doy_start: int, doy_end: int, th_cov: float) -> dict:
    """
    Resolves every input/output path for a given station/year/DoY
    range/threshold, following the results/{STATION}/{YEAR}/... layout.

    Computed from the arguments received, not from the module-level
    globals, so console and web runs resolve the same paths.
    """
    parquet_path = (
        project_paths.RESULTS_ROOT / station / str(year) / "0_parquet"
        / f"0_{station}_{year}_observations.parquet"
    )
    global_inventory_csv = (
        project_paths.RESULTS_ROOT / "global" / str(year) / f"0_raw_file_inventory_{year}.csv"
    )

    results_dir = project_paths.RESULTS_ROOT / station / str(year) / "1_completeness"
    output_tag = f"{station}_{year}_DOY{doy_start}_{doy_end}_coverageTH{th_cov}"

    return {
        "parquet_path": parquet_path,
        "global_inventory_csv": global_inventory_csv,
        "results_dir": results_dir,
        "output_csv": results_dir / f"coverage_{output_tag}.csv",
        "output_png": results_dir / f"coverage_plot_{output_tag}.png",
    }


def resolve_ndat_impact_paths(
    station: str, year: int, doy_start: int, doy_end: int, th_cov: float, ndat_mode: str,
) -> dict:
    """
    Output paths of the optional Ndat-impact DIAGNOSTIC (section [D.2]).
    Own subfolder and own file names under 1_completeness/, so they can
    never overwrite - or be read instead of - the official coverage CSV,
    which Steps 2-6 locate by its exact name.
    """
    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
    results_dir = resolve_paths(station, year, doy_start, doy_end, th_cov)["results_dir"]
    diag_dir = results_dir / "ndat_impact" / ndat_config["dir_tag"]
    tag = f"{station}_{year}_DOY{doy_start}_{doy_end}_coverageTH{th_cov}_{ndat_config['file_tag']}"

    return {
        "diag_dir": diag_dir,
        "output_csv": diag_dir / f"ndat_impact_{tag}.csv",
        "output_png": diag_dir / f"ndat_impact_plot_{tag}.png",
    }


# ======================================================================
# [D] SCIENTIFIC CALCULATION (DO NOT TOUCH)
# ======================================================================
def missing_day_note(doy: int, df_inventory: pd.DataFrame) -> str:
    """
    Reason a DoY has no rows in the station's parquet, taken from the Step 0
    RAW file inventory: missing_file, empty_file, no_data_for_station, or
    the read_error note recorded by Step 0.
    """
    if doy not in df_inventory.index:
        return "missing_file"

    inv = df_inventory.loc[doy]

    if not bool(inv["file_exists"]):
        return "missing_file"

    note = str(inv["note"])
    if note.startswith("read_error"):
        return note

    if not bool(inv["read_ok"]):
        return "empty_file"

    return "no_data_for_station"


def run_completeness(
    station: str = STATION,
    year: int = YEAR,
    doy_start: int = DOY_START,
    doy_end: int = DOY_END,
    th_cov: float = TH_COV,
) -> pd.DataFrame:
    """
    Computes daily completeness for one station/year/DoY range.

    Definition:
      minute = sec_of_day // 60
      n_minutes = # of unique minutes present (0..1440)
      coverage = n_minutes / 1440
      valid if coverage >= th_cov
    No Ndat criterion is applied here.

    Pure calculation: reads its two inputs and returns a DataFrame, with
    no prints or file writes, so console and web share it.

    Raises FileNotFoundError with an explicit message if the Step 0
    parquet or the Step 0 global inventory are missing.
    """
    paths = resolve_paths(station, year, doy_start, doy_end, th_cov)
    parquet_path = paths["parquet_path"]
    global_inventory_csv = paths["global_inventory_csv"]

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Step 0 Parquet not found for station {station}: {parquet_path}\n"
            "Run 0-toParquet.py first."
        )

    if not global_inventory_csv.exists():
        raise FileNotFoundError(
            f"Step 0 global inventory not found: {global_inventory_csv}\n"
            "Run 0-toParquet.py first."
        )

    df_station = pd.read_parquet(parquet_path, columns=PARQUET_COLUMNS)
    df_inventory = pd.read_csv(global_inventory_csv).set_index("doy")

    # Unique minutes per DoY in a single vectorised pass over the station.
    minute = (df_station["01_sec_of_day"] // 60).astype(int)
    n_minutes_by_doy = (
        df_station.assign(minute=minute)
        .groupby("14_doy_utc")["minute"]
        .nunique()
    )

    results = []

    for doy in range(doy_start, doy_end + 1):
        if doy not in n_minutes_by_doy.index:
            results.append({
                "DoY": doy,
                "n_minutes": 0,
                "coverage": 0.0,
                "status": "invalid",
                "note": missing_day_note(doy, df_inventory)
            })
            continue

        n_minutes = int(n_minutes_by_doy.loc[doy])
        coverage = n_minutes / 1440.0
        status = "valid" if coverage >= th_cov else "invalid"

        results.append({
            "DoY": doy,
            "n_minutes": n_minutes,
            "coverage": coverage,
            "status": status,
            "note": "ok"
        })

    return pd.DataFrame(results).sort_values("DoY").reset_index(drop=True)


# ======================================================================
# [D.2] DIAGNOSTIC: NDAT IMPACT ON COMPLETENESS (does not affect status)
# ======================================================================
def run_ndat_impact(
    df_results: pd.DataFrame,
    station: str,
    year: int,
    doy_start: int,
    doy_end: int,
    th_cov: float,
    ndat_mode: str,
    coverage_band: float = NDAT_COVERAGE_BAND,
) -> tuple[pd.DataFrame, dict]:
    """
    Informative only: how much the given Ndat criterion reduces each day's
    coverage. The official valid/invalid status (computed WITHOUT Ndat by
    run_completeness(), passed in as df_results) is copied as-is, never
    recomputed - Steps 2-6 keep using exactly those days.

      Parquet -> ndat_filter.apply_ndat_filter() -> unique minutes/day
              -> coverage_after_ndat = n_minutes_after_ndat / 1440

    Pure function: reads the parquet, returns (df_impact, summary); no
    prints, no file writes.

    df_impact has one row per DoY of the range: the official
    DoY/n_minutes/coverage/status plus observation counts and the
    post-Ndat minutes/coverage. The summary keeps its two scopes apart:
      - filter_on_valid_days: observations and station-minutes on the
        Step 1 valid days only - the data that actually enters Steps 2-6;
      - completeness_all_doys: baseline vs post-Ndat coverage over every
        DoY of the range. The filter only removes rows, so no invalid day
        can reach th_cov after it: the days that stay >= th_cov are always
        a subset of the valid ones;
      - post_ndat_coverage_of_valid_days: the Step 1 valid days split by
        coverage_after_ndat into three exclusive groups - >= th_cov,
        [th_cov - coverage_band, th_cov) and < th_cov - coverage_band.
    All are sums over rows of df_impact, so the CSV reproduces them.
    """
    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
    paths = resolve_paths(station, year, doy_start, doy_end, th_cov)
    df = pd.read_parquet(
        paths["parquet_path"],
        columns=PARQUET_COLUMNS + [ndat_filter.NDAT_PARQUET_COLUMN],
    )
    df["minute"] = (df["01_sec_of_day"] // 60).astype(int)  # same definition as run_completeness()
    doys = df_results["DoY"].to_numpy()

    def per_doy(frame: pd.DataFrame):
        by_doy = frame.groupby("14_doy_utc")
        n_obs = by_doy.size().reindex(doys, fill_value=0).to_numpy()
        n_minutes = by_doy["minute"].nunique().reindex(doys, fill_value=0).to_numpy()
        return n_obs, n_minutes

    n_obs, n_minutes_before = per_doy(df)
    # Guard: the unfiltered minute count must be exactly the official one,
    # or the two definitions have drifted apart and the comparison is void.
    if not (n_minutes_before == df_results["n_minutes"].to_numpy()).all():
        raise RuntimeError(
            "Ndat-impact diagnostic: unfiltered unique minutes do not match "
            "run_completeness() - the minute definitions have diverged."
        )
    n_obs_retained, n_minutes_after = per_doy(ndat_filter.apply_ndat_filter(df, ndat_mode))

    df_impact = df_results[["DoY", "n_minutes", "coverage", "status"]].copy()
    df_impact["n_obs"] = n_obs
    df_impact["n_obs_retained"] = n_obs_retained
    df_impact["n_obs_discarded"] = n_obs - n_obs_retained
    df_impact["n_minutes_after_ndat"] = n_minutes_after
    df_impact["coverage_after_ndat"] = n_minutes_after / 1440.0
    df_impact["coverage_loss_pp"] = 100.0 * (df_impact["coverage"] - df_impact["coverage_after_ndat"])
    df_impact["above_th_after_ndat"] = df_impact["coverage_after_ndat"] >= th_cov

    valid = df_impact[df_impact["status"] == "valid"]
    below = valid[~valid["above_th_after_ndat"]]
    n_obs_valid = int(valid["n_obs"].sum())
    n_kept_valid = int(valid["n_obs_retained"].sum())
    minutes_before = int(valid["n_minutes"].sum())
    minutes_after = int(valid["n_minutes_after_ndat"].sum())
    n_doys = len(df_impact)
    overall_before = 100.0 * float(df_impact["n_minutes"].sum()) / (n_doys * 1440) if n_doys else 0.0
    overall_after = 100.0 * float(df_impact["n_minutes_after_ndat"].sum()) / (n_doys * 1440) if n_doys else 0.0

    summary = {
        "ndat_mode": ndat_mode,
        "ndat_label": ndat_config["label"],
        "th_cov": th_cov,
        "filter_on_valid_days": {
            "n_valid_days": len(valid),
            "n_obs": n_obs_valid,
            "n_obs_retained": n_kept_valid,
            "n_obs_discarded": n_obs_valid - n_kept_valid,
            "pct_obs_discarded": (
                round(100.0 * (n_obs_valid - n_kept_valid) / n_obs_valid, 2) if n_obs_valid else 0.0
            ),
            "station_minutes_before": minutes_before,
            "station_minutes_after": minutes_after,
            "station_minutes_lost": minutes_before - minutes_after,
            "pct_station_minutes_lost": (
                round(100.0 * (minutes_before - minutes_after) / minutes_before, 2) if minutes_before else 0.0
            ),
        },
        "completeness_all_doys": {
            "n_doys": n_doys,
            "n_valid_days_baseline": len(valid),
            "n_days_above_th_after_ndat": int(df_impact["above_th_after_ndat"].sum()),
            "n_valid_days_below_th_after_ndat": len(below),
            "doys_below_th_after_ndat": [int(d) for d in below["DoY"]],
            "overall_coverage_baseline_pct": round(overall_before, 2),
            "overall_coverage_after_ndat_pct": round(overall_after, 2),
            "overall_coverage_loss_pp": round(overall_before - overall_after, 2),
        },
        "post_ndat_coverage_of_valid_days": _post_ndat_coverage_groups(valid, th_cov, coverage_band),
    }
    return df_impact, summary


def _post_ndat_coverage_groups(valid: pd.DataFrame, th_cov: float, coverage_band: float) -> dict:
    """
    The official valid days (rows of df_impact with status == "valid")
    split by coverage_after_ndat into three exclusive groups, reusing the
    above_th_after_ndat column. Diagnostic only - the status is unchanged.
    """
    # Rounded so that e.g. 0.75 - 0.05 is exactly 0.70, the same float as
    # 1008/1440: a day at exactly the lower bound falls in the band.
    lower_bound = round(th_cov - coverage_band, 10)
    above = valid["above_th_after_ndat"]
    in_band = ~above & (valid["coverage_after_ndat"] >= lower_bound)
    below_band = ~above & (valid["coverage_after_ndat"] < lower_bound)
    groups = {
        "threshold": th_cov,
        "band": coverage_band,
        "lower_bound": lower_bound,
        "n_valid_days": len(valid),
        "n_at_or_above_threshold": int(above.sum()),
        "n_in_band_below_threshold": int(in_band.sum()),
        "n_below_band": int(below_band.sum()),
    }
    # The three groups must cover every valid day exactly once.
    if (groups["n_at_or_above_threshold"] + groups["n_in_band_below_threshold"]
            + groups["n_below_band"] != groups["n_valid_days"]):
        raise RuntimeError(
            "Ndat-impact diagnostic: post-Ndat coverage groups do not add up "
            "to the number of valid days."
        )
    return groups


# ======================================================================
# [E] SUMMARY STATISTICS (derived from run_completeness() output)
# ======================================================================
def compute_monthly_coverage(df_results: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Groups the per-day completeness table by calendar month (from the DoY).

    Pure function: no prints, no file writes. Used by
    compute_completeness_summary() for the highest/lowest monthly rates.

    Returns one row per month present in df_results: month, n_days,
    n_valid_days, valid_fraction.
    """
    df = df_results.copy()
    df["month"] = df["DoY"].apply(
        lambda doy: datetime.strptime(f"{year}-{int(doy):03d}", "%Y-%j").month
    )

    grouped = (
        df.groupby("month")
        .agg(
            n_days=("DoY", "size"),
            n_valid_days=("status", lambda s: int((s == "valid").sum())),
        )
        .reset_index()
    )
    grouped["valid_fraction"] = grouped["n_valid_days"] / grouped["n_days"]

    return grouped.sort_values("month").reset_index(drop=True)


def compute_completeness_summary(df_results: pd.DataFrame, year: int) -> dict:
    """
    Summary statistics of the per-day table produced by run_completeness()
    (completeness is not recomputed). Pure function: no prints, no file
    writes. Used by web_server.py for the Step 1 table.

    Three different coverage statistics, not to be confused:
      coverage_pct                = % of days that individually reach th_cov
                                    (valid-day rate).
      overall_coverage_pct        = % of all possible minutes of the DoY range
                                    actually observed, invalid days included:
                                    100 * sum(n_minutes) / (n_total_days * 1440).
      mean_valid_day_coverage_pct = the same formula over the valid days only;
                                    None when no day is valid (undefined, not 0).
    None of them is a validity threshold for the year.

    n_invalid_zero_min/n_invalid_partial split the invalid days into days
    with no observed minute (n_minutes == 0, whatever the reason in
    missing_day_note()) and days with data below th_cov.

    longest_invalid_streaks and best_months/worst_months keep ALL entries
    tied for the maximum/minimum (exact valid_fraction, compared before
    rounding); idxmax()/idxmin() or a strict ">" would keep only the first.
    Each month entry carries its valid-day count, not just its name.
    """
    n_total_days = int(len(df_results))
    n_valid_days = int((df_results["status"] == "valid").sum())
    n_invalid_days = n_total_days - n_valid_days
    coverage_pct = round(100.0 * n_valid_days / n_total_days, 2) if n_total_days else 0.0
    overall_coverage_pct = (
        round(100.0 * float(df_results["n_minutes"].sum()) / (n_total_days * 1440), 2)
        if n_total_days else 0.0
    )

    valid_days = df_results[df_results["status"] == "valid"]
    mean_valid_day_coverage_pct = (
        round(100.0 * float(valid_days["n_minutes"].sum()) / (n_valid_days * 1440), 2)
        if n_valid_days else None
    )

    invalid_days = df_results[df_results["status"] != "valid"]
    n_invalid_zero_min = int((invalid_days["n_minutes"] == 0).sum())
    n_invalid_partial = int((invalid_days["n_minutes"] > 0).sum())

    ordered = df_results.sort_values("DoY")
    doys = ordered["DoY"].to_numpy()
    is_invalid = (ordered["status"] != "valid").to_numpy()

    streaks = []
    current_streak = 0
    current_start_doy = None
    for doy, invalid in zip(doys, is_invalid):
        if invalid:
            if current_streak == 0:
                current_start_doy = doy
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append((current_streak, int(current_start_doy), int(doy) - 1))
            current_streak = 0
    if current_streak > 0:
        streaks.append((current_streak, int(current_start_doy), int(doys[-1])))

    longest_invalid_streak = max((length for length, _, _ in streaks), default=0)
    longest_invalid_streaks = [
        {"start_doy": start, "end_doy": end}
        for length, start, end in streaks
        if length == longest_invalid_streak
    ]

    monthly = compute_monthly_coverage(df_results, year)
    max_fraction = monthly["valid_fraction"].max()
    min_fraction = monthly["valid_fraction"].min()

    def _month_entries(rows: pd.DataFrame) -> list[dict]:
        return [
            {
                "month": calendar.month_abbr[int(row["month"])],
                "n_valid_days": int(row["n_valid_days"]),
                "n_days": int(row["n_days"]),
                "pct": round(100.0 * float(row["valid_fraction"]), 1),
            }
            for _, row in rows.iterrows()
        ]

    best_months = _month_entries(monthly[monthly["valid_fraction"] == max_fraction])
    worst_months = _month_entries(monthly[monthly["valid_fraction"] == min_fraction])

    return {
        "n_valid_days": n_valid_days,
        "n_total_days": n_total_days,
        "n_invalid_days": n_invalid_days,
        "coverage_pct": coverage_pct,
        "overall_coverage_pct": overall_coverage_pct,
        "mean_valid_day_coverage_pct": mean_valid_day_coverage_pct,
        "n_invalid_zero_min": n_invalid_zero_min,
        "n_invalid_partial": n_invalid_partial,
        "longest_invalid_streak": int(longest_invalid_streak),
        "longest_invalid_streaks": longest_invalid_streaks,
        "best_months": best_months,
        "worst_months": worst_months,
    }


# ======================================================================
# [F] OUTPUT HELPERS
# ======================================================================
def save_results_csv(df_results: pd.DataFrame, output_csv: Path) -> None:
    """Saves the daily completeness table to CSV."""
    df_results.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")


@plot_style.with_font_sizes(sizes=plot_style.font_sizes_plus(3))
def create_coverage_plot(
    df_results: pd.DataFrame,
    th_cov: float,
    plot_title: str,
    output_png: Path,
    show_plot: bool = True,
) -> None:
    """
    Saves the annual coverage-vs-DoY plot. The PNG is always written;
    show_plot only controls whether an interactive window also opens
    (console mode only).
    """
    n_valid = int((df_results["status"] == "valid").sum())
    n_total = len(df_results)
    annual_valid_pct = 100.0 * n_valid / n_total if n_total > 0 else 0.0

    plt.figure(figsize=(10, 5))

    # Main series: daily coverage
    plt.plot(
        df_results["DoY"],
        df_results["coverage"],
        linestyle="-",
        label="Daily coverage"
    )

    # Threshold line (th_cov)
    plt.axhline(
        y=th_cov,
        linestyle="--",
        linewidth=1,
        label=f"Coverage threshold ({th_cov})"
    )

    plt.xlabel("Day of Year (DoY)")
    plt.ylabel("Daily coverage (n_minutes / 1440)")
    plt.title(plot_title)
    plt.grid(True, alpha=0.3)

    # Legend inside the plot, centre-right: clear of the top band where
    # the daily coverage usually lies
    plt.legend(loc="center right")

    # th_cov is a per-day threshold; the footer states it as such so the
    # valid-day rate is not read as a value that must itself reach th_cov.
    footer_text = (
        f"Valid days: {n_valid}/{n_total} ({annual_valid_pct:.2f}%) — "
        f"daily threshold: coverage >= {th_cov * 100:.0f}%"
    )

    plt.figtext(
        0.5, 0.01,
        footer_text,
        ha="center",
        fontsize=plt.rcParams["font.size"]
    )

    # Bottom band reserved for the footer figtext, which tight_layout()
    # ignores - otherwise it touches the x-axis label.
    plt.tight_layout(rect=(0, 0.06, 1, 1))

    # Always save (traceability)
    plt.savefig(output_png, dpi=200)
    print(f"Plot saved to {output_png}")

    if show_plot:
        try:
            plt.show()
        except Exception as e:
            print(f"(INFO) Could not open plot window: {e}")

    plt.close()


@plot_style.with_font_sizes(sizes=plot_style.font_sizes_plus(3))
def create_ndat_impact_plot(
    df_impact: pd.DataFrame,
    th_cov: float,
    plot_title: str,
    ndat_label: str,
    output_png: Path,
    show_plot: bool = True,
) -> None:
    """
    DIAGNOSTIC plot, separate from the official one: daily coverage
    without and with the Ndat criterion, with the same th_cov line. The
    footer gives the official valid days and how many of them would stay
    >= th_cov after the filter.
    """
    is_valid = df_impact["status"] == "valid"
    n_valid = int(is_valid.sum())
    n_kept = int((is_valid & df_impact["above_th_after_ndat"]).sum())
    th_label = f"{th_cov * 100:g}%"  # 0.75 -> "75%", 0.7 -> "70%"
    ndat_short = ndat_label.replace(" = ", "=")  # "Ndat = 60" -> "Ndat=60"

    plt.figure(figsize=(10, 5))

    plt.plot(df_impact["DoY"], df_impact["coverage"], linestyle="-", label="Original daily coverage")
    plt.plot(
        df_impact["DoY"],
        df_impact["coverage_after_ndat"],
        linestyle="-",
        label=f"Daily coverage after {ndat_label}",
    )
    plt.axhline(y=th_cov, linestyle="--", linewidth=1, label=f"Coverage threshold ({th_label})")

    plt.xlabel("Day of Year (DoY)")
    plt.ylabel("Daily coverage (n_minutes / 1440)")
    plt.title(plot_title)
    plt.grid(True, alpha=0.3)
    # "best": with three entries a fixed position may cover the coverage series.
    plt.legend(loc="best")

    footer_text = f"Valid days before Ndat: {n_valid} — Valid days after {ndat_short}: {n_kept}"
    plt.figtext(0.5, 0.01, footer_text, ha="center", fontsize=plt.rcParams["font.size"])

    # Bottom band reserved for the footer figtext, which tight_layout()
    # ignores - otherwise it touches the x-axis label.
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(output_png, dpi=200)
    print(f"Ndat-impact plot saved to {output_png}")

    if show_plot:
        try:
            plt.show()
        except Exception as e:
            print(f"(INFO) Could not open plot window: {e}")

    plt.close()


def print_ndat_impact_summary(summary: dict) -> None:
    """
    Console report of run_ndat_impact()'s summary, with the same wording as
    the web table and the diagnostic plot. Observations and minutes refer
    to the valid days; annual coverage to every DoY.
    """
    f = summary["filter_on_valid_days"]
    c = summary["completeness_all_doys"]
    ndat = summary["ndat_label"].replace(" = ", "=")  # "Ndat = 60" -> "Ndat=60"
    th = f"{summary['th_cov'] * 100:g}%"  # 0.75 -> "75%"
    rows = [
        ("Observations discarded [%]", f"{f['pct_obs_discarded']:.2f}%"),
        ("Temporal minutes lost [%]", f"{f['pct_station_minutes_lost']:.2f}%"),
        ("Valid days before Ndat", c["n_valid_days_baseline"]),
        (f"Valid days after {ndat}", c["n_days_above_th_after_ndat"]),
        (f"Days below {th} after Ndat", c["n_valid_days_below_th_after_ndat"]),
        ("Annual temporal coverage before -> after Ndat",
         f"{c['overall_coverage_baseline_pct']:.2f}% -> {c['overall_coverage_after_ndat_pct']:.2f}%"),
    ]
    g = summary["post_ndat_coverage_of_valid_days"]
    lo = f"{g['lower_bound'] * 100:g}%"  # th_cov - band, e.g. "70%"
    rows += [
        (f"Valid days >= {th} after Ndat", g["n_at_or_above_threshold"]),
        (f"Valid days {lo} to < {th} after Ndat", g["n_in_band_below_threshold"]),
        (f"Valid days < {lo} after Ndat", g["n_below_band"]),
    ]
    width = max(len(label) for label, _ in rows) + 1
    print(f"\nNdat impact on completeness ({ndat}, coverage threshold {th}) - informative only")
    for label, value in rows:
        print(f"  {label + ':':<{width}} {value}")


# ======================================================================
# [G] MAIN PROGRAM (console mode)
# ======================================================================
def main() -> None:
    df_results = run_completeness(STATION, YEAR, DOY_START, DOY_END, TH_COV)

    print("\nFinal daily completeness table (first rows):")
    print(df_results.head(30))

    n_valid = int((df_results["status"] == "valid").sum())
    n_total = len(df_results)
    print(f"\nSummary: valid days = {n_valid} / {n_total}  (threshold={TH_COV})")

    annual_valid_pct = 100.0 * n_valid / n_total if n_total > 0 else 0.0
    print(f"Annual total coverage (days with coverage >= {TH_COV}): {annual_valid_pct:.2f}%")

    paths = resolve_paths(STATION, YEAR, DOY_START, DOY_END, TH_COV)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)

    save_results_csv(df_results, paths["output_csv"])

    create_coverage_plot(
        df_results,
        th_cov=TH_COV,
        plot_title=PLOT_TITLE,
        output_png=paths["output_png"],
        show_plot=SHOW_PLOT,
    )

    # Optional diagnostic - runs after the official outputs are written and
    # only adds its own files under 1_completeness/ndat_impact/.
    if SHOW_NDAT_IMPACT:
        df_impact, impact = run_ndat_impact(
            df_results, STATION, YEAR, DOY_START, DOY_END, TH_COV, NDAT_MODE,
        )
        impact_paths = resolve_ndat_impact_paths(STATION, YEAR, DOY_START, DOY_END, TH_COV, NDAT_MODE)
        impact_paths["diag_dir"].mkdir(parents=True, exist_ok=True)
        save_results_csv(df_impact, impact_paths["output_csv"])
        create_ndat_impact_plot(
            df_impact,
            th_cov=TH_COV,
            plot_title=f"Ndat impact on daily coverage - Station {STATION} ({YEAR})",
            ndat_label=impact["ndat_label"],
            output_png=impact_paths["output_png"],
            show_plot=SHOW_PLOT,
        )
        print_ndat_impact_summary(impact)


if __name__ == "__main__":
    main()
