#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Location of the Step 1 completeness outputs of each index and Ndat criterion.

Step 1 writes them and the later steps read them; all of them resolve the
paths here, so the naming convention is defined in one place. Completeness
depends on the index (its Ndat column and which values are finite), so the
index is part of the path - except for ROTI L1, which keeps the historical
names it had before the index could be chosen:

  ROTI L1 (index_config.DEFAULT_INDEX):
  results/{STATION}/{YEAR}/1_completeness/<dir_tag>/
      coverage_{STATION}_{YEAR}_DOY{start}_{end}_coverageTH{th_cov}_<NDAT_TAG>.csv
      coverage_plot_{STATION}_{YEAR}_DOY{start}_{end}_coverageTH{th_cov}_<NDAT_TAG>.png

  any other index, e.g. s4_l1:
  results/{STATION}/{YEAR}/1_completeness/s4_l1/<dir_tag>/
      coverage_{STATION}_{YEAR}_S4L1_DOY{start}_{end}_coverageTH{th_cov}_<NDAT_TAG>.csv
      coverage_plot_{STATION}_{YEAR}_S4L1_DOY{start}_{end}_coverageTH{th_cov}_<NDAT_TAG>.png

dir_tag and NDAT_TAG are the dir_tag and file_tag that
ndat_filter.validate_ndat_mode() returns for the criterion. Index folders
never start with "ndat_", so they are never mistaken for a criterion.

run_tag() gives the DoY range / coverage threshold fragment of these names
("DOY1_366_coverageTH0.75"). Steps 2-6 and the comparisons put the same
fragment in their own file names, so results of different DoY ranges or
coverage thresholds never overwrite or read each other. The threshold is
written by format_th_cov(), so equal values always give the same name.

list_ndat_modes() finds which criteria Step 1 has actually produced for one
station/year/index/DoY range/threshold, from the <dir_tag> folders on disk.
"""

import numbers
from pathlib import Path

import numpy as np
import pandas as pd

import index_config as indices
import ndat_filter
import project_paths

# Name shown for "all" in criterion lists. validate_ndat_mode() keeps its
# own label ("No Ndat filter"), which plots and titles use.
ALL_NDAT_LABEL = "ALL_NDAT"


def format_th_cov(th_cov: float) -> str:
    """
    Canonical text of a coverage threshold in file names: the shortest
    decimal that reads back as the same float, without exponent or trailing
    zeros (1 and 1.0 -> "1", 0.75 -> "0.75", 0.751 -> "0.751"). Only the
    text is normalised; the value is not rounded and the threshold used in
    the calculation is not touched. Raises TypeError if th_cov is not a
    real number (strings and booleans included) and ValueError if it is
    NaN or infinite.
    """
    if isinstance(th_cov, bool) or not isinstance(th_cov, numbers.Real):
        raise TypeError(f"th_cov must be a real number, got {th_cov!r} ({type(th_cov).__name__}).")
    value = float(th_cov)
    if not np.isfinite(value):
        raise ValueError(f"th_cov must be a finite number, got {th_cov!r}.")
    return np.format_float_positional(value, trim="-")


def run_tag(doy_start: int, doy_end: int, th_cov: float) -> str:
    """DoY range and coverage threshold of a run, as written in every file name."""
    return f"DOY{doy_start}_{doy_end}_coverageTH{format_th_cov(th_cov)}"


def index_name_tag(station: str, year: int, index: str) -> str:
    """
    Station/year/index part of the Step 1 file names: "KOUG_2024" for ROTI
    L1 (historical names), "KOUG_2024_S4L1" for any other index.
    """
    config = indices.get_index(index)
    if index == indices.DEFAULT_INDEX:
        return f"{station}_{year}"
    return f"{station}_{year}_{config['file_tag']}"


def resolve_completeness_paths(
    station: str, year: int, doy_start: int, doy_end: int, th_cov: float, ndat_mode: str,
    *, index: str,
) -> dict:
    """
    Step 1 completeness paths for one station/year/index/DoY range/threshold/
    Ndat criterion: completeness_dir (1_completeness/, or
    1_completeness/<index>/ for an index other than ROTI L1), results_dir
    (its <dir_tag> subfolder), output_csv and output_png. Raises ValueError
    if index or ndat_mode is unknown.
    """
    ndat_config = ndat_filter.validate_ndat_mode(ndat_mode)
    completeness_dir = project_paths.RESULTS_ROOT / station / str(year) / "1_completeness"
    if index != indices.DEFAULT_INDEX:
        completeness_dir = completeness_dir / index
    results_dir = completeness_dir / ndat_config["dir_tag"]
    output_tag = (
        f"{index_name_tag(station, year, index)}_{run_tag(doy_start, doy_end, th_cov)}"
        f"_{ndat_config['file_tag']}"
    )

    return {
        "completeness_dir": completeness_dir,
        "results_dir": results_dir,
        "output_csv": results_dir / f"coverage_{output_tag}.csv",
        "output_png": results_dir / f"coverage_plot_{output_tag}.png",
    }


def require_completeness_csv(
    station: str, year: int, doy_start: int, doy_end: int, th_cov: float, ndat_mode: str,
    *, index: str,
) -> Path:
    """
    The Step 1 completeness CSV of this exact combination. Raises
    FileNotFoundError if Step 1 has not produced it - there is no fallback
    to the completeness of another criterion or index.
    """
    csv_path = resolve_completeness_paths(
        station, year, doy_start, doy_end, th_cov, ndat_mode, index=index
    )["output_csv"]
    if not csv_path.exists():
        label = ndat_filter.validate_ndat_mode(ndat_mode)["label"]
        index_label = indices.get_index(index)["label"]
        raise FileNotFoundError(
            f"Step 1 completeness of {index_label} for Ndat criterion '{ndat_mode}' ({label}) "
            f"not found: {csv_path}\n"
            "Run Step 1 with the same index and Ndat criterion first."
        )
    return csv_path


def ndat_display_label(ndat_mode: str) -> str:
    """Label of a criterion in criterion lists: ALL_NDAT_LABEL for "all", otherwise its validate_ndat_mode() label."""
    if ndat_mode == ndat_filter.ALL_MODE:
        return ALL_NDAT_LABEL
    return ndat_filter.validate_ndat_mode(ndat_mode)["label"]


def _mode_order(ndat_mode: str) -> tuple:
    """Sort key: "all" first, then by operator and value (eq.., ge.., lt..)."""
    config = ndat_filter.validate_ndat_mode(ndat_mode)
    return (config["operator"] != ndat_filter.ALL_MODE, config["operator"], config["value"] or 0)


def list_ndat_modes(
    station: str, year: int, doy_start: int, doy_end: int, th_cov: float, *, index: str,
) -> list[dict]:
    """
    Criteria whose Step 1 completeness exists for exactly this station, year,
    index, DoY range and th_cov: one {"mode", "label", "n_valid_days"} per
    criterion, "all" first and then by operator and value. Empty if there is
    none.

    The candidates are the ndat_* folders of the index's completeness_dir,
    not a fixed list: a folder counts only if its name is a valid criterion
    (so ndat_impact, index folders and anything non-canonical are ignored)
    and it holds the completeness CSV that resolve_completeness_paths()
    gives for this exact combination. n_valid_days is the number of
    status == "valid" rows of that CSV.
    """
    completeness_dir = resolve_completeness_paths(
        station, year, doy_start, doy_end, th_cov, ndat_filter.ALL_MODE, index=index
    )["completeness_dir"]
    if not completeness_dir.is_dir():
        return []

    found = []
    for folder in completeness_dir.iterdir():
        if not folder.is_dir() or not folder.name.startswith("ndat_"):
            continue
        mode = folder.name[len("ndat_"):]
        try:
            ndat_filter.validate_ndat_mode(mode)
        except ValueError:
            continue
        csv_path = resolve_completeness_paths(
            station, year, doy_start, doy_end, th_cov, mode, index=index
        )["output_csv"]
        if not csv_path.is_file():
            continue
        status = pd.read_csv(csv_path, usecols=["status"])["status"]
        found.append({
            "mode": mode,
            "label": ndat_display_label(mode),
            "n_valid_days": int((status == "valid").sum()),
        })
    return sorted(found, key=lambda entry: _mode_order(entry["mode"]))


def list_ndat_modes_for_stations(
    stations: list[str], year: int, doy_start: int, doy_end: int, th_cov: float, *, index: str,
) -> list[dict]:
    """
    Union of list_ndat_modes() over several stations for one index: one
    entry per criterion that at least one of them has, with mode, label,
    stations (those that have it, in the order given), n_stations,
    total_stations and n_valid_days_by_station. Same order as
    list_ndat_modes().
    """
    by_mode: dict[str, dict] = {}
    for station in stations:
        for entry in list_ndat_modes(station, year, doy_start, doy_end, th_cov, index=index):
            item = by_mode.setdefault(entry["mode"], {
                "mode": entry["mode"], "label": entry["label"], "stations": [], "n_valid_days_by_station": {},
            })
            item["stations"].append(station)
            item["n_valid_days_by_station"][station] = entry["n_valid_days"]

    return [
        {**item, "n_stations": len(item["stations"]), "total_stations": len(stations)}
        for item in sorted(by_mode.values(), key=lambda item: _mode_order(item["mode"]))
    ]
