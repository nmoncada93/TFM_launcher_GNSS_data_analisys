#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ndat quality criterion for ROTI L1 scientific analyses (Steps 2-6).

Single source of truth for the four Ndat modes and the one function that
applies them - every step script that needs to restrict observations by
Ndat(ROTI L1) imports this module instead of reimplementing the filter
(CLAUDE.md section 14: this is a methodological criterion, not a
technical detail to duplicate per script).

Deliberately does not use the existing 23_qc_ndat_ok flag (it only
represents Ndat==60, not the other three modes) nor 25_qc_all_ok (bundles
elevation/plausibility filters that are separate, still-undecided QC
criteria - CLAUDE.md section 9, qc-open-decisions.md memory) - works
directly off 08_ndat_roti_l1 instead, for all four modes alike.

Step 0 and Step 1 do not use this module: Step 0 keeps every observation
unfiltered, and Step 1 measures temporal availability, independent of
Ndat (CLAUDE.md section 7-8).
"""

import pandas as pd

NDAT_PARQUET_COLUMN = "08_ndat_roti_l1"
NDAT_DEFAULT_MODE = "eq60"

NDAT_MODES = {
    "eq60": {
        "label": "Ndat = 60",
        "description": "Complete interval - TFM official default",
        "dir_tag": "ndat_eq60",
        "file_tag": "NDATEQ60",
    },
    "ge30": {
        "label": "30 <= Ndat <= 60",
        "description": "Complete and sufficiently sampled partial intervals",
        "dir_tag": "ndat_ge30",
        "file_tag": "NDATGE30",
    },
    "all": {
        "label": "No Ndat filter",
        "description": "Diagnostic only",
        "dir_tag": "ndat_all",
        "file_tag": "NDATALL",
    },
    "lt30": {
        "label": "Ndat < 30",
        "description": "Low-sample diagnostic only",
        "dir_tag": "ndat_lt30",
        "file_tag": "NDATLT30",
    },
}


def validate_ndat_mode(mode: str) -> dict:
    """
    Checks that `mode` is one of the four supported Ndat criteria and
    returns its NDAT_MODES entry - same role as validate_index_supported()
    in 2-ccdf.py, one more axis.
    """
    if mode not in NDAT_MODES:
        raise ValueError(
            f"Unknown Ndat mode '{mode}'. Known modes: {sorted(NDAT_MODES)}"
        )
    return NDAT_MODES[mode]


def apply_ndat_filter(
    df: pd.DataFrame,
    mode: str,
    ndat_column: str = NDAT_PARQUET_COLUMN,
) -> pd.DataFrame:
    """
    Returns a new DataFrame with only the rows of `df` that satisfy the
    given Ndat mode - never mutates `df`. Row order is preserved (boolean
    masking only, no sort/groupby).

    "all" is a pure pass-through and does not require `ndat_column` to be
    present. The other three modes do, and raise KeyError with an
    explicit message if it is missing, instead of failing silently.
    """
    validate_ndat_mode(mode)

    if mode == "all":
        return df.copy()

    if ndat_column not in df.columns:
        raise KeyError(
            f"Ndat mode '{mode}' requires column '{ndat_column}', "
            f"which is not present in the given DataFrame."
        )

    ndat = df[ndat_column]
    if mode == "eq60":
        mask = ndat == 60
    elif mode == "ge30":
        mask = (ndat >= 30) & (ndat <= 60)
    elif mode == "lt30":
        mask = ndat < 30

    return df.loc[mask].copy()
