#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ndat criterion for ROTI L1 observations (Steps 2-6).

Ndat (column 08_ndat_roti_l1) is the number of samples used to compute each
ROTI L1 value; Ndat = 60 corresponds to a complete one-minute interval.
NDAT_MODES defines the supported criteria, and "eq60" is the official
criterion of the study. Every step imports this module, so the criterion is
defined and applied in a single place.

The filter works directly on 08_ndat_roti_l1 rather than on the Step 0 QC
flags: 23_qc_ndat_ok only encodes Ndat == 60, and 25_qc_all_ok also combines
elevation and plausibility checks that are not applied in this study.

The filter selects observations, not days. Valid days are decided in Step 1
from temporal coverage without Ndat, and the coverage threshold is not
re-applied after filtering. Step 1 uses this module only for its optional
Ndat-impact diagnostic, which never changes the valid/invalid status.
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
    """Return the NDAT_MODES entry for `mode`; raise ValueError if unknown."""
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
    Return a copy of `df` restricted to the rows that satisfy the Ndat mode.
    Row order and index are preserved. "all" returns every row and does not
    need `ndat_column`; the other modes raise KeyError if it is missing.
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
