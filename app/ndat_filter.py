#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ndat criterion of the observations (Steps 1-6 and comparisons).

Ndat is the number of samples used to compute each value: column
08_ndat_roti_l1 for ROTI L1, S4 L1, S4 L2 and ROTI L2, column
09_ndat_roti_lgf for ROTI LGF. Which column applies to the analysed index
comes from index_config.py (ndat_column), passed to apply_ndat_filter();
NDAT_PARQUET_COLUMN is only its default, ROTI L1's. NDAT_MAX = 60 samples
is a complete one-minute interval, so Ndat is a whole number between
NDAT_MIN and NDAT_MAX.

A criterion is identified by a canonical mode string and interpreted here;
no criterion value is hard-coded:

  "all"     every observation (ALL_NDAT)
  "eq<N>"   Ndat == N
  "ge<N>"   N <= Ndat <= NDAT_MAX
  "lt<N>"   Ndat < N

N is a whole number between NDAT_MIN and NDAT_MAX, written without sign or
leading zeros (e.g. "eq60", "ge30", "lt30", "eq33"), so every criterion has
exactly one name. validate_ndat_mode() returns its label, dir_tag and
file_tag, from which every step builds its folders and file names;
build_ndat_mode() builds the mode from an operator symbol and a number
(e.g. for a form), so free text never has to be parsed. "eq60" is the
official criterion of the study.

The filter works directly on the Ndat column rather than on the Step 0 QC
flags: 23_qc_ndat_ok only encodes Ndat(ROTI L1) == 60, and 25_qc_all_ok also
combines elevation and plausibility checks that are not applied in this study.

The filter selects observations. Each criterion has its own Step 1
completeness (valid days), and Steps 2-6 use the valid days and the
observations of the same criterion.
"""

import numbers
import re

import pandas as pd

NDAT_PARQUET_COLUMN = "08_ndat_roti_l1"
NDAT_DEFAULT_MODE = "eq60"

# Samples in a complete one-minute interval: the largest Ndat.
NDAT_MAX = 60
# At least one sample is needed for a value.
NDAT_MIN = 1

ALL_MODE = "all"

# Supported operators: symbol (build_ndat_mode), label and mask. "ge" keeps
# the NDAT_MAX upper bound of the original ge30 criterion (30 <= Ndat <= 60).
_OPERATORS = {
    "eq": {"symbol": "=", "label": "Ndat = {n}", "mask": lambda ndat, n: ndat == n},
    "ge": {"symbol": ">=", "label": "{n} <= Ndat <= " + str(NDAT_MAX),
           "mask": lambda ndat, n: (ndat >= n) & (ndat <= NDAT_MAX)},
    "lt": {"symbol": "<", "label": "Ndat < {n}", "mask": lambda ndat, n: ndat < n},
}
_MODE_PATTERN = re.compile(r"^(eq|ge|lt)([+-]?\d+)$")


def validate_ndat_mode(mode: str) -> dict:
    """
    Interprets a canonical Ndat mode and returns its configuration: mode,
    operator ("all", "eq", "ge" or "lt"), value (None for "all"), label,
    dir_tag ("ndat_<mode>") and file_tag ("NDAT<MODE>"). Raises ValueError
    with an explicit message for anything else (unknown operator, missing
    or non-integer number, number outside [NDAT_MIN, NDAT_MAX], non-canonical
    spelling, non-string input).
    """
    if mode == ALL_MODE:
        return {"mode": ALL_MODE, "operator": ALL_MODE, "value": None,
                "label": "No Ndat filter", "dir_tag": "ndat_all", "file_tag": "NDATALL"}

    match = _MODE_PATTERN.match(mode) if isinstance(mode, str) else None
    if match is None:
        raise ValueError(
            f"Unknown Ndat mode {mode!r}: expected 'all' or eq/ge/lt followed by a whole "
            "number of samples, e.g. 'eq60', 'ge40' or 'lt30'."
        )

    operator, value = match.group(1), int(match.group(2))
    if not NDAT_MIN <= value <= NDAT_MAX:
        raise ValueError(
            f"Invalid Ndat mode {mode!r}: Ndat is a number of samples between "
            f"{NDAT_MIN} and {NDAT_MAX}."
        )
    canonical = f"{operator}{value}"
    if mode != canonical:
        raise ValueError(f"Non-canonical Ndat mode {mode!r}: use {canonical!r}.")

    return {
        "mode": canonical,
        "operator": operator,
        "value": value,
        "label": _OPERATORS[operator]["label"].format(n=value),
        "dir_tag": f"ndat_{canonical}",
        "file_tag": f"NDAT{canonical.upper()}",
    }


def build_ndat_mode(operator: str, value: int | None = None) -> str:
    """
    Canonical mode from a structured choice: ("all") -> "all",
    ("=", 33) -> "eq33", (">=", 40) -> "ge40", ("<", 30) -> "lt30".
    value must be a whole number; the result is validated like any mode.
    """
    if operator == ALL_MODE:
        if value is not None:
            raise ValueError(f"Ndat operator 'all' takes no value (got {value!r}).")
        return ALL_MODE
    by_symbol = {spec["symbol"]: name for name, spec in _OPERATORS.items()}
    if operator not in by_symbol:
        raise ValueError(
            f"Unknown Ndat operator {operator!r}: expected 'all', "
            + ", ".join(repr(s) for s in by_symbol) + "."
        )
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"Invalid Ndat value {value!r}: expected a whole number of samples.")
    mode = f"{by_symbol[operator]}{int(value)}"
    validate_ndat_mode(mode)
    return mode


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
    config = validate_ndat_mode(mode)

    if config["operator"] == ALL_MODE:
        return df.copy()

    if ndat_column not in df.columns:
        raise KeyError(
            f"Ndat mode '{mode}' requires column '{ndat_column}', "
            f"which is not present in the given DataFrame."
        )

    mask = _OPERATORS[config["operator"]]["mask"](df[ndat_column], config["value"])
    return df.loc[mask].copy()
