#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The scientific indices that Steps 1-6 and the comparisons can analyse - the
only definition of them in the project.

Every index is a column of the Step 0 Parquet, which keeps all the RAW
columns, so the index is chosen in the analysis, not in Step 0. Official
RAW format (rotiS4.YYYYDDD, 12 whitespace-separated columns, no header):

     1 Second of the day (GPS time)       7 ROTI from LGF [TECU/min]
     2 Receiver name                      8 Ndat(ROTI L1): data used for ROTI L1 and S4
     3 Satellite number (1-32 GPS,        9 Ndat(ROTI LGF): data used for ROTI LGF
       >40 Galileo)                      10 S4 from L1
     4 Satellite elevation [0,90] deg    11 S4 from L2
     5 Satellite azimuth (-180,180] deg  12 ROTI from L2 [TECU/min]
       (0 North, 90 East)
     6 ROTI from L1 [TECU/min]

Each index uses the Ndat of its own computation: column 9 for ROTI LGF,
column 8 for the others (ROTI L2 included, as confirmed by the dataset's
author). Any Ndat criterion of ndat_filter applies to all of them.

A value counts as valid when it is finite: 0 is a valid value (S4 can be 0);
only NaN and infinities are not.

Fields of each entry:
  parquet_column            column of the Step 0 Parquet with the values
  ndat_column               Ndat column the Ndat criterion is applied to
  file_tag                  index tag in result file names (e.g. "ROTIL1")
  label, unit               display name and unit ("" = dimensionless: no "[]")
  plot_x_lim                x range of the Step 2 CCDF plots; None = from 0
                            to the largest value plotted
  allowed_ndat              ndat_filter operators accepted for this index
  scientifically_supported  False would keep the index out of every analysis

DEFAULT_INDEX is ROTI L1: results written before the index could be chosen
are ROTI L1 results, and keep their historical paths (completeness_paths).
"""

import math

DEFAULT_INDEX = "roti_l1"

NDAT_L1_COLUMN = "08_ndat_roti_l1"
NDAT_LGF_COLUMN = "09_ndat_roti_lgf"

# Every ndat_filter operator: ALL_NDAT, = N, >= N, < N.
ALL_NDAT_OPERATORS = ("all", "eq", "ge", "lt")

# Step 2's historical CCDF x range for ROTI, in TECU/min.
ROTI_PLOT_X_LIM = (0, 20)

INDEX_CONFIG = {
    "roti_l1": {
        "parquet_column": "06_roti_l1",
        "ndat_column": NDAT_L1_COLUMN,
        "file_tag": "ROTIL1",
        "label": "ROTI L1",
        "unit": "TECU/min",
        "plot_x_lim": ROTI_PLOT_X_LIM,
        "allowed_ndat": ALL_NDAT_OPERATORS,
        "scientifically_supported": True,
    },
    "roti_lgf": {
        "parquet_column": "07_roti_lgf",
        "ndat_column": NDAT_LGF_COLUMN,
        "file_tag": "ROTILGF",
        "label": "ROTI LGF",
        "unit": "TECU/min",
        "plot_x_lim": ROTI_PLOT_X_LIM,
        "allowed_ndat": ALL_NDAT_OPERATORS,
        "scientifically_supported": True,
    },
    "roti_l2": {
        "parquet_column": "12_roti_l2",
        "ndat_column": NDAT_L1_COLUMN,
        "file_tag": "ROTIL2",
        "label": "ROTI L2",
        "unit": "TECU/min",
        "plot_x_lim": ROTI_PLOT_X_LIM,
        "allowed_ndat": ALL_NDAT_OPERATORS,
        "scientifically_supported": True,
    },
    "s4_l1": {
        "parquet_column": "10_s4_l1",
        "ndat_column": NDAT_L1_COLUMN,
        "file_tag": "S4L1",
        "label": "S4 L1",
        "unit": "",
        "plot_x_lim": None,
        "allowed_ndat": ALL_NDAT_OPERATORS,
        "scientifically_supported": True,
    },
    "s4_l2": {
        "parquet_column": "11_s4_l2",
        "ndat_column": NDAT_L1_COLUMN,
        "file_tag": "S4L2",
        "label": "S4 L2",
        "unit": "",
        "plot_x_lim": None,
        "allowed_ndat": ALL_NDAT_OPERATORS,
        "scientifically_supported": True,
    },
}


def get_index(index: str) -> dict:
    """INDEX_CONFIG entry of `index`. Raises ValueError for an unknown index."""
    if not isinstance(index, str) or index not in INDEX_CONFIG:
        raise ValueError(f"Unknown index {index!r}: expected one of {', '.join(INDEX_CONFIG)}.")
    return INDEX_CONFIG[index]


def validate_index_supported(index: str) -> dict:
    """
    INDEX_CONFIG entry of `index`, which must be known and enabled for the
    analysis. Raises ValueError otherwise.
    """
    config = get_index(index)
    if not config["scientifically_supported"]:
        raise ValueError(
            f"Index {index!r} ({config['label']}) is in the Parquet but not enabled "
            "for the analysis (index_config.py: scientifically_supported)."
        )
    return config


def validate_ndat_for_index(index: str, ndat_config: dict) -> None:
    """Raises ValueError if the Ndat criterion (ndat_filter.validate_ndat_mode()) is not allowed for `index`."""
    config = get_index(index)
    if ndat_config["operator"] not in config["allowed_ndat"]:
        raise ValueError(
            f"Ndat criterion '{ndat_config['mode']}' ({ndat_config['label']}) is not "
            f"available for {config['label']}."
        )


def axis_label(label: str, unit: str) -> str:
    """Axis label: ROTI L1 [TECU/min], or just S4 L1 for a dimensionless index (never "[]")."""
    return f"{label} [{unit}]" if unit else label


def value_with_unit(value_text: str, unit: str) -> str:
    """A value with its unit: 0.123 TECU/min, or just 0.123 for a dimensionless index."""
    return f"{value_text} {unit}" if unit else value_text


def plot_x_lim(config: dict, max_value: float | None) -> tuple:
    """
    x range of a Step 2 CCDF plot: the index's fixed plot_x_lim, or for
    plot_x_lim None, 0 to the largest plotted value (+5 %), so no value is
    cut off. max_value is that largest value (None or NaN if nothing is
    plotted).
    """
    if config["plot_x_lim"] is not None:
        return tuple(config["plot_x_lim"])
    if max_value is None or not math.isfinite(max_value) or max_value <= 0:
        return (0, 1)
    return (0, max_value * 1.05)


def public_list() -> list[dict]:
    """Indices for the web UI, in definition order: key, label, unit, supported."""
    return [
        {
            "index": key,
            "label": config["label"],
            "unit": config["unit"],
            "file_tag": config["file_tag"],
            "supported": config["scientifically_supported"],
            "default": key == DEFAULT_INDEX,
        }
        for key, config in INDEX_CONFIG.items()
    ]
