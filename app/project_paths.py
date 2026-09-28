#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single source of truth for where results are written and read.

Every step script (0-6 and the X.2 comparisons) and web_server.py import
this module instead of writing Path("results"). A bare Path("results") is
resolved against the current working directory, so the same script could
write to <project>/results/ or to <project>/app/results/ depending only on
the folder it was started from. Anchoring to this file instead makes the
location independent of the CWD: console and web, started from the
project root, from app/ or from anywhere else, all use the same
<project>/results/ folder.

Only the root of the tree is fixed here - each script still builds its
own results/<station>/<year>/... layout below it, unchanged. The RAW input
folder (0-toParquet.py's DATA_DIR) is not handled here.
"""

from pathlib import Path

# app/project_paths.py -> app/ -> project root. resolve() gives an
# absolute, normalized path (no "..", same form in every process), so
# web_server.py's path.relative_to(RESULTS_ROOT) always matches the paths
# the step scripts build from this same constant.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
