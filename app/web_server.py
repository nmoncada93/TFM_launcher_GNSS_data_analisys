#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local web interface of the pipeline.

Serves static/index.html, the results files (GET /results/...), the
station list (GET /api/stations) and one POST endpoint per action
(/api/run/step0 ... /api/run/step6, the cross-station comparisons and
/api/browse-folder). Every endpoint calls the step scripts' own functions
in this process - no subprocess and no second implementation of the
science. Standard library only (http.server).

Results are written to the same results/{STATION}/{YEAR}/... paths as in
console mode: the web UI is another way to run the same pipeline.

Steps 1-3 and the comparisons 2.2/3.2/4.2/5.2 expose pure functions that
receive the request parameters. Steps 0, 4, 5 and 6 read their parameters
from module globals in main(); they are run through _patched_globals,
which restores the console defaults afterwards.

Step 0 has no `station` field: it processes every station listed in
stations.json in one pass (each daily RAW file mixes all stations). Its
`raw_dir` field is the folder with that year's RAW files (default
step0.DATA_DIR), used directly as a local path - this process has the same
disk access as the console script.

/api/browse-folder opens a native Windows folder picker (Shell.Application's
BrowseForFolder, hosted by powershell.exe - see FOLDER_PICKER_PS1) so that
index.html can fill `raw_dir`. Browsers never expose an absolute path from
<input type="file" webkitdirectory>, and tkinter is not guaranteed in the
embeddable Python distribution; PowerShell and Shell32 are part of Windows,
so no extra dependency is needed.
"""

import contextlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# Makes pandas/numpy/matplotlib/pyarrow importable when this file is run by
# app/python_portable/python.exe, an embeddable distribution whose python311._pth
# has `import site` disabled and therefore never sees app/python_libraries on its own.
sys.path.insert(0, str(Path(__file__).parent / "python_libraries"))

import matplotlib
# Must run before the first `import matplotlib.pyplot` anywhere - including
# inside the step modules loaded below - or the backend may default to
# TkAgg. TkAgg is a GUI backend that is not safe outside the main thread,
# and ThreadingHTTPServer handles every request in its own thread; worker
# threads can then crash the whole process with "Tcl_AsyncDelete: async
# handler deleted by the wrong thread". Agg is headless and safe from any
# thread, with the same savefig() output.
matplotlib.use("Agg")

import pandas as pd

import project_paths

HOST = "127.0.0.1"
PORT = 8000

APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "static"
# <project>/results whatever the CWD - the same object every step script
# builds its paths from (project_paths.py), so the /results/... URLs below
# (path.relative_to(RESULTS_ROOT)) always match the files the steps write.
RESULTS_ROOT = project_paths.RESULTS_ROOT

# Stations used by the WEB mode only: codes and fixed UTC offsets. Located
# next to this file (not relative to CWD). The scientific scripts never read
# it - web_server.py injects it into them (explicit `stations=` arguments or
# _patched_globals), and each script keeps its own console defaults.
STATIONS_JSON = APP_DIR / "stations.json"


class StationsConfigError(Exception):
    """stations.json is missing or invalid - reported to the web, no fallback."""


def load_stations_config(path: Path = STATIONS_JSON) -> list[tuple[str, float]]:
    """
    Reads and validates stations.json, returning [(code, utc_offset_hours)]
    in file order. Read on every request that needs it, so edits to the file
    take effect without restarting the server. Any problem raises
    StationsConfigError - deliberately no silent fallback to a hardcoded list.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StationsConfigError(f"stations.json not found: {path}")
    except json.JSONDecodeError as exc:
        raise StationsConfigError(f"stations.json is not valid JSON: {exc}")

    stations = raw.get("stations") if isinstance(raw, dict) else None
    if not isinstance(stations, list):
        raise StationsConfigError('stations.json must contain a "stations" list.')
    if not stations:
        raise StationsConfigError('stations.json: the "stations" list is empty.')

    config = []
    for i, entry in enumerate(stations):
        if not isinstance(entry, dict):
            raise StationsConfigError(f"stations.json: entry {i} is not an object.")
        code = entry.get("code")
        if not isinstance(code, str) or not code.strip():
            raise StationsConfigError(f'stations.json: entry {i} has an empty or non-text "code".')
        if code in (c for c, _ in config):
            raise StationsConfigError(f'stations.json: duplicated station code "{code}".')
        if "utc_offset_hours" not in entry:
            raise StationsConfigError(f'stations.json: station "{code}" has no "utc_offset_hours".')
        offset = entry["utc_offset_hours"]
        # Any finite number, fractional offsets included (e.g. 5.5, 5.75).
        if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset):
            raise StationsConfigError(
                f'stations.json: station "{code}" has a non-numeric or non-finite '
                f'"utc_offset_hours": {offset!r}.'
            )
        config.append((code, offset))
    return config


def _unconfigured(requested: list[str], config: list[tuple[str, float]]) -> list[str]:
    """Requested station codes that are not in stations.json."""
    known = {code for code, _ in config}
    return [s for s in requested if s not in known]

# Shell.Application.BrowseForFolder is the classic Shell32 COM folder
# dialog (no .NET, no owner window), which comes to the foreground on its
# own. A WinForms FolderBrowserDialog owned by a hidden form may never
# become visible, leaving the request waiting with no dialog.
#
# `RootFolder` is left at 0 (Desktop: "This PC" and every drive) rather than
# the current raw_dir: BrowseForFolder's RootFolder *restricts* the visible
# tree to that folder, which would prevent choosing a folder on another
# drive. The script is static and never interpolated with request data
# ($InitialDir is not used), so a crafted `initial_dir` cannot inject
# anything.
FOLDER_PICKER_PS1 = """\
param([string]$InitialDir)

$shell = New-Object -ComObject Shell.Application
$BIF_RETURNONLYFSDIRS = 0x0001
$BIF_NEWDIALOGSTYLE = 0x0040
$folder = $shell.BrowseForFolder(0, "Select the RAW data folder", ($BIF_RETURNONLYFSDIRS -bor $BIF_NEWDIALOGSTYLE), 0)

if ($null -ne $folder) {
    Write-Output $folder.Self.Path
} else {
    Write-Output "__CANCELLED__"
}
"""

# matplotlib's pyplot state (current figure/axes, rcParams) is not
# thread-safe, and ThreadingHTTPServer handles every request in its own
# thread, so two concurrent plotting requests can corrupt each other's
# figures. Every handler that calls a save_*_plot()/create_*_plot() function
# is wrapped with @_serialize_plots so only one thread uses matplotlib at a
# time. Handlers that only read/check files are not wrapped, so they never
# wait behind plot generation.
_PLOT_LOCK = threading.Lock()


def _serialize_plots(handler):
    def wrapped(self, *args, **kwargs):
        with _PLOT_LOCK:
            return handler(self, *args, **kwargs)
    return wrapped


def _load_module(filename: str, module_name: str):
    """
    Dynamically loads a script whose filename is not a valid Python
    module name (starts with a digit).
    """
    spec = importlib.util.spec_from_file_location(module_name, APP_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step0 = _load_module("0-toParquet.py", "step0_to_parquet")
step1 = _load_module("1-completitud.py", "step1_completitud")
step2 = _load_module("2-ccdf.py", "step2_ccdf")
step2_2 = _load_module("2.2-ccdf_comparison.py", "step2_2_ccdf_comparison")
step3 = _load_module("3-temporal_variability.py", "step3_temporal_variability")
step3_2 = _load_module("3.2-temporal_comparison.py", "step3_2_temporal_comparison")
step4 = _load_module("4-hourly_variability.py", "step4_hourly_variability")
step4_2 = _load_module("4.2-hourly_comparison.py", "step4_2_hourly_comparison")
step5 = _load_module("5-daypart_variability.py", "step5_daypart_variability")
step5_2 = _load_module("5.2-daypart_comparison.py", "step5_2_daypart_comparison")
step6 = _load_module("6-month_hour_heatmaps.py", "step6_month_hour_heatmaps")


@contextlib.contextmanager
def _patched_globals(module, **overrides):
    """
    Temporarily overwrites module-level globals (for Steps 0 and 4-6, whose
    main() reads its parameters from globals) and restores the original
    values on exit, even if the wrapped call raises, so a web request never
    changes the console defaults.
    """
    originals = {name: getattr(module, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


def _json_safe(value):
    """NaN is not valid JSON - a station/month with no data becomes null."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _table_records(df):
    return [
        {key: _json_safe(v) for key, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _hours_at_max(df_hourly, freq_col):
    """
    All hours tied for the maximum of freq_col - idxmax() alone would
    silently return only the first one, hiding a genuine tie (e.g. two
    hours both at exactly 14.52%).
    """
    max_value = df_hourly[freq_col].max()
    tied = df_hourly[df_hourly[freq_col] == max_value]
    return tied["hour_local"].tolist(), float(max_value), int(tied["n_values"].iloc[0])


def _dayparts_at_max(df_daypart, freq_col):
    """
    Same tie-aware logic as _hours_at_max, keyed on daypart name instead
    of hour_local.
    """
    max_value = df_daypart[freq_col].max()
    tied = df_daypart[df_daypart[freq_col] == max_value]
    return tied["daypart"].tolist(), float(max_value), int(tied["n_values"].iloc[0])


MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".png": "image/png",
    ".csv": "text/csv; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


def guess_mime(path: Path) -> str:
    return MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404, f"Not found: {path}")
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", guess_mime(path))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stations_config(self):
        """stations.json, or None after answering with its error (no analysis runs)."""
        try:
            return load_stations_config()
        except StationsConfigError as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})
            return None

    def _reject_unconfigured(self, requested: list[str], config) -> bool:
        """True (after answering 400) if any requested station is not in stations.json."""
        unknown = _unconfigured(requested, config)
        if unknown:
            self._send_json(400, {
                "ok": False,
                "error": f"Station(s) not configured in stations.json: {', '.join(unknown)}",
            })
            return True
        return False

    def do_GET(self) -> None:
        route = urlparse(self.path).path

        if route in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html")
            return

        if route == "/api/stations":
            try:
                config = load_stations_config()
            except StationsConfigError as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})
                return
            self._send_json(200, {
                "ok": True,
                "stations": [{"code": c, "utc_offset_hours": o} for c, o in config],
            })
            return

        if route.startswith("/results/"):
            relative = route[len("/results/"):]
            if ".." in relative.split("/"):
                self.send_error(400, "Invalid path.")
                return
            self._send_file(RESULTS_ROOT / relative)
            return

        self.send_error(404, f"Unknown route: {route}")

    def do_POST(self) -> None:
        route = urlparse(self.path).path

        if route == "/api/browse-folder":
            self._handle_browse_folder()
            return

        if route == "/api/run/step0":
            self._handle_run_step0()
            return

        if route == "/api/run/step1":
            self._handle_run_step1()
            return

        if route == "/api/run/step2":
            self._handle_run_step2()
            return

        if route == "/api/run/step2/ccdf-compare":
            self._handle_run_step2_ccdf_compare()
            return

        if route == "/api/run/step2/ccdf-grid":
            self._handle_run_step2_ccdf_grid()
            return

        if route == "/api/run/step2/cross-station-ccdf":
            self._handle_run_step2_cross_station_ccdf()
            return

        if route == "/api/run/step2.2":
            self._handle_run_step2_2()
            return

        if route == "/api/run/step3":
            self._handle_run_step3()
            return

        if route == "/api/run/step3/comparison":
            self._handle_run_step3_comparison()
            return

        if route == "/api/run/step4":
            self._handle_run_step4()
            return

        if route == "/api/run/step4/comparison":
            self._handle_run_step4_comparison()
            return

        if route == "/api/run/step5":
            self._handle_run_step5()
            return

        if route == "/api/run/step5/comparison":
            self._handle_run_step5_comparison()
            return

        if route == "/api/run/step6":
            self._handle_run_step6()
            return

        if route == "/api/run/step6/cross-station-maxima":
            self._handle_run_step6_cross_station_maxima()
            return

        self.send_error(404, f"Unknown route: {route}")

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _handle_browse_folder(self) -> None:
        """
        Opens a native Windows folder picker and returns the chosen path
        (see FOLDER_PICKER_PS1 and the module docstring).

        -STA is explicit, although Shell.Application does not strictly
        require it. stdin=DEVNULL: any unexpected interactive prompt fails
        immediately instead of leaving the request hanging.

        90 s timeout, so the request never waits indefinitely; index.html's
        browseRawDir() also aborts client-side around the same time. If the
        user needs longer, "Browse..." can be clicked again.

        subprocess.run() blocks only this request's thread
        (ThreadingHTTPServer), not other concurrent requests.
        """
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        try:
            initial_dir = str(params.get("initial_dir") or step0.DATA_DIR)

            script_path = Path(tempfile.gettempdir()) / "gnss_web_folder_picker.ps1"
            script_path.write_text(FOLDER_PICKER_PS1, encoding="utf-8")

            command = [
                "powershell.exe", "-NoProfile", "-NonInteractive", "-STA",
                "-ExecutionPolicy", "Bypass", "-File", str(script_path),
                initial_dir,
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=90,
                stdin=subprocess.DEVNULL,
            )

            if result.returncode != 0:
                self._send_json(500, {
                    "ok": False,
                    "error": (
                        f"Folder picker failed (exit {result.returncode}). "
                        f"stderr: {result.stderr.strip() or '(empty)'} | "
                        f"stdout: {result.stdout.strip() or '(empty)'}"
                    ),
                })
                return

            selected = result.stdout.strip()
            if not selected or selected == "__CANCELLED__":
                self._send_json(200, {"ok": True, "cancelled": True})
                return

            self._send_json(200, {"ok": True, "cancelled": False, "path": selected})
        except subprocess.TimeoutExpired as exc:
            partial_out = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            partial_err = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            self._send_json(504, {
                "ok": False,
                "error": (
                    "Folder picker timed out after 90s with no response - the dialog may not "
                    "have become visible. Partial stdout: " + (partial_out.strip() or "(empty)") +
                    " | Partial stderr: " + (partial_err.strip() or "(empty)")
                ),
            })
        except FileNotFoundError:
            self._send_json(500, {"ok": False, "error": "powershell.exe not found on this system."})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def _handle_run_step0(self) -> None:
        """
        Step 0's main() reads YEAR/DATA_DIR/DOY_START/DOY_END and the
        station settings from module globals and returns nothing, so it is
        run through _patched_globals, as Steps 4-6. No `station` field:
        Step 0 processes every stations.json station in one pass. `raw_dir`
        is the folder with that year's RAW files (default step0.DATA_DIR),
        used directly as a local path.

        Output paths are re-resolved afterwards with
        step0.resolve_global_paths(year) from the request's own `year`, not
        from step0's globals, which _patched_globals has already restored.

        Not wrapped with @_serialize_plots: Step 0 does no plotting.
        """
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        config = self._stations_config()  # stations.json - web-only station list/offsets
        if config is None:
            return

        try:
            year = int(params.get("year", step0.YEAR))
            doy_start = int(params.get("doy_start", step0.DOY_START))
            doy_end = int(params.get("doy_end", step0.DOY_END))
            raw_dir = Path(params.get("raw_dir") or step0.DATA_DIR)

            if not raw_dir.is_dir():
                self._send_json(404, {
                    "ok": False,
                    "error": f"RAW directory not found: {raw_dir}",
                })
                return

            # Web mode: station subset and fixed UTC offsets come from
            # stations.json (stations in the RAW files but not in the JSON are
            # ignored). _patched_globals restores step0's console defaults.
            with _patched_globals(
                step0, YEAR=year, DATA_DIR=raw_dir,
                DOY_START=doy_start, DOY_END=doy_end,
                USE_SELECTED_STATIONS=True,
                SELECTED_STATIONS=[code for code, _ in config],
                STATION_UTC_OFFSETS=dict(config),
            ):
                step0.main()  # the real, unmodified main()

            global_paths = step0.resolve_global_paths(year)
            df_summary = pd.read_csv(global_paths["station_summary_csv"])

            self._send_json(200, {
                "ok": True,
                "year": year,
                "raw_dir": str(raw_dir),
                "doy_start": doy_start,
                "doy_end": doy_end,
                "table": _table_records(df_summary),
                "station_summary_csv_url": f"/results/{global_paths['station_summary_csv'].relative_to(RESULTS_ROOT).as_posix()}",
                "raw_inventory_csv_url": f"/results/{global_paths['inventory_csv'].relative_to(RESULTS_ROOT).as_posix()}",
            })
        except FileNotFoundError as exc:
            self._send_json(404, {"ok": False, "error": str(exc)})
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @_serialize_plots
    def _handle_run_step1(self) -> None:
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        config = self._stations_config()  # stations.json - web-only station list/offsets
        if config is None:
            return

        try:
            station = str(params.get("station", step1.STATION))
            if self._reject_unconfigured([station], config):
                return
            year = int(params.get("year", step1.YEAR))
            doy_start = int(params.get("doy_start", step1.DOY_START))
            doy_end = int(params.get("doy_end", step1.DOY_END))
            th_cov = float(params.get("th_cov", step1.TH_COV))

            df_results = step1.run_completeness(
                station=station, year=year,
                doy_start=doy_start, doy_end=doy_end, th_cov=th_cov,
            )

            paths = step1.resolve_paths(station, year, doy_start, doy_end, th_cov)
            paths["results_dir"].mkdir(parents=True, exist_ok=True)

            step1.save_results_csv(df_results, paths["output_csv"])
            step1.create_coverage_plot(
                df_results,
                th_cov=th_cov,
                plot_title=f"Daily data coverage - Station {station} ({year})",
                output_png=paths["output_png"],
                show_plot=False,
            )

            summary = step1.compute_completeness_summary(df_results, year)

            # Optional Ndat-impact DIAGNOSTIC, after the official outputs.
            # Only adds its own files and an extra "ndat_impact" key - the
            # rest of the response and the valid/invalid status are the
            # same with or without it. Unknown/missing ndat_mode -> ValueError -> 400.
            extra = {}
            if params.get("ndat_impact"):
                ndat_mode = str(params.get("ndat_mode"))
                df_impact, impact = step1.run_ndat_impact(
                    df_results, station, year, doy_start, doy_end, th_cov, ndat_mode,
                )
                impact_paths = step1.resolve_ndat_impact_paths(
                    station, year, doy_start, doy_end, th_cov, ndat_mode,
                )
                impact_paths["diag_dir"].mkdir(parents=True, exist_ok=True)
                step1.save_results_csv(df_impact, impact_paths["output_csv"])
                step1.create_ndat_impact_plot(
                    df_impact,
                    th_cov=th_cov,
                    plot_title=f"Ndat impact on daily coverage - Station {station} ({year})",
                    ndat_label=impact["ndat_label"],
                    output_png=impact_paths["output_png"],
                    show_plot=False,
                )
                extra["ndat_impact"] = {
                    **impact,
                    "csv_url": f"/results/{impact_paths['output_csv'].relative_to(RESULTS_ROOT).as_posix()}",
                    "plot_url": f"/results/{impact_paths['output_png'].relative_to(RESULTS_ROOT).as_posix()}",
                }

            self._send_json(200, {
                "ok": True,
                "station": station,
                "year": year,
                "th_cov": th_cov,
                "csv_url": f"/results/{paths['output_csv'].relative_to(RESULTS_ROOT).as_posix()}",
                "plot_url": f"/results/{paths['output_png'].relative_to(RESULTS_ROOT).as_posix()}",
                **summary,
                **extra,
            })
        except FileNotFoundError as exc:
            self._send_json(404, {"ok": False, "error": str(exc)})
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @_serialize_plots
    def _handle_run_step2(self) -> None:
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        config = self._stations_config()  # stations.json - web-only station list/offsets
        if config is None:
            return

        try:
            station = str(params.get("station", step2.STATION))
            if self._reject_unconfigured([station], config):
                return
            year = int(params.get("year", step2.YEAR))
            doy_start = int(params.get("doy_start", step2.DOY_START))
            doy_end = int(params.get("doy_end", step2.DOY_END))
            th_cov = float(params.get("th_cov", step2.TH_COV))
            ndat_mode = params.get("ndat_mode", None)
            value_col = "roti_l1"  # only scientifically supported index - not a form field

            results = step2.run_ccdf_analysis(
                station=station, year=year,
                doy_start=doy_start, doy_end=doy_end, th_cov=th_cov,
                value_col=value_col, ndat_mode=ndat_mode,
            )
            index_config = results["index_config"]
            df_pcts = results["df_pcts"]

            paths = step2.resolve_paths(
                station, year, doy_start, doy_end, th_cov, value_col, index_config,
                ndat_mode=ndat_mode,
            )
            paths["ndat_dir"].mkdir(parents=True, exist_ok=True)
            paths["percentile_dir"].mkdir(parents=True, exist_ok=True)
            paths["ccdf_dir"].mkdir(parents=True, exist_ok=True)

            df_pcts.to_csv(paths["output_pcts_csv"], index=False)
            step2.save_monthly_percentiles_plot(
                df_pcts=df_pcts,
                out_png=paths["output_monthly_percentiles_png"],
                station=station, year=year,
                index_label=index_config["label"],
                index_unit=index_config["unit"],
                min_valid_days=step2.MIN_VALID_DAYS_PER_MONTH,
                show_plot=False,
            )

            # Individual monthly CCDF - same as main()'s SAVE_INDIVIDUAL_CCDF
            # block, so "Run Step 2" via web keeps the ccdf/ folder up to
            # date for this exact station/year/doy/th_cov combination.
            for month, (x, ccdf_y, _n) in results["ccdf_by_month"].items():
                step2.save_ccdf_plot(
                    month=month, x=x, ccdf=ccdf_y,
                    out_png=step2.individual_ccdf_png(paths, doy_start, doy_end, month),
                    station=station, year=year,
                    index_label=index_config["label"],
                    index_unit=index_config["unit"],
                    show_plot=False,
                )

            summary = step2.compute_ccdf_summary(df_pcts)

            self._send_json(200, {
                "ok": True,
                "station": station,
                "year": year,
                "csv_url": f"/results/{paths['output_pcts_csv'].relative_to(RESULTS_ROOT).as_posix()}",
                "plot_url": f"/results/{paths['output_monthly_percentiles_png'].relative_to(RESULTS_ROOT).as_posix()}",
                "min_valid_days_per_month": step2.MIN_VALID_DAYS_PER_MONTH,
                **summary,
            })
        except FileNotFoundError as exc:
            self._send_json(404, {"ok": False, "error": str(exc)})
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @_serialize_plots
    def _handle_run_step2_ccdf_compare(self) -> None:
        """Overlay CCDF of several months within one station (save_multi_ccdf_plot)."""
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        config = self._stations_config()  # stations.json - web-only station list/offsets
        if config is None:
            return

        try:
            station = str(params.get("station", step2.STATION))
            if self._reject_unconfigured([station], config):
                return
            year = int(params.get("year", step2.YEAR))
            doy_start = int(params.get("doy_start", step2.DOY_START))
            doy_end = int(params.get("doy_end", step2.DOY_END))
            th_cov = float(params.get("th_cov", step2.TH_COV))
            ndat_mode = params.get("ndat_mode", None)
            value_col = "roti_l1"
            months = sorted({int(m) for m in params.get("months", [])})

            if not months:
                self._send_json(400, {"ok": False, "error": "Select at least one month."})
                return

            results = step2.run_ccdf_analysis(
                station=station, year=year,
                doy_start=doy_start, doy_end=doy_end, th_cov=th_cov,
                value_col=value_col, ndat_mode=ndat_mode,
            )
            index_config = results["index_config"]

            paths = step2.resolve_paths(
                station, year, doy_start, doy_end, th_cov, value_col, index_config,
                ndat_mode=ndat_mode,
            )
            paths["ccdf_comparisons_dir"].mkdir(parents=True, exist_ok=True)

            out_png = step2.multi_ccdf_png(paths, doy_start, doy_end, months)
            step2.save_multi_ccdf_plot(
                ccdf_by_month=results["ccdf_by_month"],
                pcts_by_month=results["pcts_by_month"],
                months_to_compare=months,
                out_png=out_png,
                station=station, year=year,
                index_label=index_config["label"],
                index_unit=index_config["unit"],
                show_plot=False,
            )

            # One station only here, so "low support" is unambiguous per
            # month - no station-mixing risk (unlike the cross-station
            # endpoint below, where the same month can be low-support for
            # one station and fine for another).
            n_days_by_month = dict(zip(results["df_pcts"]["month"], results["df_pcts"]["n_days"]))
            low_support_months = [
                m for m in months
                if n_days_by_month.get(m, 0) < step2.MIN_VALID_DAYS_PER_MONTH
            ]

            self._send_json(200, {
                "ok": True,
                "station": station,
                "year": year,
                "months": months,
                "low_support_months": low_support_months,
                "min_valid_days_per_month": step2.MIN_VALID_DAYS_PER_MONTH,
                "plot_url": f"/results/{out_png.relative_to(RESULTS_ROOT).as_posix()}",
            })
        except FileNotFoundError as exc:
            self._send_json(404, {"ok": False, "error": str(exc)})
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @_serialize_plots
    def _handle_run_step2_cross_station_ccdf(self) -> None:
        """Overlays CCDF curves across stations for one or more months, on one shared axis."""
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        config = self._stations_config()  # stations.json - web-only station list/offsets
        if config is None:
            return

        try:
            year = int(params.get("year", step2.YEAR))
            doy_start = int(params.get("doy_start", step2.DOY_START))
            doy_end = int(params.get("doy_end", step2.DOY_END))
            th_cov = float(params.get("th_cov", step2.TH_COV))
            ndat_mode = params.get("ndat_mode", None)
            value_col = "roti_l1"
            stations = sorted({str(s) for s in params.get("stations", [])})
            if self._reject_unconfigured(stations, config):
                return
            months = sorted({int(m) for m in params.get("months", [])})

            if not stations:
                self._send_json(400, {"ok": False, "error": "Select at least one station."})
                return
            if not months:
                self._send_json(400, {"ok": False, "error": "Select at least one month."})
                return

            ccdf_by_station_month = {}
            missing_stations = {}
            low_support = {}
            index_config = None

            for station in stations:
                try:
                    results = step2.run_ccdf_analysis(
                        station=station, year=year,
                        doy_start=doy_start, doy_end=doy_end, th_cov=th_cov,
                        value_col=value_col, ndat_mode=ndat_mode,
                    )
                    index_config = results["index_config"]
                    ccdf_by_station_month[station] = results["ccdf_by_month"]

                    # Low support is per (station, month), not per month alone -
                    # the same month can be low-support for one station and
                    # fine for another, so this must not be collapsed into one
                    # month-only list the way the single-station endpoint does.
                    n_days_by_month = dict(zip(results["df_pcts"]["month"], results["df_pcts"]["n_days"]))
                    low_support[station] = [
                        m for m in months
                        if n_days_by_month.get(m, 0) < step2.MIN_VALID_DAYS_PER_MONTH
                    ]
                except FileNotFoundError as exc:
                    missing_stations[station] = str(exc)

            stations_used = list(ccdf_by_station_month.keys())
            if not stations_used:
                self._send_json(404, {
                    "ok": False,
                    "error": "No Step 0/1 data for any selected station at these settings.",
                    "missing_stations": missing_stations,
                })
                return

            out_png = step2.resolve_cross_station_ccdf_path(
                year, doy_start, doy_end, stations_used, months, index_config,
                ndat_mode=ndat_mode,
            )
            out_png.parent.mkdir(parents=True, exist_ok=True)

            step2.save_cross_station_ccdf_plot(
                ccdf_by_station_month=ccdf_by_station_month,
                stations=stations_used,
                months=months,
                out_png=out_png,
                year=year,
                index_label=index_config["label"],
                index_unit=index_config["unit"],
                show_plot=False,
            )

            self._send_json(200, {
                "ok": True,
                "year": year,
                "stations_used": stations_used,
                "missing_stations": missing_stations,
                "low_support": {s: low_support[s] for s in stations_used},
                "min_valid_days_per_month": step2.MIN_VALID_DAYS_PER_MONTH,
                "months": months,
                "plot_url": f"/results/{out_png.relative_to(RESULTS_ROOT).as_posix()}",
            })
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def _handle_run_step2_ccdf_grid(self) -> None:
        """
        Existence manifest for the 12 x 4 individual-CCDF grid - no station
        field (compares whichever stations already have Step 2 output, same
        pattern as step2.2), no recomputation, just checks which of the
        already-known file paths exist on disk.
        """
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        config = self._stations_config()  # stations.json - web-only station list/offsets
        if config is None:
            return

        try:
            year = int(params.get("year", step2.YEAR))
            doy_start = int(params.get("doy_start", step2.DOY_START))
            doy_end = int(params.get("doy_end", step2.DOY_END))
            th_cov = float(params.get("th_cov", step2.TH_COV))
            ndat_mode = params.get("ndat_mode", None)
            value_col = "roti_l1"
            index_config = step2.validate_index_supported(value_col)

            grid = {}
            for station in [code for code, _ in config]:
                paths = step2.resolve_paths(
                    station, year, doy_start, doy_end, th_cov, value_col, index_config,
                    ndat_mode=ndat_mode,
                )
                grid[station] = {}
                for month in range(1, 13):
                    png = step2.individual_ccdf_png(paths, doy_start, doy_end, month)
                    grid[station][month] = (
                        f"/results/{png.relative_to(RESULTS_ROOT).as_posix()}"
                        if png.exists() else None
                    )

            self._send_json(200, {"ok": True, "year": year, "grid": grid})
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @_serialize_plots
    def _handle_run_step2_2(self) -> None:
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        config = self._stations_config()  # stations.json - web-only station list/offsets
        if config is None:
            return

        try:
            year = int(params.get("year", step2_2.YEAR))
            doy_start = int(params.get("doy_start", step2_2.DOY_START))
            doy_end = int(params.get("doy_end", step2_2.DOY_END))
            th_cov = float(params.get("th_cov", step2_2.TH_COV))
            ndat_mode = params.get("ndat_mode", None)
            value_col = "roti_l1"  # only scientifically supported index

            result = step2_2.run_ccdf_comparison(
                stations=[code for code, _ in config],
                year=year, doy_start=doy_start, doy_end=doy_end,
                value_col=value_col, th_cov=th_cov, ndat_mode=ndat_mode,
            )
            index_config = result["index_config"]
            table = result["table"]

            paths = step2_2.resolve_comparison_paths(
                year, value_col, index_config, ndat_mode=ndat_mode
            )
            step2_2.save_comparison_table(table, paths["output_csv"])
            step2_2.save_comparison_plot(
                table, result["available_stations"], index_config, paths["output_png"],
                step2_2.MIN_VALID_DAYS_PER_MONTH, step2_2.PLOT_DPI, show_plot=False,
            )

            self._send_json(200, {
                "ok": True,
                "year": year,
                "available_stations": result["available_stations"],
                "missing_stations": result["missing_stations"],
                "th_cov_warnings": result["th_cov_warnings"],
                "min_valid_days_per_month": step2_2.MIN_VALID_DAYS_PER_MONTH,
                "table": _table_records(table),
                "csv_url": f"/results/{paths['output_csv'].relative_to(RESULTS_ROOT).as_posix()}",
                "plot_url": f"/results/{paths['output_png'].relative_to(RESULTS_ROOT).as_posix()}",
            })
        except FileNotFoundError as exc:
            self._send_json(404, {"ok": False, "error": str(exc)})
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @_serialize_plots
    def _handle_run_step3(self) -> None:
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        config = self._stations_config()  # stations.json - web-only station list/offsets
        if config is None:
            return

        try:
            station = str(params.get("station", step3.STATION))
            if self._reject_unconfigured([station], config):
                return
            year = int(params.get("year", step3.YEAR))
            doy_start = int(params.get("doy_start", step3.DOY_START))
            doy_end = int(params.get("doy_end", step3.DOY_END))
            th_cov = float(params.get("th_cov", step3.TH_COV))
            ndat_mode = params.get("ndat_mode", None)
            value_col = "roti_l1"  # only scientifically supported index - not a form field

            results = step3.run_temporal_analysis(
                station=station, year=year,
                doy_start=doy_start, doy_end=doy_end, th_cov=th_cov,
                value_col=value_col, ndat_mode=ndat_mode,
            )
            index_config = results["index_config"]
            df_thresholds = results["df_thresholds"]
            df_daily = results["df_daily"]

            paths = step3.resolve_paths(
                station, year, doy_start, doy_end, th_cov, value_col, index_config,
                ndat_mode=ndat_mode,
            )
            paths["ndat_dir"].mkdir(parents=True, exist_ok=True)

            df_thresholds.to_csv(paths["output_thresholds_csv"], index=False)
            df_daily.to_csv(paths["output_daily_csv"], index=False)
            step3.save_daily_frequency_plot(
                df_daily=df_daily,
                threshold_high=results["threshold_high"],
                threshold_extreme=results["threshold_extreme"],
                station=station, year=year,
                index_label=index_config["label"],
                index_unit=index_config["unit"],
                doy_start=doy_start, doy_end=doy_end,
                percentile_high=step3.PERCENTILE_HIGH,
                percentile_extreme=step3.PERCENTILE_EXTREME,
                out_png=paths["output_daily_plot"],
                show_plot=False,
            )

            row = df_thresholds.iloc[0]
            self._send_json(200, {
                "ok": True,
                "station": station,
                "year": year,
                "n_values": int(row["n_values"]),
                "p90_annual": float(row["p90_annual"]),
                "p99_annual": float(row["p99_annual"]),
                "frequency_p90_pct": float(row["frequency_p90_pct"]),
                "frequency_p99_pct": float(row["frequency_p99_pct"]),
                "thresholds_csv_url": f"/results/{paths['output_thresholds_csv'].relative_to(RESULTS_ROOT).as_posix()}",
                "daily_csv_url": f"/results/{paths['output_daily_csv'].relative_to(RESULTS_ROOT).as_posix()}",
                "plot_url": f"/results/{paths['output_daily_plot'].relative_to(RESULTS_ROOT).as_posix()}",
            })
        except FileNotFoundError as exc:
            self._send_json(404, {"ok": False, "error": str(exc)})
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @_serialize_plots
    def _handle_run_step3_comparison(self) -> None:
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        config = self._stations_config()  # stations.json - web-only station list/offsets
        if config is None:
            return

        try:
            year = int(params.get("year", step3_2.YEAR))
            doy_start = int(params.get("doy_start", step3_2.DOY_START))
            doy_end = int(params.get("doy_end", step3_2.DOY_END))
            top_n = int(params.get("top_n", step3_2.TOP_N_DAYS))
            ndat_mode = params.get("ndat_mode", None)
            value_col = "roti_l1"  # only scientifically supported index

            result = step3_2.run_temporal_comparison(
                stations=[code for code, _ in config],
                year=year, doy_start=doy_start, doy_end=doy_end,
                value_col=value_col, top_n=top_n, ndat_mode=ndat_mode,
            )
            index_config = result["index_config"]

            paths = step3_2.resolve_comparison_paths(
                year, index_config, ndat_mode=ndat_mode
            )
            step3_2.save_comparison_outputs(
                result["table"], result["top_days"], result["coincidences"], paths
            )
            step3_2.save_daily_comparison_plot(
                result["available_stations"], result["daily_by_station"], index_config,
                paths["output_daily_comparison_png"],
                doy_start=doy_start, doy_end=doy_end, show_plot=False,
            )

            self._send_json(200, {
                "ok": True,
                "year": year,
                "available_stations": result["available_stations"],
                "missing_stations": result["missing_stations"],
                "top_n": top_n,
                "table": _table_records(result["table"]),
                "top_days": _table_records(result["top_days"]),
                "coincidences": _table_records(result["coincidences"]),
                "table_csv_url": f"/results/{paths['output_table_csv'].relative_to(RESULTS_ROOT).as_posix()}",
                "top_days_csv_url": f"/results/{paths['output_top_days_csv'].relative_to(RESULTS_ROOT).as_posix()}",
                "coincidences_csv_url": f"/results/{paths['output_coincidences_csv'].relative_to(RESULTS_ROOT).as_posix()}",
                "plot_url": f"/results/{paths['output_daily_comparison_png'].relative_to(RESULTS_ROOT).as_posix()}",
            })
        except FileNotFoundError as exc:
            self._send_json(404, {"ok": False, "error": str(exc)})
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @_serialize_plots
    def _handle_run_step4(self) -> None:
        """
        Unlike Steps 1-3, Step 4's main() reads STATION/YEAR/DOY_START/
        DOY_END/TH_COV/NDAT_MODE from module globals and returns nothing.
        It is run unmodified with its globals temporarily patched
        (_patched_globals restores them afterwards); the CSV it wrote is
        then read back via resolve_paths() to build the JSON summary.
        """
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        config = self._stations_config()  # stations.json - web-only station list/offsets
        if config is None:
            return

        try:
            station = str(params.get("station", step4.STATION))
            if self._reject_unconfigured([station], config):
                return
            year = int(params.get("year", step4.YEAR))
            doy_start = int(params.get("doy_start", step4.DOY_START))
            doy_end = int(params.get("doy_end", step4.DOY_END))
            th_cov = float(params.get("th_cov", step4.TH_COV))
            ndat_mode = params.get("ndat_mode", None)

            with _patched_globals(
                step4, STATION=station, YEAR=year,
                DOY_START=doy_start, DOY_END=doy_end, TH_COV=th_cov,
                NDAT_MODE=ndat_mode,
            ):
                step4.main()  # the real, unmodified main()

            # main() has no return value - rebuild the same paths (from the
            # request params, not from step4's now-restored globals) to
            # read back the CSV it just wrote, with the same ndat_mode
            # passed to main() above.
            index_config = step4.validate_index_supported(step4.VALUE_COL)
            paths = step4.resolve_paths(
                station, year, doy_start, doy_end, th_cov, step4.VALUE_COL, index_config,
                ndat_mode=ndat_mode,
            )
            df_hourly = pd.read_csv(paths["output_hourly_csv"])
            p90_hours, p90_freq, p90_n = _hours_at_max(df_hourly, "frequency_p90_pct")
            p99_hours, p99_freq, p99_n = _hours_at_max(df_hourly, "frequency_p99_pct")

            self._send_json(200, {
                "ok": True,
                "station": station,
                "year": year,
                "max_p90_hours": p90_hours,
                "max_p90_freq_pct": p90_freq,
                "max_p90_n_values": p90_n,
                "max_p99_hours": p99_hours,
                "max_p99_freq_pct": p99_freq,
                "max_p99_n_values": p99_n,
                "csv_url": f"/results/{paths['output_hourly_csv'].relative_to(RESULTS_ROOT).as_posix()}",
                "plot_url": f"/results/{paths['output_hourly_plot'].relative_to(RESULTS_ROOT).as_posix()}",
            })
        except FileNotFoundError as exc:
            self._send_json(404, {"ok": False, "error": str(exc)})
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @_serialize_plots
    def _handle_run_step4_comparison(self) -> None:
        """
        Unlike Step 4 itself, 4.2-hourly_comparison.py has a pure
        run_hourly_comparison(), called directly like the other X.2
        comparisons (station list from stations.json); no _patched_globals
        needed.
        """
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        config = self._stations_config()  # stations.json - web-only station list/offsets
        if config is None:
            return

        try:
            year = int(params.get("year", step4_2.YEAR))
            doy_start = int(params.get("doy_start", step4_2.DOY_START))
            doy_end = int(params.get("doy_end", step4_2.DOY_END))
            ndat_mode = params.get("ndat_mode", None)
            value_col = "roti_l1"  # only scientifically supported index

            result = step4_2.run_hourly_comparison(
                stations=[code for code, _ in config],
                year=year, doy_start=doy_start, doy_end=doy_end,
                value_col=value_col, ndat_mode=ndat_mode,
            )
            index_config = result["index_config"]

            paths = step4_2.resolve_comparison_paths(
                year, index_config, ndat_mode=ndat_mode
            )
            step4_2.save_comparison_table(result["table"], paths)
            step4_2.save_hourly_comparison_plot(
                result["available_stations"], result["hourly_by_station"], index_config, 90,
                paths["output_p90_comparison_png"], show_plot=False,
            )
            step4_2.save_hourly_comparison_plot(
                result["available_stations"], result["hourly_by_station"], index_config, 99,
                paths["output_p99_comparison_png"], show_plot=False,
            )

            self._send_json(200, {
                "ok": True,
                "year": year,
                "available_stations": result["available_stations"],
                "missing_stations": result["missing_stations"],
                "table": _table_records(result["table"]),
                "table_csv_url": f"/results/{paths['output_table_csv'].relative_to(RESULTS_ROOT).as_posix()}",
                "p90_plot_url": f"/results/{paths['output_p90_comparison_png'].relative_to(RESULTS_ROOT).as_posix()}",
                "p99_plot_url": f"/results/{paths['output_p99_comparison_png'].relative_to(RESULTS_ROOT).as_posix()}",
            })
        except FileNotFoundError as exc:
            self._send_json(404, {"ok": False, "error": str(exc)})
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @_serialize_plots
    def _handle_run_step5(self) -> None:
        """
        Same pattern as _handle_run_step4 (main() reads its parameters from
        globals and returns nothing; run through _patched_globals, then the
        CSV is read back), summarised by dominant daypart
        (_dayparts_at_max) instead of hour of day.
        """
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        config = self._stations_config()  # stations.json - web-only station list/offsets
        if config is None:
            return

        try:
            station = str(params.get("station", step5.STATION))
            if self._reject_unconfigured([station], config):
                return
            year = int(params.get("year", step5.YEAR))
            doy_start = int(params.get("doy_start", step5.DOY_START))
            doy_end = int(params.get("doy_end", step5.DOY_END))
            th_cov = float(params.get("th_cov", step5.TH_COV))
            ndat_mode = params.get("ndat_mode", None)

            with _patched_globals(
                step5, STATION=station, YEAR=year,
                DOY_START=doy_start, DOY_END=doy_end, TH_COV=th_cov,
                NDAT_MODE=ndat_mode,
            ):
                step5.main()  # the real, unmodified main()

            index_config = step5.validate_index_supported(step5.VALUE_COL)
            paths = step5.resolve_paths(
                station, year, doy_start, doy_end, th_cov, step5.VALUE_COL, index_config,
                ndat_mode=ndat_mode,
            )
            df_daypart = pd.read_csv(paths["output_daypart_csv"])
            p90_dayparts, p90_freq, p90_n = _dayparts_at_max(df_daypart, "frequency_p90_pct")
            p99_dayparts, p99_freq, p99_n = _dayparts_at_max(df_daypart, "frequency_p99_pct")

            self._send_json(200, {
                "ok": True,
                "station": station,
                "year": year,
                "max_p90_dayparts": p90_dayparts,
                "max_p90_freq_pct": p90_freq,
                "max_p90_n_values": p90_n,
                "max_p99_dayparts": p99_dayparts,
                "max_p99_freq_pct": p99_freq,
                "max_p99_n_values": p99_n,
                "csv_url": f"/results/{paths['output_daypart_csv'].relative_to(RESULTS_ROOT).as_posix()}",
                "plot_url": f"/results/{paths['output_daypart_plot'].relative_to(RESULTS_ROOT).as_posix()}",
            })
        except FileNotFoundError as exc:
            self._send_json(404, {"ok": False, "error": str(exc)})
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @_serialize_plots
    def _handle_run_step5_comparison(self) -> None:
        """
        Same wiring as 2.2/3.2/4.2: 5.2-daypart_comparison.py has a pure
        run_daypart_comparison(), no _patched_globals needed. Unlike 4.2,
        it also takes th_cov (coverage cross-check, see that script's
        docstring) and saves a single combined PNG (p90 and p99 stacked).
        """
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        config = self._stations_config()  # stations.json - web-only station list/offsets
        if config is None:
            return

        try:
            year = int(params.get("year", step5_2.YEAR))
            doy_start = int(params.get("doy_start", step5_2.DOY_START))
            doy_end = int(params.get("doy_end", step5_2.DOY_END))
            th_cov = float(params.get("th_cov", step5_2.TH_COV))
            ndat_mode = params.get("ndat_mode", None)
            value_col = "roti_l1"  # only scientifically supported index

            result = step5_2.run_daypart_comparison(
                stations=[code for code, _ in config],
                year=year, doy_start=doy_start, doy_end=doy_end,
                value_col=value_col, th_cov=th_cov, ndat_mode=ndat_mode,
            )
            index_config = result["index_config"]

            paths = step5_2.resolve_comparison_paths(
                year, value_col, index_config, ndat_mode=ndat_mode
            )
            step5_2.save_comparison_table(result["table"], paths["output_csv"])
            step5_2.save_comparison_plot(
                result["table"], result["available_stations"], index_config,
                paths["output_png"], show_plot=False,
            )

            self._send_json(200, {
                "ok": True,
                "year": year,
                "available_stations": result["available_stations"],
                "missing_stations": result["missing_stations"],
                "th_cov_warnings": result["th_cov_warnings"],
                "table": _table_records(result["table"]),
                "table_csv_url": f"/results/{paths['output_csv'].relative_to(RESULTS_ROOT).as_posix()}",
                "plot_url": f"/results/{paths['output_png'].relative_to(RESULTS_ROOT).as_posix()}",
            })
        except FileNotFoundError as exc:
            self._send_json(404, {"ok": False, "error": str(exc)})
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @_serialize_plots
    def _handle_run_step6(self) -> None:
        """
        Same pattern as Steps 4-5: main() reads its parameters from globals
        and returns nothing; it is run through _patched_globals and its
        outputs, including the Top-10 cells CSVs (get_top_cells()), are
        read back.
        """
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        config = self._stations_config()  # stations.json - web-only station list/offsets
        if config is None:
            return

        try:
            station = str(params.get("station", step6.STATION))
            if self._reject_unconfigured([station], config):
                return
            year = int(params.get("year", step6.YEAR))
            doy_start = int(params.get("doy_start", step6.DOY_START))
            doy_end = int(params.get("doy_end", step6.DOY_END))
            th_cov = float(params.get("th_cov", step6.TH_COV))
            ndat_mode = params.get("ndat_mode", None)

            with _patched_globals(
                step6, STATION=station, YEAR=year,
                DOY_START=doy_start, DOY_END=doy_end, TH_COV=th_cov,
                NDAT_MODE=ndat_mode,
            ):
                step6.main()  # the real, unmodified main()

            index_config = step6.validate_index_supported(step6.VALUE_COL)
            paths = step6.resolve_paths(
                station, year, doy_start, doy_end, th_cov, step6.VALUE_COL, index_config,
                ndat_mode=ndat_mode,
            )
            top_p90 = pd.read_csv(paths["output_top_cells_p90_csv"])
            top_p99 = pd.read_csv(paths["output_top_cells_p99_csv"])

            self._send_json(200, {
                "ok": True,
                "station": station,
                "year": year,
                "top_p90": _table_records(top_p90),
                "top_p99": _table_records(top_p99),
                "csv_url": f"/results/{paths['output_month_hour_csv'].relative_to(RESULTS_ROOT).as_posix()}",
                "top_p90_csv_url": f"/results/{paths['output_top_cells_p90_csv'].relative_to(RESULTS_ROOT).as_posix()}",
                "top_p99_csv_url": f"/results/{paths['output_top_cells_p99_csv'].relative_to(RESULTS_ROOT).as_posix()}",
                "plot_p90_url": f"/results/{paths['output_heatmap_p90'].relative_to(RESULTS_ROOT).as_posix()}",
                "plot_p99_url": f"/results/{paths['output_heatmap_p99'].relative_to(RESULTS_ROOT).as_posix()}",
                "plot_nvalues_url": f"/results/{paths['output_heatmap_nvalues'].relative_to(RESULTS_ROOT).as_posix()}",
            })
        except FileNotFoundError as exc:
            self._send_json(404, {"ok": False, "error": str(exc)})
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def _handle_run_step6_cross_station_maxima(self) -> None:
        """
        Cross-station summary of Step 6 (there is no 6.2 comparison): reads
        the rank-1 row of each station's saved Top-10 p90/p99 CSVs via
        step6.get_cross_station_maxima(). No plotting and no recomputation,
        so it is not wrapped with @_serialize_plots.
        """
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        config = self._stations_config()  # stations.json - web-only station list/offsets
        if config is None:
            return

        try:
            year = int(params.get("year", step6.YEAR))
            doy_start = int(params.get("doy_start", step6.DOY_START))
            doy_end = int(params.get("doy_end", step6.DOY_END))
            th_cov = float(params.get("th_cov", step6.TH_COV))
            ndat_mode = params.get("ndat_mode", None)
            value_col = "roti_l1"  # only scientifically supported index

            index_config = step6.validate_index_supported(value_col)
            result = step6.get_cross_station_maxima(
                [code for code, _ in config], year, doy_start, doy_end, th_cov, value_col,
                index_config, ndat_mode,
            )

            self._send_json(200, {
                "ok": True,
                "year": year,
                "available_stations": result["available_stations"],
                "missing_stations": result["missing_stations"],
                "table": _table_records(result["table"]),
            })
        except FileNotFoundError as exc:
            self._send_json(404, {"ok": False, "error": str(exc)})
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, format, *args):
        print(f"[web_server] {self.address_string()} - {format % args}")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Serving on http://{HOST}:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
