#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local web interface entry point (docs/roadmap.md).

Serves static/index.html and exposes one HTTP endpoint that calls
1-completitud.py's own run_completeness() / save_results_csv() /
create_coverage_plot() directly - no reimplementation of the science,
no subprocess. Standard library only (http.server), no web framework,
per the roadmap's explicit design principle: "no crear una segunda
implementacion cientifica para la web".

Results are written to the exact same results/{STATION}/{YEAR}/... paths
that console mode already uses - the web UI is another way to trigger
the same pipeline, not a parallel one.

Currently wires up Steps 0, 1, 2 (with 2.2), 3 (with 3.2), 4 (with 4.2),
5 (with 5.2) and 6 (individual only, no 6.2 yet). Steps 0, 4, 5 and 6 were
all deliberately migrated "adapt only" (CLAUDE.md section 11) and have
no pure run_x() function - they are wired via _patched_globals instead
(see _handle_run_step0/_handle_run_step4/_handle_run_step5/_handle_run_step6).
The X.2 cross-station comparisons (2.2/3.2/4.2/5.2) each expose a pure
function and are wired the same simple way as Steps 1-3.

Step 0 has no `station` field, unlike every other step - it always
processes every configured SELECTED_STATIONS station in one pass (each
daily RAW file mixes all stations together, see 0-toParquet.py). It does
have one field no other step has: `raw_dir`, the folder holding that
year's daily RAW files (defaults to step0.DATA_DIR, "datos_Estudio"),
used directly as a local filesystem path - not uploaded, since this
process already has the same disk access the console script does.

/api/browse-folder opens a native Windows folder picker (Shell.Application's
BrowseForFolder COM dialog, hosted by powershell.exe - see FOLDER_PICKER_PS1)
and returns the chosen path so index.html can fill the `raw_dir` field
without the user typing it by hand. Not <input type="file" webkitdirectory">:
browsers never expose a real absolute filesystem path from that element,
by design. Not tkinter: not guaranteed present in the embeddable Python
distribution the packaged app will eventually use (docs/roadmap.md). Not
System.Windows.Forms.FolderBrowserDialog: an earlier version used that and
a real user test hung indefinitely with no visible dialog - see
FOLDER_PICKER_PS1's own comment for the diagnosis. PowerShell and Shell32
are both part of Windows itself, not a Python dependency - nothing new to
pip install.
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
# transitively, inside the step modules loaded below - or the backend
# defaults to TkAgg. TkAgg is a GUI backend: it is not safe to touch outside
# the process main thread, and ThreadingHTTPServer handles every request in
# its own thread. Observed directly: worker threads intermittently crashing
# the whole process (not just the request) with
# "Tcl_AsyncDelete: async handler deleted by the wrong thread" while garbage
# collecting Tk-backed figure objects from a plot-heavy request. Agg is the
# standard headless/non-interactive backend - same savefig() output, no
# window, safe from any thread. Purely technical (CLAUDE.md section 14): it
# does not change any plotted data or PNG pixel output.
matplotlib.use("Agg")

import pandas as pd

HOST = "127.0.0.1"
PORT = 8000

APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "static"
RESULTS_ROOT = Path("results")  # relative to CWD, same convention as every step script

# Switched away from System.Windows.Forms.FolderBrowserDialog after a real
# user test hung indefinitely on "Waiting for dialog..." with no visible
# window - root-caused to the owner-window pattern this used to have
# (create a Form, .Show() it, then immediately .Hide() it, then pass it as
# ShowDialog()'s owner): a WinForms modal dialog owned by a hidden form can
# fail to reach the foreground reliably, and there is no way to verify this
# interactively from here (a human has to click a real dialog - see the
# conversation this was diagnosed in). Shell.Application.BrowseForFolder is
# the classic COM/Shell32 folder-browse dialog (no .NET, no owner-window
# trick, no STA-apartment subtlety to get wrong) - it is what native
# Windows Explorer-style "Browse..." buttons have used since long before
# .NET existed, and is a well-established, widely-used pattern specifically
# because it reliably surfaces in the foreground on its own.
#
# `RootFolder` is deliberately left at 0 (Desktop, i.e. "This PC" and every
# drive) rather than seeded from the current raw_dir value: unlike
# FolderBrowserDialog.SelectedPath (a pre-selection within a tree the user
# can still navigate out of), BrowseForFolder's RootFolder *restricts* the
# visible tree to that folder and everything below it - passing the current
# raw_dir there would make it impossible to browse to a folder on a
# different drive or outside that subtree, breaking the actual requirement
# (pick *any* folder on the machine). Static script, never string-
# interpolated with request data - nothing here reads $InitialDir at all,
# so there is nothing for a crafted `initial_dir` value to inject into.
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
# thread - two concurrent plot-producing requests can corrupt each other's
# figure (observed directly: a linear-scale handler raising a mathtext
# ParseException for "$\mathdefault{10^{-5}}$", a log-scale tick label that
# only a *different*, concurrently-running handler could have produced).
# Every handler that calls a save_*_plot()/create_*_plot() function is
# wrapped with @_serialize_plots so only one thread touches matplotlib at a
# time. Handlers that only read/check files (no plotting) are left
# unwrapped on purpose - serializing them too would just make them wait
# behind unrelated plot generation for no reason.
_PLOT_LOCK = threading.Lock()


def _serialize_plots(handler):
    def wrapped(self, *args, **kwargs):
        with _PLOT_LOCK:
            return handler(self, *args, **kwargs)
    return wrapped


def _load_module(filename: str, module_name: str):
    """
    Dynamically loads a script whose filename is not a valid Python
    module name (starts with a digit) - same technique already used and
    verified in 2.2-ccdf_comparison.py and 5.2-daypart_comparison.py.
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
    Temporarily overwrites module-level globals (for scripts like Step 4-6
    that read their parameters from globals, not function arguments -
    CLAUDE.md section 11, "adapt only" scope, deliberately not refactored
    into a pure function) and restores the original values on exit, even
    if the wrapped call raises - the console pipeline's own state must
    never end up different just because a web request passed through it.
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
    of hour_local - Step 5 has only 4 categories, so a near-tie (e.g.
    UNSA's Afternoon/Evening in p90, see hallazgos.md 1.4) is plausible
    even if not currently exact.
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

    def do_GET(self) -> None:
        route = urlparse(self.path).path

        if route in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html")
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
        Opens a native Windows folder picker and returns the chosen path -
        see FOLDER_PICKER_PS1 / the module docstring for why powershell.exe
        + Shell.Application, not <input type="file"> or tkinter.

        -STA: explicit, not relied on as a default - Shell.Application does
        not actually require it (unlike the WinForms dialog this used to
        call), but it costs nothing and removes any ambiguity.
        stdin=DEVNULL: if anything in this call chain ever tries to prompt
        interactively (it should not - -NonInteractive already covers
        PowerShell's own confirmation prompts), reading from stdin fails
        immediately instead of hanging forever waiting for input nobody can
        provide from an HTTP request. Direct defense against the exact
        failure mode being guarded against here: a request that hangs with
        no visible cause and no way to recover except restarting the server.

        90s timeout, deliberately shorter than a human might actually take
        to browse folders: this project's UI never leaves a request hanging
        indefinitely (CLAUDE.md's "no fallback silencioso" discipline,
        applied here as "no dejar la peticion esperando en silencio") -
        index.html's browseRawDir() also aborts client-side around the same
        time, so the UI recovers either way, whichever fires first. If a
        real dialog is up and the user just needs longer, "Browse..." can
        simply be clicked again.

        Blocking: subprocess.run() blocks this request's own thread until
        PowerShell exits or the timeout fires - contained to that one
        thread only (ThreadingHTTPServer gives every request its own
        thread), so it does not stall any other concurrent request (e.g. a
        Step running at the same time).
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
        Step 0 was also migrated "adapt only" (CLAUDE.md section 11) -
        main() reads YEAR/DATA_DIR/DOY_START/DOY_END from module globals
        and returns nothing, same shape as Steps 4-6. Same
        _patched_globals + call-the-real-main()-then-re-resolve-paths
        pattern as _handle_run_step6, with two differences: no `station`
        field (Step 0 always processes every SELECTED_STATIONS station in
        one pass), and one new field, `raw_dir` - the folder holding that
        year's daily RAW files, defaulting to step0.DATA_DIR
        ("datos_Estudio"). Used directly as a local filesystem path, not
        uploaded: this process already has the same disk access the
        console script has, and a browser directory-picker could not
        supply a real absolute path anyway (webkitdirectory only ever
        exposes paths relative to the picked folder, by browser design).

        Paths are re-resolved via step0.resolve_global_paths(year) *after*
        the _patched_globals block exits, using the request's own local
        `year` - not by reading step0's module globals post-call, which
        _patched_globals has already restored to their console defaults
        by then. Same reasoning as _handle_run_step6 re-reading via
        step6.resolve_paths(station, year, ...) with the request's own
        arguments, never via mutated-then-restored module state.

        Not wrapped with @_serialize_plots - Step 0 does no plotting
        (same reasoning as _handle_run_step2_ccdf_grid /
        _handle_run_step6_cross_station_maxima).
        """
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
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

            with _patched_globals(
                step0, YEAR=year, DATA_DIR=raw_dir,
                DOY_START=doy_start, DOY_END=doy_end,
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

        try:
            station = str(params.get("station", step1.STATION))
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

            self._send_json(200, {
                "ok": True,
                "station": station,
                "year": year,
                "th_cov": th_cov,
                "csv_url": f"/results/{paths['output_csv'].relative_to(RESULTS_ROOT).as_posix()}",
                "plot_url": f"/results/{paths['output_png'].relative_to(RESULTS_ROOT).as_posix()}",
                **summary,
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

        try:
            station = str(params.get("station", step2.STATION))
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

        try:
            station = str(params.get("station", step2.STATION))
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

        try:
            year = int(params.get("year", step2.YEAR))
            doy_start = int(params.get("doy_start", step2.DOY_START))
            doy_end = int(params.get("doy_end", step2.DOY_END))
            th_cov = float(params.get("th_cov", step2.TH_COV))
            ndat_mode = params.get("ndat_mode", None)
            value_col = "roti_l1"
            stations = sorted({str(s) for s in params.get("stations", [])})
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
                    # fine for another (e.g. UNSA August n_days=1, normal
                    # elsewhere), so this must not be collapsed into one
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

        try:
            year = int(params.get("year", step2.YEAR))
            doy_start = int(params.get("doy_start", step2.DOY_START))
            doy_end = int(params.get("doy_end", step2.DOY_END))
            th_cov = float(params.get("th_cov", step2.TH_COV))
            ndat_mode = params.get("ndat_mode", None)
            value_col = "roti_l1"
            index_config = step2.validate_index_supported(value_col)

            grid = {}
            for station in step2_2.STATIONS:
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

        try:
            year = int(params.get("year", step2_2.YEAR))
            doy_start = int(params.get("doy_start", step2_2.DOY_START))
            doy_end = int(params.get("doy_end", step2_2.DOY_END))
            th_cov = float(params.get("th_cov", step2_2.TH_COV))
            ndat_mode = params.get("ndat_mode", None)
            value_col = "roti_l1"  # only scientifically supported index

            result = step2_2.run_ccdf_comparison(
                stations=step2_2.STATIONS,
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

        try:
            station = str(params.get("station", step3.STATION))
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

        try:
            year = int(params.get("year", step3_2.YEAR))
            doy_start = int(params.get("doy_start", step3_2.DOY_START))
            doy_end = int(params.get("doy_end", step3_2.DOY_END))
            top_n = int(params.get("top_n", step3_2.TOP_N_DAYS))
            ndat_mode = params.get("ndat_mode", None)
            value_col = "roti_l1"  # only scientifically supported index

            result = step3_2.run_temporal_comparison(
                stations=step3_2.STATIONS,
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
        Step 4 was deliberately migrated "adapt only" (CLAUDE.md section 11)
        - unlike Steps 1-3, main() reads STATION/YEAR/DOY_START/DOY_END/
        TH_COV from module globals, not from arguments, and returns
        nothing. Rather than refactor that (an unrequested methodology/
        scope change - CLAUDE.md section 14), this calls the real,
        unmodified main() with its globals temporarily patched
        (_patched_globals restores them afterward), then re-reads the CSV
        main() just wrote (resolve_paths() is already pure/parametrized)
        to build the JSON summary.
        """
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        try:
            station = str(params.get("station", step4.STATION))
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
            # read back the CSV it just wrote. Same ndat_mode variable as
            # the _patched_globals call above - see CLAUDE.md section 10/14
            # discipline (one mode, two call sites, never re-derived).
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
        run_hourly_comparison() (station list comes from its own STATIONS
        config, same as 2.2/3.2 - not a per-request field) - wired the
        same simple way as the other X.2 comparison endpoints, no
        _patched_globals needed here.
        """
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        try:
            year = int(params.get("year", step4_2.YEAR))
            doy_start = int(params.get("doy_start", step4_2.DOY_START))
            doy_end = int(params.get("doy_end", step4_2.DOY_END))
            ndat_mode = params.get("ndat_mode", None)
            value_col = "roti_l1"  # only scientifically supported index

            result = step4_2.run_hourly_comparison(
                stations=step4_2.STATIONS,
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
        Same shape as _handle_run_step4: 5-daypart_variability.py was also
        migrated "adapt only" (CLAUDE.md section 11) - main() reads its
        parameters from globals and returns nothing. Same
        _patched_globals + re-read-the-CSV pattern, just summarized by
        dominant daypart (_dayparts_at_max) instead of hour of day.
        """
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        try:
            station = str(params.get("station", step5.STATION))
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
        Same wiring style as 4.2/3.2/2.2 - 5.2-daypart_comparison.py has a
        pure run_daypart_comparison(), no _patched_globals needed. Two
        differences from 4.2, not oversights: run_daypart_comparison()
        also takes th_cov (its own coverage-consistency cross-check, see
        that script's docstring), and it saves a single combined PNG
        (p90+p99 stacked), not two separate ones.
        """
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        try:
            year = int(params.get("year", step5_2.YEAR))
            doy_start = int(params.get("doy_start", step5_2.DOY_START))
            doy_end = int(params.get("doy_end", step5_2.DOY_END))
            th_cov = float(params.get("th_cov", step5_2.TH_COV))
            ndat_mode = params.get("ndat_mode", None)
            value_col = "roti_l1"  # only scientifically supported index

            result = step5_2.run_daypart_comparison(
                stations=step5_2.STATIONS,
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
        Same shape as Step 4/5: 6-month_hour_heatmaps.py was also
        migrated "adapt only" (CLAUDE.md section 11) - main() reads its
        parameters from globals and returns nothing. Same
        _patched_globals + re-read-the-outputs pattern. Unlike Step 4/5,
        no new max-detection helper is needed here - main() itself now
        saves the Top-10 CSVs (get_top_cells()), so this just reads them
        back, same as every other output file.
        """
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return

        try:
            station = str(params.get("station", step6.STATION))
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
        Cross-station summary view for Step 6 - NOT a Step 6.2 comparator
        (Step 6 has none). Reads the rank-1 row of each station's already-
        saved Top-10 p90/p99 CSVs via step6.get_cross_station_maxima() - no
        plotting, no recomputation, so unlike Step 6 itself this is not
        wrapped with @_serialize_plots (same reasoning as
        _handle_run_step2_ccdf_grid, the other read-only/no-plot handler).
        """
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
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
                step6.STATIONS, year, doy_start, doy_end, th_cov, value_col,
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
