#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local web interface of the pipeline.

Listens on http://127.0.0.1:9000 - loopback only, never reachable from
another machine. The environment variable GNSS_WEB_PORT changes the port
(e.g. GNSS_WEB_PORT=9001); the host cannot be changed. If the port is
already in use the server stops with an error instead of sharing it.
Started as a script (main(), whatever the launcher), it opens the page in
the default browser once it is listening; importing this module opens
nothing.

Serves static/index.html, the results files (GET /results/...), the
station list (GET /api/stations), the results folder (GET /api/paths) and
one POST endpoint per action (/api/run/step0 ... /api/run/step6, the
cross-station comparisons, /api/ndat-modes, /api/browse-folder and
/api/open-results-folder). Every analysis endpoint calls the step scripts'
own functions in this process - no subprocess and no second implementation
of the science. Standard library only (http.server).

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

/api/browse-folder opens the system's native folder dialog so that
index.html can fill `raw_dir` (browsers never expose an absolute path from
<input type="file" webkitdirectory>), and /api/open-results-folder opens
the results folder in the file manager. Both use platform_support.py,
which holds everything that depends on the operating system.

Every request must carry a Host header naming this server
(127.0.0.1:<port> or localhost:<port>), so a page from another site cannot
reach it through a DNS name pointing to 127.0.0.1 (DNS rebinding).
"""

import contextlib
import errno
import importlib.util
import json
import math
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

# app/python_libraries holds Windows builds (win_amd64) of pandas, numpy,
# matplotlib and pyarrow. app/python_portable/python.exe already lists it in
# its python311._pth; this line makes it usable by another Windows Python
# 3.11 as well. Never on other systems: those binaries cannot load there,
# and the Windows packages would shadow the runtime's own ones.
if sys.platform == "win32":
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

import completeness_paths
import index_config as indices
import ndat_filter
import platform_support
import project_paths

# Loopback only. Only the port can be changed (PORT_ENV_VAR), never the host.
HOST = "127.0.0.1"
DEFAULT_PORT = 9000
PORT_ENV_VAR = "GNSS_WEB_PORT"

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

BROWSE_TITLE = "Select the RAW data folder"

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


# Text of the ValueError that 2-ccdf.py's run_ccdf_analysis() raises when the
# Step 1 completeness has no valid day in the selected range.
STEP2_NO_VALID_DAYS = "No valid days were found in the selected range."


def _parse_th_cov(params: dict, default: float) -> float:
    """
    th_cov of a request body as a float. JSON true/false are rejected:
    float() would otherwise read them as 1.0 / 0.0.
    """
    value = params.get("th_cov", default)
    if isinstance(value, bool):
        raise ValueError(f"th_cov must be a number, got {json.dumps(value)}.")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"th_cov must be a number, got {json.dumps(value)}.") from None


def _parse_index(params: dict) -> str:
    """
    Index analysed by a request: its "index" field, a key of
    index_config.INDEX_CONFIG. Without it, ROTI L1 - the only index before
    it could be chosen, so older requests keep their meaning. Raises
    ValueError for an unknown or disabled index.
    """
    index = params.get("index", indices.DEFAULT_INDEX)
    indices.validate_index_supported(index)
    return index


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


def _run_step1_completeness(station, year, doy_start, doy_end, th_cov, ndat_mode, value_col):
    """
    Computes and saves (CSV + plot) one Step 1 completeness of the index
    value_col. Returns (df_results, entry), entry being mode, label,
    csv_url, plot_url and the compute_completeness_summary() fields. The
    plot title names the index; any criterion other than ALL_NDAT adds its
    label to it.
    """
    df_results = step1.run_completeness(
        station=station, year=year,
        doy_start=doy_start, doy_end=doy_end, th_cov=th_cov, ndat_mode=ndat_mode,
        value_col=value_col,
    )

    paths = step1.resolve_paths(station, year, doy_start, doy_end, th_cov, ndat_mode, value_col)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)

    plot_title = f"Daily data coverage - {indices.get_index(value_col)['label']} - Station {station} ({year})"
    if ndat_mode != ndat_filter.ALL_MODE:
        plot_title += f" - {ndat_filter.validate_ndat_mode(ndat_mode)['label']}"

    step1.save_results_csv(df_results, paths["output_csv"])
    step1.create_coverage_plot(
        df_results,
        th_cov=th_cov,
        plot_title=plot_title,
        output_png=paths["output_png"],
        show_plot=False,
    )

    return df_results, {
        "index": value_col,
        "mode": ndat_mode,
        "label": completeness_paths.ndat_display_label(ndat_mode),
        "csv_url": f"/results/{paths['output_csv'].relative_to(RESULTS_ROOT).as_posix()}",
        "plot_url": f"/results/{paths['output_png'].relative_to(RESULTS_ROOT).as_posix()}",
        **step1.compute_completeness_summary(df_results, year),
    }


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


def resolve_results_file(url_part: str) -> Path | None:
    """
    The file a /results/<url_part> request may serve, or None if it would
    lie outside RESULTS_ROOT. The URL is decoded first (%2e%2e, %5c, ...),
    then the resolved path must still be inside the resolved RESULTS_ROOT:
    this rejects ".." segments, absolute paths ("//etc/passwd",
    "C:/Windows/..."), backslashes on Windows and links leading out of the
    tree, whatever the operating system.
    """
    decoded = unquote(url_part)
    if "\x00" in decoded:
        return None
    root = RESULTS_ROOT.resolve()
    try:
        target = (root / decoded).resolve()
    except (OSError, ValueError):
        return None
    if target == root or not target.is_relative_to(root):
        return None
    return target


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

    def _host_allowed(self) -> bool:
        """
        True if the request names this server in its (single) Host header:
        127.0.0.1 or localhost with the port actually listened on, in any
        letter case. Without a port only on port 80, which browsers omit.
        Anything else - another name resolving to 127.0.0.1, another port,
        no Host - is rejected, whatever the route.
        """
        hosts = self.headers.get_all("Host") or []
        if len(hosts) != 1:
            return False
        port = self.server.server_address[1]
        allowed = {f"{HOST}:{port}", f"localhost:{port}"}
        if port == 80:
            allowed |= {HOST, "localhost"}
        return hosts[0].strip().lower() in allowed

    def _reject_host(self, has_body: bool) -> None:
        """400 for a request failing _host_allowed(); a POST body is read first."""
        if has_body:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if 0 <= length <= 1_000_000:
                self.rfile.read(length)
            else:
                self.close_connection = True
        self.send_error(400, "Invalid Host header.")

    def do_GET(self) -> None:
        if not self._host_allowed():
            self._reject_host(has_body=False)
            return
        route = urlparse(self.path).path

        if route in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html")
            return

        if route == "/api/paths":
            # Read-only: where the steps write their outputs (shown in the page).
            self._send_json(200, {"ok": True, "results_root": str(RESULTS_ROOT)})
            return

        if route == "/api/indices":
            # The Index selector's options, from index_config.py (the only definition).
            self._send_json(200, {"ok": True, "default": indices.DEFAULT_INDEX, "indices": indices.public_list()})
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
            target = resolve_results_file(route[len("/results/"):])
            if target is None:
                self.send_error(400, "Invalid path.")
                return
            self._send_file(target)
            return

        self.send_error(404, f"Unknown route: {route}")

    def do_POST(self) -> None:
        if not self._host_allowed():
            self._reject_host(has_body=True)
            return
        route = urlparse(self.path).path

        if route == "/api/browse-folder":
            self._handle_browse_folder()
            return

        if route == "/api/open-results-folder":
            self._handle_open_results_folder()
            return

        if route == "/api/run/step0":
            self._handle_run_step0()
            return

        if route == "/api/run/step1":
            self._handle_run_step1()
            return

        if route == "/api/ndat-modes":
            self._handle_ndat_modes()
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

    def _same_origin(self) -> bool:
        """
        False if the request comes from another site's page. Browsers send
        Origin with every cross-site POST, so when it is present it must be
        this server's own address (127.0.0.1 or localhost, same port). A
        request without Origin did not come from a web page.
        """
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        port = self.server.server_address[1]
        return origin in (f"http://{HOST}:{port}", f"http://localhost:{port}")

    def _handle_open_results_folder(self) -> None:
        """
        Opens RESULTS_ROOT in the system file manager
        (platform_support.open_folder). Takes no parameters, so the browser
        can never choose what is opened, and only answers this application's
        own page (Origin check) - any web page could otherwise post to
        127.0.0.1. RESULTS_ROOT is created first if it does not exist yet, as
        the first step run would do. Nothing to open it with: 501.
        """
        # The body is read before any answer, so a rejected request still gets
        # its JSON reply instead of a reset connection.
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return
        if not self._same_origin():
            self._send_json(403, {"ok": False, "error": "Request not sent by this application's page."})
            return
        if params:
            self._send_json(400, {"ok": False, "error": "This action takes no parameters: it only opens the results folder."})
            return
        try:
            RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
            platform_support.open_folder(RESULTS_ROOT)
        except platform_support.ActionUnavailableError as exc:
            self._send_json(501, {"ok": False, "error": str(exc)})
            return
        except platform_support.PlatformActionError as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})
            return
        except OSError as exc:
            self._send_json(500, {"ok": False, "error": f"Could not open {RESULTS_ROOT}: {exc}"})
            return
        self._send_json(200, {"ok": True, "results_root": str(RESULTS_ROOT)})

    def _handle_browse_folder(self) -> None:
        """
        Opens the system's native folder dialog (platform_support.
        choose_directory), starting at `initial_dir` when it is an existing
        folder, and returns the chosen absolute path:

            chosen    -> 200 {"ok": true, "cancelled": false, "path": ...}
            cancelled -> 200 {"ok": true, "cancelled": true}
            no dialog on this computer -> 501, dialog failed -> 500,
            no answer within platform_support.DIALOG_TIMEOUT_S (90 s) -> 504

        index.html's browseRawDir() also gives up client-side a little
        earlier; "Browse..." can be clicked again. The dialog blocks only this
        request's thread (ThreadingHTTPServer), not other requests.

        Only this application's own page may open the dialog (Origin check,
        as for /api/open-results-folder); the body is read first, so a
        rejected request still gets its JSON reply.
        """
        try:
            params = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
            return
        if not self._same_origin():
            self._send_json(403, {"ok": False, "error": "Request not sent by this application's page."})
            return

        if not isinstance(params, dict):
            self._send_json(400, {"ok": False, "error": "The request body must be a JSON object."})
            return

        initial_dir = str(params.get("initial_dir") or step0.DATA_DIR)
        try:
            selected = platform_support.choose_directory(BROWSE_TITLE, initial_dir)
        except platform_support.ActionUnavailableError as exc:
            self._send_json(501, {"ok": False, "error": str(exc)})
            return
        except platform_support.ActionTimeoutError as exc:
            self._send_json(504, {"ok": False, "error": str(exc)})
            return
        except platform_support.PlatformActionError as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})
            return
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return

        if selected is None:
            self._send_json(200, {"ok": True, "cancelled": True})
        else:
            self._send_json(200, {"ok": True, "cancelled": False, "path": selected})

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
        except step0.RawSourceConflictError as exc:
            # The year holds data from another (or an unverifiable) RAW input:
            # nothing was written.
            self._send_json(409, {"ok": False, "error": str(exc)})
        except FileNotFoundError as exc:
            self._send_json(404, {"ok": False, "error": str(exc)})
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @_serialize_plots
    def _handle_run_step1(self) -> None:
        """
        Step 1 completeness. The criterion is given by ndat_operator ("all",
        "=", ">=", "<") and ndat_value, and built by
        ndat_filter.build_ndat_mode(). ALL_NDAT is always computed first and
        then the criterion, if it is not "all", so ALL_NDAT is never older
        than the criterion. A request without ndat_operator (the current
        index.html) computes ALL_NDAT only.

        The top-level fields describe the ALL_NDAT completeness, as before.
        "completeness" lists every completeness computed, in that order
        (mode, label, csv_url, plot_url and its summary), and "ndat_mode" is
        the requested criterion.

        Optional Ndat-impact diagnostic (ndat_impact): ALL_NDAT against the
        requested criterion (400 if it is "all"), or, in a request without
        ndat_operator, against its ndat_mode field.
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
            station = str(params.get("station", step1.STATION))
            if self._reject_unconfigured([station], config):
                return
            year = int(params.get("year", step1.YEAR))
            doy_start = int(params.get("doy_start", step1.DOY_START))
            doy_end = int(params.get("doy_end", step1.DOY_END))
            th_cov = _parse_th_cov(params, step1.TH_COV)
            value_col = _parse_index(params)

            # Every check happens before anything is computed or written.
            operator = params.get("ndat_operator")
            if operator is None:
                if params.get("ndat_value") is not None:
                    raise ValueError("ndat_value needs ndat_operator.")
                ndat_mode = ndat_filter.ALL_MODE
                impact_mode = str(params.get("ndat_mode")) if params.get("ndat_impact") else None
            else:
                if params.get("ndat_mode") is not None:
                    raise ValueError(
                        "ndat_mode is not used with ndat_operator: the criterion is "
                        "ndat_operator/ndat_value, and the Ndat-impact diagnostic uses it."
                    )
                ndat_mode = ndat_filter.build_ndat_mode(operator, params.get("ndat_value"))
                impact_mode = ndat_mode if params.get("ndat_impact") else None
                if impact_mode == ndat_filter.ALL_MODE:
                    raise ValueError(
                        "The Ndat-impact diagnostic compares ALL_NDAT with another criterion: "
                        "choose an Ndat operator other than 'all'."
                    )
            indices.validate_ndat_for_index(value_col, ndat_filter.validate_ndat_mode(ndat_mode))
            if impact_mode is not None:
                indices.validate_ndat_for_index(value_col, ndat_filter.validate_ndat_mode(impact_mode))

            modes = [ndat_filter.ALL_MODE]
            if ndat_mode != ndat_filter.ALL_MODE:
                modes.append(ndat_mode)
            completeness = []
            for mode in modes:
                df_results, entry = _run_step1_completeness(
                    station, year, doy_start, doy_end, th_cov, mode, value_col
                )
                if mode == ndat_filter.ALL_MODE:
                    df_all = df_results
                completeness.append(entry)

            # Optional Ndat-impact DIAGNOSTIC, after the official outputs.
            # Only adds its own files and an extra "ndat_impact" key - the
            # rest of the response and the valid/invalid status are the
            # same with or without it.
            extra = {}
            if impact_mode is not None:
                df_impact, impact = step1.run_ndat_impact(
                    df_all, station, year, doy_start, doy_end, th_cov, impact_mode,
                    value_col=value_col,
                )
                impact_paths = step1.resolve_ndat_impact_paths(
                    station, year, doy_start, doy_end, th_cov, impact_mode, value_col,
                )
                impact_paths["diag_dir"].mkdir(parents=True, exist_ok=True)
                step1.save_results_csv(df_impact, impact_paths["output_csv"])
                step1.create_ndat_impact_plot(
                    df_impact,
                    th_cov=th_cov,
                    plot_title=(
                        f"Ndat impact on daily coverage - {indices.get_index(value_col)['label']} "
                        f"- Station {station} ({year})"
                    ),
                    ndat_label=impact["ndat_label"],
                    output_png=impact_paths["output_png"],
                    show_plot=False,
                )
                extra["ndat_impact"] = {
                    **impact,
                    "csv_url": f"/results/{impact_paths['output_csv'].relative_to(RESULTS_ROOT).as_posix()}",
                    "plot_url": f"/results/{impact_paths['output_png'].relative_to(RESULTS_ROOT).as_posix()}",
                }

            all_ndat = {k: v for k, v in completeness[0].items() if k not in ("mode", "label")}
            self._send_json(200, {
                "ok": True,
                "station": station,
                "year": year,
                "th_cov": th_cov,
                **all_ndat,
                "index": value_col,
                "ndat_mode": ndat_mode,
                "completeness": completeness,
                **extra,
            })
        except FileNotFoundError as exc:
            self._send_json(404, {"ok": False, "error": str(exc)})
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def _handle_ndat_modes(self) -> None:
        """
        Ndat criteria whose Step 1 completeness exists for the requested
        index (_parse_index), year, DoY range and th_cov (completeness_paths;
        only reads the 1_completeness folders, nothing is computed). With "station": that
        station's criteria (mode, label, n_valid_days). Without it: the
        union over the stations of stations.json, each criterion with the
        stations that have it (stations, n_stations, total_stations,
        n_valid_days_by_station). No completeness gives "modes": [].
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
            missing = [key for key in ("year", "doy_start", "doy_end", "th_cov") if params.get(key) is None]
            if missing:
                raise ValueError(f"Missing field(s): {', '.join(missing)}.")
            year = int(params["year"])
            doy_start = int(params["doy_start"])
            doy_end = int(params["doy_end"])
            th_cov = _parse_th_cov(params, None)
            value_col = _parse_index(params)

            if params.get("station") is not None:
                station = str(params["station"])
                if self._reject_unconfigured([station], config):
                    return
                self._send_json(200, {
                    "ok": True,
                    "station": station,
                    "index": value_col,
                    "modes": completeness_paths.list_ndat_modes(
                        station, year, doy_start, doy_end, th_cov, index=value_col
                    ),
                })
                return

            stations = [code for code, _ in config]
            self._send_json(200, {
                "ok": True,
                "stations": stations,
                "total_stations": len(stations),
                "index": value_col,
                "modes": completeness_paths.list_ndat_modes_for_stations(
                    stations, year, doy_start, doy_end, th_cov, index=value_col
                ),
            })
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
            th_cov = _parse_th_cov(params, step2.TH_COV)
            ndat_mode = params.get("ndat_mode", None)
            value_col = _parse_index(params)

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
            # date for this exact station/year/doy/th_cov combination. One x
            # range for the 12 months (step2.ccdf_x_lim()).
            x_lim = step2.ccdf_x_lim(index_config, results["ccdf_by_month"].values())
            for month, (x, ccdf_y, _n) in results["ccdf_by_month"].items():
                step2.save_ccdf_plot(
                    month=month, x=x, ccdf=ccdf_y,
                    out_png=step2.individual_ccdf_png(paths, month),
                    station=station, year=year,
                    index_label=index_config["label"],
                    index_unit=index_config["unit"],
                    x_lim=x_lim,
                    show_plot=False,
                )

            summary = step2.compute_ccdf_summary(df_pcts)

            self._send_json(200, {
                "ok": True,
                "station": station,
                "year": year,
                "index": value_col,
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
            th_cov = _parse_th_cov(params, step2.TH_COV)
            ndat_mode = params.get("ndat_mode", None)
            value_col = _parse_index(params)
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

            out_png = step2.multi_ccdf_png(paths, months)
            step2.save_multi_ccdf_plot(
                ccdf_by_month=results["ccdf_by_month"],
                pcts_by_month=results["pcts_by_month"],
                months_to_compare=months,
                out_png=out_png,
                station=station, year=year,
                index_label=index_config["label"],
                index_unit=index_config["unit"],
                x_lim=step2.ccdf_x_lim(
                    index_config, [results["ccdf_by_month"][m] for m in months if m in results["ccdf_by_month"]]
                ),
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
                "index": value_col,
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
        """
        Overlays CCDF curves across stations for one or more months, on one
        shared axis. A station without Step 0/1 data or without valid days
        for the criterion goes to missing_stations; 404 if none is left.
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
            th_cov = _parse_th_cov(params, step2.TH_COV)
            ndat_mode = params.get("ndat_mode", None)
            value_col = _parse_index(params)
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
                except ValueError as exc:
                    # A station whose completeness has no valid day is left out
                    # like a missing one (no fallback to another criterion or
                    # range); any other ValueError is a real error -> 400.
                    if str(exc) != STEP2_NO_VALID_DAYS:
                        raise
                    label = ndat_filter.validate_ndat_mode(ndat_mode)["label"]
                    missing_stations[station] = (
                        f"No valid days at {station} for {indices.get_index(value_col)['label']}, "
                        f"Ndat criterion '{ndat_mode}' ({label}), "
                        f"th_cov={th_cov}, year={year}, DOY{doy_start}_{doy_end}."
                    )

            stations_used = list(ccdf_by_station_month.keys())
            if not stations_used:
                self._send_json(404, {
                    "ok": False,
                    "error": "No Step 0/1 data for any selected station at these settings.",
                    "missing_stations": missing_stations,
                })
                return

            out_png = step2.resolve_cross_station_ccdf_path(
                year, doy_start, doy_end, th_cov, stations_used, months, index_config,
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
                x_lim=step2.ccdf_x_lim(index_config, [
                    curves[m] for curves in ccdf_by_station_month.values() for m in months if m in curves
                ]),
                show_plot=False,
            )

            self._send_json(200, {
                "ok": True,
                "year": year,
                "index": value_col,
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
            th_cov = _parse_th_cov(params, step2.TH_COV)
            ndat_mode = params.get("ndat_mode", None)
            value_col = _parse_index(params)
            index_config = indices.validate_index_supported(value_col)

            grid = {}
            for station in [code for code, _ in config]:
                paths = step2.resolve_paths(
                    station, year, doy_start, doy_end, th_cov, value_col, index_config,
                    ndat_mode=ndat_mode,
                )
                grid[station] = {}
                for month in range(1, 13):
                    png = step2.individual_ccdf_png(paths, month)
                    grid[station][month] = (
                        f"/results/{png.relative_to(RESULTS_ROOT).as_posix()}"
                        if png.exists() else None
                    )

            self._send_json(200, {"ok": True, "year": year, "index": value_col, "grid": grid})
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
            th_cov = _parse_th_cov(params, step2_2.TH_COV)
            ndat_mode = params.get("ndat_mode", None)
            value_col = _parse_index(params)

            result = step2_2.run_ccdf_comparison(
                stations=[code for code, _ in config],
                year=year, doy_start=doy_start, doy_end=doy_end,
                value_col=value_col, th_cov=th_cov, ndat_mode=ndat_mode,
            )
            index_config = result["index_config"]
            table = result["table"]

            paths = step2_2.resolve_comparison_paths(
                year, doy_start, doy_end, th_cov, value_col, index_config, ndat_mode=ndat_mode
            )
            step2_2.save_comparison_table(table, paths["output_csv"])
            step2_2.save_comparison_plot(
                table, result["available_stations"], index_config, paths["output_png"],
                step2_2.MIN_VALID_DAYS_PER_MONTH, step2_2.PLOT_DPI, show_plot=False,
            )

            self._send_json(200, {
                "ok": True,
                "year": year,
                "index": value_col,
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
            th_cov = _parse_th_cov(params, step3.TH_COV)
            ndat_mode = params.get("ndat_mode", None)
            value_col = _parse_index(params)

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
                "index": value_col,
                "unit": index_config["unit"],
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
            th_cov = _parse_th_cov(params, step3_2.TH_COV)
            top_n = int(params.get("top_n", step3_2.TOP_N_DAYS))
            ndat_mode = params.get("ndat_mode", None)
            value_col = _parse_index(params)

            result = step3_2.run_temporal_comparison(
                stations=[code for code, _ in config],
                year=year, doy_start=doy_start, doy_end=doy_end, th_cov=th_cov,
                value_col=value_col, top_n=top_n, ndat_mode=ndat_mode,
            )
            index_config = result["index_config"]

            paths = step3_2.resolve_comparison_paths(
                year, doy_start, doy_end, th_cov, value_col, index_config,
                ndat_mode=ndat_mode, top_n=top_n,
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
                "index": value_col,
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
            th_cov = _parse_th_cov(params, step4.TH_COV)
            ndat_mode = params.get("ndat_mode", None)
            value_col = _parse_index(params)

            with _patched_globals(
                step4, STATION=station, YEAR=year,
                DOY_START=doy_start, DOY_END=doy_end, TH_COV=th_cov,
                NDAT_MODE=ndat_mode, VALUE_COL=value_col,
            ):
                step4.main()  # the real, unmodified main()

            # main() has no return value - rebuild the same paths (from the
            # request params, not from step4's now-restored globals) to
            # read back the CSV it just wrote, with the same index and
            # ndat_mode passed to main() above.
            index_config = indices.validate_index_supported(value_col)
            paths = step4.resolve_paths(
                station, year, doy_start, doy_end, th_cov, value_col, index_config,
                ndat_mode=ndat_mode,
            )
            df_hourly = pd.read_csv(paths["output_hourly_csv"])
            p90_hours, p90_freq, p90_n = _hours_at_max(df_hourly, "frequency_p90_pct")
            p99_hours, p99_freq, p99_n = _hours_at_max(df_hourly, "frequency_p99_pct")

            self._send_json(200, {
                "ok": True,
                "station": station,
                "year": year,
                "index": value_col,
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
            th_cov = _parse_th_cov(params, step4_2.TH_COV)
            ndat_mode = params.get("ndat_mode", None)
            value_col = _parse_index(params)

            result = step4_2.run_hourly_comparison(
                stations=[code for code, _ in config],
                year=year, doy_start=doy_start, doy_end=doy_end, th_cov=th_cov,
                value_col=value_col, ndat_mode=ndat_mode,
            )
            index_config = result["index_config"]

            paths = step4_2.resolve_comparison_paths(
                year, doy_start, doy_end, th_cov, value_col, index_config, ndat_mode=ndat_mode
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
                "index": value_col,
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
            th_cov = _parse_th_cov(params, step5.TH_COV)
            ndat_mode = params.get("ndat_mode", None)
            value_col = _parse_index(params)

            with _patched_globals(
                step5, STATION=station, YEAR=year,
                DOY_START=doy_start, DOY_END=doy_end, TH_COV=th_cov,
                NDAT_MODE=ndat_mode, VALUE_COL=value_col,
            ):
                step5.main()  # the real, unmodified main()

            index_config = indices.validate_index_supported(value_col)
            paths = step5.resolve_paths(
                station, year, doy_start, doy_end, th_cov, value_col, index_config,
                ndat_mode=ndat_mode,
            )
            df_daypart = pd.read_csv(paths["output_daypart_csv"])
            p90_dayparts, p90_freq, p90_n = _dayparts_at_max(df_daypart, "frequency_p90_pct")
            p99_dayparts, p99_freq, p99_n = _dayparts_at_max(df_daypart, "frequency_p99_pct")

            self._send_json(200, {
                "ok": True,
                "station": station,
                "year": year,
                "index": value_col,
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
        it saves a single combined PNG (p90 and p99 stacked).
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
            th_cov = _parse_th_cov(params, step5_2.TH_COV)
            ndat_mode = params.get("ndat_mode", None)
            value_col = _parse_index(params)

            result = step5_2.run_daypart_comparison(
                stations=[code for code, _ in config],
                year=year, doy_start=doy_start, doy_end=doy_end,
                value_col=value_col, th_cov=th_cov, ndat_mode=ndat_mode,
            )
            index_config = result["index_config"]

            paths = step5_2.resolve_comparison_paths(
                year, doy_start, doy_end, th_cov, value_col, index_config, ndat_mode=ndat_mode
            )
            step5_2.save_comparison_table(result["table"], paths["output_csv"])
            step5_2.save_comparison_plot(
                result["table"], result["available_stations"], index_config,
                paths["output_png"], show_plot=False,
            )

            self._send_json(200, {
                "ok": True,
                "year": year,
                "index": value_col,
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
            th_cov = _parse_th_cov(params, step6.TH_COV)
            ndat_mode = params.get("ndat_mode", None)
            value_col = _parse_index(params)

            with _patched_globals(
                step6, STATION=station, YEAR=year,
                DOY_START=doy_start, DOY_END=doy_end, TH_COV=th_cov,
                NDAT_MODE=ndat_mode, VALUE_COL=value_col,
            ):
                step6.main()  # the real, unmodified main()

            index_config = indices.validate_index_supported(value_col)
            paths = step6.resolve_paths(
                station, year, doy_start, doy_end, th_cov, value_col, index_config,
                ndat_mode=ndat_mode,
            )
            top_p90 = pd.read_csv(paths["output_top_cells_p90_csv"])
            top_p99 = pd.read_csv(paths["output_top_cells_p99_csv"])

            self._send_json(200, {
                "ok": True,
                "station": station,
                "year": year,
                "index": value_col,
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
            th_cov = _parse_th_cov(params, step6.TH_COV)
            ndat_mode = params.get("ndat_mode", None)
            value_col = _parse_index(params)

            index_config = indices.validate_index_supported(value_col)
            result = step6.get_cross_station_maxima(
                [code for code, _ in config], year, doy_start, doy_end, th_cov, value_col,
                index_config, ndat_mode,
            )

            self._send_json(200, {
                "ok": True,
                "year": year,
                "index": value_col,
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


def resolve_port() -> int:
    """
    Port of the local server: DEFAULT_PORT, or the value of PORT_ENV_VAR if
    that variable is set. Raises ValueError unless it is a whole number from
    1 to 65535 (0 would let the system pick a random port).
    """
    raw = os.environ.get(PORT_ENV_VAR)
    if raw is None:
        return DEFAULT_PORT
    text = raw.strip()
    if not (text.isascii() and text.isdigit()) or not 1 <= int(text) <= 65535:
        raise ValueError(f"{PORT_ENV_VAR} must be a whole number from 1 to 65535, got {raw!r}.")
    return int(text)


class LocalHTTPServer(ThreadingHTTPServer):
    """
    ThreadingHTTPServer whose bind fails if the port is already in use.
    http.server sets SO_REUSEADDR, and on Windows that lets a second server
    bind a port another program is listening on - both would then share it
    silently. SO_EXCLUSIVEADDRUSE (Windows) makes the bind fail instead.
    Windows still allows binding 127.0.0.1 while another program listens on
    all interfaces (0.0.0.0) on that port, so the bind is also refused if
    anything already answers on the address.
    """
    allow_reuse_address = not hasattr(socket, "SO_EXCLUSIVEADDRUSE")

    def server_bind(self):
        try:
            probe = socket.create_connection(self.server_address, timeout=0.5)
        except OSError:
            pass  # nothing listening there
        else:
            probe.close()
            host, port = self.server_address
            raise OSError(errno.EADDRINUSE, f"another program is already listening on {host}:{port}")
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def open_in_browser(url: str) -> None:
    """Opens the page once the server listens (platform_support.open_browser); never raises."""
    if not platform_support.open_browser(url):
        print(f"Could not open a web browser automatically - open {url} in one.", flush=True)


def main() -> None:
    try:
        port = resolve_port()
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}")
    try:
        server = LocalHTTPServer((HOST, port), Handler)
    except OSError as exc:
        if exc.errno in (errno.EACCES, getattr(errno, "WSAEACCES", errno.EACCES)):
            reason = (
                f"Permission denied for port {port}: on Linux, ports below 1024 need "
                f"administrator rights; Windows may reserve some ports or let another "
                f"program use one exclusively."
            )
        else:
            reason = f"Another program is probably using port {port}: close it, or choose another port."
        raise SystemExit(
            f"ERROR: cannot listen on http://{HOST}:{port} ({exc}).\n"
            f"{reason} Set {PORT_ENV_VAR} to another port (e.g. {PORT_ENV_VAR}=9001)."
        )
    print(f"Serving on http://{HOST}:{port}  (Ctrl+C to stop)", flush=True)
    # The socket is already listening, so the page's first request just waits
    # for serve_forever(). In its own thread: a slow browser start never
    # delays the server, and a failed one only prints the address.
    threading.Thread(target=open_in_browser, args=(f"http://{HOST}:{port}/",), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
