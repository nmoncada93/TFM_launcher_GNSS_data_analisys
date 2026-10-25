# GNSS Ionospheric Analysis Pipeline

Local web application to prepare daily GNSS RAW files (ROTI and S4) and analyse their completeness, distributions and temporal variability, station by station.

---

# How to Run

## Windows (recommended)

1. Download the project: on GitHub, **Code → Download ZIP**, or `git clone` it.
2. Extract the ZIP to a short folder path, for example `C:\GNSS\`.
3. Double-click `run_web_server.bat`.
4. The application opens in your default browser.

No Python installation is required: the project includes its own Python runtime (64-bit Windows 10 or 11).

Keep the console window open while you work. Close it to stop the application.

Notes:

- Very long folder paths (for example deep OneDrive folders) can make Windows skip files while extracting. Use a short path.
- If Windows SmartScreen shows "Windows protected your PC", click **More info → Run anyway**.

## Ubuntu

Ubuntu runtime scripts are prepared; validation on a real Ubuntu system is still pending.

Target: Ubuntu 22.04 or newer, x86_64. No `sudo` is needed.

Open a terminal in the project folder.

**First time only** — prepare the Linux Python runtime (requires Internet, about 125 MB):

```bash
bash tools/linux/build_linux_runtime.sh
```

**Then, every time:**

```bash
bash run_web_server.sh
```

The application opens in your default browser. Press `Ctrl+C` in the terminal to stop it.

Work in a Linux folder (for example your home folder), not in a Windows shared folder.

**Browse...** uses `zenity` (or `kdialog`) when installed. Otherwise, type or paste the RAW folder path.

## Browser and port

The application opens automatically in the default browser. If it does not, open:

```text
http://127.0.0.1:9000
```

The application is only reachable from this computer (`127.0.0.1`).

Optional — use another port if 9000 is busy:

```text
Windows (cmd):   set GNSS_WEB_PORT=9001
                 run_web_server.bat

Ubuntu:          GNSS_WEB_PORT=9001 bash run_web_server.sh
```

---

# Requirements (optional / developers)

Recommended: use the supplied portable runtime. You normally do not need to install anything manually.

- **Windows:** the runtime is already included (`app/python_portable/` and `app/python_libraries/`, Python 3.11.9).
- **Ubuntu:** `tools/linux/build_linux_runtime.sh` downloads the same Python version and the same package versions, and checks their hashes.

The requirement files are only for development, rebuilding a runtime or custom Python environments:

| File | Content |
|---|---|
| `app/requirements.txt` | The 4 packages the code uses directly (pandas, numpy, matplotlib, pyarrow). Their own dependencies are not pinned. |
| `tools/linux/requirements-linux-x86_64.txt` | The complete pinned Linux set (14 packages, with hashes). Used by the Ubuntu build script. |

## Install requirements manually (optional)

Normally not needed. The portable runtime remains the recommended and reproducible option.

A custom environment is a development alternative. `app/requirements.txt` fixes only the main packages; pip chooses the versions of their dependencies, so the environment is not identical to the portable runtime.

Custom environments require Python 3.11. Python 3.11.9 is the tested version. Run the commands from the project root folder.

Windows (Command Prompt):

```text
python -m venv app\.venv
app\.venv\Scripts\activate
python -m pip install -r app\requirements.txt
```

Ubuntu:

```bash
python3 -m venv app/.venv
source app/.venv/bin/activate
python -m pip install -r app/requirements.txt
```

Both launchers automatically use `app/.venv` when it exists; delete that folder to go back to the portable runtime.

---

# Basic workflow

The repository contains no RAW data and no results. You need your own daily RAW files.

1. **Step 0** — choose the RAW folder (**Browse...**), the Year and the DoY range, review the stations (**Station configuration**), and click **Prepare data**.
2. **Index** — choose the variable to analyse: ROTI L1, ROTI LGF, ROTI L2, S4 L1 or S4 L2.
3. **Step 1** — completeness: coverage threshold (TH_COV) and Ndat criterion.
4. **Steps 2–6** — analyses. Sections 2.2, 3.2, 4.2, 5.2 and the Step 6 comparison compare stations.

Results are saved under `results/` in the project folder (**Open results folder** opens it).

`Clear` only clears the web page. It never deletes result files.

---

# RAW data format

One RAW file per day. Preferred names:

```text
rotiS4.YYYYDDD
rotiS4.YYYYDDD.gz
rotiS4.YYYYDDD.tmp
rotiS4.YYYYDDD.tmp.gz
```

If no preferred name exists, another name is accepted when it contains `YYYYDDD` as a complete number (no other digit directly before or after it) and has no extension or ends in `.gz`, `.tmp` or `.tmp.gz`, for example `roti.2025120.gz`.

- The date in the file name must be inside the Year and DoY range chosen in Step 0.
- More than one candidate file for the same day is an error (ambiguous).

Each line has 12 whitespace-separated columns, no header:

```text
 1  Second of the day (GPS time)
 2  Receiver name
 3  Satellite number (1–32 GPS; >40 Galileo)
 4  Satellite elevation [0, 90] deg
 5  Satellite azimuth (-180, 180] deg (0 North, 90 East)
 6  ROTI from L1 [TECU/min]
 7  ROTI from LGF [TECU/min]
 8  Ndat(ROTI L1): number of data used to calculate ROTI L1 and S4
 9  Ndat(ROTI LGF): number of data used to calculate ROTI LGF
10  S4 from L1
11  S4 from L2
12  ROTI from L2 [TECU/min]
```

A file with another number of columns on any line stops Step 0 with an error naming the file, before any result file is written.

## Index

The **Index** selector (between Step 0 and Step 1, also shown in the top bar) chooses the variable analysed by Steps 1–6 and the comparisons:

```text
Index     Value column   Ndat column   Unit
ROTI L1   6              8             TECU/min
ROTI LGF  7              9             TECU/min
ROTI L2   12             8             TECU/min
S4 L1     10             8             dimensionless
S4 L2     11             8             dimensionless
```

- Every index accepts the same Ndat criteria.
- Only NaN or infinite values are discarded: S4 = 0 is a valid value.
- Step 0 keeps every RAW column, so it does not depend on the index.

---

# Station configuration — `app/stations.json`

`stations.json` lists the stations shown in the application and their UTC offsets:

```json
{
  "code": "KOUG",
  "utc_offset_hours": -3
}
```

- The station code must match the receiver name in the RAW data (column 2).
- `utc_offset_hours` defines local time, used by the hourly, daypart and month-hour analyses.

Manage it from Step 0 (**Station configuration**):

- add or remove stations (removing one deletes no result);
- edit the **Fixed offset** (hours, decimals allowed, from -12 to 14), saved on Enter or when the field loses focus;
- **Detect from RAW**: lists the receiver names found in up to 12 RAW files of the Step 0 Year and DoY range, and marks the ones already configured. A new station can be added once its Fixed offset is entered.

The file can also be edited by hand. Reload the web page afterwards; restarting the server is not required.

Coordinates are defined separately in `STATION_METADATA` inside `app/0-toParquet.py`. A station missing there is still processed, with empty longitude and latitude.

Changing the stations or UTC offsets of a year that is already built changes its Step 0 configuration: Step 0 then refuses to rebuild that year (see "Step 0 provenance").

---

# Scripts and developer information

## Pipeline

```text
RAW
 ↓
Step 0 — Parquet
 ↓
Step 1 — Completeness
 ↓
Steps 2–6 — Scientific analyses
```

## Step scripts

| Script | Purpose |
|---|---|
| `app/0-toParquet.py` | Step 0: reads the daily RAW files and writes one annual Parquet file per station, plus RAW inventory, station summary, UTC/local-time columns, QC information and provenance. |
| `app/1-completitud.py` | Step 1: daily completeness of the selected index and valid days (TH_COV, Ndat criterion). A minute counts when it has at least one observation that meets the Ndat criterion with a finite index value. |
| `app/2-ccdf.py` | Step 2: monthly distributions, CCDFs and monthly percentiles; cross-station CCDF. |
| `app/2.2-ccdf_comparison.py` | Step 2.2: compares monthly percentiles between stations. |
| `app/3-temporal_variability.py` | Step 3: annual p90/p99 thresholds of the selected index and daily exceedance frequency. |
| `app/3.2-temporal_comparison.py` | Step 3.2: compares daily variability between stations. |
| `app/4-hourly_variability.py` | Step 4: exceedance frequency by local hour. |
| `app/4.2-hourly_comparison.py` | Step 4.2: compares hourly results between stations. |
| `app/5-daypart_variability.py` | Step 5: exceedance frequency by part of the day. |
| `app/5.2-daypart_comparison.py` | Step 5.2: compares daypart results between stations. |
| `app/6-month_hour_heatmaps.py` | Step 6: month × local-hour heatmaps and cross-station maxima. |

## Supporting files

| File | Role |
|---|---|
| `app/web_server.py` | Local HTTP server and API. Listens on `127.0.0.1` only; default port 9000 (`GNSS_WEB_PORT`). |
| `app/static/index.html` | Web interface. |
| `app/index_config.py` | The only definition of the indices: value column, Ndat column, file tag, label, unit, plot range. |
| `app/ndat_filter.py` | Ndat criteria: `ALL_NDAT`, `Ndat = N`, `N <= Ndat <= 60`, `Ndat < N` (N from 1 to 60), applied to the Ndat column of the index. |
| `app/completeness_paths.py` | Completeness paths (per index), TH_COV run tags, discovery of Step 1 Ndat configurations. |
| `app/project_paths.py` | Main project and results paths. |
| `app/platform_support.py` | Operating-system functions: folder selection, opening folders, opening the browser. |
| `tools/linux/` | Ubuntu runtime build (`build_linux_runtime.sh`) and package (`make_ubuntu_package.sh`) scripts; see `tools/linux/README.md`. |

## Run the server without the launcher

From the project root folder:

```text
Windows:   app\python_portable\python.exe app\web_server.py
Ubuntu:    app/python_portable_linux/bin/python3.11 app/web_server.py
```

## Development rules

1. Scientific calculations belong in the corresponding Step script.
2. HTTP/API handling belongs in `web_server.py`.
3. UI changes belong in `app/static/index.html`.
4. Use `project_paths.py` and `completeness_paths.py` for result paths.
5. TH_COV file identity must use `completeness_paths.run_tag()` / `format_th_cov()`.
6. Use `ndat_filter.py` for Ndat criteria and `index_config.py` for indices.
7. Use `platform_support.py` for OS-specific operations.
8. Preserve Windows and Ubuntu compatibility.
9. Keep the server restricted to `127.0.0.1`.
10. Test existing functionality before committing changes.

---

# Advanced notes

## Step 0 provenance

Step 0 protects existing studies against accidental reconstruction from different inputs. The identity of a study includes:

- the RAW folder and the resolved RAW files (size and modification time);
- the DoY range;
- the relevant station configuration;
- the QC configuration.

If the identity conflicts with an existing study, Step 0 stops before writing anything.

Moving or copying a study to another folder, computer or operating system can therefore prevent Step 0 from rebuilding it. Steps 1–6 can still use its existing results.

To rebuild a year on purpose, first move or delete its outputs yourself: `results/<STATION>/<YEAR>` for every station and `results/global/<YEAR>`. The application never deletes them.

Studies created before provenance tracking are legacy. Step 0 does not overwrite them; Steps 1–6 keep using their Parquet files and results normally.

## Results per index

Results of different indices never mix: each index has its own completeness, result folders (`<step>/<index>/...`) and file tags (`ROTIL1`, `S4L1`, ...).

ROTI L1 keeps the historical names. Results created before the index could be chosen are ROTI L1 results and stay usable without renaming or migration.

## Completeness selection

The web interface lists the Step 1 completeness configurations available for the selected station, year, DoY range and TH_COV. Each scientific Step reads exactly the selected configuration and fails clearly if it does not exist.
