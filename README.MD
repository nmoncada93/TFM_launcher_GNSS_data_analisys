# GNSS Ionospheric Analysis Pipeline

Local web application for preparing GNSS RAW data and performing completeness, distribution and temporal analyses.

The application runs locally at:

```text
http://127.0.0.1:9000
```

Derived data are stored under:

```text
results/
```

`Clear` only clears the web interface. It never deletes result files.

---

# Main scripts

## Step 0 — `app/0-toParquet.py`

Reads daily RAW files and creates one annual Parquet file per station.

It also creates the RAW inventory, station summary, UTC/local-time columns, QC information and RAW provenance metadata.

Preferred RAW naming:

```text
rotiS4.YYYYDDD
rotiS4.YYYYDDD.gz
rotiS4.YYYYDDD.tmp
rotiS4.YYYYDDD.tmp.gz
```

If no preferred name exists, another filename is accepted when it contains `YYYYDDD` as a complete number (no other digit directly before or after it) and has no extension or ends in `.gz`, `.tmp` or `.tmp.gz`, for example:

```text
roti.2025120.gz
```

More than one candidate for the same day is treated as ambiguous.

Every RAW line must have the 12 columns of the official format (see "Index"). A file with another number of columns on any line stops Step 0 with an error naming the file, before any result file is written.

The Parquet keeps every RAW column, so Step 0 does not depend on the analysed index.

## Step 1 — `app/1-completitud.py`

Calculates daily completeness of the selected index and determines valid days according to:

```text
Coverage threshold (TH_COV)
Ndat criterion
```

A minute counts when it has at least one observation that satisfies the Ndat criterion and has a finite value of the index (0 is a valid value). Each index and Ndat criterion has its own completeness result.

## Step 2 — `app/2-ccdf.py`

Calculates monthly distributions, CCDFs and monthly percentiles of the selected index.

It also contains the cross-station CCDF analysis.

## Step 2.2 — `app/2.2-ccdf_comparison.py`

Compares monthly percentile results between stations.

## Step 3 — `app/3-temporal_variability.py`

Calculates annual ROTI thresholds such as p90 and p99 and analyzes daily exceedance frequency.

## Step 3.2 — `app/3.2-temporal_comparison.py`

Compares temporal variability between stations.

## Step 4 — `app/4-hourly_variability.py`

Analyzes ROTI variability according to local hour.

## Step 4.2 — `app/4.2-hourly_comparison.py`

Compares hourly variability between stations.

## Step 5 — `app/5-daypart_variability.py`

Analyzes ROTI variability by time-of-day period.

## Step 5.2 — `app/5.2-daypart_comparison.py`

Compares daypart results between stations.

## Step 6 — `app/6-month_hour_heatmaps.py`

Generates month × local-hour heatmaps and cross-station maximum analyses.

---

# Supporting files

## `app/web_server.py`

Runs the local HTTP server and API.

The server is restricted to:

```text
127.0.0.1
```

Default port:

```text
9000
```

## `app/ndat_filter.py`

Handles dynamic Ndat criteria:

```text
ALL_NDAT
Ndat = N
N <= Ndat <= 60
Ndat < N
```

with `N` between 1 and 60. The criterion is applied to the Ndat column of the analysed index (see "Index").

## `app/index_config.py`

The only definition of the analysable indices: Parquet column, Ndat column, file tag, label, unit and plot range of each one.

## `app/completeness_paths.py`

Centralizes completeness paths (per index), TH_COV run tags and discovery of Step 1 Ndat configurations.

## `app/project_paths.py`

Defines the main project/results paths.

## `app/platform_support.py`

Contains operating-system-specific functions such as folder selection and opening the results directory.

## `app/static/index.html`

Main web interface.

---

# Index

The **Index** selector, between Step 0 and Step 1, chooses the variable that Steps 1–6 and the comparisons analyse. The active index is also shown in the top bar.

Official RAW format (`rotiS4.YYYYDDD`, 12 whitespace-separated positional columns, no header):

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

Index, value column and Ndat column:

```text
ROTI L1   column 6    Ndat column 8   TECU/min
ROTI LGF  column 7    Ndat column 9   TECU/min
ROTI L2   column 12   Ndat column 8   TECU/min
S4 L1     column 10   Ndat column 8   dimensionless
S4 L2     column 11   Ndat column 8   dimensionless
```

Every index accepts the same Ndat criteria. Only NaN or infinite values are discarded: S4 = 0 is a valid value.

Results of different indices never mix: each index has its own completeness, result folders (`<step>/<index>/...`) and file tags (`ROTIL1`, `S4L1`, ...).

ROTI L1 keeps the historical names. Results created before the index could be chosen are ROTI L1 results and stay usable without renaming or migration.

---

# Station configuration — `app/stations.json`

`stations.json` controls which stations are available in the web interface and their UTC offsets.

Example:

```json
{
  "code": "KOUG",
  "utc_offset_hours": -3
}
```

`utc_offset_hours` is used to calculate local time and therefore affects hourly, daypart and month-hour analyses.

To add a station, add another entry with its station code and UTC offset.

The station code must match the identifier present in the RAW data.

Coordinates are currently defined separately in `STATION_METADATA` inside `0-toParquet.py`. A station not present there can still be processed, but its longitude and latitude will be empty.

After editing `stations.json`, reload the web page. Restarting the server is not required.

Changing stations or UTC offsets for an already-built year changes the Step 0 scientific configuration. Provenance protection will therefore prevent Step 0 from silently rebuilding that existing study.

---

# How to RUN

## Windows

Recommended:

```text
run_web_server.bat
```

The launcher first uses:

```text
app/.venv/
```

if it exists.

Otherwise it uses the supplied portable Python runtime.

Once the server is listening, the page opens in the default browser. If no browser can be opened, open:

```text
http://127.0.0.1:9000
```

For the most reproducible results, use the supplied portable runtime rather than a custom virtual environment.

### Change the port

From `cmd.exe`:

```text
set GNSS_WEB_PORT=9001
run_web_server.bat
```

---

## Ubuntu

**Ubuntu support is prepared but the Linux runtime has not yet been built and validated on a real Ubuntu system.**

After the Linux runtime has been built:

```bash
./run_web_server.sh
```

Once the server is listening, the page opens in the default browser. If no browser can be opened, open:

```text
http://127.0.0.1:9000
```

To build the Linux runtime from the source repository:

```bash
bash tools/linux/build_linux_runtime.sh
```

This build:

- must run on Linux x86_64;
- requires Internet access while building;
- should be run inside a native Linux filesystem, not a Windows shared folder.

To create the distributable package:

```bash
bash tools/linux/make_ubuntu_package.sh
```

Change port with:

```bash
GNSS_WEB_PORT=9001 ./run_web_server.sh
```

Folder browsing uses `zenity` when available and `kdialog` as fallback. If neither is available, type or paste the path manually.

---

# Alternative — run the server directly

Run these commands from the project root folder.

## Windows

```text
app\python_portable\python.exe app\web_server.py
```

## Ubuntu

After building the Linux runtime:

```bash
app/python_portable_linux/bin/python3.11 app/web_server.py
```

---

# Optional — Python requirements

**Recommended:** use the portable runtime supplied with the application.

Both platform builds target:

```text
Python 3.11.9
```

with the same scientific dependency versions.

`app/requirements.txt` contains the four main project dependencies.

The complete reproducible Linux dependency lock, including transitive dependencies and wheel hashes, is:

```text
tools/linux/requirements-linux-x86_64.txt
```

The complete environment currently contains 14 packages:

```text
numpy
pandas
pyarrow
matplotlib
pillow
contourpy
kiwisolver
fonttools
cycler
python-dateutil
packaging
pyparsing
six
tzdata
```

Manual dependency installation is intended mainly for development. For normal use, the portable runtime is recommended.

---

# Step 0 provenance

Step 0 protects existing studies against accidental reconstruction from incompatible inputs.

Its identity includes:

- RAW folder and resolved RAW files;
- DoY range;
- relevant station configuration;
- QC configuration.

If the identity conflicts with an existing study, Step 0 stops before writing anything.

Because the identity includes the RAW folder location and each RAW file's size and modification time, moving or copying a study to another folder, computer or operating system can prevent Step 0 from rebuilding it. Steps 1–6 can still use its existing results.

To rebuild an incompatible year on purpose, first move or delete its outputs yourself: `results/<STATION>/<YEAR>` for every station and `results/global/<YEAR>`. The application never deletes them automatically.

Studies created before provenance tracking are considered legacy. Step 0 will not overwrite them automatically, although Steps 1–6 can continue using their existing Parquet/results normally.

---

# How to improve or develop new functionalities

Keep the application layers separated:

1. Scientific calculations belong in the corresponding Step script.
2. HTTP/API handling belongs in `web_server.py`.
3. UI changes belong in `app/static/index.html`.
4. Use `project_paths.py` and `completeness_paths.py` for result paths.
5. TH_COV file identity must use `completeness_paths.run_tag()` / `format_th_cov()`.
6. Use `ndat_filter.py` for Ndat criteria and `index_config.py` for indices (value column, Ndat column, labels, units).
7. Use `platform_support.py` for OS-specific operations.
8. Preserve Windows and Ubuntu compatibility.
9. Keep the server restricted to `127.0.0.1`.
10. Test existing functionality before committing changes.

Pipeline:

```text
RAW
 ↓
Step 0 — Parquet
 ↓
Step 1 — Completeness
 ↓
Steps 2–6 — Scientific analyses
```

The web interface discovers the Step 1 completeness configurations available for the selected station, year, DoY range and TH_COV. Each scientific Step then reads the exact selected completeness configuration and fails clearly if it does not exist.
