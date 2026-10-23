#!/usr/bin/env bash
# Starts the local web interface on Linux (Ubuntu 22.04 / 24.04, x86_64).
# Same role as run_web_server.bat on Windows: picks the Python runtime and
# runs app/web_server.py, which listens on 127.0.0.1 only - port 9000, or
# the one in GNSS_WEB_PORT, inherited from the environment:
#
#     ./run_web_server.sh                       (or: bash run_web_server.sh)
#     GNSS_WEB_PORT=9001 ./run_web_server.sh
#
# No sudo, nothing installed, nothing written outside the project.
set -eu

# Project root = the folder of this script, wherever it is started from.
# Relative RAW folders typed into the page then resolve against the project
# root, as with a double-clicked run_web_server.bat.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ -x "app/.venv/bin/python" ]; then
    PYTHON="app/.venv/bin/python"
    echo "Using app/.venv/bin/python"
elif [ -e "app/python_portable_linux/bin/python3.11" ]; then
    PYTHON="app/python_portable_linux/bin/python3.11"
    echo "app/.venv not found - using app/python_portable_linux/bin/python3.11"
else
    echo "ERROR: no Linux Python runtime found: neither app/.venv/bin/python nor" >&2
    echo "app/python_portable_linux/bin/python3.11 exists. The Ubuntu package includes" >&2
    echo "the second one; in a source checkout, build it with" >&2
    echo "tools/linux/build_linux_runtime.sh. Cannot start the server." >&2
    exit 1
fi

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: $PYTHON is not executable - the package was probably unpacked" >&2
    echo "without its permissions. Unpack the .tar.gz again with: tar -xzf <file>.tar.gz" >&2
    exit 1
fi

exec "$PYTHON" app/web_server.py
