#!/usr/bin/env bash
# Builds app/python_portable_linux/, the Python runtime of the Ubuntu package:
#   - CPython 3.11.9, self-contained build from python-build-standalone
#     (relocatable, needs glibc >= 2.17; checksum-verified);
#   - the exact packages of requirements-linux-x86_64.txt (Linux wheels,
#     hash-verified, the same versions as the Windows app/python_libraries/).
#
# Run ON LINUX x86_64 (Ubuntu 22.04/24.04 machine, VM or WSL), from a
# checkout of this project:
#
#     bash tools/linux/build_linux_runtime.sh
#
# Needs internet (about 21 MB of Python + 104 MB of wheels), tar, sha256sum
# and curl or wget. No sudo and nothing installed in the system: everything
# goes into app/python_portable_linux/, which is replaced if it exists.
# Building on Linux keeps the runtime's executable bits and symbolic links.
set -euo pipefail

PBS_RELEASE="20240814"
PY_VERSION="3.11.9"
ASSET="cpython-${PY_VERSION}+${PBS_RELEASE}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
ASSET_SHA256="daa487c7e73005c4426ac393273117cf0e2dc4ab9b2eeda366e04cd00eea00c9"
URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_RELEASE}/${ASSET//+/%2B}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LOCK="$HERE/requirements-linux-x86_64.txt"
TARGET="$ROOT/app/python_portable_linux"
PY="$TARGET/bin/python3.11"

fail() { echo "ERROR: $*" >&2; exit 1; }

[ "$(uname -s)" = "Linux" ] || fail "run this script on Linux (this is $(uname -s))."
[ "$(uname -m)" = "x86_64" ] || fail "only x86_64 is supported (this is $(uname -m))."
[ -f "$ROOT/app/web_server.py" ] || fail "project not found at $ROOT."
for tool in tar sha256sum; do
    command -v "$tool" >/dev/null || fail "'$tool' not found."
done
if command -v curl >/dev/null; then
    download() { curl -fL --retry 3 -o "$1" "$2"; }
elif command -v wget >/dev/null; then
    download() { wget -O "$1" "$2"; }
else
    fail "neither curl nor wget found."
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "== Python $PY_VERSION ($ASSET)"
download "$WORK/$ASSET" "$URL" || fail "download failed: $URL"
echo "$ASSET_SHA256  $WORK/$ASSET" | sha256sum -c --quiet - || fail "checksum mismatch for $ASSET."
tar -xzf "$WORK/$ASSET" -C "$WORK"          # unpacks to $WORK/python/
[ -x "$WORK/python/bin/python3.11" ] || fail "unexpected archive layout (no python/bin/python3.11)."
rm -rf "$TARGET"
mv "$WORK/python" "$TARGET"
"$PY" -c 'import sys; v = sys.version_info[:3]; assert v == (3, 11, 9), v' \
    || fail "unexpected Python version in $TARGET."

echo "== Packages of $(basename "$LOCK") (binary wheels only, hashes checked, nothing else)"
"$PY" -m pip --version >/dev/null 2>&1 || "$PY" -m ensurepip --default-pip >/dev/null \
    || fail "pip is not available in the downloaded Python."
"$PY" -m pip install --disable-pip-version-check --no-cache-dir --no-deps \
    --only-binary=:all: --require-hashes -r "$LOCK" || fail "package installation failed."
"$PY" -m pip check || fail "missing or conflicting dependencies."

echo "== Checks"
"$PY" - "$LOCK" <<'EOF'
import importlib.metadata as md
import re
import sys

pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)", open(sys.argv[1]).read(), re.M))
wrong = {name: (want, md.version(name)) for name, want in pins.items() if md.version(name) != want}
if wrong:
    sys.exit(f"version mismatch (locked, installed): {wrong}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot  # noqa: E402,F401
import numpy  # noqa: E402,F401
import pandas  # noqa: E402,F401
import PIL  # noqa: E402,F401
import pyarrow.parquet  # noqa: E402,F401

print(f"Python {sys.version.split()[0]}; {len(pins)} packages match the lock.")
EOF
(cd "$ROOT/app" && "$PY" -B -c '
import shutil
import web_server  # loads every step module; does not start the server
import platform_support
assert web_server.platform_support is platform_support
print("App modules import. Linux helpers found:",
      {tool: bool(shutil.which(tool)) for tool in ("zenity", "kdialog", "xdg-open", "gio")})
') || fail "the application modules do not import with this runtime."

echo "== Done: $TARGET ($(du -sh "$TARGET" | cut -f1))"
echo "Start the server with ./run_web_server.sh; make the package with"
echo "bash tools/linux/make_ubuntu_package.sh"
