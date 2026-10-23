#!/usr/bin/env bash
# Packs the application for Ubuntu into dist/GNSS_Ubuntu_<version>.tar.gz:
# the project files (those tracked by git plus new, not ignored ones) and
# app/python_portable_linux/, without the Windows runtime, the Windows
# launcher, results/, database/ or these build tools. tar keeps executable
# bits and symbolic links. Run on Linux, after build_linux_runtime.sh:
#
#     bash tools/linux/make_ubuntu_package.sh [version]
#
# version defaults to <date>-<git commit>. Nothing outside dist/ is written.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RUNTIME="app/python_portable_linux"

fail() { echo "ERROR: $*" >&2; exit 1; }

[ "$(uname -s)" = "Linux" ] || fail "run this script on Linux (this is $(uname -s))."
cd "$ROOT"
[ -x "$RUNTIME/bin/python3.11" ] || fail "$RUNTIME is missing: run tools/linux/build_linux_runtime.sh first."
command -v git >/dev/null || fail "'git' not found."

VERSION="${1:-$(date +%Y%m%d)-$(git rev-parse --short HEAD)}"
NAME="GNSS_Ubuntu_${VERSION}"
OUT="$ROOT/dist/$NAME.tar.gz"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$NAME" "$ROOT/dist"

# Project files: tracked + untracked-but-not-ignored, existing ones only,
# minus what is Windows-only or build-only.
git ls-files -z --cached --others --exclude-standard \
    | grep -zvE '^(app/python_portable/|app/python_libraries/|run_web_server\.bat$|tools/)' \
    | while IFS= read -r -d '' file; do if [ -e "$file" ]; then printf '%s\0' "$file"; fi; done \
    > "$STAGE/files.lst"
printf '%s\0' "$RUNTIME" >> "$STAGE/files.lst"

tar -cf - --null -T "$STAGE/files.lst" | tar -xf - -C "$STAGE/$NAME"
chmod 755 "$STAGE/$NAME/run_web_server.sh"
find "$STAGE/$NAME" -name __pycache__ -type d -prune -exec rm -rf {} +

tar -czf "$OUT" -C "$STAGE" "$NAME"
(cd "$ROOT/dist" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
echo "Package: $OUT ($(du -h "$OUT" | cut -f1))"
echo "SHA256:  $(cut -d' ' -f1 "$OUT.sha256")"
echo "Use: tar -xzf $NAME.tar.gz && cd $NAME && ./run_web_server.sh"
