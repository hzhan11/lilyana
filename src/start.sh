#!/bin/sh
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${PYTHON:-}" ]; then
    exec "$PYTHON" "$HERE/launcher.py" "$@"
elif [ -x "$HERE/../.venv/bin/python" ]; then
    exec "$HERE/../.venv/bin/python" "$HERE/launcher.py" "$@"
elif [ -x "$HERE/../.venv/Scripts/python.exe" ]; then
    exec "$HERE/../.venv/Scripts/python.exe" "$HERE/launcher.py" "$@"
elif command -v python3 >/dev/null 2>&1; then
    exec python3 "$HERE/launcher.py" "$@"
elif command -v python >/dev/null 2>&1; then
    exec python "$HERE/launcher.py" "$@"
else
    echo "Python 3 is required. Install it or set PYTHON to its executable." >&2
    exit 1
fi
