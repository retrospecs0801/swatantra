#!/usr/bin/env bash
# Sovereign AI Workbench launch script
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [ -f "$PROJECT_DIR/.venv/bin/streamlit" ]; then
    STREAMLIT_BIN="$PROJECT_DIR/.venv/bin/streamlit"
elif [ -f "/Users/yusufhabib/Documents/projects/openweight AI /.venv/bin/streamlit" ]; then
    STREAMLIT_BIN="/Users/yusufhabib/Documents/projects/openweight AI /.venv/bin/streamlit"
else
    echo "Error: Streamlit binary not found in virtual environment."
    exit 1
fi

echo "Starting Sovereign AI Workbench on http://localhost:8501 ..."
exec "$STREAMLIT_BIN" run app.py "$@"
