#!/bin/bash
# Start the accelerated Breeze-TTS-2 WebUI (Linux / WSL).
# Override REPO_DIR, MODEL_DIR, PYTHON, WEBUI_PORT, PID_FILE or LOG_FILE as needed.
set -uo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/breeze-tts-2}"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
WEBUI_PORT="${WEBUI_PORT:-7860}"
PID_FILE="${PID_FILE:-$HOME/breeze-webui.pid}"
LOG_FILE="${LOG_FILE:-$HOME/breeze-webui.log}"

cd "$REPO_DIR" || exit 2
# The depth-decoder fast stage uses torch.compile; keep the Inductor cache across
# restarts (the default lives in /tmp and is lost when WSL restarts).
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

if [ -f "$PID_FILE" ]; then
  oldpid=$(cat "$PID_FILE" 2>/dev/null || true)
  [ -n "${oldpid:-}" ] && kill "$oldpid" 2>/dev/null || true
  rm -f "$PID_FILE"
fi
pkill -f "webui_server.py" 2>/dev/null || true
sleep 1

nohup "$PYTHON" webui_server.py "$MODEL_DIR" --port "$WEBUI_PORT" \
  --max-seq-len 1024 --fast-depth-decoder --fast-codec --fast-backbone-decode \
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "started pid=$(cat "$PID_FILE")"

# The first start compiles the graph stages. Later starts reuse the Inductor cache.
for i in $(seq 1 300); do
  if curl -sf --max-time 2 "http://127.0.0.1:$WEBUI_PORT/health" >/dev/null 2>&1; then
    echo "READY after $((i*2))s"
    curl -s --max-time 3 "http://127.0.0.1:$WEBUI_PORT/health"; echo
    curl -s --max-time 3 "http://127.0.0.1:$WEBUI_PORT/api/status"; echo
    exit 0
  fi
  sleep 2
done
echo "TIMEOUT waiting for server"
tail -60 "$LOG_FILE"
exit 1
