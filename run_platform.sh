#!/usr/bin/env bash
# ==============================================================================
# run_platform.sh — AetherMed 4.0 Full-Stack Platform Launcher
# Starts FastAPI backend (port 8000) and Next.js frontend (port 3000).
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "======================================================================"
echo "      AETHERMED 4.0 — PRECISION CLINICAL DIGITAL TWIN PLATFORM        "
echo "======================================================================"

# 1. Virtual Environment Activation
if [ -d "$ROOT_DIR/venv" ]; then
    echo "[✓] Activating Python virtual environment ($ROOT_DIR/venv)..."
    source "$ROOT_DIR/venv/bin/activate"
elif [ -d "/Users/seba/Downloads/aethermed 3/venv" ]; then
    echo "[✓] Activating fallback virtual environment (/Users/seba/Downloads/aethermed 3/venv)..."
    source "/Users/seba/Downloads/aethermed 3/venv/bin/activate"
else
    echo "[!] Warning: No dedicated virtualenv found; using system Python."
fi

# 2. Trap for graceful shutdown of background processes
cleanup() {
    echo ""
    echo "[*] Shutting down AetherMed 4.0 services..."
    if [ -n "${BACKEND_PID:-}" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID:-}" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    echo "[✓] All processes stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 3. Start FastAPI Backend on Port 8000
echo "[*] Starting FastAPI Backend on http://0.0.0.0:8000 ..."
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend health check
echo "[*] Awaiting backend health verification..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
        echo "[✓] Backend is healthy and ready!"
        break
    fi
    sleep 1
done

# 4. Start Next.js Frontend on Port 3000
echo "[*] Starting Next.js Production Cockpit on http://localhost:3000 ..."
cd "$ROOT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "======================================================================"
echo "                AETHERMED 4.0 PLATFORM IS LIVE!                       "
echo "======================================================================"
echo "  • Clinical Cockpit Web UI : http://localhost:3000"
echo "  • FastAPI Swagger Docs    : http://localhost:8000/docs"
echo "  • Telemetry Endpoint      : http://localhost:8000/api/health"
echo "  • Streamlit Fallback      : streamlit run src/dashboard/app.py"
echo "======================================================================"
echo "Press Ctrl+C to stop all services."
echo ""

# Keep script running
wait
