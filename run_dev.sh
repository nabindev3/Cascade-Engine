#!/bin/bash
# Quick start for local development (without Docker)

set -e

echo "═══════════════════════════════════════════════"
echo "  Cascade Inference Engine — Dev Mode"
echo "═══════════════════════════════════════════════"

# Check prerequisites
command -v python3 >/dev/null 2>&1 || { echo "❌ python3 required"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "❌ node required"; exit 1; }

# Load .env if present
if [ -f .env ]; then
    echo "📄 Loading .env"
    export $(grep -v '^#' .env | xargs)
fi

# Install Python deps
echo "🐍 Installing Python dependencies..."
cd python_core
pip install -r requirements.txt -q
cd ..

# Install Node deps
echo "📦 Installing Node dependencies..."
cd typescript_api
npm install --silent
cd ..

# Start Python core in background
echo "🚀 Starting Python core on :8001..."
export PYTHONPATH=$(pwd)
uvicorn python_core.service:app --host 0.0.0.0 --port 8001 --reload &
CORE_PID=$!

# Wait for core to be ready
sleep 3

# Start TypeScript gateway
echo "🌐 Starting API Gateway on :3000..."
cd typescript_api
export CORE_SERVICE_URL=http://localhost:8001
npx tsx src/server.ts &
GW_PID=$!
cd ..

echo ""
echo "═══════════════════════════════════════════════"
echo "  ✅ Services running:"
echo "     Core:    http://localhost:8001"
echo "     Gateway: http://localhost:3000"
echo ""
echo "  Test it:"
echo "     curl -X POST http://localhost:3000/v1/infer \\"
echo "       -H 'Authorization: Bearer dev-key-123' \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"prompt\": \"What is the capital of France?\"}'"
echo ""
echo "  Press Ctrl+C to stop all services."
echo "═══════════════════════════════════════════════"

# Trap Ctrl+C to kill both processes
trap "kill $CORE_PID $GW_PID 2>/dev/null; exit" INT TERM
wait
