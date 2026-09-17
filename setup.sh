#!/bin/bash
# setup.sh — One-command project setup
# Usage: bash setup.sh

set -e

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║       DeepTrace AI — Project Setup           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Frontend ─────────────────────────────────────────────────────────────────
echo "📦 Setting up Frontend..."
cd frontend
npm install
cp .env.example .env
echo "   ✓ Frontend ready"
echo "   ⚠  Edit frontend/.env and add your VITE_ANTHROPIC_API_KEY"
cd ..

echo ""

# ── Backend ──────────────────────────────────────────────────────────────────
echo "🐍 Setting up Backend..."
cd backend
python3 -m venv venv
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || true
pip install -r requirements.txt -q
cp .env.example .env
echo "   ✓ Backend ready"
echo "   ⚠  Edit backend/.env and add your ANTHROPIC_API_KEY"
cd ..

echo ""

# ── ML ───────────────────────────────────────────────────────────────────────
echo "🧠 Setting up ML pipeline..."
cd ml
pip install -r requirements.txt -q
mkdir -p data checkpoints
echo "   ✓ ML pipeline ready"
echo "   ℹ  See ml/datasets/DOWNLOAD.md to download training datasets"
cd ..

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   Setup complete!                            ║"
echo "╠══════════════════════════════════════════════╣"
echo "║                                              ║"
echo "║  1. Add API keys to .env files               ║"
echo "║                                              ║"
echo "║  2. Start frontend:                          ║"
echo "║     cd frontend && npm run dev               ║"
echo "║                                              ║"
echo "║  3. Start backend:                           ║"
echo "║     cd backend && uvicorn app.main:app       ║"
echo "║          --reload                            ║"
echo "║                                              ║"
echo "║  4. Train model (optional):                  ║"
echo "║     cd ml && python training/train.py        ║"
echo "║                                              ║"
echo "║  Frontend → http://localhost:5173            ║"
echo "║  Backend  → http://localhost:8000            ║"
echo "║  API Docs → http://localhost:8000/docs       ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
