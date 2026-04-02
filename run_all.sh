#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all.sh — Full Pipeline Orchestration
# Loan Risk AI Governance System
# Run: bash run_all.sh [--skip-agents <list>]
# ─────────────────────────────────────────────────────────────────────────────

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✅ $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠️  $1${NC}"; }
error() { echo -e "${RED}[$(date '+%H:%M:%S')] ❌ $1${NC}"; exit 1; }
header() { echo -e "\n${GREEN}══════════════════════════════════════════════${NC}"; echo -e "${GREEN}  $1${NC}"; echo -e "${GREEN}══════════════════════════════════════════════${NC}\n"; }

header "🏦 Loan Risk AI Governance System"
echo "  Running full pipeline: Data → Train → Monitor → Fairness"
echo "  Project root: $PROJECT_ROOT"

# ── 0. Check Python ────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  error "Python 3 not found. Please install Python 3.9+"
fi
PYTHON=$(command -v python3)
log "Python: $($PYTHON --version)"

# ── 1. Create virtual environment (if not exists) ─────────────────────────
if [ ! -d "venv" ]; then
  header "📦 Agent 0: Setup — Creating virtual environment"
  $PYTHON -m venv venv
  log "Virtual environment created"
fi

source venv/bin/activate
log "Virtual environment activated"

# ── 2. Install dependencies ──────────────────────────────────────────────
header "📦 Agent 0: Setup — Installing dependencies"
pip install -q --upgrade pip
pip install -q -r requirements.txt
log "Dependencies installed"

# ── 3. DataAgent ──────────────────────────────────────────────────────────
header "💾 Agent 1: DataAgent — Data preprocessing pipeline"
$PYTHON src/data_preprocessing.py
log "Data preprocessing complete"

# ── 4. TrainingAgent ─────────────────────────────────────────────────────
header "🧠 Agent 2: TrainingAgent — Model training + MLflow tracking"
$PYTHON src/train.py
log "Training complete. Run 'mlflow ui' to view experiments"

# ── 5. MonitoringAgent ───────────────────────────────────────────────────
header "📊 Agent 3: MonitoringAgent — Drift detection + performance"
$PYTHON src/monitoring.py
log "Monitoring complete. Check monitoring/ for reports"

# ── 6. FairnessAgent ─────────────────────────────────────────────────────
header "⚖️  Agent 4: FairnessAgent — Fairness analysis + SHAP"
$PYTHON src/fairness.py
log "Fairness analysis complete. Check reports/ for plots"

# ── 7. Tests ──────────────────────────────────────────────────────────────
header "🧪 Running test suite"
pytest tests/ -v --tb=short 2>&1 || warn "Some tests failed — check output above"

# ── 8. Summary ───────────────────────────────────────────────────────────
header "🎉 Pipeline Complete!"
echo ""
echo "  📁 Generated artifacts:"
echo "     data/processed/     — train/test/future CSVs + scaler"
echo "     models/             — trained model (.pkl) + feature columns"
echo "     mlruns/             — MLflow experiment runs"
echo "     monitoring/         — drift_report.html + drift_results.json"
echo "     reports/            — ROC curves, confusion matrices, SHAP plots, fairness charts"
echo "     governance/         — Model Card, EU AI Act docs"
echo ""
echo "  🚀 Next steps:"
echo "     • View MLflow:   mlflow ui  (open http://localhost:5000)"
echo "     • Start API:     cd api && uvicorn main:app --reload"
echo "     • Open docs:     http://localhost:8000/docs"
echo "     • Drift report:  open monitoring/drift_report.html"
echo ""
echo "  🐳 Docker:"
echo "     docker build -t loan-risk-api -f api/Dockerfile ."
echo "     docker run -p 8000:8000 loan-risk-api"
echo ""
log "All agents completed successfully! 🏦"
