# VoiceShield Setup & Installation Guide

This document provides step-by-step instructions for running VoiceShield locally on Windows, Linux, or macOS.

---

## 1. Prerequisites

- **Python**: Python 3.10 or 3.11
- **Node.js**: Node.js 18+ (tested on Node 24) and npm
- **Git**: For cloning repositories and submodules
- **GPU (Optional)**: NVIDIA GPU with CUDA drivers (CPU execution is fully supported and blazingly fast: ~150ms per window on AASIST-L)

---

## 2. Repository Structure

Ensure the workspace contains both `voiceguard` and `voiceshield`:
```
AiVoiceDetection/
├── voiceguard/                  # Underlying VoiceGuard repository & weights
│   ├── models/weights/AASIST-L.pth
│   └── ...
├── voiceshield/
│   ├── backend/                 # FastAPI service & integration adapter
│   ├── frontend/                # React dashboard
│   └── docs/                    # Documentation
└── venv/                        # Shared virtual environment
```

---

## 3. Pretrained Weights Setup

If not already present in `voiceguard/models/weights/`, download the official pretrained weights:

```bash
python voiceguard/scripts/download_weights.py
```

This downloads:
- `AASIST-L.pth` (85K params, ~416 KB)
- `AASIST.pth` (297K params, ~1.2 MB)

---

## 4. Backend Setup

### A. Activate Virtual Environment
```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

### B. Install Python Dependencies
```bash
pip install -r voiceshield/backend/requirements.txt
```

### C. Run Automated Tests
```bash
cd voiceshield/backend
pytest -c pytest.ini tests -v
cd ../..
```
All 26 automated unit and integration tests should pass.

### D. Start the FastAPI Backend
```bash
cd voiceshield/backend
..\..\venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The backend will be available at:
- **API**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

---

## 5. Frontend Setup

### A. Install Dependencies
```bash
cd voiceshield/frontend
npm install
```

### B. Start Development Server
```bash
npm run dev
```
The frontend dashboard will be available at:
- **Dashboard**: http://localhost:5173

Vite automatically proxies `/api` calls to `http://127.0.0.1:8000`.

---

## 6. One-Click Startup Scripts (Windows)

For convenience, startup scripts are provided in the workspace root:

- `start_backend.bat`: Launches the FastAPI backend.
- `start_frontend.bat`: Launches the Vite frontend.
- `start_all.bat`: Launches both servers concurrently.
