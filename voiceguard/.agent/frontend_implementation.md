# VoiceGuard — Frontend Implementation Plan

## Overview

Build a premium, single-page web dashboard for the VoiceGuard deepfake detection system. The frontend will allow security analysts to upload audio files, record live microphone audio, and view real-time analysis results — all communicating with the existing FastAPI backend.

> [!IMPORTANT]
> The frontend will be a **single `index.html` file** served directly by FastAPI (via `StaticFiles`), using vanilla HTML/CSS/JS with no build tools. This keeps deployment simple and self-contained.

---

## Design Direction

**Aesthetic**: Dark-mode cyber-security dashboard with glowing accents, glassmorphism panels, and micro-animations. Think SOC analyst command center, not a basic form.

**Color Palette**:
| Token | Value | Usage |
|---|---|---|
| `--bg-primary` | `#0a0e1a` | Page background |
| `--bg-card` | `rgba(15, 23, 42, 0.7)` | Glassmorphism card panels |
| `--accent-green` | `#22c55e` | LOW risk / safe indicators |
| `--accent-amber` | `#f59e0b` | MEDIUM risk / caution |
| `--accent-red` | `#ef4444` | HIGH risk / alert |
| `--accent-blue` | `#3b82f6` | Primary action buttons, links |
| `--accent-cyan` | `#06b6d4` | Spectral chart accents |
| `--text-primary` | `#f1f5f9` | Main text |
| `--text-muted` | `#94a3b8` | Secondary labels |

**Typography**: `Inter` from Google Fonts (weights 400, 500, 600, 700).

---

## Page Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  HEADER — Logo + "VoiceGuard" + Health Status Indicator          │
├─────────────────────┬────────────────────────────────────────────┤
│                     │                                            │
│  LEFT PANEL         │  RIGHT PANEL                               │
│  ─────────────      │  ──────────────                            │
│                     │                                            │
│  ┌───────────────┐  │  ┌──────────────────────────────────────┐  │
│  │ Upload Zone   │  │  │  Risk Verdict Banner                 │  │
│  │ (drag & drop) │  │  │  Score: 0.87  │  🔴 HIGH RISK       │  │
│  └───────────────┘  │  └──────────────────────────────────────┘  │
│                     │                                            │
│  ┌───────────────┐  │  ┌──────────────────────────────────────┐  │
│  │ 🎙 Live Mic   │  │  │  Deepfake Score Gauge (animated arc) │  │
│  │ Record Button │  │  │            73.6%                     │  │
│  │ Waveform viz  │  │  └──────────────────────────────────────┘  │
│  └───────────────┘  │                                            │
│                     │  ┌──────────────────────────────────────┐  │
│  ┌───────────────┐  │  │  Spectral Diagnostics Panel          │  │
│  │ Analysis      │  │  │  ├─ Pitch Stability: 0.26            │  │
│  │ History Table │  │  │  ├─ Spectral Flatness: 0.38          │  │
│  │ (scrollable)  │  │  │  ├─ Harmonic Ratio: 2.34             │  │
│  └───────────────┘  │  │  └─ Zero Crossing Rate: 0.087        │  │
│                     │  └──────────────────────────────────────┘  │
│                     │                                            │
│                     │  ┌──────────────────────────────────────┐  │
│                     │  │  Inference Details / Metadata Card    │  │
│                     │  │  Latency: 45ms │ Model: AASIST-L     │  │
│                     │  └──────────────────────────────────────┘  │
│                     │                                            │
├─────────────────────┴────────────────────────────────────────────┤
│  FOOTER — System Status (uptime, total analyses, GPU status)     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Header Bar
- VoiceGuard shield icon + app title
- **Live health dot**: polls `GET /health` every 15s; green dot = healthy, red = unhealthy
- Shows model variant and GPU status in a subtle chip

### 2. Upload Zone (Left Panel)
- Large dashed-border drop zone with hover glow animation
- Accepts `.wav`, `.flac`, `.mp3` files via drag-and-drop or click-to-browse
- On drop: shows filename + duration, triggers `POST /api/v1/analyze`
- Progress indicator during upload/inference

### 3. Live Microphone Recorder (Left Panel)
- "Record" button with pulsing glow animation while active
- Uses `MediaRecorder` API to capture browser microphone audio
- Real-time waveform visualization using `<canvas>` + `AnalyserNode`
- On stop: converts to WAV blob and posts to `POST /api/v1/analyze`
- Alternative advanced mode: stream chunks via `WebSocket /api/v1/stream/{call_id}`, showing live buffer progress bar

### 4. Risk Verdict Banner (Right Panel — Top)
- Full-width banner that appears after analysis
- Color-coded background (green/amber/red glow) depending on risk level
- Large score number with animated count-up
- Recommended action label with icon
- Slide-down entrance animation

### 5. Deepfake Score Gauge (Right Panel)
- Animated arc/donut chart showing 0–100% deepfake probability
- Color transitions: green → amber → red as score increases
- Needle/arc animates on result arrival
- Built with `<canvas>` or SVG — no chart library needed

### 6. Spectral Diagnostics Panel (Right Panel)
- Card with labeled horizontal bar charts for each metric:
  - Pitch Stability
  - Spectral Flatness
  - Spectral Centroid
  - Harmonic Ratio
  - Zero Crossing Rate
  - Energy RMS
- Each bar is color-coded by how "suspicious" the value is
- Tooltip on hover explaining what each metric means for deepfakes

### 7. Inference Details Card (Right Panel)
- Metadata: latency, model(s) used, confidence, timestamp
- Call ID displayed for audit correlation
- Copy-to-clipboard button for the full JSON result

### 8. Analysis History Table (Left Panel — Bottom)
- Scrollable table of all analyses performed in the current session
- Columns: `#`, `Filename`, `Score`, `Risk`, `Latency`, `Time`
- Rows color-coded by risk level
- Click a row to re-display its full result on the right panel
- Stored in `sessionStorage` so it persists across page navigations

### 9. Footer Status Bar
- Live metrics from `GET /health/metrics`: total analyses, avg latency, GPU memory
- Uptime counter (ticking live)
- Connection status indicator

---

## Backend API Integration

| Frontend Action | API Call | Method |
|---|---|---|
| Page load health check | `GET /health` | Fetch |
| File upload analysis | `POST /api/v1/analyze` | FormData + Fetch |
| Live mic streaming | `WS /api/v1/stream/{call_id}` | WebSocket |
| Footer metric refresh | `GET /health/metrics` | Fetch (poll 15s) |

### File Upload Flow
```
User drops file → FormData { file, call_id } → POST /api/v1/analyze → JSON AnalysisResult → Render panels
```

### WebSocket Streaming Flow
```
User clicks Record → navigator.mediaDevices.getUserMedia()
  → MediaRecorder (timeslice=500ms)
  → ondataavailable: chunk → PCM convert → base64 → WS send {"type":"audio_chunk","data":"...","seq":n}
  ← WS recv {"type":"buffer_status", ...}  → Update progress bar
  ← WS recv {"type":"analysis_result", ...} → Render panels
User clicks Stop → WS send {"type":"end_stream"} → Close WS
```

---

## File Structure

```
frontend/
├── index.html          # Single-page app (all HTML structure)
├── css/
│   └── style.css       # Complete stylesheet (design tokens, layout, animations)
└── js/
    ├── app.js           # Main entry point, initialization, event wiring
    ├── api.js           # API client (fetch wrappers, WebSocket manager)
    ├── recorder.js      # Microphone recording + WAV conversion
    ├── gauge.js         # Animated donut/arc gauge canvas component
    ├── waveform.js      # Real-time waveform canvas visualization
    └── utils.js         # Helpers (format time, copy to clipboard, etc.)
```

> [!TIP]
> Serving strategy: Add a `StaticFiles` mount to the existing FastAPI app so the entire system (backend + frontend) is served from one process. No separate dev server needed.

---

## Serving via FastAPI

Add to `api/app.py`:
```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="frontend")

@app.get("/app", tags=["Frontend"])
async def serve_frontend():
    return FileResponse("frontend/index.html")
```

This allows accessing the dashboard at `http://localhost:8000/app`.

---

## Key Animations & Micro-interactions

| Element | Animation |
|---|---|
| Upload zone hover | Border glow pulse (box-shadow with `@keyframes`) |
| Record button active | Pulsing red ring (`scale` + `opacity` loop) |
| Score gauge | Arc draws from 0° to target over 800ms (ease-out) |
| Risk banner entrance | `translateY(-20px) → 0` + `opacity 0 → 1` (300ms) |
| Spectral bars | Width grows from 0% to value over 500ms (staggered) |
| History row insert | `fadeIn` + slight `translateY` slide (200ms) |
| Health dot | Subtle constant pulse (3s cycle, green glow) |
| Loading state | Skeleton shimmer on result panels while awaiting response |

---

## Responsive Considerations

- **Desktop (>1024px)**: Two-column layout as shown above
- **Tablet (768–1024px)**: Stack left panel above right panel
- **Mobile (<768px)**: Single column, full-width cards, upload button replaces drop zone

---

## Implementation Order

1. **Phase 1 — Shell & Styling**: `index.html` + `style.css` with full layout, design tokens, and empty panels
2. **Phase 2 — API Client & File Upload**: `api.js` + `app.js` wiring `POST /api/v1/analyze` to the drop zone
3. **Phase 3 — Result Rendering**: Verdict banner, score gauge (`gauge.js`), spectral bars, inference card
4. **Phase 4 — Microphone Recording**: `recorder.js` + `waveform.js` with WAV conversion and upload
5. **Phase 5 — WebSocket Streaming**: Live streaming mode with buffer progress and real-time results
6. **Phase 6 — History & Polish**: Session history table, footer metrics polling, responsive breakpoints, final animations

---

## Dependencies

**None**. Pure vanilla HTML/CSS/JS. No npm, no build step, no frameworks.

The only external resource is the Google Fonts `Inter` import via `<link>` tag.