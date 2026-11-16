/**
 * VoiceShield — Frontend API Service Client
 */

import {
  AnalysisResult,
  BatchAnalysisResult,
  HealthStatus,
  ModelInfoResponse,
} from '../types';

const API_BASE = '/api';

export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const res = await fetch(`${API_BASE}/model-info`);
  if (!res.ok) {
    throw new Error(`Failed to fetch model info: ${res.statusText}`);
  }
  return res.json();
}

export async function analyzeAudioFile(
  file: File,
  enableWindowing: boolean = true
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('enable_windowing', String(enableWindowing));

  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Analysis failed (${res.status}: ${res.statusText})`);
  }

  return res.json();
}

export async function analyzeAudioBatch(files: File[]): Promise<BatchAnalysisResult> {
  const formData = new FormData();
  files.forEach((f) => formData.append('files', f));

  const res = await fetch(`${API_BASE}/analyze/batch`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Batch analysis failed (${res.status})`);
  }

  return res.json();
}

export async function fetchWaveformPeaks(
  file: File,
  points: number = 120
): Promise<{ duration_sec: number; sample_rate: number; channels: number; peaks: number[] }> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('points', String(points));

  const res = await fetch(`${API_BASE}/waveform`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    throw new Error('Failed to extract waveform peaks');
  }

  return res.json();
}

export async function fetchHistory(): Promise<AnalysisResult[]> {
  const res = await fetch(`${API_BASE}/history`);
  if (!res.ok) {
    throw new Error(`Failed to fetch history: ${res.statusText}`);
  }
  return res.json();
}

export async function clearHistory(): Promise<void> {
  const res = await fetch(`${API_BASE}/history`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    throw new Error(`Failed to clear history: ${res.statusText}`);
  }
}

export function createStreamSocket(
  onMessage: (data: any) => void,
  onError?: (err: Event) => void,
  onClose?: () => void
): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  const wsUrl = `${protocol}//${host}/api/stream/analyze`;

  const ws = new WebSocket(wsUrl);

  ws.onmessage = (event) => {
    try {
      const parsed = JSON.parse(event.data);
      onMessage(parsed);
    } catch (e) {
      console.error('Failed to parse WebSocket message:', e);
    }
  };

  if (onError) ws.onerror = onError;
  if (onClose) ws.onclose = onClose;

  return ws;
}
