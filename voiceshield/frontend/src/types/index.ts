/**
 * VoiceShield — Frontend Type Definitions
 */

export type ClassificationState = 'LIKELY HUMAN' | 'LIKELY AI-GENERATED' | 'UNCERTAIN';

export interface AudioMetadata {
  filename: string;
  duration_sec: number;
  sample_rate: number;
  channels: number;
  format: string;
  file_size_bytes: number;
}

export interface SegmentAnalysis {
  window_index: number;
  start_sec: number;
  end_sec: number;
  score: number;
  classification: ClassificationState;
  spoof_score: number;
  bonafide_score: number;
}

export interface SpectralDiagnostics {
  pitch_stability: number;
  pitch_mean_hz: number;
  spectral_flatness: number;
  spectral_centroid_hz: number;
  spectral_rolloff_hz: number;
  harmonic_ratio: number;
  energy_rms: number;
}

export interface AnalysisResult {
  id: string;
  filename: string;
  classification: ClassificationState;
  model_score: number;
  confidence: number;
  raw_logits?: [number, number];
  model_name: string;
  model_version: string;
  audio_metadata: AudioMetadata;
  processing_duration_ms: number;
  created_at: string;
  segments: SegmentAnalysis[];
  spectral_diagnostics?: SpectralDiagnostics;
  limitations_note: string;
  is_uncertain: boolean;
  uncertainty_reason?: string;
}

export interface BatchAnalysisResult {
  total_files: number;
  human_count: number;
  ai_generated_count: number;
  uncertain_count: number;
  total_duration_sec: number;
  total_processing_ms: number;
  results: AnalysisResult[];
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  model_variant: string;
  device: string;
  cuda_available: boolean;
  gpu_name?: string;
  torch_version: string;
  uptime_sec: number;
  version: string;
}

export interface ModelInfoResponse {
  architecture: string;
  model_variant: string;
  parameter_count: number;
  input_requirements: {
    sample_rate_hz: number;
    channels: number;
    window_samples: number;
    window_duration_sec: number;
    dtype: string;
  };
  weights_path: string;
  weights_found: boolean;
  available_methods: string[];
  supported_inference_modes: string[];
  decision_thresholds: {
    likely_human_upper_bound: number;
    uncertain_lower_bound: number;
    uncertain_upper_bound: number;
    likely_ai_lower_bound: number;
  };
  calibration_method: string;
  important_limitations: string[];
}

export interface StreamResultData {
  call_id: string;
  timestamp: string;
  analysis_index: number;
  window_score: number;
  smoothed_score: number;
  classification: ClassificationState;
  alert_triggered: boolean;
  latency_ms: number;
  buffered_seconds: number;
}
