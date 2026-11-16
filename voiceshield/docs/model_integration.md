# VoiceShield Model Integration Guide

This guide details how VoiceShield interfaces with the underlying **VoiceGuard** detection engine without modifying its core weights or architecture.

---

## 1. Underlying VoiceGuard Architecture

VoiceGuard is based on **AASIST** (*Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks*, Jung et al., ICASSP 2022 / NAVER Clova AI):

1. **Sinc-Based Front-End (`SincConv`)**:
   - Instead of standard STFT or Mel-spectrograms, AASIST applies learned parametric sinc-based bandpass filters directly to raw waveform samples.
   - Preserves high-resolution phase and boundary spectral cues that vocoders often corrupt.
2. **Residual Encoder**:
   - 6 residual convolutional blocks extract 2D feature representations across frequency and temporal dimensions.
3. **Dual-Branch Graph Attention Networks**:
   - **GAT-S (Spectral Branch)**: Computes node attention across frequency sub-bands.
   - **GAT-T (Temporal Branch)**: Computes node attention across temporal frames.
4. **Heterogeneous Graph Fusion with Master Nodes**:
   - Bidirectional message passing between spectral and temporal nodes using learnable master nodes.
5. **Linear Classification Head**:
   - Concatenates 5 pooled representations (`T_max`, `T_avg`, `S_max`, `S_avg`, `master`) and projects to 2 logits:
     $$\text{output} = [z_0, z_1] = [\text{spoof\_logit}, \text{bonafide\_logit}]$$

---

## 2. Integration Layer Design (`VoiceGuardIntegrationService`)

VoiceShield wraps AASIST inside `voiceshield/backend/app/services/voiceguard_service.py`:

```python
# Slicing input audio to AASIST 64,600 samples
audio_window = pad_or_crop(audio_chunk, 64600)
tensor = torch.from_numpy(audio_window).float().unsqueeze(0).to(self.device)

# Model forward pass
with torch.no_grad():
    last_hidden, logits = self.model(tensor)

# Softmax conversion
probs = torch.nn.functional.softmax(logits, dim=1)
deepfake_score = float(probs[0, 0].item())  # Spoof probability
bonafide_score = float(probs[0, 1].item())  # Bonafide probability
```

### Why Multi-Window Analysis is Necessary
AASIST is strictly a fixed-length model ($64,600\text{ samples} \approx 4.0375\text{s}$ at $16\text{ kHz}$). In VoiceGuard's initial implementation, any audio longer than 4.04s was abruptly truncated:
```python
# Initial VoiceGuard truncation
audio = audio[:64600]
```
In VoiceShield, we implemented **Sliding-Window Segment Evaluation**:
- Full audio files are windowed into overlapping $4.04\text{s}$ segments with a $2.0\text{s}$ stride.
- Each segment receives an authentic forward pass from AASIST.
- Segments are returned in a timeline list:
  ```json
  [
    {"window_index": 0, "start_sec": 0.0, "end_sec": 4.04, "score": 0.78, "classification": "LIKELY AI-GENERATED"},
    {"window_index": 1, "start_sec": 2.0, "end_sec": 6.04, "score": 0.81, "classification": "LIKELY AI-GENERATED"}
  ]
  ```
- Overall recording score is aggregated as:
  $$\text{Score}_{\text{final}} = 0.6 \cdot \text{Score}_{\text{mean}} + 0.4 \cdot \text{Score}_{\text{max}}$$
  This balances sustained acoustic evidence with peak suspicious segment detection.

---

## 3. Calibrated 3-Tier Classification

Rather than presenting raw probabilities as uncalibrated percentages of criminal scam intent, VoiceShield implements a rigorous 3-tier boundary:

| Score Range | Classification State | Risk Level | Description |
|---|---|---|---|
| **$s < 0.30$** | `LIKELY HUMAN` | Low | Voice exhibits natural acoustic dynamics, normal pitch variations, and authentic harmonic structures. |
| **$0.30 \le s < 0.70$** | `UNCERTAIN` | Medium | Inconclusive acoustic indicators. May result from lossy compression, background noise, or borderline model confidence. |
| **$s \ge 0.70$** | `LIKELY AI-GENERATED` | High | Audio exhibits acoustic artifacts consistent with vocoder synthesis, phase distortion, or generative cloning. |

### Decision Certainty Metric
A confidence metric is computed from the distance to the maximum-entropy decision boundary ($s = 0.5$):
$$\text{Certainty} = 2 \cdot |s - 0.5|$$
- Scores near $0.0$ or $1.0$ have certainty near $100\%$.
- Scores near $0.5$ have certainty near $0\%$.
