# SafeDrive-AI 

> AI-Powered Motor Protection and Thermal Control System

---

## Version History

### v2.4 — *Latest Stable Release* 

> **Release date:** 2025  
> **Status:** Active development

####  Model Architecture — Redesigned

| Feature | v2.3 (Old) | v2.4 (New) |
|---|---|---|
| Backbone size | `7 → 32 → 16` | `7 → 64 → 32 → 16` |
| Normalization | None | `LayerNorm` after every layer |
| Skip connection | `state + x[:,1:2]` *(dimension mismatch)* | `skip_proj = nn.Linear(7,16)` *(correct projection)* |
| Head structure | Single `nn.Linear(16,1)` | `Linear(16,8) → ReLU → Linear(8,1)` |

**Critical bug fixed:** In v2.3, `state + x[:,1:2]` attempted to add a 1-dimensional slice to a 16-dimensional tensor, causing both a dimension mismatch and a physically meaningless operation. In v2.4, the `skip_proj` layer correctly projects the full input vector from 7 → 16 dimensions before the residual addition.

---

#### Training Data — Enriched

| Feature | v2.3 | v2.4 |
|---|---|---|
| Sample count | 54 | 3,000 |
| Vibration | Fixed (0.2 / 0.6) | `random.uniform(0.05, 0.95)` |
| Current | Fixed (0.3 / 0.7) | `random.uniform(0.15, 0.95)` |
| Sensor noise | None | Gaussian noise on all channels |
| Throttle cut threshold | 70°C | 68°C *(earlier intervention)* |
| Vibration → throttle effect | None | Extra cut applied when `vibration > 0.60` |
| Current → throttle effect | None | Slight extra cut when `current > 0.75` |

Training on 54 samples made overfitting inevitable. 3,000 diverse samples with realistic sensor noise were introduced to align the model with real-world sensor behavior.

---

#### ⚙️ Training Loop

| Feature | v2.3 | v2.4 |
|---|---|---|
| Epochs | 600 | 800 |
| Scheduler | None | `CosineAnnealingLR` (T_max=800) |
| Gradient clipping | None | `max_norm=1.0` |
| Model saving | None | Auto-saves to `motor_koruyucu.pth` |
| Model loading | None | Skips training if `.pth` exists |
| `ZORLA_EGIT` flag | None | Set to `True` to retrain from scratch |

---

#### PID Controller — Anti-Windup Added

In v2.3, `self.integral` could grow without bound, creating a windup risk.  
In v2.4, `np.clip` protection is applied with `integral_limit=8.0`.

```python
# v2.4
self.integral = float(np.clip(
    self.integral + error * self.dt,
    -self.integral_limit, self.integral_limit
))
```

---

#### Fail-Safe — Updated Thresholds

| Condition | v2.3 | v2.4 |
|---|---|---|
| Over-temperature | `T >= 100°C` | `T >= 98°C` |
| Excessive vibration | `vibration > 0.85` | `vibration > 0.90` |
| Low voltage | Not present | `voltage < 0.35` *(new)* |
| Mechanical fault threshold | `fault > 0.9` | `fault > 0.92` |

---

#### Motor Physics — Coefficients Recalibrated

```
dT/dt = α·P + γ·Vib - β·(T - T_ambient) + extra_load
```

| Coefficient | v2.3 | v2.4 | Notes |
|---|---|---|---|
| `alpha` (heating) | 0.60 | 0.55 | Realistic thermal load |
| `beta` (cooling) | 0.08 | 0.028 | Real motors cool slowly |
| `gamma` (vibration heat) | — | 1.40 | Vibration-induced heating added |

In v2.3, the combination of `alpha=0.6` and `beta=0.08` caused the motor to never cool down in simulation.

---

#### 📡 New: CAN Bus Simulation

v2.4 introduces a CAN Bus protocol layer. Sensor data is encoded and decoded in 8-byte frame format, aligned with OBD-II / SAE J1939 standards:

| CAN ID | Signal | Resolution |
|---|---|---|
| `0x0C8` | Engine RPM | 1 rpm |
| `0x130` | Coolant temperature | 1 °C |
| `0x1A0` | Throttle position % | 1 % |
| `0x3B0` | Battery voltage | 0.1 V |
| `0x4A0` | Vibration | 0.01 g |
| `0x5C0` | Motor current | 0.1 A |
| `0x6D0` | Vehicle speed | 1 km/h |

Pipeline: `encode → simulated CAN bus → decode → tensor → AI+PID decision`

---

#### New: Stress Test Scenarios

v2.4 runs 120-second simulations across four distinct stress scenarios:

- **Normal Drive** — stable baseline reference
- **Sudden Load Spike** — two separate load impulses (t=18–32s, t=55–68s)
- **Vibration Ramp** — sinusoidal vibration profile
- **Voltage Drop** — gradual battery degradation

---

#### New: Live Dashboard

Real-time panel based on `FuncAnimation` — 300ms update interval.

**Animation bugs from v2.3 (fixed):**
- Dependency on global variables → animation would freeze
- `fault_risk = random.uniform(0,10)` → fake data was being displayed
- `'power' scope` error → raised `NameError` at runtime

In v2.4, all state is held in the `dash` dictionary; each frame computes its own independent step with no global dependencies.

---

####  Baseline Comparison

Four control methods are benchmarked head-to-head on the sudden load scenario:

| Method | Description |
|---|---|
| **SafeDrive-AI** | NN + PID + Fail-Safe |
| Threshold | Fixed rule-based control (traditional) |
| PID Only | Classic PID with no fault awareness |
| NN Only | Raw NN output with no PID smoothing |

---

### v2.3 — *Previous Release*

- Multi-head architecture introduced (throttle cut + fault risk)
- PID + NN + Fail-Safe integration
- Stability test under sensor noise
- Decision curve visualization
- 54-sample training dataset *(expanded to 3,000 in v2.4)*

---

## Installation

```bash
pip install torch numpy matplotlib
python safedrive_ai.py
```

On first run, the model is trained and saved as `motor_koruyucu.pth`.  
Subsequent runs load the saved model directly (~30s saved per run).

To retrain from scratch:
```python
ZORLA_EGIT = True  # inside safedrive_ai.py
```

---

## Architecture Overview

```
Sensor Input (7 features)
        │
        ├─ state_net (7→64→32→16, LayerNorm+Tanh)
        │         │
        │   skip_proj (7→16)
        │         │
        └──── state (16)
                  │
        ┌─────────┴──────────┐
        │                    │
  control_head           fault_head
 (16→8→1, Sigmoid)    (16→8→1, Sigmoid)
        │                    │
  Throttle Cut [0-1]   Fault Risk [0-1]
        │
   PID Anti-Windup
        │
    Fail-Safe
        │
  Motor Power [0-1]
```
