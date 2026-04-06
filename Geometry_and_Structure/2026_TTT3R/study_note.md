# TTT3R Study Note — A Concept Mind-Map

> Paper: TTT3R: 3D Reconstruction as Test-Time Training (arXiv:2509.26645v4)
> Authors: Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, Anpei Chen
> Affiliations: Zhejiang University, Westlake University, Uni of Tübingen / Tübingen AI Center
> Venue: ICLR 2026

This note organizes every key concept in the paper as a mind-map.
Each concept is broken down into four facets:

- **Definition** — what it is, stated plainly
- **Properties** — its mathematical or behavioral characteristics
- **Application** — how the paper (or the field) uses it
- **Links** — connections to other concepts in this map

---

## 0. The Big Picture

```
         3D Reconstruction from Streaming Images
                         │
          "Need online, constant-memory inference"
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   Full Attention    RNN-based        Explicit Memory
   (VGGT, Fast3R)   (CUT3R)          (Point3R)
   O(t²) cost,      O(1) memory,     O(t) memory,
   offline only      forgetting       grows linearly
                     problem
                         │
         "Why does CUT3R forget? Can we fix it?"
                         │
                         ▼
         ┌───────────────────────────────┐
         │  Reinterpret CUT3R as TTT    │
         │                               │
         │  State S = fast weight        │
         │  Update = gradient descent    │
         │  β_t = learning rate          │
         │                               │
         │  CUT3R forces β_t = 1.0      │
         │  → always overwrites history  │
         │  → catastrophic forgetting    │
         └───────────────┬───────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │  TTT3R: Confidence-guided β_t │
         │                               │
         │  β_t = σ(Σ_m Q_{S} K_{X}^⊤)  │
         │                               │
         │  High alignment → large β_t   │
         │  Low alignment → small β_t    │
         │  → selective memory update    │
         └───────────────┬───────────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        Training-    2× pose    6 GB GPU
        free,        accuracy   memory,
        plug-and-    on long    20 FPS
        play         sequences
```

**The paper's story in one paragraph**: Modern recurrent 3D reconstruction models like CUT3R achieve constant-memory online inference but degrade on sequences longer than their training context due to catastrophic forgetting. TTT3R reframes CUT3R's state update as a Test-Time Training process — the recurrent state is a fast weight updated via gradient descent, where the learning rate $\beta_t$ controls memory plasticity. CUT3R's forgetting problem stems from an implicit $\beta_t = 1.0$ that always fully overwrites historical information with new observations. TTT3R derives a closed-form, confidence-guided $\beta_t$ from the cross-attention alignment between the memory state and incoming observations: high alignment confidence means the new observation is informative and worth writing to memory; low confidence means the update should be suppressed. This training-free intervention achieves a 2× improvement in global pose estimation over CUT3R on long sequences while preserving its 20 FPS speed and 6 GB memory footprint.

---

## 1. Sequence Modeling for Pointmap Regression

### Definition

A unified formulation that interprets all pointmap-oriented 3D reconstruction methods as sequence models. For each incoming image $\mathbf{I}_t \in \mathbb{R}^{W \times H \times 3}$, the model estimates the camera pose $\mathbf{T}_t \in \mathbb{R}^{3 \times 4}$, camera intrinsic $\mathbf{C}_t \in \mathbb{R}^{3 \times 3}$, and canonical point cloud $\mathbf{P}_t \in \mathbb{R}^{W \times H \times 3}$:

$$\mathbf{X}_t = \text{Tokenize}(\mathbf{I}_t), \quad \mathbf{S}_t = \text{Update}(\mathbf{S}_{t-1}, \mathbf{X}_t), \quad \mathbf{Y}_t = \text{Read}(\mathbf{S}_t, \mathbf{X}_t), \quad \mathbf{P}_t = \text{De-tokenize}(\mathbf{Y}_t) \tag{Eq.1}$$

where the input image $\mathbf{I}_t$ is patchified into image tokens $\mathbf{X}_t \in \mathbb{R}^{(h \times w) \times c}$ via an image tokenizer (e.g., DINO, CroCo). The tokens update the state $\mathbf{S}_{t-1}$ into $\mathbf{S}_t$, then the model reads from the updated state to produce output tokens $\mathbf{Y}_t$. Dense prediction de-tokenizers (pixel shuffle, DPT head) extract pointmaps. Camera pose $\mathbf{T}_t$ and intrinsic $\mathbf{C}_t$ are solved from pointmaps via PnP/Weiszfeld or regressed from tokens.

### Properties

- **Unifying perspective**: the Update and Read operations define the core distinction among methods — full-attention, RNN-based, and TTT-based approaches differ only in how these operations work.
- **Tokenizer-agnostic**: the framework applies regardless of the specific image tokenizer (DINO, CroCo, etc.).
- **State as memory**: $\mathbf{S}_t$ encodes all historical context — its capacity and update mechanism determine the model's ability to generalize over long sequences.

### Application

- This formulation enables comparing DUSt3R, VGGT, CUT3R, StreamVGGT, and TTT3R under one lens.
- The paper uses this framework to identify that the state update rule is the key differentiator for length generalization.

### Links

- → **Full Attention Methods**: a specific instantiation of Update/Read with growing state
- → **RNN-based Methods (CUT3R)**: a specific instantiation with fixed-size state
- → **CUT3R as TTT**: reinterprets CUT3R's update through the TTT framework
- → **Confidence-Guided State Update**: the proposed improvement to the Update operation

---

## 2. Full Attention vs. RNN-based Methods

### Definition

Two categories of sequence modeling for 3D reconstruction, distinguished by their Update and Read operations.

**Full attention** (VGGT, Fast3R): state grows by appending KV pairs.

$$\text{Update}(\mathbf{S}_{t-1}, \mathbf{X}_t) = \mathbf{S}_{t-1}.\text{append}(\mathbf{K}_{\mathbf{X}_t}, \mathbf{V}_{\mathbf{X}_t})$$
$$\text{Read}(\mathbf{S}_t, \mathbf{X}_t) = \mathbf{X}_t + \text{softmax}(\mathbf{Q}_{\mathbf{X}_t} \mathbf{K}_{\mathbf{S}_t}^\top) \mathbf{V}_{\mathbf{S}_t} \tag{Eq.2}$$

**RNN-based** (CUT3R): fixed-length state updated via one-to-one cross-attention.

$$\text{Update}(\mathbf{S}_{t-1}, \mathbf{X}_t) = \mathbf{S}_{t-1} + \text{softmax}(\mathbf{Q}_{\mathbf{S}_{t-1}} \mathbf{K}_{\mathbf{X}_t}^\top) \mathbf{V}_{\mathbf{X}_t} \tag{Eq.3}$$

where state $\mathbf{S}_{t-1} \in \mathbb{R}^{n \times c}$ consists of $n$ tokens with channel dimension $c$.

### Properties

- **Full attention**: $O(t^2)$ compute, $O(t)$ memory — all output tokens $\mathbf{Y}_1, \ldots, \mathbf{Y}_t$ must be recomputed when $\mathbf{X}_t$ arrives. Lossless but unscalable.
- **Causal attention** (StreamVGGT): $O(t)$ compute but $O(t)$ memory — still grows with sequence length.
- **RNN-based**: $O(1)$ compute and memory per step. The state is a fixed-size bottleneck that compresses all history. Scalable but suffers from the forgetting problem — performance degrades as sequence length increases beyond the training context.
- **Memory comparison** (Fig. 2): at 1000 frames, VGGT uses ~48 GB, Point3R ~35 GB, CUT3R/TTT3R ~6 GB.

### Application

- CUT3R is the primary baseline: it achieves real-time inference with constant memory but drifts on long sequences.
- VGGT serves as an offline upper bound — it preserves full history but is impractical for streaming.

### Links

- → **Sequence Modeling for Pointmap Regression**: both are instantiations of the generic framework (Eq. 1)
- → **CUT3R as TTT**: the RNN update (Eq. 3) is reinterpreted as a TTT mechanism
- → **The Forgetting Problem**: RNN-based methods suffer from forgetting due to fixed-size state

---

## 3. CUT3R Reinterpreted as Test-Time Training

### Definition

The paper's key theoretical insight: CUT3R's recurrent state update (Eq. 3) can be rewritten as a TTT gradient descent step. TTT treats the state $\mathbf{S}_{t-1} \in \mathbb{R}^{n \times c}$ as a *fast weight* updated via gradient descent:

$$\text{Update}(\mathbf{S}_{t-1}, \mathbf{X}_t) = \mathbf{S}_{t-1} - \beta_t \nabla(\mathbf{S}_{t-1}, \mathbf{X}_t) \tag{Eq.4}$$

where $\nabla(\mathbf{S}_{t-1}, \mathbf{X}_t)$ is a learned gradient function and $\beta_t$ is the learning rate.

For linear TTT (DeltaNet), the reconstruction error $\|\mathbf{S}_{t-1}\mathbf{K}_{\mathbf{X}_t} - \mathbf{V}_{\mathbf{X}_t}\|^2$ yields an analytical gradient:

$$\text{Update}(\mathbf{S}_{t-1}, \mathbf{X}_t) = \mathbf{S}_{t-1} - \overbrace{\beta \underbrace{(\mathbf{S}_{t-1}\mathbf{K}_{\mathbf{X}_t} - \mathbf{V}_{\mathbf{X}_t})}_{}\mathbf{K}_{\mathbf{X}_t}^\top}^{\nabla(\mathbf{S}_{t-1}, \mathbf{X}_t)} \tag{Eq.5}$$

### Derivation: Eq.3 → Eq.6 (CUT3R = TTT with β=1.0)

**Step 1.** Reformulate CUT3R's Eq. 3 using the TTT formulation of Eq. 4:

$$\underbrace{\mathbf{S}_{t-1} + \text{softmax}(\mathbf{Q}_{\mathbf{S}_{t-1}} \mathbf{K}_{\mathbf{X}_t}^\top) \mathbf{V}_{\mathbf{X}_t}}_{\text{CUT3R (Eq.3)}} = \mathbf{S}_{t-1} - \underbrace{\beta_t}_{1.0} \underbrace{\overbrace{-\text{softmax}(\mathbf{Q}_{\mathbf{S}_{t-1}} \mathbf{K}_{\mathbf{X}_t}^\top) \mathbf{V}_{\mathbf{X}_t}}^{\nabla(\mathbf{S}_{t-1}, \mathbf{X}_t)}}_{\text{TTT gradient}} \tag{Eq.6}$$

**Step 2.** Identify the components:
- The gradient $\nabla(\mathbf{S}_{t-1}, \mathbf{X}_t)$ is a linear combination of observation values $\mathbf{V}_{\mathbf{X}_t}$, weighted by the softmax alignment scores $\text{softmax}(\mathbf{Q}_{\mathbf{S}_{t-1}} \mathbf{K}_{\mathbf{X}_t}^\top)$.
- The key $\mathbf{K}_{\mathbf{X}_t}$ determines *where to write* in the state.
- The value $\mathbf{V}_{\mathbf{X}_t}$ specifies *what to write*.
- The learning rate $\beta_t$ controls the *intensity* of the write.

**Step 3.** Observe the critical limitation: in CUT3R, $\beta_t = 1.0$ always. The softmax normalization forces the update weights to sum to 1.0 along the observation-token dimension, meaning the model always fully incorporates new information. This prioritizes new observations over historical state, causing catastrophic forgetting.

### Properties

- **Associative memory interpretation**: the fast weight $\mathbf{S}$ functions as a dynamic associative memory. The gradient-based update writes new key-value associations; the learning rate controls plasticity.
- **Conceptual mapping**: key = *where to write*, value = *what to write*, learning rate = *how much to write*.
- **Structural limitation of CUT3R**: $\beta_t = 1.0$ means CUT3R has no mechanism to selectively retain vs. overwrite — every new observation overwrites with maximum intensity.

### Application

- This reinterpretation motivates the confidence-guided learning rate: since the forgetting problem stems from a fixed $\beta_t = 1.0$, making $\beta_t$ adaptive and data-dependent should resolve it.

### Links

- → **RNN-based Methods (CUT3R)**: CUT3R is reinterpreted through the TTT lens
- → **The Forgetting Problem**: directly explained by the $\beta_t = 1.0$ constraint
- → **Confidence-Guided State Update**: the proposed fix that makes $\beta_t$ adaptive
- → **Sequence Modeling for Pointmap Regression**: CUT3R's Update operation rewritten

---

## 4. The Forgetting Problem

### Definition

The phenomenon where RNN-based 3D reconstruction models (CUT3R) degrade in performance as the input sequence length exceeds the training context — camera poses drift, depth estimates become inconsistent, and 3D geometry breaks down.

### Properties

- **Root cause (TTT perspective)**: CUT3R uses an implicit learning rate $\beta_t = 1.0$ (Eq. 6) that always fully integrates new observations. The softmax normalization forces attention weights to sum to 1.0, meaning the model has no mechanism to suppress low-quality updates or preserve historical information.
- **State overfitting**: the recurrence drives the state into distributions not encountered during training — the "unexplored states hypothesis" (Ruiz & Gu, 2025). Models trained on 64-frame sequences produce states at frame 200+ that are out-of-distribution.
- **Low-quality updates**: not all observations carry equal information. Textureless regions, repeated patterns, and occluded areas produce low-confidence updates that pollute the state (Fig. 5: textureless walls get $\beta_t \approx 0$ in TTT3R, but CUT3R updates them with $\beta_t = 1.0$).
- **Manifestation**: drifted camera poses, broken geometry, severe distortions, ghosting artifacts in long-sequence 3D reconstruction (Fig. 10).

### Application

- CUT3R achieves competitive results on short sequences (≤64 frames) but degrades on long sequences (Fig. 7: ATE increases from ~0.05 to ~0.8 on ScanNet at 1000 frames).
- The forgetting problem is the primary motivation for TTT3R.

### Links

- → **CUT3R as TTT**: the forgetting problem is explained by the $\beta_t = 1.0$ limitation
- → **Confidence-Guided State Update**: the proposed solution
- → **State Reset Mechanism**: an orthogonal mitigation strategy that periodically clears the state
- → **Full Attention Methods**: preserve full history and thus avoid forgetting (but are unscalable)

---

## 5. Confidence-Guided State Update Rule (TTT3R)

### Definition

The core contribution of TTT3R: a closed-form, per-token learning rate $\beta_t$ derived from the alignment confidence between the memory state and incoming observations. This learning rate replaces CUT3R's implicit $\beta_t = 1.0$, enabling selective memory updates that balance retaining historical information with adapting to new observations.

**Per-token learning rate**:

$$\beta_t = \sigma\Big(\sum_m \mathbf{Q}_{\mathbf{S}_{t-1}} \mathbf{K}_{\mathbf{X}_t}^\top\Big) \tag{Eq.7}$$

where $\sum_m$ denotes a normalized mean over the spatial dimension $m = \{1, \ldots, h\} \times \{1, \ldots, w\}$ of image tokens, and $\sigma(\cdot)$ is the sigmoid function.

**Full TTT3R closed-form state update**:

$$\text{Update}(\mathbf{S}_{t-1}, \mathbf{X}_t) = \mathbf{S}_{t-1} - \beta_t \, \nabla(\mathbf{S}_{t-1}, \mathbf{X}_t) \tag{Eq.8}$$

where $\beta_t \in \mathbb{R}^{n \times 1}$ is per-token (one learning rate per state token) and $\nabla(\mathbf{S}_{t-1}, \mathbf{X}_t) = -\text{softmax}(\mathbf{Q}_{\mathbf{S}_{t-1}} \mathbf{K}_{\mathbf{X}_t}^\top) \mathbf{V}_{\mathbf{X}_t}$.

### Properties

- **Confidence as alignment**: $\mathbf{Q}_{\mathbf{S}_{t-1}} \mathbf{K}_{\mathbf{X}_t}^\top \in \mathbb{R}^{n \times (h \times w)}$ are the cross-attention scores between state queries and observation keys. High scores mean strong alignment → the observation is informative and worth writing. Low scores mean poor alignment → the update should be suppressed.
- **Per-token granularity**: $\beta_t \in \mathbb{R}^{n \times 1}$ — each of the $n$ state tokens gets its own learning rate. This allows different parts of the scene representation to be updated at different rates.
- **Closed-form derivation**: $\beta_t$ is derived analytically from the existing cross-attention mechanism — no additional learnable parameters, no hyper-network, no training required.
- **Training-free / plug-and-play**: TTT3R modifies only the state update rule of CUT3R at inference time. No fine-tuning, no additional parameters, no extra computational cost. The slow weights (frozen model parameters) remain unchanged.
- **Zero additional cost**: the cross-attention scores $\mathbf{Q}_{\mathbf{S}_{t-1}} \mathbf{K}_{\mathbf{X}_t}^\top$ are already computed in CUT3R's update — TTT3R simply reuses them to derive $\beta_t$.
- **Sigmoid gating**: the sigmoid $\sigma(\cdot)$ maps the averaged alignment score to $[0, 1]$, acting as a soft gate that interpolates between full retention ($\beta_t \approx 0$, keep old state) and full update ($\beta_t \approx 1$, write new information).
- **Visualization** (Fig. 5): informative regions (well-textured objects) get high $\beta_t \approx 1.0$; uninformative regions (textureless walls, sky) get low $\beta_t \approx 0$. CUT3R assigns $\beta_t = 1.0$ uniformly to all regions.

### Application

- Applied to CUT3R at inference time with zero modification to the model weights.
- Achieves 2× improvement in camera pose estimation accuracy over CUT3R on long sequences (Fig. 7).
- Preserves CUT3R's inference speed (~20 FPS) and memory footprint (~6 GB).

### Links

- → **CUT3R as TTT**: TTT3R is the direct application of the TTT reinterpretation with adaptive $\beta_t$
- → **The Forgetting Problem**: the confidence-guided learning rate directly addresses forgetting
- → **Per-Token Learning Rate β_t**: detailed analysis of the learning rate design space
- → **State Reset Mechanism**: an orthogonal enhancement that can be combined with TTT3R
- → **Experimental Results**: the confidence-guided update produces all experimental improvements
- → **Per-Token Learning Rate β_t**: $\beta_t$ implements this confidence-guided update

---

## 6. Per-Token Learning Rate β_t

### Definition

The learning rate $\beta_t$ in the TTT framework controls memory plasticity — how strongly new observations overwrite the existing state. Different approaches model $\beta_t$ with varying granularity:

### Properties

| Type | Model | $\beta_t$ | Granularity | Training |
|---|---|---|---|---|
| **ScalarLR** | RetNet | $\beta \in \mathbb{R}^1$ | Single scalar for all tokens | Learnable |
| **ConditionLR** | DeltaNet, TTT, Mamba-2 | $\beta_t = \sigma(\ell_\beta(\mathbf{X}_t)) \in \mathbb{R}^1$ | Input-dependent scalar | Learnable ($\ell_\beta$) |
| **TokenLR** | Gated Linear Attention | $\beta_t = \sigma(\ell_\beta(\mathbf{X}_t)) \in \mathbb{R}^{n \times 1}$ | Per-token, input-conditioned | Learnable ($\ell_\beta$) |
| **TTT3R (ours)** | — | $\beta_t = \sigma(\sum_m \mathbf{Q}_{\mathbf{S}_{t-1}} \mathbf{K}_{\mathbf{X}_t}^\top) \in \mathbb{R}^{n \times 1}$ | Per-token, alignment-based | **Training-free** |

- **Key difference**: prior approaches (ScalarLR, ConditionLR, TokenLR) all require learnable parameters — a scalar, a hyper-network $\ell_\beta$, or a gating module — that must be trained on long sequences. TTT3R derives $\beta_t$ from the existing cross-attention mechanism with no additional parameters.
- **Ablation** (Table 1): TTT3R outperforms all learnable gating mechanisms (ScalarLR, ConditionLR, TokenLR) in both camera pose and video depth estimation, despite not requiring any training.
- **Finetuning** (Table 1, Appendix A.2): finetuning TTT3R's update rule during training (TTT3R + Finetune) yields marginal improvement in pose but degrades depth — the closed-form rule is sufficient.

**Table 1: Comparison with learnable gating mechanisms** (1000f TUM-D pose, 500f KITTI depth)

| Method | ATE↓ | RPE rot↓ | Abs Rel↓ | $\delta$ <1.25↑ |
|---|---|---|---|---|
| CUT3R | 0.173 | 0.494 | 0.152 | 80.2 |
| CUT3R + ScalarLR | 0.165 | 0.502 | 0.151 | 80.6 |
| CUT3R + ConditionLR | 0.166 | 0.509 | 0.149 | 81.0 |
| CUT3R + TokenLR | 0.154 | 0.497 | 0.148 | 81.5 |
| **TTT3R** | **0.106** | **0.431** | **0.131** | **86.9** |
| TTT3R + Finetune | 0.091 | 0.434 | 0.133 | 86.3 |

### Application

- TTT3R's per-token $\beta_t$ is directly instantiated as Eq.7-8 in the confidence-guided update. No learnable parameters are needed — the alignment scores are reused from CUT3R's existing cross-attention.
- Ablation (Table 1): TTT3R's training-free $\beta_t$ outperforms all three learnable variants (ScalarLR, ConditionLR, TokenLR) that require additional training.

### Links

- → **Confidence-Guided State Update**: $\beta_t$ is the mechanism that implements the confidence-guided update
- → **CUT3R as TTT**: the learning rate is the key parameter identified by the TTT reinterpretation
- → **The Forgetting Problem**: adaptive $\beta_t$ directly addresses the fixed $\beta_t = 1.0$ limitation
- → **Experimental Results**: the per-token granularity contributes to TTT3R's consistent improvements

---

## 7. State Reset Mechanism

### Definition

An optional extension where the recurrent state $\mathbf{S}_t$ is periodically reset to its initial value $\mathbf{S}_0$ every $n$ frames. The resulting chunks are then globally aligned using metric camera poses, producing a plug-and-play solution for sequences exceeding ~1000 frames.

### Properties

- **Motivation**: TTT3R mitigates but does not fully resolve state forgetting. Beyond ~1000 frames, the state enters distributions never seen during training (the "unexplored states hypothesis"), causing failures (Fig. 14, middle).
- **Mechanism**: reset state every $n = 100$ frames (optimal, Table 2), then align chunks using global metric poses.
- **Orthogonal to TTT3R**: State Reset can be applied to both CUT3R and TTT3R. The combination (TTT3R + Reset) achieves the best overall performance (Table 5).
- **Not used in main experiments**: for simplicity, State Reset is only used in visualization demos (Fig. 1, Fig. 15) and the supplementary comparison (Fig. 13). All main-paper quantitative results use TTT3R without State Reset.

**Table 5: TTT3R vs. non-TTT baselines** (1000f TUM-D pose, 500f KITTI depth)

| Method | ATE↓ | RPE rot↓ | Abs Rel↓ | $\delta$ <1.25↑ |
|---|---|---|---|---|
| CUT3R | 0.173 | 0.494 | 0.152 | 80.2 |
| CUT3R + Reset | 0.126 | 0.403 | 0.128 | 84.9 |
| CUT3R + EMA | 0.164 | 0.525 | 0.150 | 80.7 |
| CUT3R + BurnIn | 0.144 | 3.093 | 0.151 | 80.2 |
| **TTT3R** | **0.106** | **0.431** | **0.131** | **86.9** |
| **TTT3R + Reset** | **0.093** | **0.375** | **0.115** | **88.5** |

### Application

- Best reset period: $n = 100$ frames (Table 2).
- TTT3R + Reset enables robust 3D reconstruction beyond 1000 frames (Fig. 15).
- CUT3R + Reset also improves CUT3R, but TTT3R + Reset still outperforms it.

### Links

- → **The Forgetting Problem**: State Reset is a complementary mitigation for extreme-length sequences
- → **Confidence-Guided State Update**: orthogonal mechanism — TTT3R reduces forgetting rate, Reset bounds accumulation
- → **Experimental Results**: main results use TTT3R only; State Reset results are supplementary

---

## 8. Experimental Results

### 8.1 Camera Pose Estimation (Long Sequence)

**Setup**: TUM-Dynamics and ScanNet, 50–1000 frames. Metric: Absolute Translation Error (ATE) after Sim(3) alignment.

**Key results** (Fig. 7):
- **ScanNet**: CUT3R ATE increases from ~0.05 (50f) to ~0.8 (1000f). TTT3R ATE stays below ~0.35 at 1000f — a **~2× improvement**.
- **TUM-D**: CUT3R reaches ~0.15 ATE at 1000f; TTT3R stays at ~0.08 — close to **2× improvement**.
- VGGT (offline upper bound) OOMs beyond ~200 frames on ScanNet.
- Point3R improves over CUT3R but OOMs beyond ~700 frames.
- StreamVGGT OOMs beyond ~200 frames.
- TTT3R achieves the best accuracy among all constant-memory methods while maintaining CUT3R's real-time efficiency.

**Short-sequence evaluation** (Table 6, Appendix C.3):

| Method | Online | Sintel ATE↓ | Sintel RPE trans↓ | Sintel RPE rot↓ | TUM-D ATE↓ | TUM-D RPE trans↓ | TUM-D RPE rot↓ | ScanNet ATE↓ | ScanNet RPE trans↓ | ScanNet RPE rot↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| VGGT | ✗ | 0.172 | 0.062 | 0.471 | 0.012 | 0.010 | 0.310 | 0.035 | 0.015 | 0.377 |
| CUT3R | ✓ | 0.213 | 0.046 | 0.621 | 0.056 | 0.015 | 0.473 | 0.096 | 0.022 | 0.600 |
| Point3R | ✓ | 0.351 | 0.128 | 1.822 | 0.075 | 0.029 | 0.642 | 0.106 | 0.035 | 1.946 |
| StreamVGGT | ✓ | 0.251 | 0.149 | 1.894 | 0.061 | 0.033 | 3.209 | 0.161 | 0.057 | 3.647 |
| STream3R | ✓ | 0.213 | 0.076 | 0.868 | **0.026** | **0.013** | **0.330** | 0.052 | **0.021** | 0.850 |
| **TTT3R** | ✓ | **0.201** | **0.063** | **0.617** | 0.028 | 0.012 | 0.379 | **0.064** | 0.021 | **0.592** |

- TTT3R achieves the best overall online performance, particularly on TUM-D and ScanNet.
- On short sequences, gains over CUT3R are minimal (both work well within training context).

---

### 8.2 Video Depth Estimation

**Setup**: KITTI and Bonn datasets, 50–500 frames. Metrics: Abs Rel↓, $\delta < 1.25$↑.

**Key results** (Fig. 8):
- **Bonn** (scale-invariant relative depth): TTT3R consistently achieves the best Abs Rel among online methods across all sequence lengths, outperforming CUT3R and Point3R.
- **KITTI** (metric depth): TTT3R outperforms CUT3R and Point3R at all sequence lengths. Point3R achieves strong accuracy on short sequences (≤300f) but degrades on longer ones due to forgetting-induced metric-scale errors.
- VGGT and StreamVGGT OOM beyond ~150 frames.

**Short-sequence evaluation** (Table 7, Appendix C.4): TTT3R achieves SOTA or competitive performance among online methods on KITTI for both metric and scale-invariant depth, and ranks first or second on Sintel and Bonn.

---

### 8.3 3D Reconstruction

**Setup**: 7-Scenes dataset, 50–400 frames. Metrics: Chamfer Distance↓, Normal Consistency↑.

**Key results** (Fig. 9):
- CUT3R: Chamfer Distance degrades sharply from ~0.03 (50f) to ~0.12 (400f).
- TTT3R: stays around ~0.03–0.04 across all sequence lengths.
- TTT3R achieves results comparable to VGGT (offline) while operating online in real-time with only 6 GB GPU memory.
- Qualitative comparison (Fig. 10): CUT3R produces drifted poses, broken geometry, and ghosting; TTT3R maintains consistent reconstructions.

**Efficiency** (Fig. 6):
- TTT3R runs at ~20 FPS, same as CUT3R.
- Peak GPU memory: ~6 GB, constant regardless of sequence length.
- VGGT: ~3 FPS at 100 frames, OOM at ~200 frames.
- Point3R: slower and OOMs at ~700 frames.

### Links

- → **Confidence-Guided State Update**: all improvements stem from the modified update rule
- → **The Forgetting Problem**: results quantify how much the forgetting problem hurts CUT3R
- → **State Reset Mechanism**: supplementary results show TTT3R + Reset further improves very long sequences

---

## 9. Ablation Studies

### Non-TTT Baselines (Table 5, Appendix A.3)

Three alternative mitigation strategies for the forgetting problem, all applied to CUT3R:

**CUT3R + Reset**: periodic state reset every $n$ frames, globally aligned with metric poses. Best $n = 100$ (Table 2). Effective for pose estimation but does not improve depth.

**CUT3R + EMA**: Exponential Moving Average shrinkage toward initial state: $\mathbf{S}_t = (1-\alpha)\mathbf{S}_{t-1} + \alpha\mathbf{S}_0$. Best $\alpha = 0.001$ (Table 3). Modest improvement.

**CUT3R + BurnIn**: update state only on keyframes (every $n = 100$ frames), leaving intermediate states unchanged. Good for pose but catastrophic for RPE rot (3.093 vs 0.494).

TTT3R outperforms all three strategies (Table 5), and the combination TTT3R + Reset achieves the best results.

### Finetuning Analysis (Appendix A.2)

Finetuning TTT3R's update rule during training (using CUT3R's training data, 4–64 frame sequences):
- Marginal benefit in pose: ATE 0.106 → 0.091 (Table 1).
- Degrades depth: Abs Rel 0.131 → 0.133.
- The closed-form rule is already effective without training — finetuning adds minimal value.

### Links

- → **Confidence-Guided State Update**: ablations validate the TTT3R update rule against alternatives
- → **State Reset Mechanism**: Reset is the strongest non-TTT baseline and combines well with TTT3R

---

## 10. What Existed Before and What This Paper Changes

### 10.1 Prior Approaches and Their Limitations

**Full-attention offline methods** (VGGT, Fast3R) process all frames simultaneously through global self-attention, achieving the highest reconstruction quality. Their $O(t^2)$ compute and $O(t)$ memory make them impractical for streaming: VGGT exhausts 48 GB GPU memory at ~200 frames on ScanNet. They also require re-running inference on all frames whenever a new frame arrives.

**RNN-based online methods** (CUT3R) compress history into a fixed-size state $\mathbf{S} \in \mathbb{R}^{n \times c}$, enabling $O(1)$ per-frame compute and constant memory. CUT3R runs at 20 FPS with 6 GB memory. The state acts as an implicit memory, but its capacity is bounded by training context (64 frames). Beyond this, CUT3R suffers catastrophic forgetting — camera poses drift, geometry breaks down.

**Explicit memory methods** (Point3R) extend CUT3R by anchoring history tokens to reconstructed 3D point positions, creating a growing spatial memory cache. This mitigates forgetting but causes memory to grow linearly ($O(t)$), hitting OOM at ~700 frames. Inference speed also degrades.

**KV-cache methods** (StreamVGGT) cache historical keys and values in a causal transformer, enabling incremental processing. However, the KV-cache grows at $O(t)$ — sharing the same scalability limitation as full attention.

### 10.2 What This Paper Contributes

**Contribution 1: TTT-based framework for 3D reconstruction.** A general sequence modeling framework that reinterprets recurrent 3D reconstruction models through the lens of Test-Time Training, identifying the state update rule as the key design axis for length generalization.

**Contribution 2: CUT3R ↔ TTT equivalence.** The paper shows that CUT3R's cross-attention update is mathematically equivalent to a TTT gradient descent step with fixed learning rate $\beta_t = 1.0$, explaining the forgetting problem as a consequence of this constraint.

**Contribution 3: Confidence-guided state update.** A training-free, closed-form learning rate $\beta_t = \sigma(\sum_m \mathbf{Q}_{\mathbf{S}_{t-1}} \mathbf{K}_{\mathbf{X}_t}^\top)$ derived from cross-attention alignment confidence, requiring no additional parameters, training, or computational cost.

### 10.3 Side-by-Side: Prior Art vs. This Paper

| Dimension | VGGT (offline) | CUT3R | Point3R | StreamVGGT | **TTT3R** |
|---|---|---|---|---|---|
| Processing | Offline, all frames | Online, per-frame | Online, per-frame | Online, causal | **Online, per-frame** |
| Compute | $O(t^2)$ | $O(1)$ | $O(t)$ | $O(t)$ | **$O(1)$** |
| Memory | $O(t)$, ~48GB@1k | $O(1)$, ~6GB | $O(t)$, OOM@700 | $O(t)$, OOM@200 | **$O(1)$, ~6GB** |
| Speed | ~3 FPS | ~20 FPS | ~12 FPS | ~5 FPS | **~20 FPS** |
| Long-seq accuracy | Best (if fits) | Degrades | Moderate | OOM | **2× better than CUT3R** |
| Additional training | N/A | N/A | Finetuned | Finetuned | **None (plug-and-play)** |

| Dimension | CUT3R (closest) | **TTT3R** |
|---|---|---|
| State update | $\mathbf{S} + \text{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ | $\mathbf{S} - \beta_t \cdot \text{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ |
| Learning rate $\beta_t$ | Implicit 1.0 (fixed) | Adaptive, per-token, confidence-guided |
| Forgetting | Catastrophic on long sequences | Mitigated (2× better pose) |
| Additional cost | — | Zero |
| Training required | — | None |

### 10.4 The Core Shift in Thinking

Previous work treated the forgetting problem in recurrent 3D reconstruction as a memory capacity issue — if the fixed-size state cannot hold enough information, either grow the memory (Point3R) or re-process everything (VGGT). Both approaches sacrifice the scalability that makes recurrent models attractive in the first place.

TTT3R reframes the problem: forgetting is not about capacity, it is about *update control*. CUT3R's state has enough capacity to represent long sequences, but its update rule ($\beta_t = 1.0$) forces every new observation to overwrite history at maximum intensity. A textureless wall overwrites a detailed architectural feature with the same urgency. The fix is not more memory — it is *smarter updates*.

By deriving the learning rate from the alignment confidence already computed in CUT3R's cross-attention, TTT3R achieves this smarter update at zero cost. The result is counterintuitive: a training-free, zero-overhead modification to CUT3R's inference code produces a 2× improvement in long-sequence pose estimation. The insight is that the recurrent model already has the information needed to control its own memory — it just was not using it.

---

## 11. Quick Reference Card

| # | Insight | Design Choice | Evidence |
|---|---|---|---|
| 1 | CUT3R is a TTT mechanism with fixed β_t = 1.0 | Reinterpret RNN update as gradient descent on fast weights | Eq. 6: mathematical equivalence |
| 2 | Forgetting stems from lack of update control, not lack of capacity | Introduce adaptive per-token β_t | Table 5: TTT3R > all non-TTT baselines |
| 3 | Cross-attention alignment is a natural confidence signal | Derive β_t from softmax alignment scores | Fig. 5: high β on textured regions, low on walls |
| 4 | Training-free modification suffices | No additional parameters, training, or compute | Table 1: outperforms all learnable gating mechanisms |
| 5 | Selective updates preserve memory over long sequences | σ(·) gating suppresses low-quality updates | Fig. 7: 2× better pose at 1000 frames |
| 6 | State Reset is orthogonal and complementary | Reset + TTT3R achieves best overall results | Table 5: TTT3R + Reset = 0.093 ATE vs 0.106 |
| 7 | Per-token granularity matters | β_t ∈ ℝ^{n×1} vs scalar or input-conditioned | Table 1: TTT3R > TokenLR > ConditionLR > ScalarLR |
| 8 | Online methods can approach offline quality | TTT3R matches VGGT on 7-Scenes with 6 GB | Fig. 9: comparable Chamfer Distance |

---

## 12. Open Questions

1. **State forgetting beyond 1000 frames**: TTT3R mitigates but does not fully resolve forgetting. State Reset provides a workaround, but a fundamental solution requires better length generalization in recurrent models.

2. **Gap with offline methods**: VGGT (with full attention) still outperforms TTT3R in reconstruction accuracy on short sequences. The fixed-size state is an information bottleneck.

3. **Design space of TTT for 3D reconstruction**: the TTT perspective opens a large design space — different fast-weight architectures (Titans, Mamba-2), different gradient objectives, different learning rate schedules — that remains unexplored.

4. **Training with TTT3R update rule**: finetuning with the confidence-guided rule yields mixed results (better pose, worse depth). Understanding this trade-off could unlock further gains.

---

## 13. Concept Dependency Graph

```
┌────────────────────────────────────────────────────────┐
│               PROBLEM LAYER                            │
│                                                        │
│  ┌─────────────────┐     ┌──────────────────────┐     │
│  │ Quadratic cost  │     │ Forgetting problem   │     │
│  │ of full attn    │     │ of RNN state         │     │
│  └────────┬────────┘     └──────────┬───────────┘     │
│           │                         │                  │
└───────────┼─────────────────────────┼──────────────────┘
            │                         │
            ▼                         ▼
┌───────────────────┐   ┌──────────────────────────────┐
│ Full Attention    │   │ RNN-based Methods (CUT3R)    │
│ vs. RNN Methods   │   │ Eq.3: fixed-size state       │
│ Eqs.2,3           │   │                              │
└───────────────────┘   └──────────┬───────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ CUT3R as TTT (Eq.6)          │
                    │ State = fast weight           │
                    │ β_t = 1.0 → forgetting       │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
          ┌──────────────┐ ┌────────────┐ ┌────────────────┐
          │ Confidence-  │ │ Per-Token  │ │ State Reset    │
          │ Guided β_t   │ │ Learning   │ │ Mechanism      │
          │ Eqs.7,8      │ │ Rate       │ │ (orthogonal)   │
          └──────┬───────┘ └─────┬──────┘ └────────┬───────┘
                 │               │                  │
                 └───────────────┼──────────────────┘
                                 ▼
                          TTT3R Model
                                 │
                   ┌─────────────┼─────────────┐
                   ▼             ▼             ▼
             Camera Pose    Video Depth    3D Recon.
             (2× better)   (best online)  (≈VGGT)
```

---

## 14. Key Equations

| Eq. | Name | Formula | Section |
|---|---|---|---|
| 1 | Generic sequence modeling | Tokenize → Update → Read → De-tokenize | §1 |
| 2 | Full attention Update/Read | $\mathbf{S}_{t-1}.\text{append}(\mathbf{K},\mathbf{V})$; $\mathbf{X}_t + \text{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ | §2 |
| 3 | RNN Update (CUT3R) | $\mathbf{S}_{t-1} + \text{softmax}(\mathbf{Q}_{\mathbf{S}_{t-1}}\mathbf{K}_{\mathbf{X}_t}^\top)\mathbf{V}_{\mathbf{X}_t}$ | §2 |
| 4 | TTT Update (general) | $\mathbf{S}_{t-1} - \beta_t \nabla(\mathbf{S}_{t-1}, \mathbf{X}_t)$ | §3 |
| 5 | Linear TTT Update (DeltaNet) | $\mathbf{S}_{t-1} - \beta(\mathbf{S}_{t-1}\mathbf{K}_{\mathbf{X}_t} - \mathbf{V}_{\mathbf{X}_t})\mathbf{K}_{\mathbf{X}_t}^\top$ | §3 |
| 6 | CUT3R = TTT with β=1.0 | $\mathbf{S}_{t-1} - 1.0 \cdot (-\text{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V})$ | §3 |
| 7 | Confidence-guided β_t | $\beta_t = \sigma(\sum_m \mathbf{Q}_{\mathbf{S}_{t-1}} \mathbf{K}_{\mathbf{X}_t}^\top)$ | §5 |
| 8 | TTT3R full update rule | $\mathbf{S}_{t-1} - \beta_t \nabla(\mathbf{S}_{t-1}, \mathbf{X}_t)$ with adaptive $\beta_t$ | §5 |

---

## 15. Reference Map

### Offline 3D Reconstruction
- DUSt3R (Wang et al., 2024) — geometric 3D vision made easy
- VGGT (Wang et al., 2025) — visual geometry grounded transformer
- Fast3R (Yang et al., 2025) — towards 3D reconstruction from 1000+ images
- VGGSfM (Wang et al., 2024) — visual geometry grounded deep SfM
- MASt3R (Leroy et al., 2024) — grounding image matching in 3D
- MonST3R (Zhang et al., 2025) — geometry estimation with motion

### Online 3D Reconstruction
- CUT3R (Wang et al., 2025) — continuous 3D perception with persistent state
- Point3R (Wu et al., 2025) — streaming 3D reconstruction with spatial pointer memory
- StreamVGGT (Zhuo et al., 2025) — streaming 4D visual geometry transformer
- STream3R (Lan et al., 2025) — scalable sequential 3D reconstruction
- Spann3R (Wang & Agapito, 2024) — 3D reconstruction with spatial memory
- Easi3R (Chen et al., 2025) — disentangled motion from DUSt3R

### Test-Time Training / Fast Weights
- TTT (Sun et al., 2024) — learning at test time: RNNs with expressive hidden states
- DeltaNet (Schlag et al., 2021; Yang et al., 2024) — linear transformers as fast weight programmers
- Titans (Behrouz et al., 2024) — learning to memorize at test time
- Gated Linear Attention (Yang et al., 2023) — hardware-efficient training
- RetNet (Sun et al., 2023) — retentive network
- Test-time regression (Wang et al., 2025) — unifying framework for associative memory
- Longhorn (Liu et al., 2024) — state space models are amortized online learners

### Modern RNNs / State-Space Models
- Mamba (Gu & Dao, 2023) — selective state spaces
- Mamba-2 (Dao & Gu, 2024) — transformers are SSMs
- xLSTM (Beck et al., 2024) — extended long short-term memory
- Hopfield Networks (Ramsauer et al., 2020) — associative memory viewpoint

### Length Generalization
- Ruiz & Gu (2025) — understanding and improving length generalization in recurrent models
- Stuffed Mamba (Chen et al., 2024) — oversized states lead to inability to forget
- Decimamba (Ben-Kish et al., 2024) — exploring length extrapolation of Mamba

### SLAM and Classical Methods
- DROID-SLAM (Teed & Deng, 2021) — deep visual SLAM
- ORB-SLAM (Mur-Artal et al., 2015) — versatile monocular SLAM
- COLMAP (Schönberger & Frahm, 2016) — structure-from-motion
- MegaSaM (Li et al., 2025) — structure and motion from casual videos
- VGGT-SLAM (Maggio et al., 2025) — dense RGB SLAM
- VGGT-Long (Deng et al., 2025) — pushing VGGT limits on long sequences

### Self-supervised / Scene-adaptive Test-Time Methods
- CVD (Kopf et al., 2021) — consistent video depth estimation
- Test3R (Yuan et al., 2025) — learning to reconstruct 3D at test time

### Datasets
- ScanNet (Dai et al., 2017) — richly-annotated 3D indoor scenes
- TUM-Dynamics (Sturm et al., 2012) — RGB-D SLAM evaluation
- KITTI (Geiger et al., 2013) — autonomous driving benchmark
- Sintel (Butler et al., 2012) — naturalistic optical flow benchmark
- Bonn (Palazzolo et al., 2019) — dynamic RGB-D environments
- 7-Scenes (Shotton et al., 2013) — camera relocalization
