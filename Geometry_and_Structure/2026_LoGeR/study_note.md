# LoGeR Study Note — A Concept Mind-Map

> Paper: LoGeR: Long-Context Geometric Reconstruction with Hybrid Memory (arXiv:2603.03269v1)
> Authors: Junyi Zhang, Charles Herrmann, Junhwa Hur, Chen Sun, Ming-Hsuan Yang, Forrester Cole, Trevor Darrell, Deqing Sun
> Affiliations: Google DeepMind, UC Berkeley

This note organizes every key concept in the paper as a mind-map.
Each concept is broken down into four facets:

- **Definition** — what it is, stated plainly
- **Properties** — its mathematical or behavioral characteristics
- **Application** — how the paper (or the field) uses it
- **Links** — connections to other concepts in this map

---

## 0. The Big Picture

```
              Feedforward 3D Reconstruction (DUSt3R, MonST3R, VGGT, π³)
                                    │
                 "Works well on short windows, but how to scale
                  to minutes-long videos (thousands of frames)?"
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
     Full Attention           Recurrent/SSM          Naive Stitching
     O(N²) — impossible       (CUT3R, TTT3R)        (deterministic)
     for N > 1000             Lossy compression      Preserves local
                              → detail loss           detail but no
                                                      global memory
                                                      → scale drift
                                    │
                 "A single memory mechanism is insufficient"
                                    │
                                    ▼
              ┌───────────────────────────────────────────┐
              │     Hybrid Memory Module (this paper)     │
              │                                           │
              │  SWA (local, lossless)  +  TTT (global,   │
              │  non-parametric)           parametric,    │
              │  adjacent-chunk detail     compressed)    │
              │                                           │
              │         O(N) total compute                │
              └───────────────────┬───────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              Chunk-wise     Progressive    "Data Wall"
              Processing     Curriculum     Strategy
              (in-dist.)     Training       (large-scale
                                            navigation data)
                    │             │             │
                    └─────────────┼─────────────┘
                                  ▼
                           LoGeR Model
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              KITTI (74%     VBR (30.8%     7-Scenes,
              ATE reduction) improvement,   ScanNet,
                             up to 19k      TUM, Bonn
                             frames)
```

**The paper's story in one paragraph**: Feedforward geometric foundation models produce high-quality 3D reconstructions within short context windows but cannot scale to minutes-long videos — full attention is quadratically expensive, and recurrent designs compress away the fine details needed for accurate local alignment. LoGeR breaks this impasse with a *hybrid memory module* that combines two complementary mechanisms inside each network block: a non-parametric Sliding Window Attention (SWA) that passes lossless features between adjacent chunks for precise local alignment, and a parametric Test-Time Training (TTT) layer whose fast weights compress global context to prevent scale drift over thousands of frames. Paired with a progressive curriculum and a training mixture weighted toward large-scale navigation data to break the "data wall," LoGeR processes video streams chunk-by-chunk, stays within the distribution of short-context training data, and scales to 19k frames — reducing KITTI ATE by 74% over prior feedforward methods and achieving 30.8% improvement on the VBR long-sequence benchmark.

---

## 1. Chunk-wise Processing

### Definition

LoGeR partitions an input video $\mathcal{X} = \{I_t\}_{t=1}^{T}$ into $M$ sequential chunks $\{\mathcal{C}^m\}_{m=1}^{M}$ with minimal overlap (e.g., a single overlapping frame). Each chunk $\mathcal{C}^m$ comprises a variable number of frames $\mathcal{C}^m_1, \ldots, \mathcal{C}^m_n$. Within each chunk, a bidirectional geometry backbone (e.g., VGGT or $\pi^3$) produces high-quality dense predictions. Cross-chunk coherence is maintained by the hybrid memory module.

### Properties

- **In-distribution inference**: each chunk is processed as a short multi-view sequence — the same distribution as the training data. This avoids the train/test distribution gap that plagues methods trained on short sequences but applied to long ones.
- **Linear cost in sequence length**: total compute scales as $O(T)$ — the number of chunks times the fixed per-chunk cost.
- **Overlap convention**: minimal overlap (typically 1 frame) between adjacent chunks provides the anchor for cross-chunk alignment via SWA and for LoGeR*'s feedforward pose alignment.
- **Chunk size at inference**: 64 frames for short sequences; reduced to 32 or 48 for long sequences (KITTI, VBR) to mitigate trajectory drift in the base model.

### Application

- At inference, LoGeR uses chunk size 64 with overlap 3 for short/small-scale scenes, and chunk size 32 (KITTI) or 48 (VBR) for long sequences.
- Training uses sequences of 48 frames (stage 1) scaling to 128 frames (stage 2), split into 4–20 chunks.

### Links

- → **Hybrid Memory Module**: the mechanism that propagates information across chunk boundaries
- → **Progressive Curriculum Training**: the training schedule that teaches the model to use cross-chunk memory
- → **LoGeR\* Feedforward Alignment**: optional post-hoc pose alignment using chunk overlaps
- → **The "Data Wall"**: chunk-wise processing enables training on existing short-context data

---

## 2. Hybrid Memory Module

### Definition

The core architectural contribution: a dual-component memory system inserted into every residual block of the geometry backbone. It combines:

1. **TTT (parametric, global)**: fast weights $W^m$ that compress long-range context into a fixed-size state, providing global structural consistency.
2. **SWA (non-parametric, local)**: sliding-window attention over the previous and current chunk's tokens, providing lossless high-precision features for adjacent-chunk alignment.

**Table 1: Architectural trade-offs in sequence modeling**

| Mechanism | Compute Cost | Local Context | Global Context |
|---|---|---|---|
| Full Attention | $O(N^2)$ | Lossless | Lossless |
| Sliding Window Attn. | $O(N)$ | Lossless | Limited |
| TTT / Linear Attn. | $O(N)$ | Compressed | Compressed |
| **Ours (Hybrid Memory)** | $O(N)$ | **Lossless** | **Compressed** |

### Properties

- **Complementary roles**: TTT anchors the global coordinate frame and scale, preventing drift over thousands of frames. SWA ensures seamless, high-precision transitions at chunk boundaries.
- **Linear complexity**: both components operate in $O(N)$ time, keeping total cost linear in sequence length.
- **Decoupled concerns**: the long-range parametric memory (TTT) and the short-range non-parametric memory (SWA) address fundamentally different failure modes — scale drift vs. local misalignment.
- **Ablation evidence** (Table 3): removing TTT causes ATE on 1000-frame ScanNet to increase from 0.107 to 0.162 (+51%); removing SWA causes increase from 0.107 to 0.143 (+34%). Both are essential.

### Application

- Inserted into every residual block of the $\pi^3$ backbone (18 blocks total).
- TTT layers are inserted in all 18 blocks; SWA layers only at blocks 6, 10, 14, and 18 (4 layers).
- The base network has ~954M parameters; TTT and SWA add ~296M parameters.

### Links

- → **Test-Time Training (TTT)**: the global memory component
- → **Sliding Window Attention (SWA)**: the local memory component
- → **Chunk-wise Processing**: the hybrid memory propagates information across chunks
- → **Block Architecture**: how the two components integrate with per-frame and bidirectional attention

---

## 3. Test-Time Training (TTT) — Global Memory

### Definition

TTT is a parametric associative memory that stores context from prior forward calls into *fast weights* $W$. It defines a neural network $f_W(\cdot): \mathbb{R}^d \to \mathbb{R}^d$ parameterized by fast weights $W$, which are updated at both train and inference time.

**Update operation** — gradient-based self-supervised update:

$$W \leftarrow W - \eta \nabla_W \mathcal{L}\big(f_W(\mathbf{k}), \mathbf{v}\big) \tag{Eq.1}$$

where $\eta$ is the learning rate and $\mathcal{L}(\cdot, \cdot)$ is a loss between the transformed key $f_W(\mathbf{k})$ and the value $\mathbf{v}$.

**Apply operation** — use updated weights to process queries:

$$o = f_W(\mathbf{q}) \tag{Eq.2}$$

### Properties

- **Neural memory**: the update operation (Eq. 1) encodes the KV cache into $W$ as a form of neural memory. Conceptually, $f_W$ learns to link keys with corresponding values.
- **Fixed memory footprint**: regardless of sequence length, the memory is bounded by the size of $W$ — enabling theoretically infinite context.
- **Inherently lossy**: the compression into a fixed-size $W$ discards detail. This is the fundamental limitation that motivates pairing TTT with lossless SWA.
- **Practical capacity bound**: while theoretically infinite in receptive field, practical capacity is limited by the training context length. Beyond ~1000 frames, error accumulates and periodic state resets become necessary.
- **LaCTT variant**: LoGeR uses Large-Chunk Test-Time Training (LaCTT), which performs both update and apply at the chunk level rather than per-token, making it more efficient for chunk-wise processing.
- **SwiGLU fast-weight architecture**: $f_W$ is implemented as a SwiGLU MLP with head dimension 512 and expansion factor 4.
- **Muon optimizer**: uses the Muon optimizer for test-time updates.

### Application

**Chunk-wise TTT Apply** — injects compressed history into current chunk:

$$\tilde{\mathbf{H}}^{\mathcal{C}^m} = \mathbf{H}^{\mathcal{C}^m} + f_{W^m}\Big(\text{LN}(\mathbf{H}^{\mathcal{C}^m})\Big) \tag{Eq.5}$$

**Chunk-wise TTT Update** — stores current chunk's summary for future:

$$W^{m+1} = \mathcal{U}\Big(W^m; \mathbf{H}^{\mathcal{C}^m}\Big) \tag{Eq.6}$$

where $f_{W^m}(\cdot)$ is the fast-weight module parameterized by $W^m$, and $\mathcal{U}(\cdot)$ is the online update rule (gradient-based, self-supervised). The *apply* step modulates current tokens using historical context; the *update* step compresses the current chunk's information into $W$ for future chunks.

- Placed pre-norm inside TTT to stabilize long-horizon streaming.
- Periodic state resets every 5 windows during inference to prevent unbounded drift (following Ruiz & Gu, 2025).

### Links

- → **Hybrid Memory Module**: TTT is the global (parametric) component
- → **Sliding Window Attention (SWA)**: TTT's lossy compression is complemented by SWA's lossless local context
- → **Chunk-wise Processing**: TTT operates at the chunk level via LaCTT
- → **Progressive Curriculum Training**: curriculum teaches the model to rely on TTT for long-range context
- → **LoGeR\* Feedforward Alignment**: TTT anchors global scale, making SE(3) (not SIM(3)) alignment sufficient for LoGeR*
- → **Block Architecture**: TTT is applied after SWA and before bidirectional attention within each block

---

## 4. Sliding Window Attention (SWA) — Local Memory

### Definition

SWA is a non-parametric memory that preserves uncompressed features from the previous chunk for lossless cross-chunk alignment. It inserts attention layers where each token attends to the frame attention output from both the previous chunk $\mathcal{C}^{m-1}$ and the current chunk $\mathcal{C}^m$:

$$\mathbf{H}^{\mathcal{C}^m} \leftarrow \mathbf{H}^{\mathcal{C}^m} + \text{Attn}_\text{swa}\Big(\big[\text{LN}(\mathbf{H}^{\mathcal{C}^{m-1}}), \text{LN}(\mathbf{H}^{\mathcal{C}^m})\big]; \theta\Big) \tag{Eq.4}$$

### Properties

- **Lossless information highway**: directly passes full-fidelity features from the previous chunk — no compression, no information loss at the boundary.
- **Sparse integration**: SWA layers are inserted at only 4 of 18 blocks (6th, 10th, 14th, 18th) to remain compute-bounded.
- **Initialized from $\pi^3$**: the SWA layers are initialized from the corresponding global attention layers in $\pi^3$, inheriting pre-trained cross-view matching capability.
- **Bounded context**: only spans adjacent chunks — cannot propagate information beyond one chunk boundary. This is why TTT is needed for long-range context.
- **Temporal positional embeddings**: three learnable positional embedding tokens are added to frame representations, indicating whether a frame overlaps with the previous chunk, has no overlap, or overlaps with the subsequent chunk.
- **KV-cache optimization**: at inference, a KV-cache mechanism avoids redundant token computations in the SWA layers.

### Application

- Ensures geometric consistency at chunk boundaries by letting the current chunk directly attend to full-resolution features from the neighboring chunk.
- Ablation (Table 3): removing SWA increases ATE on ScanNet 500f from 0.087 to 0.115 (+32%), and causes visible local misalignment artifacts (Fig. 10).

### Links

- → **Hybrid Memory Module**: SWA is the local (non-parametric) component
- → **Test-Time Training (TTT)**: SWA complements TTT's lossy global memory with lossless local detail
- → **Per-frame Attention**: SWA operates on the output of per-frame attention
- → **Block Architecture**: SWA is placed between per-frame attention and TTT in the block

---

## 5. Per-frame Attention

### Definition

Standard self-attention applied independently to each frame's tokens to extract spatial features:

$$\mathbf{H}^{\mathcal{C}^m} \leftarrow \mathbf{H}^{\mathcal{C}^m} + \big[\text{Attn}_\text{frame}\big(\text{LN}(\mathbf{H}^{\mathcal{C}^m_i}); \theta\big), |i \in \{1, \ldots, n\}\big] \tag{Eq.3}$$

where $[\cdot]$ is the concatenation operator and attention is applied within each frame separately.

### Properties

- **Intra-frame spatial reasoning**: captures spatial relationships within a single image (e.g., depth ordering, object boundaries).
- **No temporal context**: operates per-frame, providing no cross-frame information — temporal context comes from SWA, TTT, and bidirectional attention.
- **Inherited from backbone**: uses the same weights as the $\pi^3$ backbone's per-frame attention.

### Application

- First operation in each residual block — processes each frame's tokens independently before any cross-frame mechanisms.

### Links

- → **Block Architecture**: the first step in the processing pipeline within each block
- → **Sliding Window Attention (SWA)**: SWA operates on per-frame attention outputs
- → **Chunk-wise Bidirectional Attention**: follows the memory modules to combine all information

---

## 6. Chunk-wise Bidirectional Attention

### Definition

A bidirectional attention module that reasons over all frames within the current chunk under a bounded context window. After the memory modules (SWA + TTT) have enriched the representations with cross-chunk context, bidirectional attention performs powerful multi-frame geometric reasoning:

$$\mathbf{H}^{\mathcal{C}^m} \leftarrow \tilde{\mathbf{H}}^{\mathcal{C}^m} + \text{BiAttn}_\text{chunk}\Big(\text{LN}(\tilde{\mathbf{H}}^{\mathcal{C}^m}); \theta\Big) \tag{Eq.7}$$

where $\tilde{\mathbf{H}}^{\mathcal{C}^m}$ is the representation after TTT apply (Eq. 5).

### Properties

- **High-fidelity intra-chunk reasoning**: uses full bidirectional attention within the chunk — the same powerful mechanism that makes short-context models (VGGT, $\pi^3$) effective.
- **Bounded cost**: attention is restricted to the chunk window, keeping complexity at $O(n^2)$ per chunk where $n$ is the chunk size (fixed and small).
- **Memory-enriched input**: operates on representations already informed by TTT (global history) and SWA (adjacent-chunk detail), so the bidirectional reasoning has access to cross-chunk context indirectly.

### Application

- Final processing step in each residual block before the output is passed to the next block.
- The updated representation $\mathbf{H}^{\mathcal{C}^m}$ serves as input to the next residual block.

### Links

- → **Block Architecture**: the final step in the block's processing pipeline
- → **Hybrid Memory Module**: operates on memory-enriched representations from TTT and SWA
- → **Chunk-wise Processing**: restricted to within-chunk attention for bounded compute

---

## 7. Block Architecture — Putting It All Together

### Definition

Each of LoGeR's 18 residual blocks processes the input sequence through four ordered stages. The full data flow within one block for chunk $\mathcal{C}^m$:

```
Input: H^{C^m} (token sequence for chunk m)
  │
  ├─(1) Per-frame Attention (Eq.3)
  │     Self-attention within each frame independently
  │     H^{C^m} ← H^{C^m} + Attn_frame(LN(H^{C^m}_i))
  │
  ├─(2) Sliding Window Attention (Eq.4) [only at blocks 6,10,14,18]
  │     Attend to previous chunk's features (lossless)
  │     H^{C^m} ← H^{C^m} + Attn_swa([LN(H^{C^{m-1}}), LN(H^{C^m})])
  │
  ├─(3) Chunk-wise TTT (Eqs.5 & 6)
  │     Apply: inject global history from fast weights W^m
  │     H̃^{C^m} = H^{C^m} + f_{W^m}(LN(H^{C^m}))
  │     Update: compress current chunk into W^{m+1}
  │     W^{m+1} = U(W^m; H^{C^m})
  │
  ├─(4) Chunk-wise Bidirectional Attention (Eq.7)
  │     Full bidirectional attention within the chunk
  │     H^{C^m} ← H̃^{C^m} + BiAttn_chunk(LN(H̃^{C^m}))
  │
  └─► Output: H^{C^m} → next residual block

              ┌──────────────────────────┐
              │   Fast Weights W^{m-1}   │
              │   from previous chunk    │
              └──────────┬───────────────┘
                         │
                    ┌────┴────┐
   H^{C^{m-1}}     │  Apply  │
       │            └────┬────┘
       │                 │ W^m
       ▼                 ▼
  ┌─────────┐    ┌──────────────┐    ┌───────────────┐    ┌─────────────┐
  │ Frame   │───▶│     SWA      │───▶│   TTT Apply   │───▶│  BiAttn     │──▶ Output
  │ Attn    │    │ (4 layers)   │    │   + Update    │    │  (chunk)    │
  │ (Eq.3)  │    │ (Eq.4)       │    │ (Eqs.5,6)    │    │  (Eq.7)     │
  └─────────┘    └──────────────┘    └──────┬────────┘    └─────────────┘
                                            │
                                       ┌────┴────┐
                                       │ Update  │
                                       └────┬────┘
                                            │ W^{m+1}
                                            ▼
                                  To next chunk's TTT
```

### Properties

- **Residual connections**: each stage adds to the representation via residual connections, preserving information from earlier stages.
- **LayerNorm pre-norm**: all attention and TTT operations apply LayerNorm before processing (pre-norm style).
- **Prediction heads**: after the final (18th) block, lightweight decoders — a pointmap decoder and a camera-pose decoder (following $\pi^3$) — produce per-frame dense pointmaps in local camera coordinates and per-frame camera poses in global world coordinates.

### Links

- → **Per-frame Attention**: step 1
- → **Sliding Window Attention (SWA)**: step 2
- → **Test-Time Training (TTT)**: step 3
- → **Chunk-wise Bidirectional Attention**: step 4
- → **Hybrid Memory Module**: the block is the container for the hybrid memory system

---

## 8. Learning Objectives

### Definition

LoGeR follows $\pi^3$ and trains with three loss components: a scale-invariant local pointmap loss, an affine-invariant relative pose loss, and a global pointmap loss.

**Local pointmap loss** — scale-invariant, depth-normalized:

$$\mathcal{L}_\text{local} = \frac{1}{N|\Omega|} \sum_{i=1}^{N} \sum_{p \in \Omega} \frac{1}{z_{i,p}} \big\| s^* \hat{\mathbf{x}}_{i,p} - \mathbf{x}_{i,p} \big\|_1 \tag{Eq.8}$$

where $N$ is the number of supervised frames, $\Omega$ is the pixel index set ($|\Omega| = HW$), $\hat{\mathbf{x}}_{i,p}$ and $\mathbf{x}_{i,p}$ are predicted and ground-truth local point coordinates, $z_{i,p}$ is depth for normalization, and $s^*$ is a single per-sequence scale factor (following MoGe).

**Pose loss** — affine-invariant relative motion:

$$\mathcal{L}_\text{pose} = \sum_{(i,j) \in \mathcal{P}} \Big(\lambda_r \mathcal{L}_\text{rot}(\hat{\mathbf{R}}_{ij}, \mathbf{R}_{ij}) + \lambda_t \big\| s^* \hat{\mathbf{t}}_{ij} - \mathbf{t}_{ij} \big\|_\text{Huber}\Big) \tag{Eq.9}$$

where $\hat{\mathbf{T}}_i = [\hat{\mathbf{R}}_i \mid \hat{\mathbf{t}}_i] \in \text{SE}(3)$ is the predicted camera pose, $\hat{\mathbf{R}}_{ij}$, $\hat{\mathbf{t}}_{ij}$ are predicted relative motion, and $\mathcal{P}$ is the set of supervised frame pairs (within a chunk and/or overlap pairs across chunks).

**Global pointmap loss** — world-coordinate consistency:

$$\mathcal{L}_\text{global} = \frac{1}{N|\Omega|} \sum_{i=1}^{N} \sum_{p \in \Omega} \big\| \Pi(\hat{\mathbf{T}}_i, \hat{\mathbf{x}}_{i,p}) - \Pi(\mathbf{T}_i, \mathbf{x}_{i,p}) \big\|_1 \tag{Eq.10}$$

where $\Pi(\mathbf{T}, \mathbf{x})$ maps a local point $\mathbf{x}$ to world coordinates using pose $\mathbf{T}$.

**Total loss**:

$$\mathcal{L} = \mathcal{L}_\text{local} + \mathcal{L}_\text{pose} + \lambda_\text{global} \mathcal{L}_\text{global} \tag{Eq.11}$$

### Properties

- **Affine-invariant**: the losses do not require a reference view — pointmaps are aligned with a single per-sequence scale $s^*$ (from MoGe), and poses use relative motion.
- **Global loss over-constrains long sequences**: $\mathcal{L}_\text{global}$ forces predicted world-coordinate pointmaps (via predicted poses) to match ground truth, directly penalizing accumulated drift.
- **Loss weights**: $\lambda_r = 0.1$ for rotation, $\lambda_t = 10$ for translation, $\lambda_\text{global} = 1$ for the global pointmap loss.
- **Supervised pairs $\mathcal{P}$**: includes both intra-chunk pairs and overlap pairs across different chunks, enabling cross-chunk supervision.

### Application

- All three losses are used during training. The global pointmap loss is the key addition for long-context training — it directly penalizes trajectory drift.

### Links

- → **Chunk-wise Processing**: losses operate on frames within and across chunks
- → **LoGeR\* Feedforward Alignment**: the global pointmap loss trains the model to produce globally consistent poses, which LoGeR* further refines
- → **Progressive Curriculum Training**: the curriculum gradually exposes the model to longer sequences where global consistency matters

---

## 9. LoGeR\* Feedforward Alignment

### Definition

An optional feedforward post-processing step that aligns raw per-chunk pose predictions into a consistent global coordinate system. LoGeR* uses the single overlapping frame between adjacent chunks to compute a rigid SE(3) alignment.

Let $\hat{\mathbf{T}}_k^{(m)}$ denote the raw predicted pose of overlapping frame $k$ within chunk $\mathcal{C}^m$, and $\bar{\mathbf{T}}^{(m-1)}$ the aligned poses of chunk $\mathcal{C}^{m-1}$ (initialized as $\bar{\mathbf{T}}^{(1)} = \mathbf{T}^{(1)}$). The rigid alignment:

$$\mathbf{A}_m = \bar{\mathbf{T}}_k^{(m-1)} \big(\hat{\mathbf{T}}_k^{(m)}\big)^{-1}$$

is applied to all frames in $\mathcal{C}^m$:

$$\bar{\mathbf{T}}_t^{(m)} = \mathbf{A}_m \, \hat{\mathbf{T}}_t^{(m)}, \quad \forall t \in \mathcal{C}^m \tag{Eq.12}$$

### Properties

- **Purely feedforward**: no optimization, no RANSAC, no PnP — just a matrix multiplication using the overlap frame.
- **SE(3) alignment**: rigid transformation only (no scale correction) — because the base LoGeR model already predicts in a globally consistent scale via TTT.
- **Used at both training and inference**: LoGeR* uses the aligned poses $\bar{\mathbf{T}}_t^{(m)}$ as the final camera pose predictions, and also applies feedforward alignment during training to reduce rollout overhead.
- **Contrast with Pi3-Chunk**: Pi3-Chunk requires SIM(3) alignment (scale + rigid) because $\pi^3$ predicts pointmaps only up-to-scale within each chunk. LoGeR's TTT memory anchors global scale, making rigid SE(3) sufficient.

### Application

- LoGeR* reduces KITTI average ATE from 25.44 to 18.65, a further 27% improvement over base LoGeR.
- On VBR, LoGeR* achieves 5.27 average ATE vs. LoGeR's 5.40.
- The feedforward alignment is applied during both training (to improve curriculum efficiency) and inference.

### Links

- → **Chunk-wise Processing**: alignment uses the overlapping frame between adjacent chunks
- → **Test-Time Training (TTT)**: TTT anchors global scale, making rigid (not SIM(3)) alignment sufficient
- → **Learning Objectives**: the global pointmap loss trains the model to produce globally consistent predictions that need only rigid alignment
- → **Pi3-Chunk Baseline**: LoGeR*'s SE(3) alignment contrasts with Pi3-Chunk's SIM(3) alignment

---

## 10. Progressive Curriculum Training

### Definition

A multi-stage training schedule that progressively increases sequence length and chunk count to stabilize the optimization of recurrent TTT layers and teach the model to shift reliance from local SWA to the global TTT hidden state.

### Stages

| Stage | Hardware | Sequence Length | Chunks | Chunk Size | Overlap | Steps |
|---|---|---|---|---|---|---|
| 1 (early) | 32× H100 | 48 frames | 4 → 12 | 12 → 4 | 3 → 1 | 25,000 |
| 2 (late) | 32× H200 | 128 frames | 12 → 20 | — | — | 15,000 |

### Properties

- **Gradual difficulty increase**: starts with 48 frames split into 4 large chunks (easy: intra-chunk reasoning dominates), then gradually increases chunk density so cross-chunk memory becomes essential.
- **Hardware-aware**: stage 2 uses H200 GPUs with more memory to accommodate 128-frame sequences with 20 chunks.
- **LoGeR\* initialization**: LoGeR* initializes from the first-stage model, integrates feedforward alignment, and fine-tunes through the remaining curriculum — this reduces train-time rollout overhead and improves final performance.
- **Ablation evidence** (Table 3): removing curriculum increases ScanNet 1000f ATE from 0.107 to 0.133 (+24%) for LoGeR, and from 0.080 to 0.093 (+16%) for LoGeR*.

### Application

- Total training: ~2 days on 32 H100 GPUs + ~2 days on 32 H200 GPUs.
- Optimizer: AdamW, 40k total steps, batch size 32.
- Learning rates: $5 \times 10^{-4}$ for new TTT/SWA parameters, $1 \times 10^{-5}$ for base network (unfrozen).
- Encoder and prediction heads are frozen; their pre-trained representations are retained.

### Links

- → **Chunk-wise Processing**: curriculum varies chunk count within fixed sequence lengths
- → **Hybrid Memory Module**: curriculum teaches the model to shift from SWA (local) to TTT (global)
- → **The "Data Wall"**: curriculum works in tandem with the data strategy
- → **LoGeR\* Feedforward Alignment**: LoGeR* uses curriculum to integrate alignment during training

---

## 11. The "Data Wall" and Training Data Strategy

### Definition

The observation that architectural improvements alone are insufficient for long-context reconstruction. Strong baselines like VGGT, even with inference-time heuristics (FastVGGT), fail on large-scale scenes because models trained on short-context, small-scale data cannot generalize to city-scale trajectories. This is the "data wall."

### Properties

- **Two barriers**: (1) a "context wall" from quadratic attention complexity, and (2) a "data wall" from training data limited to short, spatially bounded scenes.
- **Solution**: construct a training mixture heavily weighted toward large-scale navigation datasets (TartanAir, TartanAirV2, Virtual KITTI 2, OmniWorld-Game) to provide long-horizon geometric reasoning signals.
- **Ablation evidence** (Table 3): removing the 5 large-scale datasets from training increases ScanNet 1000f ATE from 0.107 to 0.156 (+46%), confirming that diverse long-horizon data is essential.

### Training Data Mixture (Table 4)

| Dataset | Percentage (%) |
|---|---|
| DL3DV | 17.89 |
| TartanAirV2 | 17.89 |
| OmniWorld-Game (subset) | 17.89 |
| ARKitScenes | 10.44 |
| TartanAir | 8.94 |
| Waymo | 6.71 |
| ARKitScenes HighRes | 4.18 |
| MegaDepth | 4.18 |
| ScanNet | 4.18 |
| ScanNet++ | 3.13 |
| Virtual KITTI 2 | 2.24 |
| HyperSim | 2.08 |
| Spring | 0.22 |
| UnReal4K | 0.04 |

### Application

- 14 large-scale datasets, mixing real and synthetic, indoor and outdoor, navigation and object-centric scenes.
- Large-scale navigation datasets are heavily weighted to break the data wall.
- Input resolution: $504 \times 280$.
- Rigorous depth filtering: maximum depth thresholds (e.g., 80m for ARKitScenes/ScanNet) or percentile-based clipping (90th/98th for DL3DV, TartanAir).

### Links

- → **Chunk-wise Processing**: chunk-wise design enables training on existing short-context data
- → **Progressive Curriculum Training**: the data strategy and curriculum work together
- → **VBR Benchmark**: the VBR benchmark tests whether the data wall has been broken

---

## 12. Experimental Results

### 12.1 Long Sequences (KITTI, VBR)

**KITTI Benchmark** — Absolute Trajectory Error (ATE, meters):

**Table 2: KITTI ATE**

| Method | Type | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DROID-SLAM | Opt. | 92.10 | 344.60 | 107.61 | 2.38 | 1.00 | 118.50 | 62.47 | 21.78 | 161.60 | 72.32 | 118.70 | 100.28 |
| DPV-SLAM | Opt. | 112.80 | 11.50 | 123.53 | 2.50 | 0.81 | 57.80 | 54.86 | 18.77 | 110.49 | 76.66 | 13.65 | 53.03 |
| DPV-SLAM++ | Opt. | – | 11.86 | 39.64 | 2.50 | 0.78 | 5.74 | 11.60 | 1.52 | 110.90 | 76.70 | 13.70 | 25.75 |
| VGGT-Long | Opt. | 8.67 | 121.17 | 32.08 | 6.12 | 4.23 | 8.31 | 5.34 | 4.63 | 53.10 | 41.99 | 18.37 | 27.64 |
| FastVGGT | FF | – | 639.39 | OOM | 21.53 | 9.51 | OOM | 40.56 | 51.35 | OOM | 201.54 | 196.22 | – |
| InfiniteVGGT | FF | 186.46 | 623.62 | 289.16 | 166.74 | 68.00 | 143.84 | 117.57 | 85.33 | 221.56 | 215.41 | 156.92 | 206.78 |
| CUT3R | FF | 190.38 | 90.59 | 264.39 | 20.40 | 7.31 | 92.25 | 67.54 | 22.48 | 145.08 | 67.92 | 40.00 | 91.62 |
| TTT3R | FF | 119.94 | 99.59 | 238.07 | 16.83 | 3.98 | 36.38 | 47.20 | 11.62 | 107.33 | 86.96 | 33.58 | 72.86 |
| Pi3-Chunk | FF | 26.65 | 196.04 | 157.92 | **5.13** | **1.09** | **12.79** | 27.66 | **5.94** | 61.26 | 56.31 | 21.96 | 52.07 |
| **LoGeR** | FF | 62.34 | 41.64 | 39.64 | 4.89 | 1.82 | 41.27 | 13.99 | 16.24 | 26.46 | 22.71 | **8.84** | **25.44** |
| **LoGeR*** | FF | **30.47** | **47.91** | **36.32** | 5.38 | 1.95 | 26.34 | **6.60** | 5.55 | **24.41** | **10.12** | 10.11 | **18.65** |

**Key findings**:
- LoGeR* reduces average ATE from 72.86 (TTT3R, prior best FF) to 18.65 — a **74% reduction**.
- LoGeR* surpasses the strongest optimization-based method (VGGT-Long, 27.64) by **32.5%**.
- The advantage is strongest on open-loop trajectories (seqs 01, 03, 04, 08, 10) where loop closure is unavailable.

**VBR Benchmark** — sequences up to 19k frames, 11.5 km trajectories:

**Table 6: VBR per-sequence ATE**

| Method | colosseo_0 | campus_0 | campus_1 | pincio_0 | spagna_0 | diag_0 | ciampino_1 | Avg. |
|---|---|---|---|---|---|---|---|---|
| VGGT-SLAM | 10.05 | 9.67 | 8.47 | 8.15 | 7.55 | 5.80 | 11.14 | 8.69 |
| VGGT-Long | 6.29 | 10.89 | 9.91 | 7.31 | 7.09 | 5.55 | 13.12 | 8.60 |
| InfiniteVGGT | 9.16 | 11.12 | 10.00 | 8.41 | 7.50 | 5.62 | 13.70 | 9.36 |
| CUT3R | 9.09 | 6.50 | 6.57 | 6.83 | 6.68 | 5.35 | 13.26 | 7.75 |
| TTT3R | 8.69 | 7.71 | 7.52 | 5.82 | 6.11 | **4.30** | 13.18 | 7.62 |
| Pi3-Chunk | 8.78 | 8.86 | 8.11 | 6.48 | 6.69 | 4.88 | 10.57 | 7.77 |
| **LoGeR** | 4.88 | **4.13** | **5.18** | **5.03** | **4.55** | 5.75 | **8.30** | **5.40** |
| **LoGeR*** | **6.92** | 4.60 | 5.59 | 3.18 | 4.67 | 5.48 | 6.44 | **5.27** |

**Key findings**:
- LoGeR achieves **30.8% improvement** over the best prior feedforward method (TTT3R, 7.62 → 5.27).
- The advantage increases with sequence length (Fig. 4) — Pi3-Chunk's SIM(3) scale errors accumulate, while LoGeR's TTT anchors global scale.
- LoGeR maintains global scale consistency up to 19k frames (Fig. 5), while baselines show severe drift.

---

### 12.2 Short Sequences (7-Scenes, ScanNet, TUM, Bonn)

**7-Scenes** (3D point-cloud reconstruction, Chamfer Distance):
- LoGeR achieves **69.2% improvement** over prior work (Fig. 6).
- At 1k frames: **90.3%** and **72.1%** error reduction vs. TTT3R and VGG-T³ with comparable runtimes (Fig. 12).
- LoGeR is **84.1% faster** than VGGT with **31.0% better** performance.

**ScanNet & TUM-Dynamics** (camera pose estimation, ATE):
- **80.0% relative improvement** on ScanNet (Fig. 9).
- **66.1% relative improvement** on TUM-Dynamics (Fig. 9).
- LoGeR avoids the structural distortions visible in TTT3R and Pi3-Chunk reconstructions (Fig. 8).

**Bonn** (video depth estimation, Abs. Rel.):
- **21.05% error reduction** at 500 frames vs. TTT3R (Fig. 11).

### Links

- → **Hybrid Memory Module**: the architectural contribution validated by these results
- → **LoGeR\* Feedforward Alignment**: LoGeR* variant further improves on long-sequence benchmarks
- → **The "Data Wall"**: large-scale training data enables generalization to KITTI/VBR-scale scenes

---

## 13. Ablation Studies

### Architecture Design (Table 3, first block)

| Method | ScanNet 500f | ScanNet 1000f | TUM 500f | TUM 1000f |
|---|---|---|---|---|
| **LoGeR** | **0.087** | **0.107** | **0.033** | **0.050** |
| w/o TTT | 0.108 | 0.162 | 0.043 | 0.079 |
| w/o SWA | 0.115 | 0.143 | 0.039 | 0.053 |

- Removing TTT: ScanNet 1000f ATE increases by **51%** (0.107 → 0.162). TTT is essential for global consistency — without it, trajectory drifts over long horizons (Fig. 10, right: severe drift).
- Removing SWA: ScanNet 500f ATE increases by **32%** (0.087 → 0.115). SWA is essential for local alignment — without it, visible misalignment artifacts appear at chunk boundaries (Fig. 10, center).

### Dataset Mixture (Table 3, second block)

| Config | ScanNet 500f | ScanNet 1000f | TUM 500f | TUM 1000f |
|---|---|---|---|---|
| All datasets | **0.087** | **0.107** | **0.033** | **0.050** |
| w/o 5 large datasets | 0.102 | 0.156 | 0.050 | 0.072 |

- Removing the 5 large-scale navigation datasets increases ScanNet 1000f ATE by **46%** — diverse long-horizon data is key.

### Curriculum Training (Table 3, third block)

| Config | ScanNet 500f | ScanNet 1000f | TUM 500f | TUM 1000f |
|---|---|---|---|---|
| LoGeR | **0.087** | **0.107** | **0.033** | **0.050** |
| w/o curriculum | 0.098 | 0.133 | 0.049 | 0.062 |
| LoGeR* | **0.070** | **0.080** | **0.031** | **0.036** |
| w/o curriculum | 0.078 | 0.093 | **0.029** | 0.040 |

- Curriculum consistently improves both LoGeR and LoGeR* variants.
- The improvement is larger for LoGeR (non-aligned) since curriculum stabilizes the recurrent TTT layers.

### Inference Efficiency (Table 5)

| Chunk Size | Speed (FPS) | Memory (GB) |
|---|---|---|
| 64 | 9.3 | 27.2 |
| 48 | 10.6 | 22.3 |
| 32 | 12.1 | 18.1 |

- Measured on a single NVIDIA A100 (40GB), 500-frame sequence.
- Memory stays constant regardless of total sequence length (chunk-wise processing).
- Speed scales linearly with chunk size.

### Links

- → **Hybrid Memory Module**: ablation validates each component
- → **The "Data Wall"**: dataset ablation validates the data strategy
- → **Progressive Curriculum Training**: curriculum ablation validates the training schedule

---

## 14. What Existed Before and What This Paper Changes

### 14.1 Prior Approaches and Their Limitations

**Full-attention feedforward models** (DUSt3R, MonST3R, VGGT, $\pi^3$) produce high-quality 3D reconstructions from multi-view images. Their quadratic $O(N^2)$ attention cost limits them to short context windows — typically dozens to a few hundred frames. Scaling to thousands of frames causes out-of-memory failures. FastVGGT uses sparse attention to reduce cost at inference, but it is trained on short-context data and fails completely on city-scale scenes like VBR (Fig. 3, top-right: garbled reconstruction).

**Recurrent/state-space models** (CUT3R) process frames sequentially with $O(N)$ cost by compressing all history into a single hidden state. This compression sacrifices the dense information needed for precise adjacent-frame alignment. CUT3R's lossy hidden state cannot maintain both global consistency and local detail simultaneously.

**Frame-wise TTT** (TTT3R) uses TTT for single-frame streaming with a confidence-based update, but processes one frame at a time. This frame-wise design lacks multi-frame bidirectional reasoning — the powerful intra-chunk attention that makes short-context models effective. TTT3R cannot match the local quality of bidirectional approaches.

**Deterministic stitching** approaches can connect chunk outputs geometrically but lack any learning-based long-range memory. Scale errors accumulate exponentially over many chunks because each junction introduces independent alignment noise.

### 14.2 What This Paper Contributes

**Contribution 1: Hybrid memory architecture.** A dual-component memory system (SWA + TTT) that achieves lossless local context *and* compressed global context at $O(N)$ cost. No prior mechanism achieves both.

**Contribution 2: Chunk-wise processing with bidirectional backbones.** By processing in chunks, LoGeR keeps local inferences within the distribution of short-context training data while propagating information via memory. This breaks the context wall without requiring long-context training data.

**Contribution 3: Breaking the data wall.** A training mixture heavily weighted toward large-scale navigation datasets, paired with progressive curriculum training, enables the model to generalize from 128-frame training to 19k-frame inference.

**Contribution 4: VBR long-context benchmark.** A new evaluation protocol using the VBR dataset (8.8k–19k frames, up to 11.5 km), providing the first rigorous test of long-context feedforward 3D reconstruction.

### 14.3 Side-by-Side: Prior Art vs. This Paper

| Dimension | VGGT/π³ | CUT3R | TTT3R | FastVGGT | **LoGeR** |
|---|---|---|---|---|---|
| Compute cost | $O(N^2)$ | $O(N)$ | $O(N)$ | $O(N)$ | $O(N)$ |
| Local context | Lossless | Compressed | Compressed | Lossless | **Lossless** |
| Global context | Lossless | Compressed | Compressed | Limited | **Compressed** |
| Multi-frame reasoning | Bidirectional | None (sequential) | None (per-frame) | Bidirectional | **Bidirectional** |
| Max practical frames | ~100 | ~1000 | ~1000 | ~1000 | **~19,000** |
| KITTI Avg. ATE | OOM | 91.62 | 72.86 | – | **18.65** |

| Dimension | TTT3R (closest FF) | **LoGeR*** |
|---|---|---|
| Processing | Per-frame | Per-chunk (bidirectional) |
| Local memory | None | SWA (lossless) |
| Global memory | TTT (frame-wise) | TTT (chunk-wise LaCTT) |
| Intra-window reasoning | None | Full bidirectional attention |
| Pose alignment | None | Feedforward SE(3) |
| KITTI Avg. ATE | 72.86 | **18.65** (74% better) |
| VBR Avg. ATE | 7.62 | **5.27** (30.8% better) |

### 14.4 The Core Shift in Thinking

Previous feedforward methods assumed a single memory mechanism would suffice for long-context 3D reconstruction. Full attention is lossless but unscalable. Recurrent state (CUT3R) and TTT (TTT3R) are scalable but lossy — they compress everything into one state, losing the dense features needed for precise local alignment. Deterministic stitching preserves local detail but has no learned memory for global consistency.

LoGeR's insight is that these failure modes are *complementary*: what one mechanism loses, another preserves. A short-range lossless memory (SWA) covers the local alignment that recurrent compression destroys, while a long-range compressed memory (TTT) covers the global consistency that local stitching cannot maintain. By combining both in every network block, each mechanism operates where it is strongest.

The second insight is equally important: the "data wall" — not just the architecture — limits long-context generalization. Chunk-wise processing with progressive curriculum lets the model train on existing short-context data while learning to use cross-chunk memory, bypassing the need for massive long-horizon datasets.

---

## 15. Quick Reference Card

| # | Insight | Design Choice | Evidence |
|---|---|---|---|
| 1 | No single memory mechanism handles both local and global context | Hybrid SWA + TTT in every block | Table 1; Table 3 ablation: removing either degrades |
| 2 | TTT anchors global scale, preventing drift | Chunk-wise LaCTT with periodic resets | Fig. 10: w/o TTT causes severe trajectory drift |
| 3 | SWA provides lossless cross-chunk alignment | Sparse insertion at 4 of 18 layers | Fig. 10: w/o SWA causes local misalignment |
| 4 | Chunk-wise processing keeps inferences in-distribution | Process in chunks matching training context size | Trains on 128 frames, generalizes to 19k |
| 5 | Data wall limits long-context generalization | Weight training mixture toward navigation datasets | Table 3: removing large datasets degrades by 46% |
| 6 | Progressive curriculum stabilizes TTT training | Gradually increase chunk count and sequence length | Table 3: curriculum improves both LoGeR and LoGeR* |
| 7 | TTT makes rigid SE(3) alignment sufficient | LoGeR* uses SE(3) vs. Pi3-Chunk's SIM(3) | LoGeR* avoids Pi3-Chunk's scale drift on VBR |
| 8 | Feedforward methods can outperform SLAM on long sequences | LoGeR surpasses VGGT-Long (optimization-based) | Table 2: 18.65 vs 27.64 ATE on KITTI |

---

## 16. Open Questions

The paper identifies several directions for future work:

1. **Length generalization of TTT**: fast weights have a fixed memory footprint but struggle to generalize beyond the training context length. Beyond ~1000 frames, periodic state resets are needed, sacrificing long-term context. Future linear sequence models may resolve this.

2. **Training data scarcity**: the "data wall" remains a bottleneck. More diverse, long-horizon datasets are needed for the community to advance long-context reconstruction.

3. **Extension beyond geometric reconstruction**: the hybrid SWA + TTT architecture could apply to other domains requiring both long-term global consistency and strong local dependencies (e.g., video understanding, robotics).

4. **Inference efficiency**: gradient checkpointing and extreme memory demands during training limit practical scalability. Pruning TTT layers or strided sampling within SWA could improve efficiency.

---

## 17. Concept Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                      PROBLEM LAYER                              │
│                                                                 │
│  ┌──────────────────┐          ┌──────────────────────┐         │
│  │ "Context Wall"   │          │ "Data Wall"          │         │
│  │ O(N²) attention  │          │ Short-context data   │         │
│  └────────┬─────────┘          └──────────┬───────────┘         │
│           │                               │                     │
└───────────┼───────────────────────────────┼─────────────────────┘
            │                               │
            ▼                               ▼
┌───────────────────────┐    ┌────────────────────────────┐
│  Chunk-wise Processing│    │  Training Data Strategy     │
│  (in-distribution)    │    │  (14 datasets, navigation-  │
│                       │    │   weighted mixture)          │
└───────────┬───────────┘    └──────────┬─────────────────┘
            │                           │
            └──────────┬────────────────┘
                       ▼
         ┌──────────────────────────────┐
         │   Hybrid Memory Module       │
         │                              │
         │  ┌────────┐   ┌────────┐    │
         │  │  SWA   │   │  TTT   │    │
         │  │(local) │   │(global)│    │
         │  │Eq.4    │   │Eqs.5,6 │    │
         │  └───┬────┘   └───┬────┘    │
         │      └─────┬──────┘         │
         │            │                │
         └────────────┼────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ┌──────────┐ ┌──────────┐ ┌──────────────┐
    │Per-frame │ │Chunk-wise│ │ Block        │
    │Attention │ │BiAttn    │ │ Architecture │
    │ Eq.3     │ │ Eq.7     │ │ (18 blocks)  │
    └──────────┘ └──────────┘ └──────┬───────┘
                                     │
                       ┌─────────────┼──────────────┐
                       ▼             ▼              ▼
               ┌──────────────┐ ┌──────────┐ ┌──────────────┐
               │ Learning     │ │LoGeR*    │ │ Progressive  │
               │ Objectives   │ │Feedfwd   │ │ Curriculum   │
               │ Eqs.8-11     │ │Alignment │ │ Training     │
               │              │ │ Eq.12    │ │              │
               └──────────────┘ └──────────┘ └──────────────┘
                       │             │              │
                       └─────────────┼──────────────┘
                                     ▼
                              LoGeR / LoGeR*
```

---

## 18. Key Equations

| Eq. | Name | Formula | Section |
|---|---|---|---|
| 1 | TTT Update | $W \leftarrow W - \eta \nabla_W \mathcal{L}(f_W(\mathbf{k}), \mathbf{v})$ | §3 TTT |
| 2 | TTT Apply | $o = f_W(\mathbf{q})$ | §3 TTT |
| 3 | Per-frame attention | $\mathbf{H}^{\mathcal{C}^m} \leftarrow \mathbf{H}^{\mathcal{C}^m} + [\text{Attn}_\text{frame}(\text{LN}(\mathbf{H}^{\mathcal{C}^m_i}); \theta)]$ | §5 Per-frame |
| 4 | Sliding window attention | $\mathbf{H}^{\mathcal{C}^m} \leftarrow \mathbf{H}^{\mathcal{C}^m} + \text{Attn}_\text{swa}([\text{LN}(\mathbf{H}^{\mathcal{C}^{m-1}}), \text{LN}(\mathbf{H}^{\mathcal{C}^m})];\theta)$ | §4 SWA |
| 5 | Chunk-wise TTT Apply | $\tilde{\mathbf{H}}^{\mathcal{C}^m} = \mathbf{H}^{\mathcal{C}^m} + f_{W^m}(\text{LN}(\mathbf{H}^{\mathcal{C}^m}))$ | §3 TTT |
| 6 | Chunk-wise TTT Update | $W^{m+1} = \mathcal{U}(W^m; \mathbf{H}^{\mathcal{C}^m})$ | §3 TTT |
| 7 | Chunk-wise BiAttn | $\mathbf{H}^{\mathcal{C}^m} \leftarrow \tilde{\mathbf{H}}^{\mathcal{C}^m} + \text{BiAttn}_\text{chunk}(\text{LN}(\tilde{\mathbf{H}}^{\mathcal{C}^m}); \theta)$ | §6 BiAttn |
| 8 | Local pointmap loss | $\mathcal{L}_\text{local} = \frac{1}{N|\Omega|}\sum_i\sum_p \frac{1}{z_{i,p}}\|s^*\hat{\mathbf{x}}_{i,p} - \mathbf{x}_{i,p}\|_1$ | §8 Loss |
| 9 | Pose loss | $\mathcal{L}_\text{pose} = \sum_{(i,j)\in\mathcal{P}}(\lambda_r\mathcal{L}_\text{rot} + \lambda_t\|s^*\hat{\mathbf{t}}_{ij} - \mathbf{t}_{ij}\|_\text{Huber})$ | §8 Loss |
| 10 | Global pointmap loss | $\mathcal{L}_\text{global} = \frac{1}{N|\Omega|}\sum_i\sum_p \|\Pi(\hat{\mathbf{T}}_i,\hat{\mathbf{x}}_{i,p}) - \Pi(\mathbf{T}_i,\mathbf{x}_{i,p})\|_1$ | §8 Loss |
| 11 | Total loss | $\mathcal{L} = \mathcal{L}_\text{local} + \mathcal{L}_\text{pose} + \lambda_\text{global}\mathcal{L}_\text{global}$ | §8 Loss |
| 12 | LoGeR* alignment | $\bar{\mathbf{T}}_t^{(m)} = \mathbf{A}_m\hat{\mathbf{T}}_t^{(m)}, \forall t \in \mathcal{C}^m$ | §9 LoGeR* |

---

## 19. Reference Map

### Feedforward 3D Reconstruction (Short-context)
- DUSt3R (Wang et al., 2024) — geometric 3D vision made easy
- MonST3R (Zhang et al., 2025a) — geometry estimation with motion
- VGGT (Wang et al., 2025a) — visual geometry grounded transformer
- $\pi^3$ (Wang et al., 2026) — permutation-equivariant visual geometry learning
- MASt3R (Leroy et al., 2024) — grounding image matching in 3D
- VGG-T³ (Elflein et al., 2026) — offline feed-forward 3D reconstruction at scale

### Feedforward 3D Reconstruction (Long-context)
- CUT3R (Wang et al., 2025b) — continuous 3D perception with persistent state
- TTT3R (Chen et al., 2026) — 3D reconstruction as test-time training
- FastVGGT (Shen et al., 2026) — training-free acceleration of VGGT
- InfiniteVGGT (Yuan et al., 2026) — visual geometry transformer for endless streams
- Long3R (Chen et al., 2025) — long sequence streaming 3D reconstruction
- Stream3R (Lan et al., 2025) — scalable sequential 3D reconstruction with causal transformer
- Point3R (Wu et al., 2025) — streaming 3D reconstruction with explicit spatial pointer memory

### Optimization-based SLAM
- DROID-SLAM (Teed & Deng, 2021) — deep visual SLAM
- DPV-SLAM (Lipson et al., 2024) — deep patch visual SLAM
- VGGT-SLAM (Maggio et al., 2025) — dense RGB SLAM
- VGGT-Long (Deng et al., 2025) — pushing VGGT's limits on long sequences

### Memory and Sequence Modeling
- TTT (Sun et al., 2024) — learning at test time: RNNs with expressive hidden states
- LaCTT (Zhang et al., 2025b) — test-time training done right
- Mamba (Gu & Dao, 2024) — selective state spaces
- S4 (Gu et al., 2022) — structured state spaces
- Longformer (Beltagy et al., 2020) — long-document transformer
- Big Bird (Zaheer et al., 2020) — transformers for longer sequences
- Jamba (Lenz et al., 2025) — hybrid transformer-mamba language models
- Length generalization (Ruiz & Gu, 2025) — improving recurrent model generalization

### Scale-invariant Geometry
- MoGe (Wang et al., 2025c) — monocular geometry estimation with optimal training supervision

### Datasets
- KITTI (Geiger et al., 2012) — autonomous driving benchmark
- VBR (Brizi et al., 2024) — vision benchmark in Rome
- 7-Scenes (Shotton et al., 2013) — camera relocalization
- ScanNet (Dai et al., 2017) — richly-annotated 3D indoor scenes
- ScanNet++ (Yeshwanth et al., 2023) — high-fidelity indoor scenes
- TUM-Dynamics (Sturm et al., 2012) — RGB-D SLAM evaluation
- Bonn (Palazzolo et al., 2019) — dynamic RGB-D environments
- DL3DV (Ling et al., 2024) — large-scale 3D vision dataset
- TartanAir (Wang et al., 2020) — visual SLAM dataset
- TartanAirV2 (Patel et al., 2025) — large-scale robot perception
- ARKitScenes (Baruch et al., 2021) — mobile RGB-D data
- HyperSim (Roberts et al., 2021) — photorealistic synthetic indoor
- Spring (Mehl et al., 2023) — scene flow, optical flow, stereo
- MegaDepth (Li & Snavely, 2018) — single-view depth from photos
- Virtual KITTI 2 (Cabon et al., 2020) — synthetic driving
- OmniWorld-Game (Zhou et al., 2025) — multi-domain 4D world modeling
- UnReal4K (Wang et al., 2023) — neural video depth stabilizer
- Waymo (Schwall et al., 2020) — public road safety data
