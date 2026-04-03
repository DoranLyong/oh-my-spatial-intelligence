# GaussianGPT: Towards Autoregressive 3D Gaussian Scene Generation — Study Note (Concept Mind-Map)

> Paper: GaussianGPT: Towards Autoregressive 3D Gaussian Scene Generation (arXiv:2603.26661v1)
> Authors: Nicolas von Lützow, Barbara Rössle, Katharina Schmid, Matthias Nießner
> Affiliations: Technical University of Munich, Germany

This note organizes every key concept in the paper as a mind-map.
Each concept is broken down into four facets:

- **Definition** — what it is, stated plainly
- **Properties** — its mathematical or behavioral characteristics
- **Application** — how the paper (or the field) uses it
- **Links** — connections to other concepts in this map

---

## 0. The Big Picture

```
              3D Gaussian Scene
                     │
        "No canonical ordering for 3D data"
        "Diffusion generates holistically, not incrementally"
        "Unconstrained Gaussians lack structure"
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    Structured    Sparse 3D    Lookup-Free
    Voxel Grid    CNN AE       Quantization
         │           │           │
         └─────┬─────┘───────────┘
               ▼
     Discrete Latent Grid (codebook indices)
               │
     ┌─────────┼─────────────┐
     ▼         ▼             ▼
   xyz      Interleaved    3D RoPE
   Serial.  pos/feat       (4D: xyz + type)
     │       tokens          │
     └─────────┬─────────────┘
               ▼
      ┌──────────────────┐
      │  GPT-2 Causal     │
      │  Transformer      │
      │  (next-token pred)│
      └────────┬──────────┘
               │
     ┌─────────┼─────────────┐
     ▼         ▼             ▼
  Uncond.   Scene         Large-Scale
  Gener.    Completion    Outpainting
  (BOS→)   (prefix→)     (sliding window)
```

The paper's core claim: autoregressive transformers — the dominant paradigm in language and vision — can be applied to **3D Gaussian scene generation** by converting Gaussian scenes into structured discrete token sequences. Unlike diffusion-based 3D generation that refines scenes holistically, this formulation builds scenes **step-by-step**, enabling completion, outpainting, temperature-controlled sampling, and flexible generation horizons — all from a single model.

---

## 1. Sparse 3D Feature Grid

### Definition

The first step of the pipeline converts a continuous 3D Gaussian scene into a structured grid representation. A voxel grid is defined in world coordinates. Each Gaussian primitive (characterized by position, opacity, size, rotation, color) is assigned to its corresponding voxel based on position. Absolute positions are replaced with **relative offsets** from voxel centers. When multiple Gaussians fall into one voxel, one is randomly subsampled.

Per-voxel, a set of lightweight **encoding heads** (linear layer expanding by 16x, followed by a residual MLP block) converts each Gaussian attribute into a feature embedding. These are concatenated into a single per-voxel feature vector, forming the sparse input feature grid.

### Properties

- The grid introduces **spatial regularity** to the inherently unstructured Gaussian representation.
- Voxel size for scenes: 0.025 m base, resulting in 20 cm voxels at the latent level (after 3 downsampling stages).
- Voxel size for objects: $128^3$ grid with 2 downsampling stages.
- Sparsity: only occupied voxels carry features; empty voxels are skipped.
- Feature representations: scales in world-space (softplus), opacities in logit space (clamped $[-10,10]$), colors clamped to $[0,1]$, standardized quaternions, offsets as unbounded values.
- Features are scaled to similar magnitudes for training stability.

### Application

**Encoder heads** (per attribute): linear layer (16x expansion) + residual MLP. **Decoder heads** (per attribute): 2 residual MLP blocks (64 channels) + final linear projection. Final projection layers are zero-initialized with constant biases for controlled initial visibility.

Separate heads handle: scales, opacities, quaternions, color, coordinate offsets, and spherical harmonics.

### Links

- -> **Sparse 3D CNN Encoder-Decoder**: processes this feature grid into a compact latent representation
- -> **3D Grid Serialization**: the latent grid produced downstream is serialized into a 1D token sequence
- -> **Vocabulary Design**: the grid structure enables clean separation of position and feature tokens

---

## 2. Sparse 3D CNN Encoder-Decoder

### Definition

A sparse 3D convolutional encoder $\mathcal{E}$ and decoder $\mathcal{D}$, following the L3DG [52] architecture, that compresses the sparse input feature grid into a compact latent representation. The encoder progressively downsamples; the decoder reconstructs voxel-level features. The convolutional design preserves **spatial locality** and **translation equivariance**.

### Properties

- 3 downsampling stages for scenes (0.025 m -> 0.2 m latent voxels).
- 2 downsampling stages for objects ($128^3$ grid).
- Sparse convolutions: computation is limited to occupied voxels, making it memory-efficient.
- The occupancy predictions in the decoder upsampling layers are supervised by a binary cross-entropy loss $\mathcal{L}_{occ}$.

### Application

- Trained with Adam optimizer, learning rate $10^{-4}$.
- Training: ~4 days on 4 RTX A6000 GPUs (scenes), ~2 days (PhotoShape objects).
- Effective batch size: 8 for scenes, 24 for objects.
- Fine-tuned on 3D-FRONT for ~1 day.
- Supervision: re-rendering losses on randomly sampled images (12 images for scenes, 4 for objects).

### Links

- -> **Sparse 3D Feature Grid**: input to the encoder
- -> **Lookup-Free Quantization (LFQ)**: discretizes the encoder output
- -> **Autoencoder Training Loss**: the combined loss that trains this encoder-decoder

---

## 3. Lookup-Free Quantization (LFQ)

### Definition

Instead of maintaining a traditional codebook (as in VQ-VAE), GaussianGPT uses **Lookup-Free Quantization** (LFQ) [73]. The encoder output $\mathbf{z}$ is discretized to binary codes by thresholding at zero: each dimension is mapped to 0 or 1 based on its sign. The codebook index is the integer interpretation of the resulting binary string.

### Properties

- Codebook size: $2^{d_{\text{code}}}$ where $d_{\text{code}}$ is the latent dimension. With $d_{\text{code}} = 12$, the codebook has $2^{12} = 4{,}096$ entries.
- No codebook lookup during training or inference — the codebook is implicit in the sign function.
- LFQ improves codebook utilization compared to standard VQ [73], which suffers from codebook collapse (unused entries).
- The entropy regularizer $\mathcal{L}_{LFQ}$ encourages full codebook usage by maximizing entropy of the codebook distribution.

### Application

- Codebook size: 4,096 across all experiments.
- $\mathcal{L}_{LFQ}$ is offset by +5 and passed through softplus to ensure positive loss values:
  $\lambda_{LFQ} \cdot \text{softplus}(\mathcal{L}_{LFQ} + 5)$.
- $\lambda_{LFQ} = 0.1$.
- The resulting discrete indices form the **feature vocabulary** for the autoregressive transformer.

### Links

- -> **Sparse 3D CNN Encoder-Decoder**: quantizes the encoder output
- -> **Autoencoder Training Loss**: $\mathcal{L}_{LFQ}$ is a component of the total loss (Eq.1)
- -> **Vocabulary Design**: produces the feature codebook indices that the transformer predicts

---

## 4. Autoencoder Training Loss

### Definition

The autoencoder (encoder + decoder + LFQ) is trained with a combined loss:

$$\mathcal{L} = \underbrace{\lambda_{RGB}\mathcal{L}_{RGB} + \lambda_{perc}\mathcal{L}_{perc}}_{\text{re-rendering}} + \underbrace{\lambda_{occ}\mathcal{L}_{occ}}_{\text{occupancy}} + \underbrace{\lambda_{LFQ}\text{softplus}(\mathcal{L}_{LFQ} + 5)}_{\text{codebook entropy}} \tag{Eq.1}$$

### Properties

- $\mathcal{L}_{RGB}$: $\mathcal{L}_1$ color loss computed on rendered images.
- $\mathcal{L}_{perc}$: VGG19-based [57] perceptual loss.
- Both re-rendering losses are aggregated over a set of sampled images and poses.
- $\mathcal{L}_{occ}$: binary cross-entropy for occupancy prediction in decoder upsampling layers.
- $\mathcal{L}_{LFQ}$: entropy regularizer for codebook utilization.

### Application

Hyperparameters:

| Parameter | Scenes | Objects |
|---|---|---|
| $\lambda_{RGB}$ | 7.5 | 12.5 |
| $\lambda_{perc}$ | 0.3 | 0.1 |
| $\lambda_{occ}$ | 1.0 | 1.0 |
| $\lambda_{LFQ}$ | 0.1 | 0.1 |
| # render views | 12 | 4 |

Scenes do not model view-dependent appearance; objects use first-order spherical harmonics.

### Links

- -> **Sparse 3D CNN Encoder-Decoder**: the model being trained
- -> **Lookup-Free Quantization**: contributes $\mathcal{L}_{LFQ}$
- -> **Sparse 3D Feature Grid**: decoder heads produce per-voxel Gaussian attributes

---

## 5. 3D Grid Serialization

### Definition

The quantized 3D latent grid must be linearized into a 1D token sequence for the autoregressive transformer. GaussianGPT uses a fixed **$xyz$ column-wise traversal**: $z$ is the least significant (innermost) dimension, so the model iterates through each scene-height column at every $(x, y)$ position before moving to the next.

### Properties

- **Why $xyz$ over space-filling curves?** Despite weaker locality guarantees than Z-order or Hilbert curves, $xyz$ achieves the best training and validation cross-entropy (Tab.2). The 3D RoPE mechanism injects spatial relationships directly into attention, making the serialization order less critical for locality. Meanwhile, the regular structure of $xyz$ simplifies next-token prediction.
- The sequence grows cubically with scene size. To manage context length, the model operates on **spatial chunks** rather than entire scenes.
- Each voxel is assigned a **relative position index** within the current chunk (not absolute coordinates), enabling generalization across positions, scenes, and layouts.

### Application

Tab.2: Serialization strategy ablation (ASE + 3D-FRONT, 40 epochs):

| Ordering Strategy | Train CE | Val CE |
|---|---|---|
| Z-order | 2.379 | 2.448 |
| Trans. Z-order | 2.379 | 2.445 |
| Hilbert | 2.467 | 2.497 |
| Trans. Hilbert | 2.462 | 2.493 |
| **$xyz$** | **2.346** | **2.444** |

Hilbert curves consistently underperform. The simple $xyz$ ordering wins due to its regular structure and the 3D RoPE compensation for locality.

### Links

- -> **3D Rotary Positional Encoding**: compensates for the loss of 3D locality in the 1D sequence
- -> **Vocabulary Design**: the serialized sequence alternates position and feature tokens
- -> **Scene Generation and Completion**: defines the generation order at inference time

---

## 6. Vocabulary Design

### Definition

The interleaved token sequence alternates between two token types, each drawn from a separate vocabulary:

1. **Position tokens $p_i$**: predict the next occupied voxel index within the current chunk. Vocabulary size = number of possible voxel positions in the chunk.
2. **Feature tokens $f_i$**: predict the LFQ codebook index for the Gaussian features at the preceding position. Vocabulary size = 4,096 (codebook size).

The transformer uses a shared backbone but separate output heads for each token type. At each step, the model knows whether it should predict a position or a feature (due to the strict alternating pattern), and the cross-entropy loss is computed over the corresponding vocabulary only. Invalid vocabulary entries are masked.

### Properties

- **Decoupling geometry from appearance**: the position head handles spatial structure (where to place voxels), while the feature head handles appearance (what attributes to assign). This prevents competition between spatial and semantic indices over shared vocabulary slots.
- The chunk size and position vocabulary are independent of the feature codebook size — they can be varied independently.
- Special tokens: BOS (begin-of-sequence) starts generation; EOS (end-of-sequence) terminates it.

### Application

The alternating pattern creates sequences of the form:
```
BOS → p₁ → f₁ → p₂ → f₂ → ... → pₙ → fₙ → EOS
```

At each position step, the model outputs a distribution over voxel indices; at each feature step, it outputs a distribution over codebook entries. Already-generated positions can be masked to ensure the ordering constraint is respected.

### Links

- -> **3D Grid Serialization**: defines the ordering of position tokens
- -> **Lookup-Free Quantization**: produces the feature codebook indices
- -> **Autoregressive Training**: the cross-entropy loss alternates between position and feature vocabularies
- -> **3D Rotary Positional Encoding**: the 4th rotary dimension encodes token type (position vs feature)

---

## 7. 3D Rotary Positional Encoding (3D RoPE)

### Definition

Standard RoPE [58] encodes position along the 1D sequence index, which conflates sequence proximity with spatial proximity. In GaussianGPT, the token sequence is obtained by serializing a 3D grid, so voxels close in $(x,y,z)$ may be far apart in the 1D sequence (and vice versa).

3D RoPE injects the **actual 3D voxel coordinates** $(x,y,z)$ into the attention mechanism. The attention score becomes a function of the **relative spatial offset** between tokens rather than their sequence offset. This gives the transformer an explicit spatial inductive bias independent of the serialization order.

The encoding is extended to 4 dimensions: the 4th dimension encodes the **token type** (position vs feature), since both token types at the same voxel share the same $(x,y,z)$ coordinates. This helps the transformer further disentangle geometry from appearance within a single unified attention formulation.

### Properties

- Follows recent coordinate-aware rotary embedding work in 3D generation [69].
- Makes the model **serialization-order-agnostic** for spatial reasoning — the transformer can attend based on 3D proximity even though the sequence is in $xyz$ order.
- This is why simple $xyz$ serialization outperforms space-filling curves (Tab.2): 3D RoPE already handles locality, so the serialization order only needs to be simple and predictable.
- The 4th (token-type) dimension prevents the transformer from confusing a position token's role with the feature token at the same spatial location.

### Application

Applied inside the multi-head self-attention layers. The query-key dot product is modulated by the relative 3D+type offset rather than the relative 1D sequence offset.

### Links

- -> **3D Grid Serialization**: compensates for locality loss in the $xyz$ ordering
- -> **Vocabulary Design**: the 4th rotary dimension distinguishes position vs feature tokens
- -> **Transformer Architecture**: integrated into the attention mechanism

---

## 8. Transformer Architecture

### Definition

A decoder-only causal transformer based on **GPT-2** [51], built on the **nanochat** [31] codebase. Stacked multi-head self-attention and feed-forward blocks with the following modifications over vanilla GPT-2:

- **3D Rotary Positional Embeddings** (extended to 4D as described above)
- **Query-Key Normalization** [21]
- **Per-layer residual scaling** [30]
- **Muon optimizer** [29] for non-embedding/non-output-head parameters

No value embeddings or sliding-window attention (unlike the default nanochat config) [6,77].

### Properties

Two model scales:

| Config | Use | Context Window | Size |
|---|---|---|---|
| GPT-2 Medium | Scenes | 16,384 tokens | — |
| GPT-2 Small | Objects | 8,192 tokens | — |

- 16,384 tokens is sufficient to cover the worst-case fully occupied scene chunk.
- Temperature-controlled sampling: all results use temperature 0.9 with Nucleus Sampling [25] $p = 0.9$.

### Application

Tab.3: Per-module optimizer and learning rate configuration:

| Module | Optimizer | Learning Rate | Parameters |
|---|---|---|---|
| Output Heads | AdamW | 0.004 (scaled) | $\beta_1=0.8, \beta_2=0.95$ |
| Embeddings | AdamW | 0.2 (scaled) | $\beta_1=0.8, \beta_2=0.95$ |
| Residual Weights | AdamW | 0.005 | $\beta_1=0.8, \beta_2=0.95$ |
| Input Skip Weights | AdamW | 0.5 | $\beta_1=0.96, \beta_2=0.95$ |
| Others | Muon | 0.02 | $\beta_2=0.95, n=5$ |

- Learning rates for output heads and embeddings are scaled by embedding dimension; for scenes, divided by $\sqrt{1024/768} \approx 1.155$.
- Cosine learning rate decay to 10% of initial value.
- Training: 4 GH200 GPUs, effective batch size 64, ~1 day for scenes, ~4.5 hours for objects.
- Fine-tuning on 3D-FRONT: ~10 hours.
- Weight initialization: token embeddings from $\mathcal{N}(0,1)$, output heads from $\mathcal{N}(0, 10^{-3})$, attention projections/MLP inputs uniform $\pm\sqrt{3/d}$, attention/MLP outputs zero-initialized, residual weights 1.0, input skip weights 0.1.
- Nesterov momentum [59] increases linearly from 0.85 to 0.95 over first 300 iterations.
- AdamW uses $\epsilon = 10^{-10}$, no weight decay; Muon uses weight decay 0.025 (scenes) / 0.1 (objects), linearly decayed to 0.

### Links

- -> **3D Rotary Positional Encoding**: integrated into attention
- -> **Autoregressive Training**: trained with cross-entropy loss (Eq.2)
- -> **Vocabulary Design**: separate output heads for position and feature tokens
- -> **Scene Generation and Completion**: inference via autoregressive sampling

---

## 9. Autoregressive Training

### Definition

Given a tokenized scene sequence $\mathbf{t} = (t_1, \ldots, t_T)$, the transformer is trained to predict each token conditioned on all previous tokens using a standard autoregressive objective:

$$\mathcal{L}_{CE} = -\sum_{i=1}^{T} \log p_\theta(t_i \mid t_{<i}). \tag{Eq.2}$$

Training uses **teacher forcing**: ground-truth tokens are provided as input, and the model learns to predict the next token in the sequence.

### Properties

- The cross-entropy loss alternates between position vocabulary (at position steps) and feature vocabulary (at feature steps).
- Invalid vocabulary entries are masked: the model can only predict position tokens at position steps and feature tokens at feature steps.
- Training is performed on spatial chunks with fixed vertical positions to align floor heights across scenes and datasets.
- Chunks are randomly sampled during training, subject to minimum occupancy thresholds: 0.2 for autoencoder training, 0.3 for GPT training (latent grid is coarser).
- For ASE: supervision from rendered ground-truth Gaussians. For 3D-FRONT: supervision from previously rendered images.
- Chunk supervision is restricted: either by chunking GT Gaussians in the same manner or using projected depth maps to mask rendered images.

### Application

Camera sampling strategy:
- 3D-FRONT: sample $8\times$ required cameras, score by visible chunk area in depth maps, sample proportionally.
- ASE: heuristic based on projected chunk bounding box coverage (prefer cameras with $\geq 40\%$ coverage).

### Links

- -> **Transformer Architecture**: the model being trained
- -> **Vocabulary Design**: determines which vocabulary the loss is computed over at each step
- -> **Scene Generation and Completion**: inference uses the learned distribution

---

## 10. Scene Generation, Completion, and Outpainting

### Definition

At inference time, scenes are built autoregressively by sampling tokens from the learned distribution.

**Unconditional generation**: Start from BOS, alternate between predicting position and feature tokens until EOS is produced. The output sequence is decoded: position tokens map to voxel locations, feature tokens map to codebook entries, and the decoder $\mathcal{D}$ reconstructs Gaussian attributes.

**Scene completion**: Given a partial scene, serialize its observed voxels as a **prefix prompt**. The model continues generating from this prefix, naturally inferring missing geometry and appearance consistent with the observed context. No architectural changes needed.

**Large-scale outpainting**: Repeatedly apply completions using a **sliding window** of previously generated columns as context. The model generates new columns that extend the scene beyond the training chunk size. A resampling strategy retries empty columns up to 5 times to encourage non-trivial occupancy. Tree search can be used to resample columns without occupancy, creating more connected scenes.

### Properties

- Completion and generation are the same mechanism — completion just starts with a non-empty prefix.
- Multiple completions from the same partial input produce diverse but structurally consistent outputs (Fig.6, Fig.11).
- Position tokens can be masked to exclude already-generated locations, ensuring ordering constraints.
- The generation is **compositional**: the model makes sequential spatial decisions, each conditioned on the full history.
- Sampling: temperature 0.9, Nucleus Sampling $p = 0.9$.

### Application

Generation speed (on GH200 GPU):
- Single $4\text{m} \times 4\text{m}$ chunk: ~90 seconds (~4.4 columns/s).
- $12\text{m} \times 12\text{m}$ scene via outpainting: ~6,000 seconds (~0.6 columns/s due to sliding window overhead).
- Without resampling: approximately 2x faster.

Large scene generation procedure (3 stages):
1. **Seed**: generate an initial contiguous block via single-chunk sampling.
2. **Bootstrap**: expand along $x$-direction by sliding the window and generating 20-column slices.
3. **Outpainting loop**: grow the scene along $y$-direction until target size is reached.

Chunk window positioning at inference: predict columns with $y \in [5, 10)$ (provides context on both sides), aim for $x \approx 5$ to match the distribution of training chunks (which have an unknown section).

### Links

- -> **Autoregressive Training**: uses the learned next-token distribution
- -> **3D Grid Serialization**: defines generation order
- -> **Vocabulary Design**: position masking prevents duplicate voxel generation
- -> **Sparse 3D CNN Encoder-Decoder**: decoder $\mathcal{D}$ reconstructs Gaussians from generated tokens

---

## 11. What Existed Before and What This Paper Changes

### 11.1 Prior Approaches and Their Limitations

**Diffusion/flow-matching on 3D representations** (DiffRF [44], L3DG [52], DiffuScene [61], WorldExplorer [54]). These methods generate 3D scenes by iteratively denoising a latent or explicit 3D representation. DiffRF operates on radiance fields; L3DG introduced latent 3D Gaussian diffusion with a sparse 3D CNN autoencoder. They produce high visual fidelity. The problem: diffusion treats generation as a **global denoising process**, refining the entire scene simultaneously. This makes incremental editing, structured completion, and progressive scene extension unnatural — the model has no mechanism to condition on a partial scene prefix and continue building.

**Layout-then-object decomposition** (ATISS [49], SceneFormer [65], LayoutGPT [16]). These methods decompose scene generation into structured layout prediction followed by object retrieval or generation. They handle compositionality well but rely on explicit semantic decomposition (object categories, bounding boxes) and cannot directly model fine-grained geometry and appearance in a unified manner.

**2D/video-prior-based scene generation** (Text2Room [26], DreamScene360 [76], WorldGrow [37]). These methods use pre-trained 2D image or video generative models to synthesize novel views and lift them to 3D. They produce photorealistic results but depend on external 2D priors, multi-stage pipelines, or optimization-heavy procedures. They do not learn a native 3D generative model.

**Autoregressive 3D modeling (objects)** (MeshGPT [56], Octree Transformer [27], G3PT [74], Can3Tok [18], MAR-3D [10], VAR-3D [19]). These works apply autoregressive transformers to 3D objects — primarily meshes, octrees, or structured geometric tokens. They show that autoregressive modeling can produce high-quality 3D shapes. However, they focus on individual objects, not full scenes with multi-object composition, large spatial extent, and complex layouts.

### 11.2 What This Paper Contributes

**Contribution 1: Structured tokenization.** GaussianGPT defines a complete pipeline from continuous 3D Gaussian scenes to discrete token sequences: voxel grid -> sparse 3D CNN -> LFQ -> interleaved position/feature tokens. This tokenization preserves spatial structure and enables autoregressive modeling over scenes (not just objects).

**Contribution 2: Autoregressive scene generation framework.** A single GPT-2-based transformer with 3D RoPE generates, completes, and outpaints 3D Gaussian scenes via next-token prediction. Generation, completion, and outpainting are all the same mechanism (prefix conditioning). Temperature and nucleus sampling provide explicit control over diversity. The model can extend scenes beyond the training horizon via sliding-window outpainting.

### 11.3 Side-by-Side Comparison

| Dimension | L3DG [52] (diffusion) | ATISS [49] (layout AR) | MeshGPT [56] (object AR) | **GaussianGPT** |
|---|---|---|---|---|
| Generation paradigm | Diffusion (holistic denoising) | Autoregressive (layout) | Autoregressive (mesh) | **Autoregressive (Gaussians)** |
| Representation | Latent 3D Gaussians | Bounding boxes + retrieval | Triangle meshes | **Latent 3D Gaussians** |
| Scope | Full normalized scenes | Room layouts | Single objects | **Scene chunks + outpainting** |
| Completion | Not native | Partial (layout only) | Not native | **Native (prefix conditioning)** |
| Outpainting | No | No | No | **Yes (sliding window)** |
| Controllability | No temperature control | Limited | Temperature | **Temperature + nucleus sampling** |
| External priors | None | Object database | None | **None** |

| Dimension | L3DG [52] | **GaussianGPT** |
|---|---|---|
| Generative model | Diffusion | Autoregressive (GPT-2) |
| Quantization | VQ (standard codebook) | LFQ (sign-based, no lookup) |
| Positional encoding | Standard (1D or none) | 3D RoPE (4D: xyz + type) |
| Token structure | No explicit pos/feat split | Interleaved position + feature |
| Scene extent | Fixed (normalized) | Flexible (chunks + outpainting) |
| Objects: FID | 8.49 | **5.68** |
| Objects: KID (x$10^3$) | 3.147 | **1.835** |
| Objects: COV | 63.80 | **67.40** |
| Objects: MMD (x$10^3$) | **4.241** | 4.278 |

### 11.4 The Core Shift in Thinking

Diffusion-based 3D generation treats a scene as a single holistic entity to be denoised from noise. The scene's structure and content emerge simultaneously during the denoising process. This works well for bounded, normalized scenes but creates friction when you want to build a scene incrementally — add a room, extend a corridor, complete a partial observation.

GaussianGPT reframes 3D scene generation as **structured text generation**: a scene is a sequence of spatial decisions (where to place content) interleaved with appearance decisions (what to place there). Each decision is conditioned on all previous decisions through causal attention. This framing inherits the strengths of autoregressive language models: compositional generation, causal reasoning over context, explicit sampling control, and natural support for prefix conditioning.

The key technical insight that makes this possible is the combination of **structured voxel tokenization** (providing the discrete vocabulary) with **3D rotary positional encoding** (preserving spatial relationships despite 1D serialization). Without 3D RoPE, the transformer would need to learn 3D proximity from sequence proximity — which fails because $xyz$ serialization breaks spatial locality. Without structured tokenization, the transformer would need to model raw, unstructured Gaussian attributes — which is too high-dimensional and lacks the regularity that discrete tokens provide.

---

## 12. Quick Reference Card

| # | Insight | Design Choice | Evidence |
|---|---|---|---|
| 1 | 3D Gaussians lack canonical ordering for autoregressive modeling | Voxel grid + $xyz$ serialization + interleaved pos/feat tokens | Fig.2 pipeline overview |
| 2 | Standard VQ suffers from codebook collapse | LFQ (sign-based quantization, no lookup table) | Codebook size 4,096 with full utilization |
| 3 | 1D positional encoding conflates sequence and spatial proximity | 3D RoPE with 4th dimension for token type | Tab.2: $xyz$ + 3D RoPE beats space-filling curves |
| 4 | Space-filling curves (Hilbert, Z-order) do not improve over $xyz$ | Use simple $xyz$; let 3D RoPE handle locality | Tab.2: Hilbert Val CE 2.497 vs $xyz$ Val CE 2.444 |
| 5 | Diffusion generates holistically; cannot condition on partial prefixes | Autoregressive formulation: completion = generation with prefix | Fig.6: diverse completions from same partial input |
| 6 | Scene size is not fixed at training time | Sliding-window outpainting extends scenes indefinitely | Fig.5, Fig.8: 12m x 12m scenes from chunk-trained model |
| 7 | Position and feature tokens compete over shared vocabulary | Separate vocabularies + separate output heads | Decouples geometry from appearance |
| 8 | Autoregressive models enable explicit sampling control | Temperature 0.9 + Nucleus Sampling $p = 0.9$ | Controllable diversity in generation |
| 9 | AR 3D generation can match or beat diffusion on objects | Best FID, KID, COV on PhotoShape Chairs | Tab.1: FID 5.68 vs L3DG 8.49 |

---

## 13. Experimental Results

### Tab.1: Unconditional Shape Generation — PhotoShape Chairs (1000 shapes)

| Method | FID $\downarrow$ | KID ($\times 10^3$) $\downarrow$ | COV $\uparrow$ | MMD ($\times 10^3$) $\downarrow$ |
|---|---|---|---|---|
| $\pi$-GAN [8] | 52.71 | 13.64 | 39.92 | 7.387 |
| EG3D [7] | 16.54 | 8.412 | 47.55 | 5.619 |
| DiffRF [44] | 15.95 | 7.935 | 58.93 | 4.416 |
| L3DG [52] | 8.49 | 3.147 | 63.80 | **4.241** |
| **Ours** | **5.68** | **1.835** | **67.40** | 4.278 |

GaussianGPT achieves the best FID, KID, and COV. MMD is competitive (within 0.037 of L3DG). Rendering quality and sample diversity both exceed all baselines. Qualitative comparison in Fig.3 shows sharper structures and cleaner renderings compared to DiffRF and L3DG.

Metrics: FID and KID on $128 \times 128$ rendered views [4,22,46]. COV and MMD use Chamfer Distance on meshes extracted from generated Gaussians [1].

### Scene synthesis qualitative results

- **Unconditional scene generation** (Fig.4): Both L3DG and GaussianGPT produce coherent indoor layouts. L3DG generates full normalized scenes; GaussianGPT generates scene chunks. Fig.9 shows 3D-FRONT-only results; Fig.10 shows ASE-pretrained + 3D-FRONT fine-tuned results with improved diversity.
- **Large-scale outpainting** (Fig.5, Fig.8): 12m x 12m scenes via sliding-window autoregressive outpainting. Maintains floor alignment, consistent room layouts, and plausible structural continuation across chunk boundaries.
- **Scene completion** (Fig.6, Fig.11): Given 1/4 or 1/2 of a validation chunk, produces diverse and semantically plausible completions. Multiple samples from the same prefix show meaningful variation in layout and object configuration.
- **Real-world completion** (Fig.7): Fine-tuned on ScanNet++ v2 (895 scenes). Despite the small dataset, the model produces varied completions on real-world scans, though autoencoder fidelity is the limiting factor for high-frequency details.
- **Teaser** (Fig.1): Summarizes the three capabilities — generation, completion, and large-scale synthesis — from a single autoregressive model.

### Tab.2: Serialization Strategy Ablation

(Reproduced in Sec.5 above.)

### Tab.3: GPT Training Hyperparameters

(Reproduced in Sec.8 above.)

---

## 14. Open Questions

1. **Real-world data.** The current framework is validated on synthetic datasets (ASE, 3D-FRONT, PhotoShape). Extending to real-world data (ScanNet++) introduces autoencoder fidelity challenges: high-frequency details and partially observed regions degrade reconstruction quality (Appendix A).
2. **Uncertainty modeling.** Real-world scans contain missing and unobserved regions. Autoregressive formulations are well-suited for this — uncertain regions can be masked as discrete decisions, enabling probabilistic reasoning over partial observations.
3. **Long-horizon stability.** Outpainting via sliding window accumulates context over many steps. Improving stability and coherence over very long generation horizons is an open problem.
4. **Beyond fixed spatial chunks.** The current generation context is bounded by chunk size. Extending the context to cover larger areas without chunking could improve global coherence.

---

## 15. Concept Dependency Graph

```
  ┌─────────────────────────────┐
  │   3D Gaussian Scene (Input)  │
  └──────────┬──────────────────┘
             │
             ▼
  ┌─────────────────────────┐
  │  Sparse 3D Feature Grid  │
  │  (voxelize + encode)     │
  └──────────┬──────────────┘
             │
             ▼
  ┌─────────────────────────┐     ┌─────────────────────┐
  │  Sparse 3D CNN E/D       │────▶│  Autoencoder Loss    │
  │  (encoder ε, decoder D)  │     │  (Eq.1)              │
  └──────────┬──────────────┘     └─────────────────────┘
             │
             ▼
  ┌─────────────────────────┐
  │  Lookup-Free Quantization│
  │  (LFQ → codebook indices)│
  └──────────┬──────────────┘
             │
             ▼
  ┌─────────────────────────┐
  │  Discrete Latent Grid    │
  └──────────┬──────────────┘
             │
    ┌────────┼────────┐
    ▼        ▼        ▼
┌────────┐┌────────┐┌──────────┐
│  xyz   ││Vocab.  ││ 3D RoPE  │
│Serial. ││Design  ││ (4D)     │
└───┬────┘└───┬────┘└────┬─────┘
    │         │          │
    └─────────┼──────────┘
              ▼
    ┌──────────────────┐
    │  GPT-2 Causal     │
    │  Transformer      │◀── Autoregressive
    └────────┬─────────┘    Training (Eq.2)
             │
    ┌────────┼────────────┐
    ▼        ▼            ▼
┌────────┐┌──────────┐┌──────────┐
│Uncond. ││Completion ││Outpaint. │
│Gener.  ││(prefix)   ││(sliding  │
│        ││           ││ window)  │
└────────┘└──────────┘└──────────┘
```

---

## 16. Key Equations

| # | Equation | Description | Section |
|---|---|---|---|
| Eq.1 | $\mathcal{L} = \lambda_{RGB}\mathcal{L}_{RGB} + \lambda_{perc}\mathcal{L}_{perc} + \lambda_{occ}\mathcal{L}_{occ} + \lambda_{LFQ}\text{softplus}(\mathcal{L}_{LFQ}+5)$ | Autoencoder training loss | 3.1 |
| Eq.2 | $\mathcal{L}_{CE} = -\sum_{i=1}^{T} \log p_\theta(t_i \mid t_{<i})$ | Autoregressive cross-entropy loss | 3.2 |

---

## 17. Datasets

| Dataset | Type | Size | Resolution / Notes |
|---|---|---|---|
| Aria Synthetic Environments (ASE) [2] | Synthetic indoor scenes | 25,000 scenes | Via SceneSplat++ [42]; diverse layouts |
| 3D-FRONT [17] | Synthetic furnished rooms | 4,472 scenes (after filtering) | Gaussians via Scaffold-GS [41], 60k iters; 8x augmentation (rotation + flipping) |
| PhotoShape [48] | Chairs (objects) | 15,576 chairs | Normalized to $128^3$ grid; 200 rendered views each |
| ScanNet++ v2 [72] | Real-world indoor scans | 895 scenes | Fine-tuning only; Gaussians via SceneSplat++ |

---

## 18. Reference Map (Citations by Topic)

**3D Gaussian Splatting:**
[32] Kerbl et al. (2023) — original 3DGS; [41] Lu et al. (2023) — Scaffold-GS; [42] Ma et al. (2025) — SceneSplat++

**Latent 3D generation (Gaussians):**
[52] Roessle et al. (2024) — L3DG (direct baseline, diffusion); [69] Xiang et al. (2025) — native structured latents; [70] Xiang et al. (2025) — structured 3D latents

**Diffusion/flow 3D generation:**
[24] Ho et al. (2020) — DDPM; [39] Lipman et al. (2023) — flow matching; [44] Müller et al. (2023) — DiffRF; [55] Shim et al. (2023) — diffusion SDF; [61] Tang et al. (2024) — DiffuScene

**Autoregressive 3D (objects):**
[56] Siddiqui et al. (2023) — MeshGPT; [27] Ibing et al. (2021) — Octree Transformer; [74] Zhang et al. (2024) — G3PT; [18] Gao et al. (2025) — Can3Tok; [10] Chen et al. (2025) — MAR-3D; [19] Han et al. (2026) — VAR-3D; [20] Hao et al. (2024) — Meshtron

**Scene generation / completion:**
[49] Paschalidou et al. (2021) — ATISS; [65] Wang et al. (2021) — SceneFormer; [16] Feng et al. (2023) — LayoutGPT; [26] Höllein et al. (2023) — Text2Room; [54] Schneider et al. (2025) — WorldExplorer; [37] Li et al. (2025) — WorldGrow

**Transformer foundations:**
[6] Brown et al. (2020) — GPT-3; [51] Radford et al. (2019) — GPT-2; [64] Vaswani et al. (2017) — Attention; [58] Su et al. (2023) — RoPE; [31] Karpathy (2025) — nanochat; [29] Jordan (2024) — Muon optimizer

**Vector quantization:**
[73] Yu et al. (2024) — LFQ (tokenizer is key to visual generation)

**Datasets:**
[2] Avetisyan et al. (2024) — ASE/SceneScript; [17] Fu et al. (2021) — 3D-FRONT; [48] Park et al. (2018) — PhotoShape; [72] Yeshwanth et al. (2023) — ScanNet++

**Evaluation metrics:**
[1] Achlioptas et al. (2018) — COV/MMD; [4] Bikowski et al. (2021) — MMD-GANs; [22] Heusel et al. (2018) — FID; [46] Obukhov et al. (2020) — torch-fidelity
