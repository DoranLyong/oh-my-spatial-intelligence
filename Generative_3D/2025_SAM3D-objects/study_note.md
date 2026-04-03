# SAM 3D: 3Dfy Anything in Images — Study Note (Concept Mind-Map)

> Paper: SAM 3D: 3Dfy Anything in Images
> Authors: SAM 3D Team (Meta Superintelligence Labs) — Xingyu Chen*, Fu-Jen Chu*, Pierre Gleize*, Kevin J Liang*, Alexander Sax*, Hao Tang*, et al.
> Project Leads: Piotr Dollar, Georgia Gkioxari, Matt Feiszli, Jitendra Malik
> Code: https://github.com/facebookresearch/sam-3d-objects
> Demo: https://www.aidemos.meta.com/segment-anything/editor/convert-image-to-3d

This note organizes every key concept in the paper as a mind-map.
Each concept is broken down into four facets:

- **Definition** — what it is, stated plainly
- **Properties** — its mathematical or behavioral characteristics
- **Application** — how the paper (or the field) uses it
- **Links** — connections to other concepts in this map

---

## 0. The Big Picture

```
              Single Image + Object Mask
                        │
           "Natural images are hard: occlusion,
            clutter, diverse objects, no 3D GT"
                        │
           ┌────────────┼────────────────┐
           ▼            ▼                ▼
    Synthetic Data   Data Engine    LLM-Style
    (Iso-3DO,        (MITL:         Training
     RP-3DO)         human+model)   Recipe
           │            │                │
           └────────┬───┘                │
                    ▼                    │
         ┌──────────────────────┐        │
         │  SAM 3D Architecture  │◀───────┘
         │                      │
         │  Stage 1: Geometry    │  (1.2B MoT)
         │  → coarse shape O    │  → layout (R,t,s)
         │  → voxel 64³         │
         │                      │
         │  Stage 2: Texture &   │  (600M sparse flow)
         │  Refinement           │  → mesh or Gaussians
         │  → fine shape + tex   │
         └──────────┬───────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   Per-Object    Full Scene   Layout
   3D Assets    Composition   (R,t,s)
```

SAM 3D is a **foundation model** for single-image 3D object reconstruction. Given a single image and an object mask, it predicts full 3D shape (including occluded parts), texture, and pose (rotation, translation, scale) relative to the camera. The key technical challenge is the **3D data barrier**: natural images paired with 3D ground truth are extremely difficult to obtain at scale. SAM 3D breaks this barrier with a human- and model-in-the-loop (MITL) data engine that collects ~1M real-world 3D annotations, combined with a multi-stage training recipe (synthetic pretraining → semi-synthetic mid-training → real-world post-training + DPO alignment) adapted from the LLM playbook. The result: 5:1 human preference win rate over prior SOTA on real-world objects and scenes.

---

## 1. Problem Formulation

### Definition

A photograph maps a 3D object to 2D pixels. SAM 3D inverts this map. Given an image $I$ and an object mask $M$, the object has shape $S$, texture $T$, rotation $R$, translation $t$, and scale $s$ in camera coordinates. Since the 3D-to-2D projection is lossy, the reconstruction problem is modeled as a **conditional distribution**:

$$p(S, T, R, t, s \mid I, M)$$

The goal is to train a generative model $q(S, T, R, t, s \mid I, M)$ that approximates $p$ as closely as possible.

### Properties

- The problem is **inherently ambiguous**: a single 2D image does not uniquely determine 3D shape (the back of an object is never observed).
- SAM 3D treats this as a **generative** problem (sample plausible shapes) rather than a deterministic regression problem.
- Shape $S$ is represented as coarse voxels ($64^3$), refined to mesh or 3D Gaussian splats.
- Texture $T$ is a per-vertex or per-point color attribute.
- Layout $(R, t, s)$ is predicted **jointly** with shape — this is a key differentiator from prior work that predicts shape and pose separately.
- The model handles **multi-object scenes**: each object is reconstructed independently, then composed using predicted layouts.

### Application

This formulation enables: (1) single-object reconstruction from natural images, (2) full scene decomposition into individual 3D assets, and (3) re-rendering from novel viewpoints.

### Links

- -> **Two-Stage Architecture**: the model $q$ that approximates $p$
- -> **Multi-Stage Training Recipe**: how $q$ is trained
- -> **SA-3DAO Benchmark**: evaluation of how well $q$ approximates $p$

---

## 2. Two-Stage Architecture

### Definition

SAM 3D uses a two-stage latent flow matching architecture (Fig.2):

**Stage 1: Geometry Model** — models $p(O, R, t, s \mid I, M)$. A 1.2B parameter flow transformer with the Mixture-of-Transformers (MoT) architecture. Conditioned on image and mask encodings, it jointly predicts coarse shape $O \in \mathbb{R}^{64^3}$ (voxels) and layout $(R, t, s)$.

**Stage 2: Texture & Refinement Model** — models $p(S, T \mid I, M, O)$. A 600M parameter sparse latent flow transformer. Takes the active voxels from Stage 1's coarse shape, refines geometric details, and synthesizes object texture.

**3D Decoders** — the latent representations from Stage 2 are decoded to either mesh or 3D Gaussian splats via a pair of VAE decoders $\mathcal{D}_m$, $\mathcal{D}_g$. These decoders share the same VAE encoder and structured latent space.

### Properties

```
            Input
              │
    ┌─────────┼──────────┐
    ▼         ▼          ▼
 Cropped   Full       Point Map
 Image+    Image+     (optional)
 Mask      Mask
    │         │          │
    ▼         ▼          ▼
  DINOv2    DINOv2    Point
  Encoder   Encoder   Encoder
    │         │          │
    └─────────┼──────────┘
              ▼
    ┌──────────────────────┐
    │  Geometry Model       │
    │  (1.2B MoT)           │
    │                       │
    │  Shape Decoder → O    │
    │  Layout Decoder → R,t,s│
    └──────────┬────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Texture & Refinement │
    │  (600M sparse flow)   │
    │                       │
    │  Mesh Decoder → mesh  │
    │  GS Decoder → splats  │
    └───────────────────────┘
```

- **Input encoding**: DINOv2 extracts features from two pairs of images: (1) cropped image + cropped binary mask, (2) full image + full image binary mask. This provides both high-resolution object detail and global scene context.
- **Optional point map conditioning**: SAM 3D can condition on a coarse scene point map $P$ (e.g., from LiDAR or monocular depth estimation), enabling integration with other pipelines.
- The MoT architecture uses a two-stream approach with information sharing in multi-modal self-attention. Shape tokens and layout tokens have separate QKV projections, FFNs, and cross-attention, but share the self-attention layer. A stop-gradient mask prevents layout tokens from influencing shape tokens during attention (Fig.2 right).
- **Distillation**: a final distillation stage reduces the number of function evaluations (NFE) from 25 to 4 for the Geometry model, enabling sub-second shape and layout prediction.

### Application

- Geometry model: 1.2B parameters, trained with rectified conditional flow matching.
- Texture & Refinement: 600M parameters, sparse latent flow transformer.
- 3D decoders: separately-trained VAE decoders sharing the same encoder/latent space.
- Inference: 4 NFE (after distillation) for geometry, producing results in sub-second time.

### Links

- -> **Mixture-of-Transformers (MoT)**: the architecture used in the Geometry model
- -> **Multi-Stage Training Recipe**: how each stage is trained
- -> **Problem Formulation**: implements the generative model $q$
- -> **Input Encoding**: DINOv2 features provide conditioning

---

## 3. Input Encoding

### Definition

SAM 3D uses DINOv2 to extract features from **two pairs** of images, producing 4 sets of conditioning tokens:

1. **Cropped object**: image $I$ cropped by mask $M$ + corresponding cropped binary mask → focused, high-resolution view of the target object.
2. **Full image**: full image $I$ + full image binary mask → global scene context and recognition cues absent from the crop.

Optionally, a **point map** $P$ (coarse 3D scene structure from LiDAR or monocular depth) can be encoded via a separate point encoder.

### Properties

- The dual-view design captures both local detail and global context — critical for handling occlusion, where the visible portion of an object may be small or ambiguous.
- DINOv2 features provide strong visual recognition capabilities, enabling the "familiar object" cue: recognizing what an object is helps predict its full 3D shape.
- Point map conditioning is optional — shape and texture quality do not depend on it (Section E.5), but layout estimation (translation/scale) requires ground-truth depth/pointmap for metric accuracy.

### Application

Both pairs are encoded independently by DINOv2 and concatenated before being fed to the Geometry model's MoT backbone.

### Links

- -> **Two-Stage Architecture**: provides conditioning tokens to the Geometry model
- -> **Mixture-of-Transformers (MoT)**: processes the concatenated conditioning tokens

---

## 4. Mixture-of-Transformers (MoT)

### Definition

The Geometry model uses the MoT architecture for jointly modeling shape and layout. MoT extends the standard transformer with a two-stream design: **shape tokens** and **layout tokens** have separate QKV projections, separate FFN blocks, and separate cross-attention layers, but share the **multi-modal self-attention** layer.

A stop-gradient attention mask ensures that layout token information flows into shape tokens, but **not** vice versa — shape tokens cannot attend to layout tokens. This prevents the layout prediction from interfering with shape generation.

### Properties

- 1.2B total parameters.
- The asymmetric attention mask (stop-gradient) is critical: without it, the two modalities would interfere, degrading both shape and layout quality.
- The MoT architecture is adapted from recent multimodal foundation models.
- Conditioned on input image/mask encodings and optional point map.
- Outputs: coarse voxel shape $O \in \mathbb{R}^{64^3}$ + 6D rotation $R \in \mathbb{R}^6$ + translation $t \in \mathbb{R}^3$ + scale $s \in \mathbb{R}^3$.
- Rotation uses the 6D continuous representation for stable training.

### Application

Trained with rectified conditional flow matching to generate multiple 3D modalities (shape + layout) simultaneously. After distillation, runs in 4 NFE.

### Links

- -> **Two-Stage Architecture**: MoT is the backbone of the Geometry model
- -> **Input Encoding**: receives DINOv2 conditioning tokens
- -> **Multi-Stage Training Recipe**: trained across all stages

---

## 5. Multi-Stage Training Recipe

### Definition

SAM 3D follows an LLM-inspired multi-stage training recipe (Fig.4, Tab.1):

**Step 1: Pretraining** — synthetic isolated 3D objects. Trains on renders of 2.7M meshes from Objaverse-XL and licensed datasets (24 viewpoints each). Learns shape vocabulary and texture priors. Dataset: *Iso-3DO*. Training budget: 2.5 trillion tokens.

**Step 1.5: Mid-Training** — semi-synthetic data. Teaches mask-following, occlusion handling, and layout estimation. Created by rendering textured meshes into natural images via alpha compositing ("render-paste"). Contains occluder-occludee pairs and object-swap pairs. Dataset: *RP-3DO* (61M samples, 2.8M unique meshes). Training budget: 2.7 trillion tokens.

**Step 2: Post-Training** — real-world alignment via SFT + DPO. Adapts the model from synthetic to real images using MITL-collected data (*MITL-3DO*) and artist-created data (*Art-3DO*). SFT uses the MITL and artist annotations; DPO uses $D^+/D^-$ preference pairs from the data engine. Final post-training budget: 0.5 trillion tokens.

### Properties

Tab.1: Training stages overview:

| Stage | Model | Modalities | Dataset | Conditioning |
|---|---|---|---|---|
| Pre-training | Geometry | $S, R$ | Iso-3DO | object-centric crop |
| Mid-training | Geometry | $S, R$ / $S, R, t, s$ | RP-3DO | full image, pointmap* |
| SFT | Geometry | $S, R, t, s$ | MITL, Art-3DO | full image, pointmap* |
| Alignment | Geometry | $S$ | MITL preference | full image, pointmap* |
| Pre-training | Texture | $T$ | Iso-3DO-500K | object-centric crop |
| Mid-training | Texture | $T$ | RP-3DO | full image |
| SFT | Texture | $T$ | MITL | full image |
| Alignment | Texture | $T$ | MITL preference | full image |

Tab.4: Cascading improvements on SA-3DAO and Preference Set:

| Training Stage | F1@0.01 | vIoU | Chamfer $\downarrow$ | EMD $\downarrow$ | Texture WR |
|---|---|---|---|---|---|
| Pre-training (Iso-3DO) | 0.1349 | 0.1202 | 0.1036 | 0.2396 | — |
| + Mid-training (RP-3DO) | 0.1705 | 0.1683 | 0.0760 | 0.1821 | 60.7 |
| + SFT (MITL-3DO) | 0.2027 | 0.2025 | 0.0578 | 0.1510 | 66.9 |
| + DPO (MITL-3DO) | 0.2156 | 0.2156 | 0.0498 | 0.1367 | 66.4 |
| + SFT (Art-3DO) | 0.2331 | **0.2337** | 0.0445 | 0.1257 | — |
| + DPO (Art-3DO) | **0.2344** | 0.2311 | **0.0400** | **0.1211** | — |

Each stage yields near-monotonic improvement. Art-3DO SFT suppresses common failures (floaters, bottomless meshes, missing symmetry). DPO eliminates remaining undesirable outputs.

### Application

- Pre- and mid-training use rectified conditional flow matching.
- Post-training uses SFT followed by DPO using $D^+/D^-$ pairs from the data engine.
- Distillation (25 → 4 NFE) is the final stage for the Geometry model.
- The entire recipe mirrors LLM training: pretraining builds broad capabilities, mid-training adds skills, post-training aligns with human preferences.

### Links

- -> **MITL Data Engine**: produces the post-training data
- -> **Two-Stage Architecture**: the model being trained
- -> **Problem Formulation**: each stage progressively improves $q$'s approximation of $p$
- -> **SA-3DAO Benchmark**: measures the impact of each stage

---

## 6. Model-in-the-Loop (MITL) Data Engine

### Definition

The core data innovation. A virtuous cycle of data collection and model improvement that produces visually grounded 3D reconstruction data at scale. The pipeline has 4 stages (Fig.5):

**Stage 1: Choosing target objects** $(I, M)$. Select diverse images and object masks from multiple real-world datasets (SA-1B, MetaCLIP, Ego4D, etc.). Use a 3D-oriented taxonomy of ~1,200 object categories to balance the distribution.

**Stage 2: Object model ranking and selection** $(S, T)$. Human annotators choose the best 3D shape and texture from $N = 8$ candidates (from an ensemble of computational and retrieval-based models). Annotators rate quality $r$ and reject candidates below threshold $\alpha$.

**Stage 2.5: Hard example triage (Artists)**. When no model produces a reasonable shape, professional 3D artists create the mesh directly. This produces the *Art-3DO* dataset — small but high-quality.

**Stage 3: Aligning objects to 2.5D scene** $(R, t, s)$. Annotators manipulate the selected 3D model's translation, rotation, and scale relative to a point cloud to match the object's pose in the image.

### Properties

- Total: ~1M images annotated with ~3.14M untextured meshes and ~100K textured meshes.
- The data engine is **iterative**: as the model improves, it becomes a better candidate generator, increasing the acceptance rate and quality. In early iterations, the retrieval ensemble dominates (~80% of candidates). Later, SAM 3D itself produces ~80% of accepted candidates (Fig.10b).
- Elo rating tracks improvement: near-linear scaling with data engine iterations (Fig.10a). 400-point Elo difference corresponds to 10:1 odds in a preference test.
- The cold-start problem is solved by using an ensemble of existing retrieval and learned models to bootstrap the first iteration.
- Negative examples ($D^-$) from rejected candidates are used for DPO training.

### Application

- The data engine is the key to SAM 3D's performance on real-world images. Models trained only on synthetic data struggle with in-the-wild objects.
- The accept/reject threshold $\alpha$ increases over time (similar to the cross-entropy method), progressively raising the quality bar.
- Fig.3 shows examples from each data type: Iso-3DO (isolated synthetic), RP-3DO (render-paste), MITL-3DO (real-world with model-selected shapes), Art-3DO (artist-created).

### Links

- -> **Multi-Stage Training Recipe**: MITL data feeds into post-training (SFT + DPO)
- -> **SA-3DAO Benchmark**: Art-3DO annotations form the evaluation set
- -> **Problem Formulation**: the data engine provides $(I, M, S, T, R, t, s)$ tuples

---

## 7. SA-3DAO Benchmark

### Definition

A new evaluation benchmark for real-world 3D object reconstruction: **SAM 3D Artist Objects (SA-3DAO)**. Contains 1,000 image-3D pairs where professional 3D artists create shapes from natural images. Objects range from churches, ski lifts, and large structures to animals, household items, and rare objects.

### Properties

- Artist-created meshes represent an **expert human upper bound** for visually grounded 3D reconstruction.
- No prior benchmark existed for real-world 3D reconstruction of objects in natural scenes with cluttered backgrounds and occlusion.
- Metrics: F1@0.01 (voxel overlap), vIoU (volumetric intersection-over-union), Chamfer distance, EMD (Earth Mover's Distance).
- ISO3D (from 3D Arena) is used as a complementary benchmark for isolated objects; it has no geometric GT, so perceptual metrics (ULIP, Uni3D) are used.
- Human preference evaluation uses curated sets (Pref Set from MetaCLIP/SA-1B/LVIS).

### Application

Tab.2: 3D shape comparison on SA-3DAO and ISO3D:

| Model | F1@0.01 $\uparrow$ | vIoU $\uparrow$ | Chamfer $\downarrow$ | EMD $\downarrow$ | ULIP $\uparrow$ | Uni3D $\uparrow$ |
|---|---|---|---|---|---|---|
| Trellis | 0.1475 | 0.1392 | 0.0902 | 0.2131 | 0.1473 | 0.3698 |
| HY3D-2.1 | 0.1399 | 0.1266 | 0.1126 | 0.2432 | 0.1293 | 0.3546 |
| HY3D-2.0 | 0.1574 | 0.1504 | 0.0866 | 0.2049 | 0.1484 | 0.3662 |
| Direct3D-S2 | 0.1513 | 0.1465 | 0.0962 | 0.2160 | 0.1405 | 0.3653 |
| TripoSG | 0.1533 | 0.1445 | 0.0844 | 0.2057 | **0.1529** | 0.3687 |
| Hi3DGen | 0.1629 | 0.1531 | 0.0937 | 0.2134 | 0.1419 | 0.3594 |
| **SAM 3D** | **0.2344** | **0.2311** | **0.0400** | **0.1211** | 0.1488 | **0.3707** |

SAM 3D outperforms all baselines on SA-3DAO by a large margin (F1 59% higher than next best, Chamfer 53% lower). On ISO3D (isolated objects without scene context), SAM 3D is competitive with the best methods. Qualitative comparisons in Fig.6 show SAM 3D produces shapes much closer to the artist-created ground truth than Trellis, HY3D, Direct3D-S2, TripoSG, and Hi3DGen, especially for complex real-world objects in cluttered scenes.

### Links

- -> **MITL Data Engine**: Art-3DO annotations form the benchmark's ground truth
- -> **Multi-Stage Training Recipe**: Tab.4 measures stage-by-stage gains on SA-3DAO
- -> **Problem Formulation**: evaluates how well $q$ approximates $p$

---

## 8. Scene-Level Reconstruction and Layout

### Definition

SAM 3D predicts not just individual object shapes but also their **layout** — rotation $R$, translation $t$, and scale $s$ in camera coordinates. This enables composing multiple independently-reconstructed objects into a coherent 3D scene (Fig.1, Fig.7).

### Properties

- Layout prediction is a new capability: prior methods either reconstruct isolated objects or require separate pose estimation pipelines.
- Tab.3 shows layout comparison on SA-3DAO and Aria Digital Twin (ADT):

| Generation | Model | SA-3DAO 3D IoU $\uparrow$ | SA-3DAO ADD-S $\downarrow$ | SA-3DAO ADD-S@0.1 $\uparrow$ | ADT 3D IoU $\uparrow$ | ADT ADD-S $\downarrow$ | ADT ADD-S@0.1 $\uparrow$ |
|---|---|---|---|---|---|---|---|
| Pipeline | Trellis + Megapose | 0.2449 | 0.5391 | 0.2831 | 0.2531 | 0.4358 | 0.1971 |
| Pipeline | HY3D-2.0 + FP | 0.2937 | 0.3705 | 0.5396 | 0.3864 | 0.1026 | 0.4129 |
| Pipeline | SAM 3D + FP | 0.2837 | 0.3848 | 0.5079 | 0.3661 | 0.0930 | 0.6495 |
| Joint | MIDI | — | — | — | 0.0336 | 2.5278 | 0.0175 |
| Joint | **SAM 3D** | **0.4254** | **0.2661** | **0.7232** | **0.4970** | **0.0765** | **0.7673** |

SAM 3D's joint prediction (shape + layout together) massively outperforms both pipeline approaches (separate shape + pose estimation) and joint baselines (MIDI).

- On ADT, ADD-S@0.1 improves from 0.12% (pipeline HY3D) to 77% (SAM 3D joint) — a step change in capability.

### Application

- Scene reconstruction: predict shape and layout for each detected object, compose them in a common 3D coordinate frame.
- Human preference: 6:1 win rate over prior SOTA for scene reconstruction (Fig.8).
- Fig.7 shows qualitative scene comparisons: SAM 3D produces more complete and geometrically accurate multi-object reconstructions than HY3D + FoundationPose and MIDI.

### Links

- -> **Two-Stage Architecture**: the Geometry model jointly predicts shape and layout
- -> **Mixture-of-Transformers (MoT)**: the architecture enabling joint shape/layout prediction
- -> **MITL Data Engine**: Stage 3 collects layout annotations

---

## 9. What Existed Before and What This Paper Changes

### 9.1 Prior Approaches and Their Limitations

**Single-view 3D reconstruction from synthetic data** (Trellis, HY3D, Direct3D-S2, TripoSG). These methods train on synthetic 3D datasets (ShapeNet, Objaverse) and reconstruct isolated objects from clean images. They achieve strong results on synthetic benchmarks but struggle with real-world images: occlusion, clutter, diverse objects, and unusual viewpoints cause significant degradation. On SA-3DAO (real-world objects in natural scenes), the best prior method (Hi3DGen) achieves F1@0.01 of only 0.1629 vs SAM 3D's 0.2344.

**Pipeline approaches** (Trellis/HY3D + Megapose/FoundationPose). These decompose scene reconstruction into separate shape prediction and pose estimation steps. The two-stage pipeline introduces error accumulation: if the shape is wrong, the pose estimator receives incorrect input. SAM 3D's joint prediction avoids this entirely.

**Joint scene generation** (MIDI). MIDI generates multi-object scenes jointly but struggles with layout accuracy (ADD-S@0.1 of 0.0175 on ADT vs SAM 3D's 0.7673). The gap is orders of magnitude.

**Real-world 3D data scarcity**. All prior methods are limited by the lack of real-world 3D training data. Existing datasets (CO3D, MVImgNet) are small and mostly indoor. Without real-world post-training data, models cannot generalize beyond synthetic distributions.

### 9.2 What This Paper Contributes

**Contribution 1: SAM 3D foundation model.** A generative model that predicts full 3D shape, texture, and layout from a single image, handling real-world complexity (occlusion, clutter, diverse objects). Joint shape+layout prediction is a new capability.

**Contribution 2: MITL data engine.** A human-and-model-in-the-loop pipeline that annotates ~1M real-world images with 3D shapes, textures, and poses — breaking the 3D data barrier at unprecedented scale.

**Contribution 3: LLM-style training recipe for 3D.** Multi-stage training (synthetic pretraining → semi-synthetic mid-training → real-world SFT → DPO alignment) adapts the LLM scaling paradigm to 3D reconstruction.

**Contribution 4: SA-3DAO benchmark.** 1K artist-created 3D shapes paired with natural images, providing the first real-world benchmark for visually grounded 3D reconstruction.

### 9.3 Side-by-Side Comparison

| Dimension | Trellis/HY3D (prior SOTA) | MIDI (joint) | **SAM 3D** |
|---|---|---|---|
| Input | Isolated object image | Scene image | **Scene image + mask** |
| Output | Shape + texture | Shape + layout | **Shape + texture + layout** |
| Training data | Synthetic only | Synthetic + limited real | **Synthetic + 1M real (MITL)** |
| Joint layout | No (needs separate pose est.) | Yes (poor accuracy) | **Yes (high accuracy)** |
| Real-world perf. (SA-3DAO F1) | 0.16 | — | **0.23** |
| Scene layout (ADT ADD-S@0.1) | 0.41 (pipeline) | 0.02 | **0.77** |
| Human preference | Baseline | — | **5:1 (objects), 6:1 (scenes)** |

### 9.4 The Core Shift in Thinking

Prior single-image 3D reconstruction methods treated the problem as a perception task: train a feed-forward model on synthetic data and apply it to real images. The implicit assumption was that synthetic data (Objaverse, ShapeNet) would transfer to real-world distribution. This assumption fails because synthetic objects lack the diversity, occlusion patterns, and contextual cues of natural images.

SAM 3D reframes the problem through the lens of **foundation model training**: the key insight is that real-world 3D annotations can be collected at scale, not by asking humans to create 3D models (impossible for non-experts), but by asking them to **select and rank** 3D model candidates produced by a mix of computational and retrieval-based methods. This is the same insight that made RLHF work for language models — humans are better judges than creators.

The MITL data engine creates a flywheel: better models produce better candidates, which lead to higher acceptance rates and better training data, which produce better models. The near-linear Elo scaling with data engine iterations (Fig.10a) confirms this flywheel effect. Combined with the LLM-style multi-stage training recipe (pretraining for capabilities, post-training for alignment), SAM 3D establishes 3D reconstruction as a foundation model problem, not just a computer vision one.

---

## 10. Quick Reference Card

| # | Insight | Design Choice | Evidence |
|---|---|---|---|
| 1 | Synthetic 3D data does not transfer well to real images | Multi-stage training: synthetic → semi-synthetic → real + DPO | Tab.4: each stage improves F1 from 0.13 to 0.23 |
| 2 | Humans can't create 3D meshes but can select the best one | MITL data engine: rank N=8 candidates, select best | ~1M annotated images, ~3.14M meshes |
| 3 | Shape and layout should be predicted jointly | MoT with stop-gradient: joint prediction, no interference | Tab.3: joint SAM 3D beats all pipeline approaches |
| 4 | DPO alignment suppresses failure modes | Use rejected candidates as $D^-$, accepted as $D^+$ | Tab.4: DPO adds +0.01 F1 after SFT |
| 5 | Render-paste creates useful semi-synthetic training data | Alpha-composite meshes into natural images for mid-training | Tab.4: mid-training adds +0.04 F1 |
| 6 | Art-3DO data is small but high-impact | Professional artists handle the hardest cases | Tab.4: Art SFT+DPO adds +0.02 F1 |
| 7 | Data engine iterations yield near-linear Elo gains | Iterative MITL: collect → train → collect again | Fig.10a: ~400 Elo points over 6 iterations |
| 8 | 5:1 preference win on objects, 6:1 on scenes | Human evaluation on diverse real-world sets | Fig.8: across SA-3DAO, LVIS, ADT, Pref Set |
| 9 | Distillation enables sub-second inference | 25 → 4 NFE via shortcut models | Section C.4 |

---

## 11. Open Questions

1. **Limitations in-the-wild.** Common failure modes include bottomless meshes, missing symmetry, and floaters — though DPO with Art-3DO data reduces these (Appendix F).
2. **Texture quality.** SAM 3D's texture is competitive (Fig.9) but the Texture & Refinement model is less detailed than the architecture section's description. Multi-view diffusion-based texture methods remain strong alternatives.
3. **Scalability of MITL.** The data engine requires human annotators for selection and pose alignment. Scaling to orders of magnitude more data may require further automation of quality assessment.
4. **Multi-view consistency.** SAM 3D reconstructs from a single image. Multi-view inputs could improve accuracy, especially for layout estimation where depth ambiguity is fundamental.
5. **Dynamic objects.** The current model assumes rigid objects. Extending to articulated or deformable objects is unexplored.

---

## 12. Concept Dependency Graph

```
  ┌─────────────────────────────┐
  │  Problem Formulation         │
  │  p(S,T,R,t,s | I,M)         │
  └──────────┬──────────────────┘
             │
    ┌────────┼────────────────┐
    ▼        ▼                ▼
┌────────┐┌──────────┐┌────────────┐
│ Input  ││ MoT      ││ Two-Stage  │
│Encoding││ Arch.    ││ Arch.      │
│(DINOv2)││ (1.2B)   ││ (Geom+Tex) │
└───┬────┘└────┬─────┘└─────┬──────┘
    └──────────┼────────────┘
               │
    ┌──────────┼──────────────┐
    ▼          ▼              ▼
┌────────┐ ┌────────────┐ ┌──────────┐
│Multi-  │ │ MITL Data   │ │ SA-3DAO  │
│Stage   │ │ Engine      │ │ Benchmark│
│Training│ │ (1M annot.) │ │ (1K GT)  │
└───┬────┘ └──────┬─────┘ └────┬─────┘
    └─────────────┼────────────┘
                  │
                  ▼
         ┌────────────────┐
         │ Scene-Level     │
         │ Reconstruction  │
         │ & Layout        │
         └─────────────────┘
```

---

## 13. Key Equations

This paper uses **problem formulation notation** rather than numbered equations:

| Concept | Formulation | Description |
|---|---|---|
| Problem | $p(S, T, R, t, s \mid I, M)$ | Conditional distribution of 3D attributes given image + mask |
| Goal | Train $q(S, T, R, t, s \mid I, M) \approx p$ | Generative model approximating the true distribution |
| Geometry Model | $p(O, R, t, s \mid I, M)$ | Coarse shape ($O \in \mathbb{R}^{64^3}$) + layout |
| Texture Model | $p(S, T \mid I, M, O)$ | Fine shape + texture conditioned on coarse geometry |
| Rotation | $R \in \mathbb{R}^6$ | 6D continuous rotation representation |
| Data engine API | $q(S, T, R, t, s \mid I, M) \to (D^+, r, D^-)$ | Produces training samples, quality score, and negative pairs |

---

## 14. Datasets

| Dataset | Type | Size | Use |
|---|---|---|---|
| Iso-3DO | Synthetic isolated objects | 2.7M meshes, 24 views each | Pre-training |
| RP-3DO | Semi-synthetic render-paste | 61M samples, 2.8M meshes | Mid-training |
| MITL-3DO | Real-world MITL-annotated | ~1M images, ~3.14M meshes | Post-training SFT |
| Art-3DO | Real-world artist-created | ~100K textured meshes | Post-training SFT + benchmark |
| SA-3DAO | Benchmark (artist GT) | 1K image-mesh pairs | Evaluation |
| ISO3D | Benchmark (isolated objects) | — | Evaluation (perceptual metrics) |
| ADT | Aria Digital Twin | — | Layout evaluation |

---

## 15. Reference Map (Citations by Topic)

**Single-view 3D reconstruction:**
Xiang et al. (2025) — Trellis (structured 3D latents); Yang et al. (2024b) — HY3D 1.0; Team (2025) — HY3D 2.0; Wu et al. (2025) — Direct3D-S2; Li et al. (2025) — TripoSG; Ye et al. (2025) — Hi3DGen

**Layout / pose estimation:**
Labbé et al. (2022) — Megapose; Wen et al. (2024) — FoundationPose; Huang et al. (2025) — MIDI

**Foundational architectures:**
Oquab et al. (2023) — DINOv2; Liang et al. (2025a) — MoT; Deng et al. (2025) — emerging multimodal pretraining; Peebles & Xie (2023) — DiT; Liu et al. (2022) — rectified flow matching

**3D datasets:**
Deitke et al. (2023) — Objaverse-XL; Chang et al. (2015) — ShapeNet; Pan et al. (2023) — ADT; Ebert (2025) — 3D Arena / ISO3D

**LLM training paradigm:**
Hernandez et al. (2021) — scaling laws for transfer; Rafailov et al. (2023) — DPO; Brown et al. (2020) — GPT-3; Ouyang et al. (2022) — InstructGPT/RLHF; Dong et al. (2023) — RAFT

**Classical 3D reconstruction:**
Roberts (1963) — recognition enables 3D; Koenderink et al. (1992) — surface perception from images; Debevec et al. (2023) — hybrid reconstruction; Gkioxari et al. (2019) — Mesh R-CNN

**Data annotation:**
Kirillov et al. (2023) — SAM; Anthony et al. (2017) — best-of-N search; Ouyang et al. (2022) — reward from human feedback
