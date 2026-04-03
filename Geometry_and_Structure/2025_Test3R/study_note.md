# Test3R: Learning to Reconstruct 3D at Test Time — Study Note (Concept Mind-Map)

> Paper: Test3R: Learning to Reconstruct 3D at Test Time (arXiv:2506.13750v1)
> Authors: Yuheng Yuan, Qiuhong Shen, Shizun Wang, Xingyi Yang, Xinchao Wang*
> Affiliations: National University of Singapore
> Code: https://github.com/nopQAQ/Test3R

This note organizes every key concept in the paper as a mind-map.
Each concept is broken down into four facets:

- **Definition** — what it is, stated plainly
- **Properties** — its mathematical or behavioral characteristics
- **Application** — how the paper (or the field) uses it
- **Links** — connections to other concepts in this map

---

## 0. The Big Picture

```
          Multi-view Images {I₁, I₂, ..., Iₙ}
                      │
                      ▼
             DUSt3R (pairwise prediction)
                      │
       "Pointmaps are imprecise and inconsistent"
       "Different source views → different pointmaps for same ref"
       "Limited generalization to unseen scenes"
                      │
                      ▼
         ┌────────────────────────────┐
         │  Test3R: Test-Time Training │
         │                            │
         │  1. Form image triplets    │
         │     (I_ref, I_src1, I_src2)│
         │                            │
         │  2. Self-supervised loss:   │
         │     ‖X₁ʳᵉᶠ - X₂ʳᵉᶠ‖ → 0   │
         │     (Eq.7)                 │
         │                            │
         │  3. Optimize only visual   │
         │     prompts (0.79M params) │
         └────────────┬───────────────┘
                      │
              ┌───────┼───────┐
              ▼       ▼       ▼
         Improved  Improved  Adapts to
         Precision Consist.  Unseen Scenes
              │       │       │
              └───────┼───────┘
                      ▼
           Global Alignment (Eq.6)
                      │
                      ▼
            High-Quality 3D Reconstruction
```

The paper's core claim: DUSt3R's pairwise prediction paradigm produces **imprecise** and **mutually inconsistent** pointmaps. Test3R fixes both problems with a single self-supervised objective: at test time, maximize the geometric consistency between pointmaps of the same reference view obtained from different source views. This is implemented through visual prompt tuning — only 0.79M parameters are optimized, keeping the 571M-parameter backbone frozen. The result: SOTA 3D reconstruction and multi-view depth estimation with ~30s of test-time training overhead.

---

## 1. DUSt3R Pairwise Prediction Paradigm

### Definition

DUSt3R [12] takes a pair of images $(I^{ref}, I^{src})$ and predicts dense **pointmaps** — 3D coordinates for every pixel of both images, expressed in the coordinate frame of the reference view $I^{ref}$.

The encoder is a weight-sharing ViT [57] with $N_e$ layers:

$$F^{ref} = \text{Encoder}(I^{ref}), \quad F^{src} = \text{Encoder}(I^{src}) \tag{Eq.1}$$

The decoder has $N_d$ layers where each branch attends to both branches via cross-attention:

$$G_i^{ref} = \text{DecoderBlock}_i^{ref}(G_{i-1}^{ref}, G_{i-1}^{src}) \tag{Eq.2}$$

$$G_i^{src} = \text{DecoderBlock}_i^{src}(G_{i-1}^{src}, G_{i-1}^{ref}) \tag{Eq.3}$$

with $G_0^{ref} = F^{ref}$ and $G_0^{src} = F^{src}$. Regression heads produce pointmaps and confidence maps:

$$X^{ref,ref}, C^{ref,ref} = \text{Head}^{ref}(G_0^{ref}, \ldots, G_{N_d}^{ref}) \tag{Eq.4}$$

$$X^{src,ref}, C^{src,ref} = \text{Head}^{src}(G_0^{src}, \ldots, G_{N_d}^{src}) \tag{Eq.5}$$

### Properties

- Each pointmap $X \in \mathbb{R}^{W \times H \times 3}$ represents per-pixel 3D coordinates in the reference view's frame.
- The model processes only **two images at a time** — no multi-view reasoning within a single forward pass.
- The confidence map $C$ estimates per-pixel reliability.
- Trained on large-scale datasets with ground-truth pointmaps derived from depth.

### Application

DUSt3R serves as the **backbone** for Test3R. The entire DUSt3R model (encoder + decoder + heads) is kept frozen during test-time training; only the visual prompts are optimized.

### Links

- -> **Cross-pair Inconsistency Problem**: the pairwise paradigm is the root cause of inconsistency
- -> **Global Alignment**: the post-hoc step that attempts to reconcile pairwise predictions
- -> **Visual Prompt Tuning**: the adaptation mechanism applied to the encoder

---

## 2. Cross-pair Inconsistency Problem

### Definition

When DUSt3R predicts pointmaps for the same reference view $I_1$ but paired with different source views $I_2$ and $I_3$, the resulting pointmaps $X_1^{ref,ref}$ (from pair $(I_1, I_2)$) and $X_2^{ref,ref}$ (from pair $(I_1, I_3)$) are **different** — even though they describe the same 3D scene from the same viewpoint.

This inconsistency manifests as:
1. **Imprecision**: each individual pointmap may have geometric errors, especially for short-baseline pairs where triangulation is poor.
2. **Inter-pair inconsistency**: pointmaps from different pairs do not agree, producing conflicting 3D geometry for the same pixels.

### Properties

- Visualized in Fig.2: pointmaps of $I_1$ with source views $I_2$ and $I_3$ are aligned using ICP and overlaid. Large regions show inconsistent colors (blue vs red), indicating spatial disagreement.
- The problem is **structural**: since DUSt3R estimates geometry from only two images at a time, the correspondences differ for each pair, and the resulting pointmaps diverge.
- The limited generalization capability of DUSt3R amplifies both issues on unseen or diverse scenes.
- Global alignment (Eq.6) cannot fully correct these errors — if the input pointmaps are inconsistent, the optimization starts from a poor initialization.

### Application

This problem is the motivation for Test3R's entire design. The triplet consistency objective (Eq.7) directly targets this inconsistency by forcing $X_1^{ref,ref} \approx X_2^{ref,ref}$.

Fig.6 shows the effect: DUSt3R depth maps from different source views have large inconsistencies in overlapping regions; Test3R depth maps are consistent.

### Links

- -> **DUSt3R Pairwise Prediction**: the paradigm that causes the inconsistency
- -> **Triplet Consistency Objective**: the self-supervised loss designed to eliminate inconsistency
- -> **Global Alignment**: downstream step that suffers when pointmaps are inconsistent

---

## 3. Triplet Consistency Objective

### Definition

The core contribution. Given a triplet of images $(I^{ref}, I^{src1}, I^{src2})$, Test3R forms two pairs: $(I^{ref}, I^{src1})$ and $(I^{ref}, I^{src2})$. Each pair is independently fed through the (prompt-augmented) DUSt3R model, producing two pointmaps for the reference view: $X_1^{ref,ref}$ and $X_2^{ref,ref}$.

The self-supervised training objective minimizes the difference between these two pointmaps:

$$\ell = \left\| X_1^{ref,ref} - X_2^{ref,ref} \right\| \tag{Eq.7}$$

### Properties

- **Self-supervised**: no ground-truth depth or 3D supervision is needed. The signal comes entirely from the requirement that the same reference view should produce the same 3D geometry regardless of which source view it is paired with.
- **Triplet sampling**: given $n$ images of a test scene, the total number of possible triplets is $n^3$. For computational efficiency, if the total exceeds 165, 165 triplets are randomly sampled.
- **Three-in-one solution**: (1) resolves inter-pair inconsistency by aligning predictions; (2) improves precision by averaging out errors from different baselines; (3) adapts the model to the specific test scene's distribution.
- The objective pushes the pointmaps closer to an overall global prediction, reducing the local errors of any single pairwise estimate.

### Application

- Training: 1 epoch over the sampled triplets per test scene.
- Optimizer: Adam [68].
- Learning rates (dataset-specific): 0.00001 (7Scenes), 0.00008 (NRGBD), 0.00004 (DTU), 0.00001 (ETH3D).
- Only the visual prompt parameters are updated (Eq.7 backpropagates through the prompts only).
- Time cost: ~30s of test-time training on a single RTX 4090 GPU.

### Links

- -> **Cross-pair Inconsistency Problem**: this objective directly solves the inconsistency
- -> **Visual Prompt Tuning**: the mechanism through which the objective modulates the model
- -> **Global Alignment**: benefits from more consistent pointmap inputs
- -> **Triplet Sampling Strategy**: determines which triplets are used for optimization

---

## 4. Visual Prompt Tuning (VPT) for Test-Time Training

### Definition

Instead of fine-tuning the full DUSt3R backbone at test time (571M parameters), Test3R inserts learnable **visual prompt tokens** into each Transformer layer of the encoder and optimizes only these prompts.

For a ViT encoder with $N_e$ layers, an image is first divided into patches and embedded as tokens $\mathbf{E}_0 = \{\mathbf{e}_0^k \in \mathbb{R}^D | k \in \mathbb{N}, 1 \leq k \leq N_t\}$. At each layer $i$, a set of learnable prompt tokens $\mathbf{P}_{i-1} = \{\mathbf{p}_{i-1}^k \in \mathbb{R}^D | k \in \mathbb{N}, 1 \leq k \leq N_p\}$ is prepended to the image tokens:

$$[\_, \mathbf{E}_i] = L_i(\langle \mathbf{P}_{i-1}, \mathbf{E}_{i-1} \rangle) \tag{Eq.8}$$

where $L_i$ is the $i$-th Transformer layer, $\langle \cdot, \cdot \rangle$ denotes concatenation, and $\_$ discards the prompt output positions. Only $\mathbf{P}_{i-1}$ are trainable; all other parameters remain frozen.

### Properties

- **Parameter efficiency**: only 0.79M trainable parameters (prompts) vs 571.17M total (DUSt3R encoder). This is 0.14% of the model.
- **Memory overhead**: 3M additional memory for prompts vs 2178.85M total → 2181.85M (+0.14%).
- **Two variants**:
  - **Test3R-S** (shallow): prompts inserted only in the first Transformer layer. Prompt tokens accompany image tokens through encoding but are then discarded.
  - **Test3R** (deep): distinct prompts at each layer. Better performance because feature distributions vary across layers, making layer-specific prompts more effective.
- **Overfitting prevention**: test-time objectives are noisy and self-supervised. Optimizing the full model would quickly overfit; prompts provide a controlled, low-dimensional adaptation space.
- Prompt length $N_p = 32$ per layer (default). Tab.4 ablates lengths 8, 16, 32, 64.

### Application

Tab.4: Prompt design ablation on 7Scenes:

| Variant | $N_p=8$ Acc | $N_p=8$ Comp | $N_p=16$ Acc | $N_p=16$ Comp | $N_p=32$ Acc | $N_p=32$ Comp | $N_p=64$ Acc | $N_p=64$ Comp |
|---|---|---|---|---|---|---|---|---|
| Test3R-S | 0.133 | 0.142 | 0.123 | 0.159 | 0.120 | 0.158 | 0.119 | 0.163 |
| Test3R | 0.118 | 0.131 | 0.122 | 0.155 | 0.105 | 0.136 | 0.149 | 0.170 |

- Increasing prompt length from 8 to 32 improves performance (more capacity).
- At $N_p = 64$, performance degrades — too many parameters become harder to optimize in 1 epoch of self-supervised training.
- Test3R (deep) outperforms Test3R-S (shallow) at optimal length ($N_p = 32$), but is more sensitive to over-parameterization ($N_p = 64$).

### Links

- -> **DUSt3R Pairwise Prediction**: prompts are inserted into the DUSt3R encoder
- -> **Triplet Consistency Objective**: the loss that drives prompt optimization
- -> **Test-Time Training**: VPT is the mechanism that makes TTT parameter-efficient

---

## 5. Test-Time Training (TTT)

### Definition

The general paradigm of adapting a pre-trained model to a specific test input using a self-supervised objective, without any labeled data. The model $f_s$ trained on source data $\mathcal{I}_s$ is adapted to test data $\mathcal{I}_t$ by minimizing a self-supervised objective $\ell$ at test time:

$$f_t: \mathcal{I}_t \to \hat{\mathcal{X}}_t$$

The key properties: (1) no ground-truth labels for the test data; (2) the self-supervised objective must be a reliable proxy for the actual task; (3) adaptation happens per test scene (transductive learning).

### Properties

- Originated from the transductive learning paradigm [39,40] (Vapnik, 1990s-2000s).
- TTT [46] (Sun et al., 2020) formalized the framework: use self-supervision to produce a different model for every test input, adapting to the target domain at test time.
- The challenge: the self-supervised objective must correlate well with downstream performance. If it doesn't, test-time training can hurt.
- In Test3R: the cross-pair consistency objective (Eq.7) is a strong proxy because geometric consistency is a necessary condition for correct 3D reconstruction.

### Application

Test3R applies TTT to 3D reconstruction for the first time. The adaptation is per-scene: for each test scene (set of images), the prompts are optimized from scratch for 1 epoch over the sampled triplets.

Tab.5: Overhead comparison (on Office-seq-09, 7Scenes):

| | Acc $\downarrow$ | Comp $\downarrow$ | NC $\uparrow$ | TTT Time | Total Time | Prompt Params | Total Params | Prompt Mem | Total Mem |
|---|---|---|---|---|---|---|---|---|---|
| DUSt3R | 0.62 | 0.51 | 0.54 | — | ~10s | — | 571.17M | — | 2178.85M |
| Test3R | 0.14 | 0.20 | 0.69 | ~30s | ~40s | 0.79M | 571.96M | 3M | 2181.85M |

Test-time training adds ~30s and 0.79M parameters. The accuracy improvement is massive: Acc drops from 0.62 to 0.14 (4.4x better), Comp from 0.51 to 0.20 (2.6x better).

### Links

- -> **Triplet Consistency Objective**: the self-supervised loss used for TTT
- -> **Visual Prompt Tuning**: the parameter-efficient mechanism for TTT adaptation
- -> **Cross-pair Inconsistency Problem**: TTT directly addresses this by scene-specific adaptation

---

## 6. Global Alignment

### Definition

After pairwise pointmap prediction, DUSt3R aligns all pointmaps into a single global coordinate system. Given $N_t$ images $\{I_i\}_{i=1}^{N_t}$, DUSt3R constructs a connectivity graph $\mathcal{G}(\mathcal{V}, \mathcal{E})$ where vertices are images and edges are image pairs. It estimates depth maps $D := \{\mathbf{D}_k\}$ and camera poses $\pi := \{\pi_k\}$ by minimizing:

$$\arg\min_{\mathbf{D}, \pi, \sigma} \sum_{e \in \mathcal{E}} \sum_{v \in e} \mathbf{C}_v^e \left\| \mathbf{D}_v - \sigma_e P_e(\pi_v, \mathbf{X}_v^e) \right\|_2^2 \tag{Eq.6}$$

where $\sigma_e$ are per-edge scale factors, $P_e(\pi_v, \mathbf{X}_v^e)$ projects the predicted pointmap to a depth map using pose $\pi_v$, and $\mathbf{C}_v^e$ are confidence weights.

### Properties

- This is a post-hoc optimization step, separate from the neural network.
- It tries to find globally consistent depth maps and poses by minimizing the discrepancy between predicted pointmaps and their depth-projected versions.
- **Garbage in, garbage out**: if the input pointmaps are inconsistent (the cross-pair problem), global alignment starts from a poor solution and cannot fully recover.
- Test3R improves global alignment indirectly by providing more consistent pointmaps as input.

### Application

The relationship between pointmaps and depth maps used in global alignment is (Eq.9, Appendix C):

$$X_{i,j} = K^{-1}[i D_{i,j},\; j D_{i,j},\; D_{i,j}] \tag{Eq.9}$$

where $(i,j)$ are pixel coordinates, $K \in \mathbb{R}^{3 \times 3}$ is the camera intrinsic matrix, and $D_{i,j}$ is the depth value.

### Links

- -> **DUSt3R Pairwise Prediction**: produces the pointmaps that global alignment reconciles
- -> **Cross-pair Inconsistency Problem**: inconsistent pointmaps degrade alignment quality
- -> **Triplet Consistency Objective**: improved consistency leads to better alignment

---

## 7. Framework Generalization

### Definition

Test3R is designed to be universally applicable to any model sharing the DUSt3R pairwise prediction pipeline. The paper validates this by applying Test3R to two additional models: **MAST3R** [13] and **MonST3R** [16].

### Properties

- MAST3R: a follow-up to DUSt3R focused on improving matching quality.
- MonST3R: extends DUSt3R to dynamic scenes (motion estimation).
- Both share the same encoder-decoder-head architecture, so visual prompts can be inserted into their encoders identically.

### Application

Tab.3: Generalization study on 7Scenes:

| Method | Acc $\downarrow$ Mean | Acc $\downarrow$ Med. | Comp $\downarrow$ Mean | Comp $\downarrow$ Med. | NC $\uparrow$ Mean | NC $\uparrow$ Med. |
|---|---|---|---|---|---|---|
| MAST3R [13] | 0.189 | 0.109 | 0.211 | 0.110 | 0.687 | 0.766 |
| MAST3R (w. Test3R) | 0.179 | 0.104 | 0.177 | 0.059 | 0.702 | 0.788 |
| MonST3R [16] | 0.240 | 0.180 | 0.268 | 0.167 | 0.672 | 0.758 |
| MonST3R (w. Test3R) | 0.218 | 0.167 | 0.251 | 0.160 | 0.687 | 0.775 |

Both models improve across all metrics when augmented with Test3R, confirming the framework's generality.

### Links

- -> **Visual Prompt Tuning**: the same VPT mechanism is applied to other models' encoders
- -> **Triplet Consistency Objective**: the same loss applies regardless of the backbone model

---

## 8. What Existed Before and What This Paper Changes

### 8.1 Prior Approaches and Their Limitations

**DUSt3R** [12] predicts dense pointmaps from image pairs end-to-end, replacing the traditional multi-stage pipeline (keypoints → matching → SfM → MVS). It works without camera parameters and achieves strong results on in-distribution data. The problem: pairwise prediction produces pointmaps that are **imprecise** (especially for short-baseline pairs) and **mutually inconsistent** (different source views yield different geometry for the same reference view). Global alignment partially mitigates this but cannot fully correct the errors.

**MAST3R** [13], **CUT3R** [35], **Spann3R** [31], **MonST3R** [16]. Follow-up works that improve DUSt3R's efficiency, quality, or applicability (e.g., to dynamic scenes). They all inherit the pairwise prediction paradigm and its associated consistency problems. CUT3R and Spann3R improve inference speed but do not address cross-pair consistency.

**Single forward-pass models** (Fast3R [38], VGGT [29]). These process multiple images in a single forward pass, which could inherently improve multi-view reasoning. Fast3R shows poor generalization to unseen scenes (Tab.6: Acc 0.377 on NRGBD vs Test3R's 0.083). VGGT performs well but requires massive computational resources for training and still underperforms Test3R on several metrics.

**Traditional MVS pipelines** (COLMAP [9,11], MVSNet [27]). Require camera intrinsics and/or poses. Produce geometrically consistent results when cameras are well-calibrated but cannot operate without camera parameters. Test3R achieves comparable or better results without any camera information (Tab.2).

### 8.2 What This Paper Contributes

**Contribution 1: Test-time training for 3D reconstruction.** Test3R is the first to apply test-time training to the DUSt3R family. The triplet consistency objective (Eq.7) provides a self-supervised signal that requires no ground truth and adapts the model to each specific test scene.

**Contribution 2: Universal applicability.** The approach works with any model sharing DUSt3R's pipeline (validated on MAST3R and MonST3R, Tab.3). The prompt-based adaptation adds only 0.79M parameters (0.14% of the model) and ~30s of overhead.

**Contribution 3: SOTA results.** Test3R achieves state-of-the-art on 3D reconstruction (7Scenes, NRGBD) and multi-view depth estimation (DTU, ETH3D), even surpassing methods that use camera poses and intrinsics (Tab.1, Tab.2).

### 8.3 Side-by-Side Comparison

| Dimension | DUSt3R [12] | CUT3R [35] | VGGT [29] | **Test3R** |
|---|---|---|---|---|
| Prediction paradigm | Pairwise | Pairwise (sequential) | Multi-view (single pass) | **Pairwise + TTT** |
| Camera params needed | No | No | No | **No** |
| Cross-pair consistency | No explicit mechanism | No | Implicit (multi-view attn) | **Explicit (Eq.7)** |
| Test-time adaptation | No | No | No | **Yes (prompts)** |
| Trainable params at test | 0 | 0 | 0 | **0.79M** |
| 7Scenes Acc (Mean) | 0.146 | 0.126 | — | **0.105** |
| NRGBD Acc (Mean) | 0.144 | 0.099 | 0.076 | **0.083** |
| DTU rel $\downarrow$ | 3.3 | — | — | **2.0** |

| Dimension | DUSt3R [12] (baseline) | **Test3R** |
|---|---|---|
| Acc 7Scenes (Mean/Med) | 0.146 / 0.078 | **0.105 / 0.051** |
| Comp 7Scenes (Mean/Med) | 0.181 / 0.067 | **0.136 / 0.035** |
| NC 7Scenes (Mean/Med) | 0.736 / 0.839 | **0.746 / 0.855** |
| Acc NRGBD (Mean/Med) | 0.144 / 0.019 | **0.083 / 0.021** |
| DTU rel $\downarrow$ | 3.3 | **2.0** |
| DTU $\tau$ $\uparrow$ | 69.9 | **84.1** |
| TTT overhead | 0s | ~30s |
| Extra params | 0 | 0.79M (0.14%) |

### 8.4 The Core Shift in Thinking

Previous works in the DUSt3R family treat 3D reconstruction as a **feed-forward prediction problem**: train a large model on diverse data, then apply it at test time without modification. The implicit assumption is that the model has seen enough during training to generalize to any test scene. When this assumption fails — on out-of-distribution scenes, unusual viewpoints, or challenging baselines — the model produces inconsistent, imprecise pointmaps, and there is no recourse.

Test3R introduces a different perspective: treat the test scene itself as a source of supervision. The cross-pair consistency objective requires no labels — it only asks that the model be self-consistent. If two pairs involving the same reference view produce different pointmaps, the model is wrong about at least one of them. By minimizing this discrepancy, the model is forced to converge on a single, more accurate reconstruction.

The use of visual prompt tuning makes this adaptation practical: the 571M-parameter backbone retains its learned knowledge while 0.79M prompt parameters absorb scene-specific adjustments. This is the right balance — enough capacity to adapt, not enough to overfit. The 30-second overhead is small relative to the reconstruction quality gains (4x improvement in accuracy on some scenes).

The deeper insight is that **geometric consistency is a free supervisory signal** for 3D reconstruction. Any method that predicts 3D geometry from multiple viewpoints can use cross-view consistency to self-improve, without needing any ground-truth data. This principle should extend beyond DUSt3R to any pairwise or multi-view 3D system.

---

## 9. Quick Reference Card

| # | Insight | Design Choice | Evidence |
|---|---|---|---|
| 1 | Pairwise prediction produces inconsistent pointmaps for the same reference view | Triplet consistency loss: $\|X_1^{ref,ref} - X_2^{ref,ref}\|$ (Eq.7) | Fig.2: large color disagreement in ICP-aligned pointmaps |
| 2 | Full backbone fine-tuning at test time causes overfitting with noisy self-supervision | Visual prompt tuning: freeze backbone, optimize only 0.79M prompt parameters | Tab.4: Test3R-S and Test3R both outperform DUSt3R |
| 3 | Layer-specific prompts outperform shared prompts | Test3R (deep) > Test3R-S (shallow) at $N_p = 32$ | Tab.4: Acc 0.105 vs 0.120 |
| 4 | Prompt length 32 is optimal | Longer prompts (64) degrade due to optimization difficulty in 1 epoch | Tab.4: Acc increases from 8→32, then drops at 64 |
| 5 | The approach generalizes to MAST3R and MonST3R | Same VPT + triplet loss applied to other backbones | Tab.3: improvements across all metrics |
| 6 | Test3R surpasses methods requiring camera parameters | Adapts to test scene distribution without any calibration | Tab.2: DTU rel 2.0 vs COLMAP 0.7, but Test3R needs no poses |
| 7 | Overhead is minimal: 30s, 0.79M params, 3MB memory | Prompt tuning is efficient by design | Tab.5: 0.14% parameter increase |
| 8 | Cross-pair consistency improves depth map agreement | Depth maps from different pairs converge after optimization | Fig.6: consistent depth maps across pairs |

---

## 10. Experimental Results

### Tab.1: 3D Reconstruction on 7Scenes and NRGBD

| Method | 7Scenes Acc$\downarrow$ Mean | Med. | Comp$\downarrow$ Mean | Med. | NC$\uparrow$ Mean | Med. | NRGBD Acc$\downarrow$ Mean | Med. | Comp$\downarrow$ Mean | Med. | NC$\uparrow$ Mean | Med. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MAST3R [13] | 0.189 | 0.109 | 0.211 | 0.110 | 0.687 | 0.766 | 0.085 | 0.033 | 0.063 | 0.028 | 0.794 | 0.928 |
| MonST3R [16] | 0.240 | 0.180 | 0.268 | 0.167 | 0.672 | 0.758 | 0.272 | 0.114 | 0.287 | 0.110 | 0.758 | 0.843 |
| Spann3R [31] | 0.298 | 0.226 | 0.205 | 0.112 | 0.650 | 0.730 | 0.416 | 0.323 | 0.417 | 0.285 | 0.684 | 0.789 |
| CUT3R [35] | **0.126** | **0.047** | 0.154 | **0.031** | 0.727 | 0.834 | **0.099** | **0.031** | **0.076** | **0.026** | 0.837 | **0.971** |
| DUSt3R [12] | 0.146 | 0.078 | 0.181 | 0.067 | 0.736 | 0.839 | 0.144 | 0.019 | 0.154 | 0.018 | 0.871 | 0.982 |
| **Ours** | 0.105 | 0.051 | **0.136** | 0.035 | **0.746** | **0.855** | 0.083 | 0.021 | 0.079 | 0.019 | **0.870** | **0.983** |

Test3R outperforms DUSt3R on all metrics for 7Scenes and most metrics for NRGBD. Only CUT3R and MAST3R beat it on some individual metrics. Qualitative comparisons in Fig.4 show that Test3R produces more accurate pointmaps than DUSt3R and CUT3R on Office-09 and Kitchen scenes from 7Scenes and NRGBD.

### Tab.2: Multi-view Depth Evaluation (DTU, ETH3D)

| Method | GT Pose | GT Range | GT Intrinsics | Align | DTU rel$\downarrow$ | DTU $\tau$$\uparrow$ | ETH3D rel$\downarrow$ | ETH3D $\tau$$\uparrow$ | AVG rel$\downarrow$ | AVG $\tau$$\uparrow$ |
|---|---|---|---|---|---|---|---|---|---|---|
| COLMAP [9,11] | $\checkmark$ | $\times$ | $\checkmark$ | $\times$ | 0.7 | 96.5 | 16.4 | 55.1 | 8.6 | 75.8 |
| MVSNet [27] | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$ | (1.8) | (86.0) | 35.4 | 31.4 | 18.6 | 58.7 |
| Robust MVD [63] | $\checkmark$ | $\times$ | $\checkmark$ | $\checkmark$ | 2.7 | 82.0 | 9.0 | 42.6 | 5.9 | 62.3 |
| DUSt3R [12] | $\times$ | $\times$ | $\times$ | med | 3.3 | 69.9 | 3.3 | 73.0 | 3.3 | 71.5 |
| **Test3R** | $\times$ | $\times$ | $\times$ | med | **2.0** | **84.1** | **3.2** | **74.0** | **2.6** | **79.1** |

Test3R achieves the best average performance among methods **not requiring camera parameters**. On DTU, it reduces relative error from 3.3 to 2.0 (39% improvement) and increases inlier ratio from 69.9 to 84.1. Qualitative depth comparisons in Fig.5 show that Test3R captures finer details (computer chassis, table edges on ETH3D) and better understands background regions (white-background DTU scenes) compared to DUSt3R and RobustMVD.

### Tab.6: Comparison with Single Forward-Based Models (NRGBD)

| Method | Acc$\downarrow$ Mean | Med. | Comp$\downarrow$ Mean | Med. | NC$\uparrow$ Mean | Med. |
|---|---|---|---|---|---|---|
| Fast3R [38] | 0.377 | 0.215 | 0.254 | 0.152 | 0.695 | 0.804 |
| VGGT [29] | **0.076** | **0.043** | **0.084** | **0.045** | **0.901** | **0.980** |
| DUSt3R [12] | 0.144 | 0.019 | 0.154 | 0.018 | 0.871 | 0.982 |
| Ours | 0.083 | 0.021 | 0.079 | 0.019 | 0.870 | 0.983 |

VGGT achieves the best accuracy on NRGBD but requires massive training resources. Test3R is competitive and surpasses VGGT on NC (median) and Comp (median).

---

## 11. Open Questions

1. **In-the-wild robustness.** Test3R still struggles with occlusions, dynamic objects, and varying illumination in unconstrained real-world data (Appendix D). Fig.1 (teaser) and Fig.7 (appendix) show that Test3R produces more detailed and consistent reconstructions than DUSt3R, with fewer outliers and better cross-view consistency — but challenges remain in ambiguous regions. Fig.3 provides the full pipeline overview showing how triplets flow through prompt-augmented encoders to produce consistent pointmaps.
2. **Source view pointmaps.** Currently, Test3R only considers pointmaps from reference views, discarding the source view pointmaps. Using camera poses to align source view pointmaps could provide additional supervision.
3. **Scalability.** Triplet count grows as $n^3$ with the number of views. For large view sets, efficient triplet sampling or alternative consistency formulations are needed.
4. **Beyond prompt tuning.** Other parameter-efficient adaptation methods (LoRA, adapters) could be explored as alternatives to visual prompts.

---

## 12. Concept Dependency Graph

```
  ┌──────────────────────────────────┐
  │  DUSt3R Pairwise Prediction      │
  │  (Eq.1–5: Encoder→Decoder→Head)  │
  └──────────┬───────────────────────┘
             │
             ├──────────────────────────────┐
             │                              │
             ▼                              ▼
  ┌──────────────────────┐     ┌─────────────────────────┐
  │ Cross-pair            │     │ Global Alignment         │
  │ Inconsistency Problem │     │ (Eq.6)                   │
  │ (Fig.2)               │     └────────────┬────────────┘
  └──────────┬────────────┘                  │
             │                               │
             ▼                               │
  ┌──────────────────────┐                   │
  │ Triplet Consistency   │                   │
  │ Objective (Eq.7)      │───────────────────┘
  └──────────┬────────────┘     (improved inputs)
             │
             ▼
  ┌──────────────────────┐     ┌─────────────────────┐
  │ Visual Prompt Tuning  │────▶│ Test-Time Training   │
  │ (Eq.8, 0.79M params)  │     │ (TTT paradigm)       │
  └──────────┬────────────┘     └─────────────────────┘
             │
             ▼
  ┌──────────────────────┐
  │ Framework General.    │
  │ (MAST3R, MonST3R)     │
  └───────────────────────┘
```

---

## 13. Key Equations

| # | Equation | Description | Section |
|---|---|---|---|
| Eq.1 | $F^{ref} = \text{Encoder}(I^{ref})$, $F^{src} = \text{Encoder}(I^{src})$ | ViT encoder feature extraction | 3 |
| Eq.2 | $G_i^{ref} = \text{DecoderBlock}_i^{ref}(G_{i-1}^{ref}, G_{i-1}^{src})$ | Decoder block (ref branch, cross-attention) | 3 |
| Eq.3 | $G_i^{src} = \text{DecoderBlock}_i^{src}(G_{i-1}^{src}, G_{i-1}^{ref})$ | Decoder block (src branch, cross-attention) | 3 |
| Eq.4 | $X^{ref,ref}, C^{ref,ref} = \text{Head}^{ref}(G_0^{ref}, \ldots, G_{N_d}^{ref})$ | Regression head output (ref) | 3 |
| Eq.5 | $X^{src,ref}, C^{src,ref} = \text{Head}^{src}(G_0^{src}, \ldots, G_{N_d}^{src})$ | Regression head output (src) | 3 |
| Eq.6 | $\arg\min_{\mathbf{D},\pi,\sigma} \sum_{e} \sum_{v} \mathbf{C}_v^e \|\mathbf{D}_v - \sigma_e P_e(\pi_v, \mathbf{X}_v^e)\|_2^2$ | **Global alignment** objective | 3 |
| Eq.7 | $\ell = \|X_1^{ref,ref} - X_2^{ref,ref}\|$ | **Triplet consistency** objective (core) | 4.2 |
| Eq.8 | $[\_, \mathbf{E}_i] = L_i(\langle \mathbf{P}_{i-1}, \mathbf{E}_{i-1} \rangle)$ | **Visual prompt tuning** formulation | 4.3 |
| Eq.9 | $X_{i,j} = K^{-1}[iD_{i,j}, jD_{i,j}, D_{i,j}]$ | Pointmap-to-depthmap conversion | C |

---

## 14. Datasets

| Dataset | Type | Size | Task | Views per Scene |
|---|---|---|---|---|
| 7Scenes [66] | Indoor (real) | 7 scenes | 3D Reconstruction | 3–5 views |
| NRGBD [67] | Indoor (real) | Multiple scenes | 3D Reconstruction | 2–4 views |
| DTU [58] | Object-centric (real) | 124 scenes | Multi-view Depth | Multiple |
| ETH3D [59] | Outdoor/indoor (real) | Multiple scenes | Multi-view Depth | Multiple |

---

## 15. Reference Map (Citations by Topic)

**DUSt3R family:**
[12] Wang et al. (2024) — DUSt3R (baseline); [13] Leroy et al. (2024) — MAST3R; [16] Zhang et al. (2024) — MonST3R; [35] Wang et al. (2025) — CUT3R; [31] Wang & Agapito (2024) — Spann3R; [29] Wang et al. (2025) — VGGT; [38] Yang et al. (2025) — Fast3R; [32] Jang et al. (2025) — Pow3R

**Traditional MVS / SfM:**
[9] Schönberger & Frahm (2016) — COLMAP SfM; [11] Schönberger et al. (2016) — COLMAP MVS; [17] Ullman (1979) — SfM theory; [27] Yao et al. (2018) — MVSNet; [10] Gu et al. (2020) — CasMVSNet

**Test-time training:**
[46] Sun et al. (2020) — TTT framework; [39] Gammerman et al. (2013) — transduction; [40] Vapnik (2006) — empirical learning; [47] Hansen et al. (2020) — self-supervised policy TTT; [49] Liu et al. (2021) — TTT++; [50] Yuan et al. (2023) — robust TTT

**Visual prompt tuning:**
[15] Jia et al. (2022) — VPT; [51] Mann et al. (2020) — language prompts; [54] Lester et al. (2021) — prompt tuning at scale; [56] Liu et al. (2021) — P-tuning v2; [60] Gao et al. (2022) — visual prompt tuning for domain adaptation

**Benchmarks:**
[58] Aanæs et al. (2016) — DTU; [59] Schöps et al. (2017) — ETH3D; [63] Schröppel et al. (2022) — RobustMVD; [66] Shotton et al. (2013) — 7Scenes; [67] Azinović et al. (2022) — NRGBD
