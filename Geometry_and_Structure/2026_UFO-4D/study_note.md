# UFO-4D Study Note — A Concept Mind-Map

> Paper: UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images (arXiv:2602.24290v2)
> Authors: Junhwa Hur, Charles Herrmann, Songyou Peng, Philipp Henzler, Zeyu Ma, Todd Zickler, Deqing Sun
> Affiliations: Google, Princeton University, Harvard University
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
                    4D Scene Reconstruction from Two Images
                                    |
                "How to recover 3D geometry + 3D motion jointly?"
                                    |
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
     Test-time Optimization   Task-specific FF       Unified FF
     (MoSca, MoDGS)          (DynaDUSt3R,           (this paper)
     Hours per scene          MonST3R, ZeroMSF)
                              Separate tasks,
                              no motion-geometry
                              coupling
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  Core Insight:                 │
                    │  A single Dynamic 3D Gaussian  │
                    │  representation can render     │
                    │  image, depth, AND motion —    │
                    │  enabling self-supervision     │
                    │  that couples all modalities   │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              Differentiable   Semi-supervised   Feedforward
              4D Rasterizer    Loss (sup+self)   Pose Estimation
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                             UFO-4D Model
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              3D Geometry     3D Motion        4D Interpolation
              (depth, point)  (scene flow)     (novel view+time)
```

**The paper's story in one paragraph**: Existing feedforward methods for 4D reconstruction either produce fragmented per-task outputs (depth-only, flow-only) or rely on per-pixel representations that cannot be rendered. UFO-4D predicts a unified set of Dynamic 3D Gaussians from two unposed images in a single forward pass. Because these Gaussians are renderable, the model can use a self-supervised photometric loss that is independent of sparse ground truth. Because geometry and motion share the same Gaussian primitives, supervising one regularizes the other. This synergy yields >3x improvement on scene flow benchmarks and state-of-the-art geometry and pose estimation.

---

## 1. Dynamic 3D Gaussian Splatting (D-3DGS)

### Definition

Dynamic 3D Gaussian Splatting extends static 3D Gaussian Splatting by adding a per-Gaussian 3D motion vector. Each Gaussian primitive is a tuple:

$$(\boldsymbol{\mu}, \mathbf{v}, \mathbf{r}, \mathbf{s}, \mathbf{h}, o) \tag{Eq.1 element}$$

where:
- $\boldsymbol{\mu} \in \mathbb{R}^3$: 3D center position
- $\mathbf{v} \in \mathbb{R}^3$: 3D motion vector (velocity)
- $\mathbf{r} \in \mathbb{R}^4$: covariance rotation as quaternion
- $\mathbf{s} \in \mathbb{R}^3$: scale
- $\mathbf{h} \in \mathbb{R}^K$: view-dependent color via spherical harmonics
- $o \in [0,1]$: opacity

The full model maps a pair of unposed images to a set of dynamic Gaussians and a relative camera pose:

$$f_\theta(\mathbf{I}_t, \mathbf{I}_{t+1}) \mapsto (\mathcal{G}, \mathbf{P}), \quad \mathcal{G} = \{(\boldsymbol{\mu}, \mathbf{v}, \mathbf{r}, \mathbf{s}, \mathbf{h}, o)_\mathbf{p} \mid \mathbf{p} \in \mathcal{D}(\mathbf{I}_t) \cup \mathcal{D}(\mathbf{I}_{t+1})\} \tag{Eq.1}$$

where $\mathcal{D}(\mathbf{I})$ is the set of pixels in image $\mathbf{I}$.

### Properties

- **One Gaussian per pixel**: the model estimates one 3D Gaussian for each pixel from each input image. Gaussians from $\mathbf{I}_t$ have forward motion ($t \to t+1$); those from $\mathbf{I}_{t+1}$ have backward motion ($t+1 \to t$).
- **Canonical coordinate**: all Gaussians and the relative camera pose $\mathbf{P}$ are defined in the coordinate system of the first camera $\mathbf{I}_t$.
- **Alignment convention**: before rasterization, Gaussians from $\mathbf{I}_{t+1}$ are translated by their motion $(\boldsymbol{\mu} + \mathbf{v})$ and their motion vectors are inverted $(-\mathbf{v})$ to align with time $t$.
- **Density**: the union of Gaussians from both images forms a dense 4D scene representation covering all visible elements.
- **Linear motion assumption**: temporal evolution uses linear interpolation of position via the velocity vector, best suited for short time intervals.

### Application

- The D-3DGS representation is the central output of UFO-4D. All downstream tasks — depth, point maps, scene flow, optical flow, image synthesis, 4D interpolation — are derived from this single representation.
- Downstream 2D/3D task extraction:
  - **Depth**: last channel of the estimated 3D point
  - **Optical flow**: projection of 3D scene flow into 2D
  - **Moving object segmentation**: thresholding the scene flow magnitude

### Links

- → **Feedforward Architecture**: the ViT encoder-decoder that predicts D-3DGS parameters
- → **Differentiable 4D Rasterization**: D-3DGS primitives are rendered through the differentiable rasterizer to produce images, point maps, and flow maps
- → **Semi-Supervised Loss Framework**: supervision signals are applied both to raw Gaussian attributes (center, velocity) and to rasterized outputs
- → **Multi-Task Synergy**: all modalities share the same Gaussian primitives, creating cross-task regularization
- → **4D Spatio-Temporal Interpolation**: translating Gaussians by fractional motion enables continuous-time rendering
- → **Opacity as Learnable Confidence**: opacity $o$ emerges as a selection mechanism for which Gaussians contribute

---

## 2. Feedforward Architecture

### Definition

UFO-4D uses a ViT-based encoder-decoder architecture inspired by DUSt3R and NoPoSplat. Given two images $\mathbf{I}_t$, $\mathbf{I}_{t+1}$ and their camera intrinsics $\mathbf{K}$, the network outputs Dynamic 3D Gaussian parameters and a relative camera pose in a single forward pass.

```
Image I_t                                    Image I_{t+1}
    │                                             │
    ▼                                             ▼
┌──────────┐                              ┌──────────┐
│ ViT Enc. │ ◄──── weight-sharing ────►   │ ViT Enc. │
└────┬─────┘                              └────┬─────┘
     │                                         │
     ▼                                         ▼
  [tokens] ⊕ [intrinsic tok] ⊕ [pose tok]  [tokens] ⊕ [intrinsic tok] ⊕ [pose tok]
     │                                         │
     ▼                                         ▼
┌──────────┐      cross-attention         ┌──────────┐
│ ViT Dec. │ ◄────────────────────────►   │ ViT Dec. │
└────┬─────┘                              └────┬─────┘
     │                                         │
     ├──► Center head ──► μ                    ├──► Center head ──► μ
     ├──► Attributes head ──► r, s, h, o       ├──► Attributes head ──► r, s, h, o
     ├──► Velocity head ──► v                  ├──► Velocity head ──► v
     │                                         │
     └──► Pose head (3-layer MLP) ──► τ, q ◄──┘
```

### Properties

- **Weight-sharing encoder**: a ViT encoder (DINOv2-based) processes each image independently with shared weights.
- **Intrinsic token**: obtained by feeding camera intrinsics through a linear layer; commonly available from device metadata.
- **Pose token**: a learnable parameter optimized during training — does not require camera pose at inference time.
- **Cross-attention**: within each decoder attention block, cross-attention layers match and integrate information between the two input images.
- **DPT-based heads**: the center, attributes, and velocity heads use a DPT architecture (multi-scale feature fusion) with different output channel dimensions.
- **Pose head**: a 3-layer MLP that predicts relative camera pose as translation $\boldsymbol{\tau}$ and quaternion $\mathbf{q}$.
- **Initialization**: Gaussian head from NoPoSplat, rest from MASt3R, camera head from scratch.

### Application

- Single forward pass: two images in → Dynamic 3D Gaussians + relative pose out.
- The predicted pose is used at both training and inference time for rasterizing at time $t+1$, eliminating inference-time post-regression (PnP+RANSAC) used by MonST3R and St4RTrack.
- Direct feedforward pose estimation achieves ~16.6% higher accuracy than PnP+RANSAC baseline (Table G in Appendix).

### Links

- → **Dynamic 3D Gaussian Splatting**: the architecture's output representation
- → **Camera Pose Estimation**: the pose head and learnable pose token mechanism
- → **Differentiable 4D Rasterization**: the predicted Gaussians and pose feed into the rasterizer for rendering and supervision
- → **Prior Art (DUSt3R, NoPoSplat)**: the architecture builds directly on these foundations

---

## 3. Differentiable 4D Rasterization

### Definition

A unified, differentiable rendering pipeline that extends 3DGS rasterization to handle temporal evolution. It renders not only color images but also dense point maps and scene flow maps at any continuous time $t' \in [t, t+1]$.

**Temporal evolution**: the scene at intermediate time $t' = t + \Delta t$ (where $\Delta t \in [0,1]$) is modeled by translating each Gaussian along its velocity:

$$\mathcal{G}(t') = \{(\boldsymbol{\mu} + \Delta t \cdot \mathbf{v},\ \mathbf{v},\ \mathbf{r},\ \mathbf{s},\ \mathbf{h},\ \mathbf{c},\ o)_\mathbf{p}\} \tag{Eq.2}$$

**Image rendering**: standard alpha-blending of $\mathcal{N}'$-ordered depth-sorted Gaussians:

$$\hat{\mathbf{I}}_{t'}(\mathbf{p}) = \sum_{i \in \mathcal{N}_\mathbf{p}^{t'}} \mathbf{c}_i o_i \prod_{j=1}^{i-1}(1 - o_j) \tag{Eq.3}$$

where $\mathbf{c}_i$ is view-dependent color from spherical harmonics, $o_i$ is opacity from the projected 2D Gaussian.

**Point map rendering**: replaces color with 3D center position $\boldsymbol{\mu}_i$:

$$\mathbf{X}_{t'}(\mathbf{p}) = \sum_{i \in \mathcal{N}_\mathbf{p}^{t'}} \boldsymbol{\mu}_i o_i \prod_{j=1}^{i-1}(1 - o_j) \tag{Eq.4a}$$

**Scene flow rendering**: replaces color with velocity $\mathbf{v}_i$:

$$\mathbf{V}_{t'}(\mathbf{p}) = \sum_{i \in \mathcal{N}_\mathbf{p}^{t'}} \mathbf{v}_i o_i \prod_{j=1}^{i-1}(1 - o_j) \tag{Eq.4b}$$

### Properties

- **Full differentiability**: all rasterization outputs (image, point, flow) are differentiable with respect to all Gaussian parameters, enabling end-to-end gradient flow from any rendered output back through all prediction heads.
- **Temporal depth ordering**: depth orders of Gaussians can change with $\Delta t$, naturally handling occlusion and disocclusion at intermediate times.
- **Dense output**: unlike prior work that uses only sparse Gaussian centers, this approach produces dense, geometrically consistent maps by substituting intrinsic attributes in the alpha-blending formula.
- **Unified formulation**: Eqs. 3, 4a, 4b share the identical alpha-blending structure — only the per-Gaussian attribute being blended changes (color → position → velocity).

### Derivation: Image Rendering → Geometry/Motion Rendering

**Step 1.** Start from standard 3DGS image rendering (Eq. 3) which alpha-blends color $\mathbf{c}_i$.

**Step 2.** Observe that the blending weights $w_i = o_i \prod_{j=1}^{i-1}(1-o_j)$ are independent of what attribute is being blended.

**Step 3.** Replace color $\mathbf{c}_i$ with 3D center $\boldsymbol{\mu}_i$ → Eq. 4a (point map).

**Step 4.** Replace color $\mathbf{c}_i$ with velocity $\mathbf{v}_i$ → Eq. 4b (flow map).

**Key consequence**: supervision signals from rendered images, point maps, and flow maps all backpropagate through the same rasterizer, enabling joint optimization of all Gaussian heads.

### Application

- Used for both model training (computing self-supervised photometric loss) and downstream perception tasks (extracting depth, flow).
- Enables 4D interpolation by setting $\Delta t$ to any value in $[0,1]$ and rasterizing from any camera viewpoint.
- Rasterized point and motion outputs are used to compute the smoothness loss (Eq. 7c).

### Links

- → **Dynamic 3D Gaussian Splatting**: the primitives being rasterized
- → **Feedforward Architecture**: the prediction heads that produce the Gaussian parameters fed into this rasterizer
- → **Semi-Supervised Loss Framework**: all loss terms operate on rasterized outputs or raw Gaussian attributes
- → **4D Spatio-Temporal Interpolation**: rasterization at arbitrary $\Delta t$ enables continuous-time rendering
- → **Multi-Task Synergy**: the shared alpha-blending weights couple image, geometry, and motion gradients

---

## 4. Semi-Supervised Loss Framework

### Definition

The total training loss combines a supervised component using (sparse) ground-truth annotations and a self-supervised component from differentiable rendering:

$$L_\text{total} = L_\text{sup} + L_\text{self} \tag{Eq.5}$$

This semi-supervised design addresses the scarcity of dense, accurately annotated 4D training data.

### Properties

- **Complementary signals**: supervised losses provide direct geometric accuracy; self-supervised losses provide dense photometric consistency.
- **Robustness to sparse annotations**: the self-supervised loss fills in where ground truth is missing or noisy (e.g., Stereo4D has noisy motion annotations for static regions).
- **Gradient coupling**: because geometry, motion, and appearance share Gaussian primitives, gradients from the photometric loss propagate to geometry and motion heads.

### Links

- → **Supervised Loss**: direct regression on Gaussian centers, velocities, and pose
- → **Self-Supervised Loss**: photometric reconstruction + smoothness
- → **Multi-Task Synergy**: the loss framework is the mechanism through which synergy operates
- → **Dynamic 3D Gaussian Splatting**: losses are applied to both raw attributes and rasterized outputs

---

### 4.1 Supervised Loss

#### Definition

Penalizes differences between predictions and ground truth for motion, point, and pose:

$$L_\text{sup} = L_\text{motion} + w_\text{point} L_\text{point} + w_\text{pose} L_\text{pose} \tag{Eq.6a}$$

$$L_\text{motion} = \sum_{u \in \{t, t+1\}} \frac{1}{|\mathbf{V}_u^\text{GT}|} \big(||\mathbf{v}_u - \mathbf{V}_u^\text{GT}|| + ||\mathbf{V}_u - \mathbf{V}_u^\text{GT}||\big) \tag{Eq.6b}$$

$$L_\text{point} = \sum_{u \in \{t, t+1\}} \frac{1}{|\mathbf{X}_u^\text{GT}|} \big(||\boldsymbol{\mu}_u - \mathbf{X}_u^\text{GT}|| + ||\mathbf{X}_u - \mathbf{X}_u^\text{GT}||\big) \tag{Eq.6c}$$

$$L_\text{pose} = ||\mathbf{q} - \mathbf{q}_\text{GT}|| + ||\boldsymbol{\tau} - \boldsymbol{\tau}_\text{GT}|| \tag{Eq.6d}$$

#### Properties

- **Dual supervision**: $L_\text{motion}$ and $L_\text{point}$ are applied to both raw Gaussian attributes ($\mathbf{v}_u$, $\boldsymbol{\mu}_u$) and rasterized outputs ($\mathbf{V}_u$, $\mathbf{X}_u$).
- **Valid-pixel normalization**: normalized by the number of valid ground truth entries $|\mathbf{V}_u^\text{GT}|$ and $|\mathbf{X}_u^\text{GT}|$, handling sparse annotations.
- **Separate pose terms**: translation and quaternion are supervised independently.
- **Both timesteps**: losses are computed at both $t$ and $t+1$.

#### Application

- Applied to datasets with sparse annotations (Stereo4D, PointOdyssey, Virtual KITTI 2).
- Only computed on pixels with valid ground truth.

#### Links

- → **Self-Supervised Loss**: complements supervised loss where ground truth is sparse or noisy
- → **Camera Pose Estimation**: $L_\text{pose}$ directly supervises the pose head
- → **Differentiable 4D Rasterization**: rasterized outputs $\mathbf{V}_u$ and $\mathbf{X}_u$ are produced by the rasterizer

---

### 4.2 Self-Supervised Photometric Loss

#### Definition

Measures the reconstruction quality between input images and rasterized images, plus a smoothness regularizer:

$$L_\text{self} = L_\text{photo} + w_\text{smooth} L_\text{smooth} \tag{Eq.7a}$$

$$L_\text{photo} = \sum_{u \in \{t, t+1\}} \big(\text{mse}(\hat{\mathbf{I}}_u, \mathbf{I}_u) + w_\text{lpips} \text{lpips}(\hat{\mathbf{I}}_u, \mathbf{I}_u)\big) \tag{Eq.7b}$$

$$L_\text{smooth} = \sum_{u \in \{t, t+1\}} \sum_{\mathbf{d} \in \{\mathbf{X}, \mathbf{V}\}} \big(|\partial_x \mathbf{d}_u| e^{-|\partial_x \mathbf{I}_u|} + |\partial_y \mathbf{d}_u| e^{-|\partial_y \mathbf{I}_u|}\big) \tag{Eq.7c}$$

#### Properties

- **Photometric loss**: combines pixel-wise MSE with perceptual LPIPS loss for both structural and perceptual fidelity.
- **Edge-aware smoothness**: encourages spatial smoothness on rasterized point maps $\mathbf{X}$ and motion maps $\mathbf{V}$, weighted by image gradients — allows sharp discontinuities at edges while smoothing homogeneous regions.
- **Annotation-free**: does not require any ground truth — only the input images serve as supervision targets.
- **Gradient backpropagation**: crucially, the photometric loss gradients flow through the rasterizer back to Gaussian center and velocity heads (not just color), providing geometry and motion supervision from image reconstruction.

#### Application

- The photometric loss is the mechanism that enables training with limited ground truth. By rendering what the model predicts and comparing to the input, the model self-corrects.
- The smoothness loss cleans up floaters in rasterized point and motion outputs.
- Ablation (Table 4, row b): removing image gradient flow to center/velocity causes point and motion accuracy to degrade (Point rast. EPE: 0.841 → 0.899).
- Ablation (Table 4, row c): removing motion and point rendering losses entirely degrades all tasks (Point rast. EPE: 0.841 → 0.943).

#### Links

- → **Differentiable 4D Rasterization**: the photometric loss requires differentiable rendering of images
- → **Supervised Loss**: photometric loss complements sparse supervised loss
- → **Multi-Task Synergy**: photometric gradients propagate to geometry and motion heads through shared primitives
- → **Opacity as Learnable Confidence**: photometric loss drives opacity to learn which Gaussians matter

---

## 5. Multi-Task Synergy through Shared Primitives

### Definition

The core insight of UFO-4D: because 3D geometry (points) and 3D motion (scene flow) originate from the same set of Gaussian primitives, their respective loss signals inherently regularize each other. Supervising one modality improves the other, creating a synergistic training dynamic.

### Properties

- **Shared bottleneck**: all modalities — image, depth, point, motion — are derived from the same Gaussian parameters ($\boldsymbol{\mu}$, $\mathbf{v}$, $\mathbf{r}$, $\mathbf{s}$, $\mathbf{h}$, $o$). Improving one parameter for one task propagates benefits to others.
- **Photometric → geometric**: the photometric loss (Eq. 7b) provides dense supervision to Gaussian centers and velocities even without geometric ground truth, because rendering requires correct geometry to produce accurate images.
- **Cross-modal regularization**: imperfections in one supervisory signal are compensated by another. Noisy motion ground truth in Stereo4D is regularized by the clean photometric signal.
- **Emergent capacity sharing**: Table 4 row (d) shows that a model without rendering losses (only estimating per-pixel 3D point and motion) underperforms the full model, despite the full model sharing network capacity across more tasks. The rendering-based supervision more than compensates for the shared capacity.
- **Quantitative evidence**: UFO-4D achieves >3× lower motion EPE on Stereo4D (0.049 vs 0.164) and KITTI (0.137 vs 0.442) compared to the next-best method.

### Application

- This synergy is the primary reason UFO-4D outperforms task-specific methods that estimate geometry and motion separately.
- The representational bottleneck (everything goes through Gaussian primitives) is what makes the synergy possible — per-pixel representations cannot achieve this coupling.

### Links

- → **Dynamic 3D Gaussian Splatting**: the representation that enables shared primitives
- → **Semi-Supervised Loss Framework**: the mechanism through which synergy operates
- → **Differentiable 4D Rasterization**: the rendering pipeline that couples all modalities through shared alpha-blending weights
- → **Architecture Comparison**: Table 5 shows D-3DGS representation outperforms per-pixel alternatives

---

## 6. Camera Pose Estimation

### Definition

UFO-4D directly estimates relative camera pose between the two input frames via a feedforward pose head, eliminating the need for iterative PnP+RANSAC post-processing used by prior methods.

The pose head is a 3-layer MLP that takes decoder features and predicts:
- Translation $\boldsymbol{\tau} \in \mathbb{R}^3$
- Rotation as quaternion $\mathbf{q} \in \mathbb{R}^4$

Together they form the relative camera pose $\mathbf{P} = (\boldsymbol{\tau}, \mathbf{q})$.

### Properties

- **Learnable pose token**: a learned embedding concatenated with image tokens and intrinsic tokens before the decoder. This token attends to image features via cross-attention to extract pose-relevant information.
- **Attention pattern**: cross-attention maps (Fig. G) show the pose token attends primarily to static regions (high attention on background, low on moving objects), effectively learning to suppress dynamic content for pose estimation.
- **Feedforward vs. iterative**: direct estimation is ~16.6% more accurate than PnP+RANSAC applied to the same predicted Gaussians (Table G), and is more robust to noise.
- **No inference-time pose input**: unlike DUSt3R-family methods that require post-hoc pose recovery, UFO-4D produces pose as a native output.

### Application

- The estimated pose is used during both training and inference to rasterize the scene at time $t+1$ from the second camera viewpoint.
- Pose accuracy on Stereo4D test: ATE=0.0101, RPE_trans=0.0142, RPE_rot=0.179 (Table 3).
- Outperforms MonST3R and St4RTrack on all pose metrics across Stereo4D, Bonn, and Sintel.

### Links

- → **Feedforward Architecture**: the pose head is one of four prediction heads
- → **Supervised Loss**: $L_\text{pose}$ (Eq. 6d) directly supervises pose estimation
- → **Differentiable 4D Rasterization**: the estimated pose determines the camera viewpoint for rendering at $t+1$
- → **Multi-Task Synergy**: accurate pose enables better photometric supervision, which in turn improves geometry and motion

---

## 7. Opacity as Learnable Confidence

### Definition

An emergent property of the model: the opacity parameter $o$ of each Gaussian learns to function as a confidence score, determining how much each Gaussian contributes to rendered outputs.

### Properties

- **Occlusion handling**: in regions visible in both views, the model assigns high opacity to only one of the two corresponding Gaussians (one from each image), compactly representing the scene without redundancy.
- **Disocclusion marking**: regions that are occluded in one view but visible in the other are assigned high opacity, correctly marking these as important for the 4D representation.
- **Not explicitly trained**: this behavior emerges from the combination of the total loss (Eq. 5) and the alpha-blending formulation — no explicit confidence supervision is provided.
- **Compact representation**: by suppressing low-opacity Gaussians, the model effectively selects a compact subset of primitives from the two-image union.

### Application

- Visualized in Fig. 5: blue (high opacity) regions correspond to the selected Gaussian source (either from $\mathbf{I}_t$ or $\mathbf{I}_{t+1}$), enabling efficient 4D scene representation.
- Enables the model to handle occlusion and disocclusion without explicit occlusion reasoning.

### Links

- → **Dynamic 3D Gaussian Splatting**: opacity is a parameter of each Gaussian primitive
- → **Differentiable 4D Rasterization**: opacity controls the alpha-blending weights
- → **Semi-Supervised Loss Framework**: the loss landscape drives opacity to emerge as confidence

---

## 8. 4D Spatio-Temporal Interpolation

### Definition

A novel application enabled by the D-3DGS representation: given the estimated set of dynamic 3D Gaussians, UFO-4D can render image, point, and motion maps at any camera viewpoint and any continuous time $t' = t + \Delta t$ between the two input frames.

The interpolated scene is:

$$\mathcal{G}(t') = \{(\boldsymbol{\mu} + \Delta t \cdot \mathbf{v},\ \mathbf{v},\ \mathbf{r},\ \mathbf{s},\ \mathbf{h},\ \mathbf{c},\ o)_\mathbf{p}\} \tag{Eq.2, repeated}$$

Then standard rasterization (Eqs. 3, 4a, 4b) produces outputs at the interpolated time.

### Properties

- **Continuous in time**: $\Delta t \in [0,1]$ is continuous — not restricted to discrete frames.
- **Continuous in viewpoint**: can render from any camera position, not just the two input views.
- **Linear motion model**: assumes linear motion between frames, appropriate for short time intervals.
- **No additional computation**: interpolation reuses the same Gaussians predicted in the feedforward pass — only the translation step changes.

### Application

- Demonstrated on DAVIS dataset (Fig. 7): renders interpolated image, depth, and motion at multiple intermediate times from the canonical camera coordinate.
- The method also clearly segments moving objects in the interpolated outputs.
- Unique capability among feedforward methods — per-pixel representations cannot perform this interpolation.

### Links

- → **Dynamic 3D Gaussian Splatting**: the representation that makes interpolation possible
- → **Differentiable 4D Rasterization**: the rendering engine that produces interpolated outputs
- → **Multi-Task Synergy**: interpolation is a direct benefit of the unified representation

---

## 9. Training Strategy

### Definition

UFO-4D is trained on a mixture of three datasets with specific augmentation and initialization strategies.

### Training Configuration

| Parameter | Value |
|---|---|
| Datasets | Stereo4D (60%), PointOdyssey (20%), Virtual KITTI 2 (20%) |
| Image width | 512 (varying aspect ratios) |
| Batch size | 16 (global) |
| Iterations | 120k |
| Hardware | 4× NVIDIA A100 40GB |
| Training time | ~3 days |
| Augmentations | Random scale crop, horizontal flip, temporal flip |
| Stereo4D subset | 10% (subsampled for storage efficiency) |

### Initialization

| Component | Source |
|---|---|
| Gaussian head | NoPoSplat (Ye et al., 2025) |
| Rest of network | MASt3R (Leroy et al., 2024) |
| Camera head | Random (trained from scratch) |

- MASt3R initialization yields better overall accuracy than MonST3R (Table D).
- MonST3R initialization gives better 3D point accuracy on Bonn and KITTI specifically.

### Dataset Ablation (Table C)

| ST | PO | VK | Stereo4D Point↓ | KITTI Motion↓ | Bonn Point↓ |
|---|---|---|---|---|---|
| 0.7 | 0.15 | 0.15 | 0.065 | 0.273 | 0.314 |
| 0.85 | 0.15 | – | 0.065 | 0.298 | 0.344 |
| 0.85 | – | 0.15 | **0.064** | 0.355 | 0.291 |
| – | 0.5 | 0.5 | 0.074 | **0.267** | **0.273** |
| **1.0** | – | – | **0.064** | 0.311 | 0.316 |

- No single best dataset for generalization.
- Pure Stereo4D training gives best Stereo4D accuracy.
- Virtual KITTI2 improves KITTI motion but hurts Stereo4D and Bonn as trade-off.
- PointOdyssey improves Bonn but degrades KITTI.
- A mixture of datasets improves generalization at the cost of specialization.

### Links

- → **Semi-Supervised Loss Framework**: the loss function used during training
- → **Feedforward Architecture**: the model being trained
- → **Prior Art**: initialization from DUSt3R-family models bootstraps performance

---

## 10. Experimental Results

### 10.1 Geometry Estimation

**Setup**: Evaluate pointmap accuracy (EPE) and depth accuracy (Abs. Rel., $\delta < 1.25$) on Stereo4D test, Bonn, KITTI, and Sintel final.

**Table 1: Geometry estimation**

| Method | Stereo4D ||| Bonn ||| KITTI ||| Sintel final |||
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | EPE | Abs.Rel. | $\delta$<1.25 | EPE | Abs.Rel. | $\delta$<1.25 | EPE | Abs.Rel. | $\delta$<1.25 | EPE | Abs.Rel. | $\delta$<1.25 |
| MASt3R | 1.887 | 0.314 | 66.50 | 0.487 | 0.174 | 76.49 | 2.304 | 0.087 | 92.04 | 3.997 | 0.366 | 61.03 |
| MonST3R | 1.504 | 0.191 | 74.75 | 0.188 | **0.062** | **96.22** | 2.797 | 0.107 | 87.61 | **3.607** | 0.341 | 57.61 |
| DynaDUSt3R | 0.811 | 0.112 | 86.07 | 0.424 | 0.084 | 92.75 | 4.889 | 0.116 | 83.59 | 4.362 | **0.301** | 61.04 |
| ZeroMSF | 1.662 | 0.220 | 76.18 | 0.208 | 0.079 | 91.58 | 3.392 | 0.107 | **92.28** | 3.637 | 0.333 | 63.86 |
| St4RTrack | 1.467 | 0.209 | 71.26 | 0.181 | 0.067 | 95.17 | 4.731 | 0.125 | 82.86 | **4.145** | 0.357 | 57.14 |
| **UFO-4D** | **0.659** | **0.106** | **88.38** | **0.162** | 0.064 | 96.04 | **2.568** | **0.090** | 88.92 | 3.866 | 0.319 | **61.94** |

**Key finding**: UFO-4D substantially outperforms all direct competitors on Stereo4D and remains competitive on all other datasets. The trade-offs across datasets stem from different training data mixtures.

---

### 10.2 Motion Estimation (Scene Flow)

**Setup**: 3D end-point error (EPE) and fraction of points within 5cm ($\delta_{3D}^{0.05}$) on Stereo4D test and KITTI Scene Flow 2015 Training.

**Table 2: Motion estimation**

| Method | Stereo4D forward || Stereo4D backward || KITTI forward ||
|---|---|---|---|---|---|---|
| | EPE_{3D}↓ | $\delta_{3D}^{0.05}$↑ | EPE_{3D}↓ | $\delta_{3D}^{0.05}$↑ | EPE_{3D}↓ | $\delta_{3D}^{0.05}$↑ |
| DynaDUSt3R | 0.175 | 51.46 | 0.161 | 52.16 | 0.463 | 1.09 |
| ZeroMSF | 0.164 | 45.63 | – | – | 0.442 | 16.20 |
| St4RTrack | 0.373 | 26.62 | – | – | 0.751 | 4.01 |
| **UFO-4D** | **0.049** | **83.06** | **0.050** | **82.67** | **0.137** | **47.80** |

**Key finding**: >3× lower EPE than the best competing method on both Stereo4D and KITTI. The $\delta_{3D}^{0.05}$ metric shows 83% of predicted points are within 5cm of ground truth vs. 52% for the next best.

---

### 10.3 Pose Estimation

**Setup**: ATE and RPE (translation, rotation) on Stereo4D test, Bonn, and Sintel final.

**Table 3: Pose estimation**

| Method | Stereo4D ||| Bonn ||| Sintel |||
|---|---|---|---|---|---|---|---|---|---|
| | ATE | RPE_trans | RPE_rot | ATE | RPE_trans | RPE_rot | ATE | RPE_trans | RPE_rot |
| MonST3R | 0.0458 | 0.0647 | 0.805 | 0.0031 | 0.0043 | 0.544 | 0.0290 | 0.0410 | 1.080 |
| St4RTrack | 0.0602 | 0.0851 | 0.729 | 0.0024 | 0.0034 | 0.400 | 0.0231 | 0.0326 | 0.884 |
| **UFO-4D** | **0.0101** | **0.0142** | **0.179** | **0.0020** | **0.0028** | **0.271** | **0.0122** | **0.0172** | **0.298** |

**Key finding**: UFO-4D's feedforward pose estimation outperforms iterative PnP+RANSAC approaches on all metrics. The improvement is especially large on Stereo4D (ATE: 4.5× better than MonST3R).

**Per-valid-pixel averaging** (Tables A, B in Appendix): Under DynaDUSt3R's alternative evaluation protocol that weights each frame by its number of valid pixels, UFO-4D consistently outperforms others on Stereo4D and KITTI and remains competitive on Bonn and Sintel.

### Links

- → **Camera Pose Estimation**: the pose head produces these results directly
- → **Multi-Task Synergy**: joint training improves all tasks simultaneously
- → **Differentiable 4D Rasterization**: accurate pose enables better photometric supervision

---

## 11. Ablation Studies

### 11.1 Loss Ablation (Table 4)

Trained on Stereo4D at 256×256 resolution for 40k iterations with batch size 12.

| Config | Image PSNR↑ | Point EPE↓ | Point rast. EPE↓ | Motion EPE↓ | Motion rast. EPE↓ |
|---|---|---|---|---|---|
| **(a) Full model** | 23.929 | **0.827** | **0.841** | **0.069** | **0.064** |
| (b) W/o image gradient | **24.268** | 0.903 | 0.899 | 0.072 | 0.070 |
| (c) W/o motion+point rendering | 20.675 | 0.911 | 0.943 | 0.071 | 0.069 |
| (d) W/o motion, point, image rendering | – | 0.886 | – | 0.071 | – |

**Key findings**:
- **(a→b)**: Removing photometric gradient flow to center/velocity degrades point and motion accuracy — photometric loss provides geometry supervision.
- **(a→c)**: Removing rendering supervision on point and motion maps degrades all tasks — rendering losses sharpen object and motion boundaries.
- **(d)**: Even without rendering, per-pixel estimation alone underperforms the full model — the rendering-based supervision adds value beyond shared capacity.
- **(b)**: PSNR is highest without geometry gradients, because the model can freely optimize image quality without being constrained by geometric consistency — but this hurts downstream tasks.

---

### 11.2 Architecture Comparison (Table 5)

Different output representations trained under UFO-4D's protocol (256×256, 60k iterations).

| Output representation | Equivalent | Stereo4D Motion EPE | Stereo4D Point EPE | KITTI Motion EPE | KITTI Point EPE | Bonn Point EPE |
|---|---|---|---|---|---|---|
| **Dynamic 3DGS** | **UFO-4D (Ours)** | 0.058 | 0.781 | **0.244** | **3.553** | 0.211 |
| Per-pixel point+motion | DynaDUSt3R, ZeroMSF | **0.057** | 0.809 | 0.283 | 4.567 | **0.196** |
| Per-pixel point only | MonST3R | – | 0.804 | – | 4.562 | 0.201 |

**Key finding**: Dynamic 3DGS representation outperforms per-pixel alternatives on Stereo4D and KITTI, with substantial gains on KITTI (~22% reduction in point EPE). Performance on Bonn is slightly behind due to large, overlapping Gaussians in textureless regions (walls) introducing blending errors.

---

### 11.3 Pose and Representation Ablation (Table F)

| Output representation | Pose estimator | Stereo4D Motion | Stereo4D Point | KITTI Motion | KITTI Point | Bonn Point |
|---|---|---|---|---|---|---|
| **Dynamic 3DGS (Ours)** | **Pose head** | **0.057** | **0.781** | **0.244** | **3.553** | 0.211 |
| Per-pixel point+motion | Pose head | 0.060 | 0.811 | 0.245 | 4.396 | 0.206 |
| Per-pixel point+motion | PnP-RANSAC | 0.058 | 0.809 | 0.283 | 4.567 | **0.196** |

**Key finding**: The feedforward pose head combined with D-3DGS achieves the best results on Stereo4D and KITTI. The KITTI gain is driven by both accurate pose estimation and photometric supervision.

---

### 11.4 Pose Estimation with PnP+RANSAC (Table G)

| Method | Stereo4D ||| Bonn ||| Sintel |||
|---|---|---|---|---|---|---|---|---|---|
| | ATE | RPE_trans | RPE_rot | ATE | RPE_trans | RPE_rot | ATE | RPE_trans | RPE_rot |
| MonST3R | 0.0458 | 0.0647 | 0.805 | 0.0031 | 0.0043 | 0.544 | 0.0290 | 0.0410 | 1.080 |
| St4RTrack | 0.0602 | 0.0851 | 0.729 | 0.0024 | 0.0034 | 0.400 | 0.0231 | 0.0326 | 0.884 |
| UFO-4D (PnP+RANSAC) | 0.0120 | 0.0170 | 0.205 | 0.0024 | 0.0033 | 0.293 | 0.0168 | 0.0238 | 0.332 |
| **UFO-4D (pose head)** | **0.0101** | **0.0142** | **0.179** | **0.0020** | **0.0028** | **0.271** | **0.0122** | **0.0172** | **0.298** |

**Key finding**: Even with PnP+RANSAC, UFO-4D's geometry is more accurate than competitors. The direct feedforward pose head achieves ~16.6% higher accuracy than PnP+RANSAC, demonstrating robustness over noise-sensitive iterative solvers.

---

### 11.5 Gaussian Scale Impact on Bonn (Table E)

| Scale Range | Point EPE↓ | Depth Abs. Rel.↓ |
|---|---|---|
| 0 – 0.5 | **0.189** | 0.073 |
| 0.5 – 1.0 | 0.151 | **0.059** |
| 1.0 – 1.5 | 0.200 | 0.068 |
| 1.5 – 2.0 | 0.189 | 0.067 |
| 2.0 – 2.5 | 0.185 | 0.071 |
| 2.5 – 3.0 | 0.205 | 0.085 |

**Key finding**: Depth accuracy degrades as Gaussian scale increases. Large Gaussians in textureless regions (walls, sky) introduce blending errors — the main source of UFO-4D's slightly lower performance on Bonn compared to per-pixel methods.

### Links

- → **Multi-Task Synergy**: loss ablation quantifies the synergy effect
- → **Dynamic 3D Gaussian Splatting**: architecture comparison validates the representation choice
- → **Camera Pose Estimation**: pose ablation separates representation and estimation contributions

---

## 12. What Existed Before and What This Paper Changes

### 12.1 Prior Approaches and Their Limitations

**Test-time optimization methods** (MoSca, MoDGS, DGS-LRM) achieve high-fidelity 4D reconstruction by optimizing per-scene, but take hours to run and depend on pre-computed cues like depth and optical flow. This makes them impractical for real-time applications.

**Task-specific feedforward models** have emerged in three parallel tracks that do not communicate. DUSt3R and MASt3R estimate static 3D geometry (pointmaps) from image pairs. MonST3R extends this to dynamic scenes by fine-tuning on video depth, but does not estimate motion. DynaDUSt3R adds motion estimation to the DUSt3R framework, but geometry and motion are predicted as independent per-pixel quantities with no rendering-based coupling. ZeroMSF follows a similar per-pixel approach with a different training recipe. None of these produce a renderable 3D representation.

**Concurrent work** St4RTrack performs simultaneous 4D reconstruction and tracking, but is optimized for long-term tracking rather than short-term dense 4D reconstruction. It also requires PnP+RANSAC for pose estimation.

**Dense 4D datasets** are scarce. Synthetic data (PointOdyssey) has a domain gap. The largest real-world dataset, Stereo4D, has noisy annotations from its depth-and-motion optimization pipeline — static regions sometimes show non-zero motion (Fig. D).

### 12.2 What This Paper Contributes

**Contribution 1: Unified renderable 4D representation.** By predicting Dynamic 3D Gaussians instead of per-pixel maps, UFO-4D is the first feedforward model that produces a single explicit 4D representation from which image, depth, point, scene flow, and optical flow can all be derived.

**Contribution 2: Self-supervised training through differentiable rendering.** The renderable representation enables a photometric loss that is independent of ground truth annotations. This addresses the data scarcity problem: even when annotations are sparse or noisy, the model can still learn from image reconstruction.

**Contribution 3: 4D spatio-temporal interpolation.** A new application that cannot be achieved by per-pixel methods: rendering the scene at any intermediate time and viewpoint from a single feedforward prediction.

**Contribution 4: State-of-the-art performance.** UFO-4D achieves SOTA on 3D geometry and 3D motion benchmarks, with >3× lower motion EPE on Stereo4D and KITTI than competing methods.

### 12.3 Side-by-Side: Prior Art vs. This Paper

| Dimension | DynaDUSt3R | ZeroMSF | MonST3R | St4RTrack | **UFO-4D** |
|---|---|---|---|---|---|
| Representation | Per-pixel point+motion | Per-pixel point+motion | Per-pixel point | Per-pixel point+motion | **Dynamic 3DGS** |
| Renderable | No | No | No | No | **Yes** |
| Self-supervision | No | No | No | No | **Yes (photometric)** |
| Pose estimation | Post-hoc | Post-hoc | PnP+RANSAC | PnP+RANSAC | **Feedforward** |
| 4D interpolation | No | No | No | No | **Yes** |
| Motion estimation | Yes | Yes | No | Yes | **Yes** |
| Cross-modal coupling | No | No | N/A | No | **Yes** |

| Dimension | DynaDUSt3R (closest) | **UFO-4D** |
|---|---|---|
| Output | Point map + 3D motion per pixel | Dynamic 3D Gaussian per pixel |
| Rendering capability | None | Full image/depth/motion |
| Self-supervised loss | None | Photometric (MSE + LPIPS) + smoothness |
| Pose | From DUSt3R pipeline | Direct feedforward head |
| Stereo4D forward EPE | 0.175 | **0.049** (3.6× better) |
| KITTI motion EPE | 0.463 | **0.137** (3.4× better) |

### 12.4 The Core Shift in Thinking

Previous feedforward methods treated 4D reconstruction as a collection of per-pixel regression tasks: predict the 3D position of each pixel, optionally predict its motion, then estimate pose separately. Each output was an independent map, with no structural relationship between them. This meant ground truth was the only source of supervision — and when that ground truth was sparse or noisy, performance suffered.

UFO-4D shifts from per-pixel prediction to primitive-based prediction. Each pixel is explained by a 3D Gaussian with both geometric and dynamic attributes. This shift creates a structural constraint: the geometry and motion of each point must be consistent enough to render a correct image. The image itself becomes supervision.

The practical consequence is large: by replacing independent per-pixel maps with a single renderable representation, the model gains access to a dense, annotation-free training signal (photometric loss) that regularizes all outputs simultaneously. The result is a 3× improvement in motion estimation — not from a better backbone, not from more data, but from a better representation that couples what previous methods kept separate.

---

## 13. Quick Reference Card

| # | Insight | Design Choice | Evidence |
|---|---|---|---|
| 1 | Shared Gaussian primitives couple geometry and motion | Use D-3DGS instead of per-pixel maps | >3× lower motion EPE (Table 2) |
| 2 | Photometric loss provides geometry supervision | Backpropagate image gradients to center/velocity | Table 4: removing gradients degrades point accuracy |
| 3 | Rendering losses sharpen boundaries | Supervise rasterized point and motion maps | Fig. 6: cleaner motion boundaries |
| 4 | Feedforward pose beats iterative PnP | Dedicated pose head with learnable pose token | ~16.6% better than PnP+RANSAC (Table G) |
| 5 | Opacity emerges as confidence | No explicit confidence training | Fig. 5: opacity selects one Gaussian per visible surface |
| 6 | Representation enables interpolation | D-3DGS + linear motion model | Fig. 7: continuous view+time rendering on DAVIS |
| 7 | Dense self-supervision overcomes noisy GT | Photometric loss complements sparse annotations | Better motion on Stereo4D despite noisy GT |
| 8 | Large Gaussians hurt textureless regions | Trade-off of Gaussian vs. per-pixel representation | Table E: depth error increases with scale on Bonn |

---

## 14. Open Questions

The paper identifies several directions for future work:

1. **Memory scalability**: extending to long sequences would be memory-intensive due to linear growth of Gaussian count. Compact scene representations (C3G, ReSplat) could address this.

2. **Non-linear motion**: the linear motion assumption ($\boldsymbol{\mu} + \Delta t \cdot \mathbf{v}$) is best suited for short time intervals. Learnable, non-linear motion models and time-varying Gaussian attributes could handle complex dynamics and photometric changes.

3. **Multi-view temporal supervision**: rendered images, motion, and geometry at novel views and times can serve as additional supervision when multi-view video annotations are available, enhancing spatio-temporal consistency.

4. **Textureless region handling**: large overlapping Gaussians in textureless regions (walls, sky) cause blending errors (Table E). This is an inherent trade-off of the Gaussian representation vs. per-pixel approaches.

---

## 15. Concept Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                    FOUNDATION LAYER                         │
│                                                             │
│  ┌──────────────┐    ┌──────────────────┐                   │
│  │ 3D Gaussian  │    │ Feedforward      │                   │
│  │ Splatting     │    │ Architecture     │                   │
│  │ (3DGS)       │    │ (ViT enc-dec)    │                   │
│  └──────┬───────┘    └────────┬─────────┘                   │
│         │                     │                             │
└─────────┼─────────────────────┼─────────────────────────────┘
          │                     │
          ▼                     ▼
┌──────────────────────────────────────────────────────┐
│              CORE REPRESENTATION                     │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │  Dynamic 3D Gaussian Splatting (D-3DGS)      │    │
│  │  Eq.1: f_θ(I_t, I_{t+1}) → (G, P)           │    │
│  └──────────────────┬───────────────────────────┘    │
│                     │                                │
└─────────────────────┼────────────────────────────────┘
                      │
        ┌─────────────┼─────────────────┐
        ▼             ▼                 ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│ Differentiable│ │ Camera Pose  │ │ Opacity as       │
│ 4D Rasterizer│ │ Estimation   │ │ Learnable        │
│ Eqs.2-4b     │ │ (pose head)  │ │ Confidence       │
└──────┬───────┘ └──────┬───────┘ └──────────────────┘
       │                │
       ▼                ▼
┌──────────────────────────────────┐
│  Semi-Supervised Loss Framework  │
│  Eq.5: L_total = L_sup + L_self │
│         │                │      │
│    ┌────┘                └───┐  │
│    ▼                         ▼  │
│ Supervised Loss       Self-Supervised│
│ Eqs.6a-6d            Loss Eqs.7a-7c│
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│   Multi-Task Synergy through     │
│   Shared Primitives              │
│   (core insight of the paper)    │
└──────────────┬───────────────────┘
               │
     ┌─────────┼─────────┐
     ▼         ▼         ▼
  3D Geom.  3D Motion  4D Interp.
  (depth,   (scene     (novel
   point)    flow)      view+time)
```

---

## 16. Key Equations

| Eq. | Name | Formula | Section |
|---|---|---|---|
| 1 | Model formulation | $f_\theta(\mathbf{I}_t, \mathbf{I}_{t+1}) \mapsto (\mathcal{G}, \mathbf{P})$ | §1 D-3DGS |
| 2 | Temporal evolution | $\mathcal{G}(t') = \{(\boldsymbol{\mu} + \Delta t \cdot \mathbf{v}, \ldots)\}$ | §3 Rasterization |
| 3 | Image rendering | $\hat{\mathbf{I}}_{t'}(\mathbf{p}) = \sum_i \mathbf{c}_i o_i \prod_{j<i}(1-o_j)$ | §3 Rasterization |
| 4a | Point map rendering | $\mathbf{X}_{t'}(\mathbf{p}) = \sum_i \boldsymbol{\mu}_i o_i \prod_{j<i}(1-o_j)$ | §3 Rasterization |
| 4b | Scene flow rendering | $\mathbf{V}_{t'}(\mathbf{p}) = \sum_i \mathbf{v}_i o_i \prod_{j<i}(1-o_j)$ | §3 Rasterization |
| 5 | Total loss | $L_\text{total} = L_\text{sup} + L_\text{self}$ | §4 Loss |
| 6a | Supervised loss | $L_\text{sup} = L_\text{motion} + w_\text{point}L_\text{point} + w_\text{pose}L_\text{pose}$ | §4.1 Sup. Loss |
| 6b | Motion loss | $L_\text{motion} = \sum_u \frac{1}{|V_u^{GT}|}(\|\mathbf{v}_u - \mathbf{V}_u^{GT}\| + \|\mathbf{V}_u - \mathbf{V}_u^{GT}\|)$ | §4.1 Sup. Loss |
| 6c | Point loss | $L_\text{point} = \sum_u \frac{1}{|X_u^{GT}|}(\|\boldsymbol{\mu}_u - \mathbf{X}_u^{GT}\| + \|\mathbf{X}_u - \mathbf{X}_u^{GT}\|)$ | §4.1 Sup. Loss |
| 6d | Pose loss | $L_\text{pose} = \|\mathbf{q} - \mathbf{q}_{GT}\| + \|\boldsymbol{\tau} - \boldsymbol{\tau}_{GT}\|$ | §4.1 Sup. Loss |
| 7a | Self-supervised loss | $L_\text{self} = L_\text{photo} + w_\text{smooth}L_\text{smooth}$ | §4.2 Self-sup. |
| 7b | Photometric loss | $L_\text{photo} = \sum_u (\text{mse}(\hat{\mathbf{I}}_u, \mathbf{I}_u) + w_\text{lpips}\text{lpips}(\hat{\mathbf{I}}_u, \mathbf{I}_u))$ | §4.2 Self-sup. |
| 7c | Smoothness loss | $L_\text{smooth} = \sum_u \sum_{\mathbf{d}}(|\partial_x \mathbf{d}_u|e^{-|\partial_x \mathbf{I}_u|} + |\partial_y \mathbf{d}_u|e^{-|\partial_y \mathbf{I}_u|})$ | §4.2 Self-sup. |

---

## 17. Reference Map

### 3D Reconstruction (Static)
- DUSt3R (Wang et al., 2024b) — geometric 3D vision made easy
- MASt3R (Leroy et al., 2024) — grounding image matching in 3D
- NoPoSplat (Ye et al., 2025) — 3D Gaussian splats from sparse unposed images
- CroCo v2 (Weinzaepfel et al., 2023) — cross-view completion pre-training
- VGGT (Wang et al., 2025a) — visual geometry grounded transformer

### 3D Reconstruction (Dynamic)
- DynaDUSt3R (Jin et al., 2025) — Stereo4D: learning how things move in 3D
- MonST3R (Zhang et al., 2025a) — geometry estimation in presence of motion
- ZeroMSF (Liang et al., 2025b) — zero-shot monocular scene flow
- St4RTrack (Feng et al., 2025) — simultaneous 4D reconstruction and tracking
- D²USt3R (Han et al., 2025) — enhancing 3D reconstruction with 4D pointmaps

### Dynamic 3D Gaussians
- 3D Gaussian Splatting (Kerbl et al., 2023) — 3DGS for real-time rendering
- Dynamic 3D Gaussians (Luiten et al., 2024) — tracking by persistent dynamic view synthesis
- C3G (An et al., 2026) — compact 3D representations with 2K Gaussians
- ReSplat (Xu et al., 2025a) — learning recurrent Gaussian splats

### Test-Time Optimization Methods
- MoSca (Lei et al., 2025) — dynamic Gaussian fusion via 4D motion scaffolds
- MoDGS (Liu et al., 2025) — dynamic Gaussian splatting with depth priors
- Shape of Motion (Wang et al., 2025b) — 4D reconstruction from a single video

### Dense Prediction Architectures
- ViT (Dosovitskiy et al., 2021) — vision transformers
- DPT (Ranftl et al., 2021) — vision transformers for dense prediction

### Datasets
- Stereo4D (Jin et al., 2025) — real-world stereo video with 4D annotations
- PointOdyssey (Zheng et al., 2023) — synthetic dataset for long-term point tracking
- Virtual KITTI 2 (Gaidon et al., 2016) — synthetic driving scenes
- KITTI (Geiger et al., 2012; Menze et al., 2018) — autonomous driving benchmark
- Bonn (Palazzolo et al., 2019) — dynamic RGB-D environments
- Sintel (Butler et al., 2012) — naturalistic optical flow benchmark

### Losses and Metrics
- LPIPS (Zhang et al., 2018) — perceptual similarity metric
- Edge-aware smoothness (Godard et al., 2017; 2019) — self-supervised depth estimation
- EPnP (Lepetit et al., 2009) — efficient PnP solver
- RANSAC (Fischler & Bolles, 1981) — random sample consensus

### 4D Generation
- SV4D (Xie et al., 2025) — dynamic 3D content generation
- Diffusion4D (Liang et al., 2024) — 4D generation via video diffusion
- 4Real (Yu et al., 2024) — photorealistic 4D scene generation
