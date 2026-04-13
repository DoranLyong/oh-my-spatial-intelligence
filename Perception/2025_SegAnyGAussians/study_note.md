# Segment Any 3D Gaussians (SAGA) Study Note — A Concept Mind-Map

> Paper: Segment Any 3D Gaussians (arXiv:2312.00860v3)
> Authors: Jiazhong Cen, Jiemin Fang, Chen Yang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, Qi Tian
> Affiliations: MoE Key Lab of AI, Shanghai Jiao Tong University; Huawei Technologies Co., Ltd.
> Published: AAAI 2025
> Code: https://github.com/Jumpat/SegAnyGAussians

This note organizes every key concept in the paper as a mind-map.
Each concept is broken down into four facets:

- **Definition** — what it is, stated plainly
- **Properties** — its mathematical or behavioral characteristics
- **Application** — how the paper (or the field) uses it
- **Links** — connections to other concepts in this map

---

## 0. The Big Picture

```
                    2D Foundation Models (SAM)
                    "Segment anything in 2D"
                              │
                    "How to lift this to 3D?"
                              │
              ┌───────────────┼───────────────────┐
              ▼               ▼                   ▼
       NeRF-based        Implicit Fields       3D Gaussian
       (SA3D, ISRF)      (GARField)            Splatting
       slow iterative    MLP-based, slow       explicit, fast
       refinement        multi-query            real-time
              │               │                   │
              │               │                   ▼
              │               │          SAGA: attach affinity
              │               │          features to each Gaussian
              │               │                   │
              │               └─── inspires ──────┤
              │                 scale-conditioned  │
              │                 features           │
              │                                    │
              └────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             Scale-Gate           Scale-Aware
             Mechanism            Contrastive
             (Eq.3-5)             Learning (Eq.7-9)
                    │                   │
                    └─────────┬─────────┘
                              ▼
                   Multi-Granularity 3D
                   Segmentation in ~3 ms
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
               Promptable  Scene     Open-Vocab
               Segment.    Decomp.   Segment.
```

---

## 1. 3D Gaussian Splatting (3D-GS)

### Definition

3D Gaussian Splatting is an explicit 3D scene representation that models a scene as a set of colored 3D Gaussians $\mathcal{G} = \{g_1, g_2, \ldots, g_K\}$, where $K$ is the number of Gaussians. Each Gaussian $g$ has a mean (position), covariance (shape), color (spherical harmonics), and opacity. Given a camera pose, 3D-GS projects the Gaussians onto the 2D image plane and renders the color of a pixel $\mathbf{p}$ via alpha-blending:

$$C(\mathbf{p}) = \sum_{i \in \mathcal{N}} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j) \tag{Eq.1}$$

where $c_i$ is the color of Gaussian $g_i$ and $\alpha_i$ is computed by evaluating the projected 2D Gaussian with the learned per-Gaussian opacity.

### Properties

- **Explicit representation**: each Gaussian is an independent entity with its own parameters, unlike implicit NeRF representations that encode the scene in MLP weights.
- **Differentiable rasterization**: 3D-GS uses a novel differentiable rasterization pipeline for efficient training and rendering, orders of magnitude faster than volumetric ray-marching.
- **Intrinsic attribute extensibility**: because each Gaussian is an explicit primitive, new per-Gaussian attributes (beyond color and opacity) can be attached without adding separate modules. SAGA exploits this to attach affinity features.
- **High-frequency texture capture**: 3D-GS preserves fine geometric details that implicit radiance fields tend to smooth away (Fig. 4). This enables SAGA to segment thin, fine-grained structures.

### Application

- SAGA uses 3D-GS as the scene backbone. The rendering equation (Eq.1) is extended to also render the attached affinity features to 2D, enabling 2D supervision from SAM masks.
- The differentiable rasterizer backpropagates the contrastive learning signal from 2D rendered feature maps back to 3D Gaussian affinity features.

### Links

- → **Gaussian Affinity Features**: 3D-GS provides the carrier; each Gaussian gets an affinity feature vector $\mathbf{f}_g \in \mathbb{R}^D$ attached.
- → **Feature Norm Regularization**: the alpha-blending rendering (Eq.1) applied to affinity features creates a linear combination of 3D features, motivating the norm regularization.
- → **Scale-Gate Mechanism**: the scale gate operates on 3D-GS features directly, made possible by the explicit representation.

---

## 2. Multi-Granularity Segmentation

### Definition

Multi-granularity segmentation is the problem of segmenting a 3D scene at varying levels of detail — a single 3D Gaussian may belong to different parts or objects depending on the granularity level. For example, at a fine scale, the wheel of a truck is a separate segment; at a coarser scale, the entire truck is one segment.

The physical scale $s_\mathbf{M}$ of a 2D mask $\mathbf{M}$ is computed by projecting its pixels into 3D using depth from 3D-GS and measuring the standard deviation of the resulting 3D point positions:

$$s_\mathbf{M} = \sigma\bigl(\{P_i\}_{i \in \delta(\mathbf{M})}\bigr) \tag{Eq.2}$$

where $\sigma(\cdot)$ denotes the standard deviation and $\delta(\cdot)$ denotes the set of positive pixels in mask $\mathbf{M}$. Since these scales are computed in 3D space, they are consistent across different views.

### Properties

- **View-consistency**: 3D physical scales remain stable across viewpoints, unlike 2D pixel-based scales which change with camera distance and angle.
- **Ambiguity**: a single Gaussian can simultaneously be part of a small part (fine scale) and a large object (coarse scale). This is the central challenge SAGA addresses.
- **Scale continuity**: scale is treated as a continuous variable $s \in [0, 1]$, not discrete categories, enabling smooth transitions between granularity levels.

### Application

- The computed mask scales serve as conditioning input to the scale gate, allowing SAGA to produce different segmentations at different scales.
- During training, SAM-extracted masks with their computed 3D physical scales form the supervision signal.

### Links

- → **Scale-Gate Mechanism**: the scale gate takes the physical scale $s$ as input and gates feature channels accordingly to enable multi-granularity output.
- → **Scale-Aware Pixel Identity Vector**: mask scales determine how pixel identity vectors are constructed for contrastive learning.
- → **Gaussian Affinity Features**: affinity features must encode information at all granularity levels simultaneously.

---

## 3. Gaussian Affinity Features

### Definition

SAGA attaches a Gaussian affinity feature $\mathbf{f}_g \in \mathbb{R}^D$ to each 3D Gaussian $g$ in the scene. $D$ is the feature dimension. These features encode segmentation information: the cosine similarity between two Gaussians' affinity features indicates whether they belong to the same 3D target.

The affinity features are rendered to 2D using the same alpha-blending as color rendering (Eq.1), replacing $c_i$ with $\mathbf{f}_{g_i}$:

$$\mathbf{F}(\mathbf{p}) = \sum_{i \in \mathcal{N}} \mathbf{f}_{g_i} \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j) \tag{Eq.4}$$

This produces a 2D affinity feature map $\mathbf{F} \in \mathbb{R}^{H \times W \times D}$ that can be supervised using 2D masks.

### Properties

- **Per-Gaussian storage**: each feature is an intrinsic attribute of its Gaussian, not stored in a separate implicit field. This avoids the over-smoothing that MLP-based feature fields exhibit (as in GARField).
- **Scale-agnostic base**: the raw feature $\mathbf{f}_g$ is scale-independent. Multi-granularity behavior is achieved by the scale gate applied on top, not by encoding scale into the features themselves.
- **Dimension**: $D$ is a hyperparameter. The paper uses $D = 256$ by default.
- **Efficiency**: only 1% of Gaussians are selected for clustering during inference, keeping segmentation fast.

### Application

- After training, the affinity features enable promptable segmentation: given a 2D point prompt on the rendered feature map, SAGA retrieves the corresponding query features and segments the 3D target by computing cosine similarities with all other Gaussians' (scale-gated) features.
- For automatic scene decomposition, SAGA clusters the affinity features using HDBSCAN.
- Code: https://github.com/Jumpat/SegAnyGAussians

### Links

- → **3D Gaussian Splatting**: affinity features are attached as an additional per-Gaussian attribute, rendered via the same differentiable rasterizer.
- → **Scale-Gate Mechanism**: the raw affinity feature $\mathbf{f}_g$ is transformed into a scale-conditioned feature $\mathbf{f}_g^s$ by the scale gate.
- → **Local Feature Smoothing**: noisy Gaussians are smoothed by averaging features from K-nearest neighbors.
- → **Feature Norm Regularization**: 3D features are normalized to unit vectors before rendering to enforce alignment along rays.
- → **Scale-Aware Contrastive Learning**: the rendered 2D affinity features are the target of the contrastive distillation loss.

---

## 4. Scale-Gate Mechanism

### Definition

The scale gate is a lightweight function $\mathcal{S}: [0,1] \to [0,1]^D$ that maps a scalar physical scale $s$ to a $D$-dimensional gate vector. It is implemented as a single linear layer followed by a sigmoid activation:

$$\mathcal{S}(s) = \sigma(\mathbf{w} s + \mathbf{b}) \tag{Eq.3a}$$

where $\mathbf{w} \in \mathbb{R}^D$, $\mathbf{b} \in \mathbb{R}^D$ are learnable parameters and $\sigma$ is the element-wise sigmoid. The scale-gated affinity feature for a Gaussian $g$ at scale $s$ is:

$$\mathbf{f}_g^s = \mathcal{S}(s) \odot \mathbf{f}_g \tag{Eq.3}$$

where $\odot$ denotes the Hadamard (element-wise) product. Since all Gaussians share the same scale gate, during training the gate can be applied after rendering to 2D for efficiency:

$$\mathbf{F}'(\mathbf{p}) = \mathcal{S}(s) \odot \mathbf{F}(\mathbf{p}) \tag{Eq.5}$$

During inference, the scale gate is applied directly to the 3D Gaussian affinity features.

### Properties

- **Extreme simplicity**: one linear layer + sigmoid. No MLP, no multi-head attention, no learned embeddings — just $2D$ parameters ($\mathbf{w}$ and $\mathbf{b}$). For $D = 256$, that is 512 scalar parameters.
- **Negligible overhead**: applying the scale gate adds near-zero computation to segmentation. Changing scale at inference requires only re-applying the gate — no re-rendering.
- **Channel selection**: the sigmoid output $\in [0,1]^D$ acts as a soft binary mask on feature channels. Each channel is either suppressed (gate $\approx 0$) or preserved (gate $\approx 1$) at a given scale.
- **Interpretability** (Appendix A.4): across 47 scenes with 32 scale gates each (1,504 entries total), 36.1% (543) are positive and 63.9% (961) are negative. A typical scene has ~12 positive gates and ~20 negative gates. At larger (coarser) scales, more gates close and fewer features remain active; at finer scales, more gates open to capture detailed information.
- **Contrast with GARField**: GARField uses a multi-layer perceptron to model scale-conditioned features in an implicit field. This requires repeated queries at different scales. SAGA's scale gate is applied once per scale to explicit per-Gaussian features, with no additional computation.

### Application

- The scale gate enables real-time scale switching: a user can adjust the segmentation granularity in milliseconds by simply changing $s$.
- Demonstrated on 3D-GS (primary), GARField (replacing its MLP, Fig. A1), and Instant-NGP (Fig. A2), showing the mechanism is representation-agnostic.

### Links

- → **Gaussian Affinity Features**: the gate modulates the raw feature $\mathbf{f}_g$ to produce $\mathbf{f}_g^s$.
- → **Multi-Granularity Segmentation**: the gate is the mechanism that enables different segmentation outputs at different scales.
- → **Scale-Aware Contrastive Learning**: the gate is trained end-to-end through the contrastive loss — gradients flow through $\mathcal{S}(s)$ to update $\mathbf{w}$ and $\mathbf{b}$.
- → **Feature Norm Regularization**: applied to the gated features $\mathbf{F}'(\mathbf{p})$.

---

## 5. Local Feature Smoothing (LFS)

### Definition

Local Feature Smoothing replaces each Gaussian's affinity feature with the average of features from its $K$-nearest neighbors in 3D space. During training, for a Gaussian $g$:

$$\mathbf{f}_g \leftarrow \frac{1}{K} \sum_{g' \in \text{KNN}(g)} \mathbf{f}_{g'} \tag{LFS}$$

where $\text{KNN}(g)$ denotes the $K$-nearest neighbors of $g$ in the set $\mathcal{G}$. After training, the smoothed feature is saved as the final affinity feature.

### Properties

- **Eliminates outliers**: many noisy Gaussians in 3D exhibit unexpectedly high feature similarities with incorrect segmentation targets. These arise from small alpha weights in rasterization or incorrect geometry learned by 3D-GS. LFS smooths these out.
- **Spatial locality prior**: exploits the assumption that nearby Gaussians in 3D space should have similar segmentation features — a reasonable inductive bias for object boundaries.
- **Ablation evidence** (Fig. 5): without LFS, segmentation at cosine similarity threshold 0.75 produces many false positives. With LFS, outliers are eliminated.

### Application

- Applied post-training as a one-time smoothing pass before inference.
- The smoothing is applied to the raw features $\mathbf{f}_g$, not the scale-gated features.

### Links

- → **Gaussian Affinity Features**: LFS operates directly on the per-Gaussian feature vectors.
- → **Feature Norm Regularization**: both are regularization techniques, but LFS targets spatial outliers while FNR targets ray-wise alignment. They complement each other (Fig. 5).
- → **3D Gaussian Splatting**: LFS uses the 3D spatial structure (KNN in 3D coordinates) of the Gaussians.

---

## 6. Feature Norm Regularization (FNR)

### Definition

Feature Norm Regularization addresses the misalignment between 2D rendered features and 3D Gaussian features. When rendering affinity features via alpha-blending (Eq.4), the result is a linear combination of multiple 3D features, each potentially pointing in different directions. A rendered 2D feature map can exhibit high cosine similarity between pixels even when the underlying 3D Gaussians have inconsistent features.

To fix this, SAGA first normalizes all 3D Gaussian features to unit vectors before rendering. Because the rendered feature is a convex combination of unit vectors, its norm satisfies $\|\mathbf{F}'(\mathbf{p})\|_2 \in [0, 1]$. The norm equals 1 only when all contributing 3D features are perfectly aligned. The regularization loss penalizes deviation from this ideal:

$$\ell_{\text{fnr}}(\mathbf{p}) = 1 - \|\mathbf{F}'(\mathbf{p})\|_2 \tag{Eq.10}$$

The total SAGA loss combines the correspondence distillation loss and FNR:

$$\mathcal{L}_{\text{SAGA}} = \sum_{\mathbf{p} \in \Omega} \mathcal{L}_{\text{corr}} + \lambda \, \ell_{\text{fnr}} \tag{Eq.12}$$

where $\Omega$ is the set of pixels in the image and $\lambda$ controls the FNR weight.

### Properties

- **Alignment incentive**: minimizing $\ell_{\text{fnr}}$ forces 3D features along each ray to point in the same direction. This ensures that the 2D feature similarity faithfully reflects the 3D feature similarity.
- **No direct 3D supervision needed**: the regularization acts on the 2D rendered norm but propagates gradients back to align 3D features, achieved entirely through the differentiable rasterizer.
- **Ablation evidence** (Fig. 5): unlike LFS which primarily eliminates outliers, FNR shapes the internal structure of feature clusters. Raising the similarity threshold to 0.95, the segmented region with FNR maintains its shape while the one without FNR becomes translucent (indicating misaligned 3D features that happen to project similarly in 2D).

### Application

- FNR is applied during training at every iteration as part of $\mathcal{L}_{\text{SAGA}}$ (Eq.12).
- The 3D features are normalized to unit vectors during both training and inference.

### Links

- → **Gaussian Affinity Features**: FNR regularizes the direction consistency of features along rendering rays.
- → **3D Gaussian Splatting**: the rendered feature norm is a byproduct of the alpha-blending rendering equation.
- → **Local Feature Smoothing**: complementary regularization — LFS handles spatial outliers, FNR handles ray-wise alignment.
- → **Scale-Aware Contrastive Learning**: FNR is combined with $\mathcal{L}_{\text{corr}}$ in the total loss (Eq.12).

---

## 7. Scale-Aware Pixel Identity Vector

### Definition

The scale-aware pixel identity vector $\mathbf{V}(s, \mathbf{p}) \in \{0, 1\}^{N_s}$ is a binary vector assigned to each pixel $\mathbf{p}$ at each scale $s$, indicating which 2D masks the pixel belongs to at that scale. $N_s$ is the number of masks at scale $s$.

To construct $\mathbf{V}(s, \mathbf{p})$, SAGA first sorts all masks $\mathcal{M}_\mathbf{I}$ from image $\mathbf{I}$ in descending order of their mask scales to get an ordered list $\mathcal{O}_\mathbf{I} = (\mathbf{M}_\mathbf{I}^{(1)}, \ldots, \mathbf{M}_\mathbf{I}^{(N_I)})$ where $s_{\mathbf{M}_\mathbf{I}^{(1)}} > \cdots > s_{\mathbf{M}_\mathbf{I}^{(N_I)}}$. Then for each pixel $\mathbf{p}$:

$$\mathbf{V}'(s, \mathbf{p}) = \begin{cases} \mathbf{M}_\mathbf{I}^{(i)}(\mathbf{p}) & \text{if } s_{\mathbf{M}_\mathbf{I}^{(i)}} < s \text{ or } \mathcal{C}(\mathbf{p}) \\ 0 & \text{otherwise} \end{cases} \tag{Eq.6}$$

where $\mathcal{C}(\mathbf{p}) \triangleq \{\forall \mathbf{M} \in \{\mathbf{M}_\mathbf{I}^{(j)} \mid s \leq s_{\mathbf{M}_\mathbf{I}^{(j)}} < s_{\mathbf{M}_\mathbf{I}^{(i)}}\}, \mathbf{M}(\mathbf{p}) = 0\}$.

**Intuition**: if a pixel belongs to a specific mask at a given scale, it continues to belong to that mask at all larger scales. Masks smaller than the queried scale $s$ are included directly. Masks larger than $s$ are included only if no intermediate-scale mask claims the pixel.

### Properties

- **Scale-consistency**: the construction ensures that pixel assignments are monotonically inclusive as scale increases — a pixel never "loses" a mask membership at a larger scale.
- **SAM-derived**: the masks come from running SAM on each training image, producing an automatic set of multi-scale masks without manual annotation.
- **Binary encoding**: each entry is 0 or 1, making the mask correspondence (Eq.7) a simple dot-product check.

### Application

- The identity vectors serve as the ground-truth supervision signal for contrastive learning. Two pixels sharing at least one mask (dot product > 0) should have similar scale-gated features; pixels sharing no masks should have dissimilar features.

### Links

- → **Multi-Granularity Segmentation**: identity vectors encode the multi-scale mask hierarchy.
- → **Scale-Aware Contrastive Learning**: identity vectors define the mask correspondence $\text{Corr}_m$ (Eq.7) that supervises the feature learning.
- → **Gaussian Affinity Features**: the identity vectors are the 2D supervision signal that trains the 3D affinity features.

---

## 8. Scale-Aware Contrastive Learning

### Definition

Scale-aware contrastive learning is SAGA's training strategy that distills SAM's 2D segmentation capability into the 3D Gaussian affinity features. It adapts the correspondence distillation loss from Hamilton et al. (2022) to be scale-aware.

For two pixels $\mathbf{p}_1, \mathbf{p}_2$ at a given scale $s$:

**Mask correspondence** (whether they share a mask):

$$\text{Corr}_m(s, \mathbf{p}_1, \mathbf{p}_2) = \mathbb{1}\bigl(\mathbf{V}(s, \mathbf{p}_1) \cdot \mathbf{V}(s, \mathbf{p}_2)\bigr) \tag{Eq.7}$$

where $\mathbb{1}(\cdot)$ equals 1 when the input is greater than 0, and 0 otherwise.

**Feature correspondence** (cosine similarity of scale-gated features):

$$\text{Corr}_f(s, \mathbf{p}_1, \mathbf{p}_2) = \langle \mathbf{F}^s(\mathbf{p}_1), \mathbf{F}^s(\mathbf{p}_2) \rangle \tag{Eq.8}$$

where $\mathbf{F}^s(\mathbf{p})$ is the normalized scale-gated feature at pixel $\mathbf{p}$.

**Correspondence distillation loss**:

$$\mathcal{L}_{\text{corr}}(s, \mathbf{p}_1, \mathbf{p}_2) = (1 - 2 \cdot \text{Corr}_m(s, \mathbf{p}_1, \mathbf{p}_2)) \cdot \max(\text{Corr}_f(s, \mathbf{p}_1, \mathbf{p}_2), 0) \tag{Eq.9}$$

### Derivation: How Eq.9 Works

**Case 1: Same mask** ($\text{Corr}_m = 1$):

$(1 - 2 \cdot 1) \cdot \max(\text{Corr}_f, 0) = -\max(\text{Corr}_f, 0)$

Minimizing this loss *maximizes* the feature correspondence — pushes features of same-mask pixels together.

**Case 2: Different masks** ($\text{Corr}_m = 0$):

$(1 - 2 \cdot 0) \cdot \max(\text{Corr}_f, 0) = \max(\text{Corr}_f, 0)$

Minimizing this loss *minimizes* the feature correspondence — pushes features of different-mask pixels apart. The clipping at 0 prevents the loss from encouraging negative correlations beyond zero, stabilizing training.

### Properties

- **Scale-conditioned**: the same pixel pair can have different loss values at different scales, because the mask correspondence (Eq.7) and the feature correspondence (Eq.8) both depend on $s$.
- **No manual annotation**: all supervision comes from SAM masks run automatically on training images.
- **Gradient flow**: gradients from $\mathcal{L}_{\text{corr}}$ flow through the differentiable rasterizer to update both the per-Gaussian affinity features $\mathbf{f}_g$ and the scale gate parameters $\mathbf{w}, \mathbf{b}$.

### Application

- Trained for approximately 10 min 40 sec on a single NVIDIA RTX 3090 GPU (Table 4).
- The total training loss combines this with FNR: $\mathcal{L}_{\text{SAGA}} = \mathcal{L}_{\text{corr}} + \lambda \, \ell_{\text{fnr}}$ (Eq.12).

### Links

- → **Scale-Aware Pixel Identity Vector**: provides the mask correspondence $\text{Corr}_m$ (Eq.7).
- → **Scale-Gate Mechanism**: the scale-gated features $\mathbf{F}^s$ are what the loss optimizes.
- → **Feature Norm Regularization**: combined in the total loss $\mathcal{L}_{\text{SAGA}}$ (Eq.12).
- → **Additional Training Strategy**: resampling and re-weighting address data imbalance in the contrastive loss.

---

## 9. Additional Training Strategy (Resampling & Re-weighting)

### Definition

The contrastive learning in SAGA faces three data imbalance problems:

1. **Scale-sensitivity imbalance**: most pixel pairs are insensitive to scale changes — their mask correspondence stays constant across scales. This makes the scale gate collapse (output constant regardless of scale).
2. **Positive-negative imbalance**: the majority of pixel pairs are negatives (different masks), overwhelming the positive signal.
3. **Target-size imbalance**: large objects occupy more pixels, biasing the loss toward segmenting large targets while ignoring small ones.

### Resampling

In each training iteration, SAGA randomly selects $N_s$ scales and $N_p$ pixels within the image, forms $N_s \times N_p$ pixel pairs, and computes a scale-conditioned correspondence matrix. The pixel pairs are split into three sets:

- $\mathcal{R}_1$: positive pairs ($\text{Corr}_m = 1$) that remain positive across scales — **consistent positives**
- $\mathcal{R}_2$: negative pairs ($\text{Corr}_m = 0$) that remain negative across scales — **consistent negatives**
- $\mathcal{R}_3$: pairs whose correspondence changes across scales — **scale-involved pairs**

SAGA randomly selects equal numbers ($2_{sel}^{\mathcal{Q}}$) of pixel pairs from each set, ensuring scale-sensitive pairs ($\mathcal{R}_3$) are adequately represented. Hard samples (feature correspondence $> 0.5$ for negatives, $< 0.75$ for positives) are also over-sampled.

### Re-weighting

When considering two masks $\mathbf{M}_i, \mathbf{M}_j$ in $\mathcal{M}_I$, the probability that a uniformly sampled pair is from $\mathbf{M}_i \times \mathbf{M}_j$ is proportional to $|\mathbf{M}_i| \cdot |\mathbf{M}_j|$, which biases training toward large-mask pairs. To counteract this, all weights in a training iteration are min-max normalized to $[0, 10]$.

### Properties

- **Design rationale**: without resampling, the scale gate receives too little gradient from scale-varying pairs, causing it to output near-constant values regardless of scale.
- **Hard sample mining**: pixel pairs with feature correspondence in the ambiguous range ($0.5$–$0.75$) receive more training attention, improving boundary precision.

### Application

- Applied at every training iteration as part of the sampling strategy for the contrastive loss.
- See Appendix A.1 for full details on the resampling and re-weighting formulations.

### Links

- → **Scale-Aware Contrastive Learning**: directly modifies the sampling distribution of pixel pairs fed to the contrastive loss.
- → **Scale-Gate Mechanism**: ensures the scale gate receives sufficient gradient from scale-sensitive pairs.
- → **Multi-Granularity Segmentation**: prevents the model from ignoring small objects.

---

## 10. Promptable 3D Segmentation

### Definition

At inference, SAGA supports three modes of promptable 3D segmentation:

1. **2D point prompt**: the user clicks a point on a rendered view. SAGA renders the affinity feature map, retrieves the scale-gated feature at the clicked pixel, and segments the 3D scene by computing cosine similarities between this query feature and all Gaussians' scale-gated features. Segmentation takes **2–5 ms** per prompt (Table 4).

2. **Scale-based scene decomposition**: by varying the scale $s$, SAGA produces hierarchical segmentations from fine-grained parts ($s \approx 0.01$) to coarse regions ($s \approx 1.0$), as shown in Fig. 3.

3. **Automatic scene decomposition**: SAGA applies HDBSCAN clustering on the (scale-gated) affinity features of a randomly selected 1% of Gaussians, then assigns all remaining Gaussians to the nearest cluster centroid.

### Properties

- **Real-time**: 2–5 ms per segmentation query — at least 3 orders of magnitude faster than SA3D (45 s) and SA-GS (15 s) which require iterative mask refinement.
- **No re-rendering for scale change**: changing the scale $s$ only requires re-applying the scale gate $\mathcal{S}(s)$ to the existing features, not re-rendering the full scene.
- **Multi-view consistent**: because segmentation operates on 3D Gaussians (not 2D masks), results are inherently view-consistent.

### Application

Datasets and results:

| Dataset | Metric | SAGA | Best Competitor |
|---|---|---|---|
| NVOS | mIoU / mAcc | 92.6 / 98.6 | SA3D-GS: 92.6 / 98.5 |
| SPIn-NeRF | mIoU / mAcc | 93.4 / 99.2 | OmniSeg3D: 94.3 / 99.3 |

### Links

- → **Gaussian Affinity Features**: the cosine similarity between query and Gaussian features determines segmentation.
- → **Scale-Gate Mechanism**: the gate enables real-time granularity adjustment.
- → **Open-Vocabulary Segmentation**: extends promptable segmentation from point prompts to text prompts.

---

## 11. Open-Vocabulary Segmentation

### Definition

SAGA extends to open-vocabulary 3D segmentation by combining its affinity features with CLIP visual features through a vote-based clustering pipeline. The pipeline has three stages:

**Stage 1 — Constructing global features via clustering**: for a 2D mask $\mathbf{M}$ with scale $s_\mathbf{M}$, compute a scale-conditioned feature by averaging the rendered features of its pixels:

$$\mathbf{f}_\mathbf{M} = \frac{1}{|\delta(\mathbf{M})|} \sum_{\mathbf{p} \in \delta(\mathbf{M})} \mathbf{F}^{s_\mathbf{M}}(\mathbf{p}) \tag{Eq.14}$$

Then compute similarities between $\mathbf{f}_\mathbf{M}$ and anchor Gaussian features sampled from $\mathcal{G}$, and use HDBSCAN to cluster the 2D masks based on their distance in feature space.

**Stage 2 — Extracting CLIP features**: for each cluster centroid $\mathbf{T}$ in the vote graph $\mathcal{V}$, extract CLIP visual features by feeding masked images $\mathbf{I} \odot \mathbf{M}$ to the CLIP visual encoder.

**Stage 3 — Relevancy scoring**: given a text prompt, compute the relevancy score for each cluster:

$$r_\mathbf{T} = \frac{1}{|\mathcal{V}_\mathbf{T}|} \sum_{\mathbf{M} \in \mathcal{V}_\mathbf{T}} r_\mathbf{M} \tag{Eq.15}$$

where $\mathcal{V}_\mathbf{T}$ is the set of 2D masks assigned to cluster $\mathbf{T}$, and $r_\mathbf{M}$ is the relevancy score of mask $\mathbf{M}$ obtained by comparing CLIP textual and visual features.

### Properties

- **No language features in 3D**: SAGA does not distill language features into the 3D representation. It uses CLIP only at inference time on 2D masked images.
- **Limitation — CLIP ambiguity at large scales**: when a mask covers a large region (e.g., a bowl of noodles with an egg), CLIP may misclassify the contents because its visual encoder processes the entire masked region. SAGA sometimes misclassifies larger objects as their most salient sub-part (e.g., "egg" for the bowl).
- **Limitation — CLIP context dependency**: CLIP's effectiveness depends on seeing enough context. A wall segment without surrounding context may fail because CLIP cannot recognize it from the masked crop alone.

### Application

Results on the 3D-OVS dataset (Table 3, mIoU):

| Method | bed | bench | room | sofa | lawn | mean |
|---|---|---|---|---|---|---|
| LERF | 73.5 | 53.2 | 46.6 | 27.0 | 73.7 | 54.8 |
| 3D-OVS | 89.5 | 89.3 | 92.8 | 74.0 | 88.2 | 86.8 |
| LangSplat | 92.5 | 94.2 | 94.1 | 90.0 | 96.1 | 93.4 |
| N2F2 | 93.8 | 92.6 | 93.5 | 92.1 | 96.3 | 93.9 |
| **SAGA** | **97.4** | **95.4** | **96.8** | **93.5** | **96.6** | **96.0** |

SAGA achieves the best mean mIoU of 96.0, surpassing N2F2 (93.9) by 2.1 points.

### Links

- → **Gaussian Affinity Features**: clustering relies on the scale-gated affinity features as global descriptors for multi-view mask association.
- → **Scale-Gate Mechanism**: the scale-conditioned feature $\mathbf{f}_\mathbf{M}$ (Eq.14) uses the mask's physical scale.
- → **Promptable 3D Segmentation**: open-vocabulary extends the promptable interface from point prompts to language prompts.

---

## 12. What Existed Before and What This Paper Changes

### 12.1 Prior Approaches and Their Limitations

**NeRF-based iterative methods (SA3D, ISRF).**
SA3D uses SAM to segment a single view, then iteratively projects and refines the mask across views using the NeRF's volumetric rendering. ISRF trains a lightweight MLP on custom-designed 3D features to select objects. Both methods require iterative refinement — SA3D takes 45 seconds per segmentation and ISRF takes 3.3 seconds. Neither handles multi-granularity: they segment at a single level of detail per query.

**Implicit feature field methods (GARField, OmniSeg3D).**
GARField trains an implicit feature field conditioned on physical scale, using an MLP that takes 3D position and scale as input and outputs an affinity feature. This handles multi-granularity but requires multiple forward passes through the MLP for different scales, and the implicit representation tends to over-smooth features — small objects are lost at larger scales because the MLP's continuous interpolation merges them into the background. GARField takes 30 minutes to train and 30'70 ms for inference. OmniSeg3D uses hierarchical contrastive learning on multi-view SAM masks but requires 15 hours 40 minutes of training and 50'100 ms inference.

**3D-GS + video tracking methods (SA-GS, GaussianGrouping, SA3D-GS).**
SA-GS and SA3D-GS use video tracking to propagate SAM masks across views and assign labels to 3D Gaussians. This avoids iterative refinement but introduces an additional mask refinement pipeline that is separate from 3D-GS, adding engineering complexity. These methods do not explicitly address multi-granularity segmentation.

### 12.2 What This Paper Contributes

**Contribution 1: Gaussian affinity features.** SAGA attaches a learned affinity feature to each 3D Gaussian, converting segmentation from a post-hoc operation into an intrinsic property of the 3D representation. No separate feature field or iterative pipeline is needed.

**Contribution 2: Scale-gate mechanism.** A 512-parameter linear+sigmoid gate that maps scale to a channel-wise modulation of affinity features. This replaces GARField's MLP-based scale conditioning with a near-zero-cost alternative that better preserves small objects (because features are explicit, not interpolated by an MLP).

**Contribution 3: Scale-aware contrastive training.** A correspondence distillation loss that transfers SAM's 2D segmentation knowledge into 3D, with a custom resampling strategy that prevents scale-sensitivity collapse.

### 12.3 Side-by-Side: Prior Art vs. This Paper

| Dimension | SA3D | GARField | OmniSeg3D | SAGA |
|---|---|---|---|---|
| 3D Representation | NeRF | NeRF | NeRF | 3D-GS |
| Feature storage | volume rendering | implicit MLP | implicit MLP | per-Gaussian explicit |
| Multi-granularity | no | yes (MLP) | yes (hierarchical) | yes (scale gate) |
| Training time | — | 20 min 60 s | 15 h 40 min | 10 min 40 s |
| Inference time | 45 s | 30 min 70 ms | 50 min 100 ms | 2–5 ms |
| Scale change cost | re-segment | re-query MLP | re-query | re-apply gate (~0 ms) |
| Preserves small objects | — | no (over-smoothing) | — | yes (explicit features) |

### 12.4 The Core Shift in Thinking

Previous methods treat segmentation as something that happens *on top of* or *alongside* the radiance field — either by iteratively refining masks through the renderer, or by training a separate implicit feature field that mirrors the scene structure. Both approaches inherit the limitations of their auxiliary representations: iterative methods are slow because each step requires a full render, and implicit feature fields over-smooth because MLPs interpolate continuously.

SAGA makes segmentation an *intrinsic property* of each 3D Gaussian. Each Gaussian carries its own affinity feature, just as it carries its own color and opacity. The scale gate — a trivially small module — converts these features into scale-dependent segmentation outputs. This shift from "field that is queried" to "attribute that is read" eliminates both the speed bottleneck (no MLP queries or iterative refinement) and the over-smoothing problem (no continuous interpolation). The result is a method that segments in milliseconds while preserving fine-grained details that implicit methods lose.

---

## 13. Quick Reference Card

| # | Finding | Design Choice | Evidence |
|---|---|---|---|
| 1 | Multi-granularity is a channel-selection problem | Linear layer + sigmoid scale gate (512 params) | Comparable to GARField MLP; §A.4 interpretability |
| 2 | Explicit per-Gaussian features avoid over-smoothing | Attach $\mathbf{f}_g \in \mathbb{R}^D$ to each Gaussian | Fig. 4: 60% Gaussian shrinkage reveals fine structure |
| 3 | 2D→3D distillation via correspondence loss | Scale-aware contrastive learning (Eq.9) | NVOS 92.6 mIoU, SPIn-NeRF 93.4 mIoU |
| 4 | Ray-wise feature alignment matters | Feature norm regularization (Eq.10) | Fig. 5: FNR vs. no-FNR at threshold 0.95 |
| 5 | Spatial smoothing eliminates rasterization artifacts | K-nearest-neighbor feature smoothing | Fig. 5: LFS removes false positives |
| 6 | Scale-sensitive pairs need oversampling | 3-set resampling + re-weighting (§A.1) | Prevents scale gate collapse |
| 7 | Inference must be millisecond-scale for interaction | 3D-GS + explicit features + gate | 2–5 ms vs. 45 s (SA3D), 30 s (GARField) |
| 8 | Scale gate is representation-agnostic | Adapted to GARField and Instant-NGP | Fig. A1, A2: competitive performance |
| 9 | CLIP open-vocab struggles at large scales | Vote-based clustering + CLIP scoring | Table 3: 96.0 mean mIoU on 3D-OVS |

---

## 14. Open Questions

1. **Generalization beyond SAM**: SAGA inherits SAM's detection limitations — objects not appearing in SAM's 2D masks cannot be segmented (Fig. 6). This is particularly problematic for small or rare targets. Can the affinity features learn to generalize beyond SAM's coverage, perhaps through self-supervised objectives on 3D structure?

2. **CLIP context dependency in open-vocabulary**: CLIP's visual encoder requires sufficient context to recognize objects (e.g., a wall seen in isolation is hard to classify). The current masking strategy feeds cropped regions to CLIP, losing context. Addressing the multi-granularity ambiguity in CLIP semantics (e.g., "egg" vs. "bowl with egg") is a promising direction.

3. **Dynamic scenes**: SAGA operates on static 3D-GS reconstructions. Extending to 4D Gaussian Splatting for dynamic scene segmentation is unexplored.

4. **Scaling the affinity feature dimension**: the paper uses $D = 256$. What is the trade-off between dimension and segmentation quality? Can lower dimensions (e.g., $D = 32$) still achieve competitive results with the scale gate?

5. **Integration with language-grounded 3D**: rather than the current two-stage approach (segment with affinity features, then label with CLIP), could language features be directly distilled alongside affinity features in a unified training procedure?

---

## 15. Concept Dependency Graph

```
                           SAM (2D Foundation Model)
                                    │
                            extracts 2D masks
                                    │
                                    ▼
                    ┌──── Scale-Aware Pixel ────┐
                    │     Identity Vector       │
                    │         (Eq.6)             │
                    │              │              │
                    │     mask correspondence     │
                    │         (Eq.7)             │
                    │              │              │
                    │              ▼              │
    3D Gaussian ───►│   Scale-Aware Contrastive  │
    Splatting       │      Learning (Eq.7-9)     │
    (Eq.1)         │              │              │
        │           │         trains              │
        │           └──────────┬──────────────────┘
        │                      │
        │                      ▼
        │           Gaussian Affinity Features
        ├──────────►      (Eq.4)
        │                      │
        │               ┌──────┴──────┐
        │               ▼             ▼
        │        Local Feature   Feature Norm
        │        Smoothing       Regularization
        │          (LFS)           (Eq.10-12)
        │               │             │
        │               └──────┬──────┘
        │                      │
        │                      ▼
        │            Scale-Gate Mechanism
        │               (Eq.3, Eq.5)
        │                      │
        │           ┌──────────┼──────────┐
        │           ▼          ▼          ▼
        │     Promptable    Scene      Open-Vocab
        │     3D Segment.   Decomp.    Segment.
        │     (point prompt) (HDBSCAN)  (Eq.14-15)
        │                                 │
        │                            CLIP features
        │                                 │
        └─────────────────────────────────┘
               (shared 3D-GS backbone)

  Additional Training Strategy ──► Scale-Aware Contrastive Learning
  (Resampling & Re-weighting)       (prevents data imbalance)
```

---

## 16. Key Equations

| # | Equation | Description | Section |
|---|---|---|---|
| Eq.1 | $C(\mathbf{p}) = \sum_i c_i \alpha_i \prod_{j<i}(1 - \alpha_j)$ | 3D-GS alpha-blending rendering | §3.1 |
| Eq.2 | $s_\mathbf{M} = \sigma(\{P_i\}_{i \in \delta(\mathbf{M})})$ | Physical scale of a 2D mask in 3D | §3.1 |
| Eq.3 | $\mathbf{f}_g^s = \mathcal{S}(s) \odot \mathbf{f}_g$ | Scale-gated affinity feature (3D) | §3.3 |
| Eq.4 | $\mathbf{F}(\mathbf{p}) = \sum_i \mathbf{f}_{g_i} \alpha_i \prod_{j<i}(1 - \alpha_j)$ | Rendered 2D affinity feature map | §3.3 |
| Eq.5 | $\mathbf{F}'(\mathbf{p}) = \mathcal{S}(s) \odot \mathbf{F}(\mathbf{p})$ | Scale gate applied to 2D features (training) | §3.3 |
| Eq.6 | $\mathbf{V}'(s,\mathbf{p})$ piecewise definition | Scale-aware pixel identity vector | §3.4 |
| Eq.7 | $\text{Corr}_m = \mathbb{1}(\mathbf{V}(s,\mathbf{p}_1) \cdot \mathbf{V}(s,\mathbf{p}_2))$ | Mask correspondence (binary) | §3.4 |
| Eq.8 | $\text{Corr}_f = \langle \mathbf{F}^s(\mathbf{p}_1), \mathbf{F}^s(\mathbf{p}_2) \rangle$ | Feature correspondence (cosine sim.) | §3.4 |
| Eq.9 | $\mathcal{L}_{\text{corr}} = (1-2\text{Corr}_m)\max(\text{Corr}_f, 0)$ | Correspondence distillation loss | §3.4 |
| Eq.10 | $\ell_{\text{fnr}}(\mathbf{p}) = 1 - \|\mathbf{F}'(\mathbf{p})\|_2$ | Feature norm regularization | §3.5 |
| Eq.12 | $\mathcal{L}_{\text{SAGA}} = \mathcal{L}_{\text{corr}} + \lambda \, \ell_{\text{fnr}}$ | Total training loss | §3.5 |
| Eq.14 | $\mathbf{f}_\mathbf{M} = \frac{1}{|\delta(\mathbf{M})|}\sum_\mathbf{p} \mathbf{F}^{s_\mathbf{M}}(\mathbf{p})$ | Scale-conditioned mask feature (OVS) | §A.3 |
| Eq.15 | $r_\mathbf{T} = \frac{1}{|\mathcal{V}_\mathbf{T}|}\sum_{\mathbf{M} \in \mathcal{V}_\mathbf{T}} r_\mathbf{M}$ | Cluster relevancy score (OVS) | §A.3 |

---

## 17. Reference Map

**3D Gaussian Splatting**
- Kerbl et al. 2023 — original 3D-GS paper

**2D Promptable Segmentation**
- Kirillov et al. 2023 — SAM (Segment Anything Model)
- Zou et al. 2023 — SEEM (similar foundation model)

**3D Segmentation in Radiance Fields**
- Cen et al. 2023 — SA3D (iterative SAM projection in NeRF)
- Cen et al. 2024 — SA3D-GS (3D-GS version of SA3D)
- Goel et al. 2023 — ISRF (lightweight MLP for 3D selection)
- Ye et al. 2024 — GaussianGrouping (video tracking + 3D-GS)
- Hu et al. 2024 — SA-GS (SAM + 3D-GS with video tracking)
- Ying et al. 2024 — OmniSeg3D (hierarchical contrastive learning)

**Scale-Conditioned Features**
- Kim et al. 2024 — GARField (scale-conditioned implicit feature field)
- Kerr et al. 2023 — LERF (language-embedded radiance fields, relevancy score)

**Open-Vocabulary 3D Segmentation**
- Qin et al. 2024 — LangSplat (3D language Gaussian splatting)
- Bhalgat et al. 2024 — N2F2 (neural feature fusion fields)
- Liu et al. 2024a — 3D-OVS dataset

**Contrastive / Distillation Methods**
- Hamilton et al. 2022 — correspondence distillation loss (adapted by SAGA)
- Tschernezki et al. 2022 — N3F (neural feature fusion fields)
- Fan et al. 2023 — NeRF-SOS (unsupervised 3D segmentation)

**Feature Field Distillation**
- Kobayashi et al. 2022 — DFF (distilled feature fields)
- Siddiqui et al. 2023 — panoptic lifting for 3D understanding

**Interactive Segmentation**
- Boykov and Jolly 2001; Grady 2006; Rother et al. 2004 — classic 2D interactive segmentation (GrabCut, graph cuts, random walks)
