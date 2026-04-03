# CoMe: Confidence-Based Mesh Extraction from 3D Gaussians — Study Note (Concept Mind-Map)

> Paper: Confidence-Based Mesh Extraction from 3D Gaussians (arXiv:2603.24725v1)
> Authors: Lukas Radl\*, Felix Windisch\*, Andreas Kurz\*, Thomas Köhler, Michael Steiner, Markus Steinberger
> Affiliations: Graz University of Technology, Austria; Huawei Technologies, Austria
> Code: https://r4dl.github.io/CoMe/

This note organizes every key concept in the paper as a mind-map.
Each concept is broken down into four facets:

- **Definition** — what it is, stated plainly
- **Properties** — its mathematical or behavioral characteristics
- **Application** — how the paper (or the field) uses it
- **Links** — connections to other concepts in this map

---

## 0. The Big Picture

```
           Multi-view Images
                  │
                  ▼
     ┌──────────────────────┐
     │  3D Gaussian Splatting │
     │   (alpha-blending)     │
     └──────────┬─────────────┘
                │
    "Geometry and appearance are coupled"
    "Photometric loss dominates in hard regions"
    "View-dependent effects faked via geometry"
                │
    ┌───────────┼───────────────────────┐
    │           │                       │
    ▼           ▼                       ▼
 Decoupled   Confidence-Aware       Variance-Reducing
 Appearance  Gaussian Splatting     Losses
    │           │                       │
    │     ┌─────┼─────┐          ┌──────┼──────┐
    │     ▼           ▼          ▼             ▼
    │  L_conf    Confidence-  L_color-var  L_normal-var
    │  (Eq.9)   Steered       (Eq.13)      (Eq.14)
    │            Densif.
    │            (Eq.12)
    │                │
    └────────┬───────┘
             ▼
    Marching Tetrahedra
             │
             ▼
    ┌──────────────────┐
    │  High-Fidelity    │
    │  Unbounded Mesh   │
    │  (F1 = 0.521 T&T) │
    └──────────────────┘
```

The paper's core argument: previous 3DGS-based mesh extraction methods either rely on multi-view geometric constraints, monocular depth priors, or mesh-aware optimization — all adding computational overhead. CoMe instead introduces three **purely self-supervised** mechanisms: (1) a confidence framework that dynamically balances photometric and geometric losses, (2) variance-reducing losses that align individual Gaussians to surfaces, and (3) a decoupled appearance model that separates illumination from structure in the D-SSIM loss. Together these achieve state-of-the-art unbounded mesh extraction without any external priors.

---

## 1. The Geometry-Appearance Coupling Problem

### Definition

In 3DGS, each Gaussian primitive simultaneously encodes **geometry** (position, orientation, scale) and **appearance** (SH coefficients). Because the photometric loss (Eq.4) operates on the final blended pixel color, the optimizer can reduce photometric error by deforming geometry rather than adjusting appearance — especially for high-frequency view-dependent effects that SH (limited to low-frequency components) cannot represent.

The symptom: highly opaque Gaussians placed *behind* the true surface, blocked by semitransparent Gaussians from certain views, to fake specular highlights or reflections. This produces correct renderings but corrupted geometry.

### Properties

- SH coefficients are limited to low-frequency view-dependent effects (typically degree 3, 48 coefficients per Gaussian).
- When the appearance model is too weak to explain the observed color variation, the optimizer resorts to geometric deformation — the only remaining degree of freedom.
- This coupling is amplified by alpha-blending (Eq.1): since only the final blended color is supervised, individual primitives have no incentive to model correct per-primitive radiance.
- Regions with varying illumination, reflections, or semi-transparent surfaces are most affected.

### Application

This coupling problem is the central motivation for all three contributions in CoMe:
- **Confidence framework** (Sec.3.3): lets the optimizer downweight photometric loss in regions where appearance cannot be explained, preventing geometric corruption.
- **Variance losses** (Sec.3.4): force individual Gaussians to model correct color/normals, counteracting the free-riding enabled by alpha-blending.
- **Decoupled appearance** (Sec.3.2): absorbs illumination variation that would otherwise be baked into geometry.

### Links

- -> **Alpha-Blending Rendering**: the rendering equation that enables the free-riding problem
- -> **Confidence-Aware Gaussian Splatting**: the primary mechanism to resolve the coupling
- -> **Color Variance Loss**: directly penalizes per-primitive color disagreement caused by coupling
- -> **Decoupled D-SSIM Appearance**: absorbs illumination to prevent it from corrupting geometry

---

## 2. 3DGS Preliminaries and Alpha-Blending Rendering

### Definition

3D Gaussian Splatting [26] represents a scene with $N$ anisotropic 3D Gaussians. Each primitive $i$ is characterized by:
- Position $\boldsymbol{\mu}_i \in \mathbb{R}^3$
- Rotation $\boldsymbol{R}_i \in SO(3)$
- Diagonal scaling matrix $\boldsymbol{S}_i \in \text{diag}(\mathbb{R}^3_+)$
- Opacity $o_i \in [0,1]$
- Spherical harmonics coefficients $\boldsymbol{\theta}_i$

The covariance matrix is $\boldsymbol{\Sigma}_i = \boldsymbol{R}_i \boldsymbol{S}_i \boldsymbol{S}_i^T \boldsymbol{R}_i^T$.

To render a pixel, Gaussians are sorted front-to-back along the corresponding ray $\boldsymbol{r}(t) = \boldsymbol{o} + t\boldsymbol{d}$ and accumulated via alpha-blending:

$$\hat{\boldsymbol{I}}(\boldsymbol{r}) = \sum_{i=0}^{N-1} w_i(\boldsymbol{r}) \; \text{sh}(\boldsymbol{\theta}_i, \boldsymbol{d}), \tag{Eq.1}$$

where $w_i(\boldsymbol{r}) = \alpha_i(\boldsymbol{r}) \prod_{j=1}^{i-1}(1-\alpha_j(\boldsymbol{r}))$ are the blending weights.

Following recent works [19,55,70], opacity is evaluated in 3D at the point $\boldsymbol{x}_i^*$ of maximum contribution along the ray:

$$\alpha_i(\boldsymbol{r}) = o_i \exp\left(-\frac{1}{2}(\boldsymbol{x}_i^* - \boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}_i^{-1} (\boldsymbol{x}_i^* - \boldsymbol{\mu}_i)\right), \tag{Eq.2}$$

where $\boldsymbol{x}_i^*$ minimizes the Mahalanobis distance between the ray and the Gaussian center:

$$\boldsymbol{x}_i^* = \boldsymbol{o} + \frac{\boldsymbol{d}^T \boldsymbol{\Sigma}_i^{-1}(\boldsymbol{\mu}_i - \boldsymbol{o})}{\boldsymbol{d}^T \boldsymbol{\Sigma}_i^{-1} \boldsymbol{d}} \boldsymbol{d}. \tag{Eq.3}$$

### Properties

- Alpha-blending produces smooth, differentiable renderings but allows individual Gaussians to "hide" behind others — only the aggregated pixel color is supervised.
- The blending weight $w_i$ decays exponentially with depth ordering: front Gaussians dominate.
- 3D opacity evaluation (Eq.2–3) replaces the older 2D splatting approach, providing more geometrically accurate depth.
- Complexity: rendering is $O(N \cdot K)$ per pixel where $K$ is the average number of Gaussians per ray.

### Application

This rendering equation is the foundation for all losses in the paper. The photometric loss (Eq.4), confidence loss (Eq.9), variance losses (Eq.13–14), and confidence rendering (Eq.11) all operate through or alongside this alpha-blending pipeline.

### Links

- -> **The Geometry-Appearance Coupling Problem**: alpha-blending enables geometric free-riding
- -> **Color Variance Loss**: addresses the lack of per-primitive supervision in Eq.1
- -> **Normal Variance Loss**: same principle applied to normals
- -> **Confidence Rendering**: uses the same blending weights $w_i(\boldsymbol{r})$ to render confidence maps

---

## 3. Photometric Reconstruction Loss

### Definition

The standard 3DGS training loss combines $\mathcal{L}_1$ and D-SSIM [61]:

$$\mathcal{L}_{\text{rgb}} = (1 - \lambda_{\text{rgb}})\mathcal{L}_1(\boldsymbol{I}, \hat{\boldsymbol{I}}) + \lambda_{\text{rgb}}\mathcal{L}_{\text{D-SSIM}}(\boldsymbol{I}, \hat{\boldsymbol{I}}), \tag{Eq.4}$$

with $\lambda_{\text{rgb}} = 0.2$. The D-SSIM dissimilarity decomposes into three terms:

$$\mathcal{L}_{\text{D-SSIM}}(\boldsymbol{I}, \hat{\boldsymbol{I}}) = 1 - l(\boldsymbol{I}, \hat{\boldsymbol{I}}) \cdot c(\boldsymbol{I}, \hat{\boldsymbol{I}}) \cdot s(\boldsymbol{I}, \hat{\boldsymbol{I}}), \tag{Eq.5}$$

where $l$, $c$, $s$ are the **luminance**, **contrast**, and **structure** terms respectively.

### Properties

- The luminance term $l(\cdot)$ is highly sensitive to illumination changes (Fig.2 shows 9x error reduction when appearance compensation is applied to the luminance term alone).
- The contrast and structure terms are inherently illumination-invariant — they measure relative variations rather than absolute intensity.
- When the appearance model $\hat{\boldsymbol{I}}^{\text{app}}$ is applied to all three D-SSIM terms (as done by prior work), CNN upsampling artifacts from $\hat{\boldsymbol{I}}^{\text{app}}$ pollute the contrast and structure gradients.

### Application

This loss is the backbone of 3DGS optimization. CoMe modifies it in two ways:
1. The confidence framework (Eq.9) wraps $\mathcal{L}_{\text{rgb}}$ with learnable per-pixel confidence weights.
2. The decoupled appearance module (Eq.8) selectively applies $\hat{\boldsymbol{I}}^{\text{app}}$ only to the luminance term.

### Links

- -> **Decoupled D-SSIM Appearance**: modifies Eq.5 to selectively apply appearance correction
- -> **Confidence-Aware Gaussian Splatting**: wraps this loss with confidence weighting (Eq.9)
- -> **Total Loss Function**: $\mathcal{L}_{\text{rgb}}$ appears inside $\mathcal{L}_{\text{conf}}$

---

## 4. Appearance Modeling (VastGaussian Baseline)

### Definition

The appearance model from VastGaussian [35] compensates for camera-specific image sensor processing (ISP) differences across views. It uses per-image latent codes $\boldsymbol{\rho}_i \in \mathbb{R}^{64}$ concatenated with a downsampled rendering, processed by a CNN $F_\Theta$:

$$\boldsymbol{M}_i = F_\Theta\left([\text{ds}_{32}(\hat{\boldsymbol{I}}), \boldsymbol{\rho}_i]\right), \tag{Eq.6}$$

producing a per-channel corrective mapping $\boldsymbol{M}_i \in \mathbb{R}^{H \times W \times 3}$. The rendered image is transformed as:

$$\hat{\boldsymbol{I}}^{\text{app}} = \hat{\boldsymbol{I}} \odot \sigma(\boldsymbol{M}_i), \tag{Eq.7}$$

where $\sigma(\cdot)$ is the sigmoid activation. $\hat{\boldsymbol{I}}^{\text{app}}$ is only used for the $\mathcal{L}_1$ term in Eq.4.

### Properties

- The CNN takes a 32x-downsampled rendering concatenated with the 64-dim latent, producing a full-resolution correction map.
- Prior work applies $\hat{\boldsymbol{I}}^{\text{app}}$ to $\mathcal{L}_1$ only, leaving $\mathcal{L}_{\text{D-SSIM}}$ with the original rendering — on the assumption that SSIM is "inherently structural."
- This assumption is partially wrong: the luminance term of D-SSIM *is* sensitive to illumination changes.

### Application

CoMe improves this baseline in several ways (Sec. A.1):
- Replaces sigmoid with $\exp(\cdot)$ to handle overexposed images.
- Zero-initializes the final conv layer ($\boldsymbol{W}=\boldsymbol{0}$, $\boldsymbol{b}=\boldsymbol{0}$) for stable early training.
- Detaches $\text{ds}_{32}(\hat{\boldsymbol{I}})$ to prevent appearance gradients from flowing into the 3DGS model.
- Uses reflection-padding instead of center-cropping for downsampling.
- Adds positional encoding $(u, v, r(u,v))$ with $r(u,v) = \sqrt{u^2+v^2}$ (Eq.16) to compensate for vignetting.
- CNN input dimensionality increases to 70 (vs 67 for VastGaussian).
- Custom fused CUDA kernel for 5x speedup over naive PyTorch.

These improvements alone yield F1-score gains: 0.475 -> 0.484 on Tanks & Temples (Tab.4).

### Links

- -> **Decoupled D-SSIM Appearance**: extends this model to selectively handle D-SSIM terms
- -> **Photometric Reconstruction Loss**: $\hat{\boldsymbol{I}}^{\text{app}}$ modifies the $\mathcal{L}_1$ portion of $\mathcal{L}_{\text{rgb}}$
- -> **The Geometry-Appearance Coupling Problem**: insufficient appearance modeling forces geometric compensation

---

## 5. Decoupled D-SSIM Appearance

### Definition

The key observation: the luminance term $l(\cdot)$ in D-SSIM (Eq.5) is highly dependent on illumination, while contrast $c(\cdot)$ and structure $s(\cdot)$ are inherently illumination-invariant. Prior work applies the appearance-corrected image $\hat{\boldsymbol{I}}^{\text{app}}$ only to $\mathcal{L}_1$, leaving the full $\mathcal{L}_{\text{D-SSIM}}$ with the raw rendering.

CoMe selectively applies $\hat{\boldsymbol{I}}^{\text{app}}$ only to the luminance term of D-SSIM:

$$\mathcal{L}_{\text{D-SSIM}}^{\text{dec}} = 1 - l(\boldsymbol{I}, \hat{\boldsymbol{I}}^{\text{app}}) \cdot c(\boldsymbol{I}, \hat{\boldsymbol{I}}) \cdot s(\boldsymbol{I}, \hat{\boldsymbol{I}}). \tag{Eq.8}$$

### Properties

- **Why not apply $\hat{\boldsymbol{I}}^{\text{app}}$ to all terms?** Using $\hat{\boldsymbol{I}}^{\text{app}}$ for contrast and structure allows the appearance model to compensate for *incorrect renderings* (not just illumination), which degrades 3D coherence. Additionally, CNN upsampling artifacts in $\hat{\boldsymbol{I}}^{\text{app}}$ pollute the gradients for these terms (visible in Fig.2).
- **Why not drop the luminance term entirely?** Removing $l(\cdot)$ produces results comparable to the unmodified VastGaussian baseline (Tab.4: w/o $l(\cdot)$ = 0.483 vs VastG. Imp. = 0.484), showing the luminance term alone contributes nothing unless appearance is decoupled.
- The decoupled approach yields F1 = 0.490 (w/o SSIM Dec.) vs 0.493 (Ours, full decoupling) on Tanks & Temples (Tab.4).

### Application

In the full pipeline:
1. $\hat{\boldsymbol{I}}^{\text{app}}$ is used for the $\mathcal{L}_1$ term (as in prior work).
2. $\hat{\boldsymbol{I}}^{\text{app}}$ is used **only for the luminance term** in $\mathcal{L}_{\text{D-SSIM}}$.
3. Contrast and structure terms use the original rendering $\hat{\boldsymbol{I}}$.

This produces a more consistent 3D scene representation, especially on datasets with varying illumination (ScanNet++, Tanks & Temples — Caterpillar and Ignatius scenes).

### Links

- -> **Photometric Reconstruction Loss**: modifies how $\mathcal{L}_{\text{D-SSIM}}$ is computed within Eq.4
- -> **Appearance Modeling**: extends the VastGaussian baseline with selective term application
- -> **The Geometry-Appearance Coupling Problem**: prevents illumination from corrupting geometry through the D-SSIM loss channel

---

## 6. Confidence-Aware Gaussian Splatting

### Definition

The core contribution. Each Gaussian is augmented with a learnable scalar confidence value. The photometric loss $\mathcal{L}_{\text{rgb}}$ is weighted by a rendered confidence map $\hat{C}$, with a logarithmic penalty to prevent trivial solutions:

$$\mathcal{L}_{\text{conf}} = \mathcal{L}_{\text{rgb}} \cdot \hat{C} - \beta \cdot \log \hat{C}, \tag{Eq.9}$$

where $\beta \in \mathbb{R}_+$ balances the two terms.

**Intuition**: When $\hat{C} = 1$, the loss reduces to $\mathcal{L}_{\text{rgb}}$ (standard behavior). When $\hat{C} < 1$, the model trades photometric reconstruction quality for uncertainty — downweighting regions it cannot explain well. The $-\beta \log \hat{C}$ term prevents the trivial solution of $\hat{C} \to 0$ everywhere.

**Geometric losses remain untouched** by the confidence weighting. This is critical: the confidence value can thus balance the *relative influence* of photometric vs geometric supervision.

### Properties

- The gradient w.r.t. $\hat{C}$ is (Eq.18):

$$\frac{\partial \mathcal{L}_{\text{conf}}}{\partial \hat{C}} = \mathcal{L}_{\text{rgb}} - \frac{\beta}{\hat{C}}.$$

At equilibrium: $\hat{C}^* = \beta / \mathcal{L}_{\text{rgb}}$. Regions with high photometric error get low confidence; easy regions get high confidence.

- $\hat{C}$ is clamped to $[0.001, 5.0]$ for numerical stability. With $\beta = 7.5 \times 10^{-2}$, the maximum gradient magnitude is $-75$ (at $\hat{C}_{\min} = 0.001$).
- The $\max \hat{C} = 5.0$ cap ensures gradients flowing to individual Gaussians remain reasonably scaled, preventing over-densification of already well-reconstructed regions.
- Inspired by Bayesian uncertainty estimation [25] and feed-forward reconstruction models [59] (DUSt3R).

### Application

- Confidence learning rate: $2.5 \times 10^{-4}$.
- $\beta = 7.5 \times 10^{-2}$ (ablated in Fig.7; best F1-score among $\beta \in [0, 0.2]$).
- $\mathcal{L}_{\text{conf}}$ is enabled from iteration 500 (at the start of densification). Earlier activation does not improve stability (Tab.5).
- Ablation (Tab.3): adding $\mathcal{L}_{\text{conf}}$ to the improved appearance baseline yields +0.016 F1 on Tanks & Temples (0.493 -> 0.509) and a large +0.030 gain on ScanNet++ (0.625 -> 0.655).

### Derivation: Confidence Equilibrium

At the stationary point $\frac{\partial \mathcal{L}_{\text{conf}}}{\partial \hat{C}} = 0$:

**Step 1.** Set $\mathcal{L}_{\text{rgb}} - \beta / \hat{C} = 0$.

**Step 2.** Solve: $\hat{C}^* = \beta / \mathcal{L}_{\text{rgb}}$.

**Consequence**: In well-reconstructed regions ($\mathcal{L}_{\text{rgb}}$ small), $\hat{C}^*$ is large — the model is confident. In challenging regions ($\mathcal{L}_{\text{rgb}}$ large), $\hat{C}^*$ shrinks — photometric loss is downweighted, letting geometric losses guide optimization.

### Links

- -> **Confidence Rendering**: defines how per-primitive $\tilde{\gamma}_i$ are aggregated into $\hat{C}$
- -> **Confidence-Aware Densification**: adapts the densification threshold using per-primitive confidence
- -> **Photometric Reconstruction Loss**: $\mathcal{L}_{\text{rgb}}$ is the quantity being weighted
- -> **Total Loss Function**: $\mathcal{L}_{\text{conf}}$ replaces $\mathcal{L}_{\text{rgb}}$ in the final loss
- -> **The Geometry-Appearance Coupling Problem**: confidence downweights photometric loss where coupling is worst

---

## 7. Confidence Rendering

### Definition

Each Gaussian is endowed with a learnable scalar $\gamma_i$, initialized to 0. The per-primitive confidence is:

$$\tilde{\gamma}_i = \exp(\gamma_i), \tag{Eq.10}$$

ensuring an initial value of 1 (neutral confidence). The rendered confidence map is obtained via alpha-blending with the same weights as color rendering:

$$\hat{C}(\boldsymbol{r}) = \sum_{i=0}^{N-1} w_i(\boldsymbol{r}) \; \tilde{\gamma}_i. \tag{Eq.11}$$

### Properties

- The $\exp(\cdot)$ parameterization ensures $\tilde{\gamma}_i > 0$ without explicit clamping.
- Initialization at $\gamma_i = 0$ means $\tilde{\gamma}_i = 1$, so $\hat{C} \approx 1$ initially — the confidence loss starts as the standard $\mathcal{L}_{\text{rgb}}$.
- The rendered $\hat{C}$ is clamped to $[0.001, 5.0]$ (Sec. A.2).
- Confidence maps effectively isolate reflective surfaces, thin foliage, and rarely observed areas (Fig.3: Barn roof, Meetingroom reflections, Truck chrome details).
- Low-confidence regions are still well-reconstructed — the geometric losses are not downweighted.

### Application

- Confidence values are visualized in Fig.3 across three scenes.
- Indoor scenes show higher average confidence (3.22 on Mip-NeRF 360 indoor test views) than outdoor scenes (1.66), reflecting the higher photometric ambiguity of outdoor environments (Fig.12).
- The learned confidence correlates directly with the mean squared error map (Fig.12), validating that it identifies under-reconstructed regions.

### Links

- -> **Confidence-Aware Gaussian Splatting**: $\hat{C}$ is the confidence map used in $\mathcal{L}_{\text{conf}}$ (Eq.9)
- -> **Confidence-Aware Densification**: per-primitive $\tilde{\gamma}_i$ modulates the densification threshold
- -> **Alpha-Blending Rendering**: uses the same blending weights $w_i(\boldsymbol{r})$

---

## 8. Confidence-Aware Densification

### Definition

Standard gradient-based densification in 3DGS [26,66,70] splits or clones Gaussians when accumulated positional gradients exceed a threshold $\tau_{\text{grad}} \in \mathbb{R}_+$. The confidence-based balancing (Eq.9) changes gradient magnitudes: when confidence is large and $\mathcal{L}_{\text{rgb}}$ is small, the loss $\mathcal{L}_{\text{conf}}$ can become negative, and in low-confidence regions, absolute gradients decrease.

To prevent over-densification in difficult-to-reconstruct regions (where confidence is low), CoMe scales the threshold by the inverse of per-primitive confidence:

$$\tilde{\tau}_{\text{grad}} = \frac{\tau_{\text{grad}}}{\min(\tilde{\gamma}_i, 1)}. \tag{Eq.12}$$

### Properties

- Low-confidence Gaussians ($\tilde{\gamma}_i < 1$) get a *higher* densification threshold, making them harder to split/clone — preventing the accumulation of small, spurious Gaussians in challenging regions.
- High-confidence Gaussians ($\tilde{\gamma}_i \geq 1$) retain the standard threshold ($\tilde{\tau}_{\text{grad}} = \tau_{\text{grad}}$) due to the $\min(\tilde{\gamma}_i, 1)$ clamp.
- The $\min$ clamp is one-sided: it only *raises* the threshold for unconfident Gaussians, never lowers it for confident ones.
- Disabling confidence-steered densification yields slightly lower F1 and 20% more primitives (Tab.5: 0.521 -> 0.516, 1.27M -> 1.53M Gaussians).

### Application

This mechanism controls model complexity. With $\beta = 0.075$, the final model uses slightly *fewer* primitives than the SOF baseline while achieving higher F1 (Tab.5: 1.27M vs SOF's baseline).

The $\beta$ parameter indirectly controls primitive count (Fig.7):
- $\beta = 0$ (no incentive to maximize $\hat{C}$): under-reconstruction, very few primitives.
- $\beta = 0.2$ (too harsh): the $-\beta\log\hat{C}$ penalty is so strong that $\hat{C}$ stays near 1, so $\mathcal{L}_{\text{rgb}}$ is never downweighted, and large absolute gradients cause excessive densification.
- $\beta = 0.075$: sweet spot.

### Links

- -> **Confidence Rendering**: uses per-primitive $\tilde{\gamma}_i$ from Eq.10
- -> **Confidence-Aware Gaussian Splatting**: the confidence loss (Eq.9) changes gradient magnitudes, motivating this adaptation
- -> **Total Loss Function**: densification operates on gradients of the total loss

---

## 9. Color Variance Loss

### Definition

Alpha-blending (Eq.1) supervises only the final blended pixel color, so individual Gaussians have no incentive to predict correct per-primitive colors. View-dependent effects are often faked by placing highly opaque Gaussians behind surfaces, blocked by semitransparent ones from certain angles.

The color variance loss penalizes the weighted per-primitive squared deviation from the ground-truth pixel color $\boldsymbol{I}$:

$$\mathcal{L}_{\text{color-var}} = \sum_{i=0}^{N-1} w_i(\boldsymbol{r}) \|\text{sh}(\boldsymbol{\theta}_i, \boldsymbol{d}) - \boldsymbol{I}\|_2^2. \tag{Eq.13}$$

Under the assumption that $\boldsymbol{I}$ represents the rendered color, this is equivalent to minimizing the blended color variance along the ray.

### Properties

- This is a *weighted variance* measure: each primitive's deviation is weighted by its blending contribution $w_i$.
- Cannot naively replace the standard loss with per-sample supervision (as done in [74] for NeRF) because the D-SSIM term operates on image patches and cannot be applied per-primitive.
- The loss is complementary to (not replacing) the standard photometric loss.
- Yields the strongest gains on Tanks & Temples (outdoor scenes with abundant view-dependent effects): Tab.3 shows +0.010 F1 on T&T vs +0.003 on ScanNet++.

### Derivation: Gradients of $\mathcal{L}_{\text{color-var}}$ (Appendix C.1)

The gradient w.r.t. the view-dependent color $\text{sh}(\boldsymbol{\theta}_i, \boldsymbol{d})$:

$$\frac{\partial \mathcal{L}_{\text{color-var}}}{\partial \text{sh}(\boldsymbol{\theta}_i, \boldsymbol{d})} = 2w_i(\boldsymbol{r})(\text{sh}(\boldsymbol{\theta}_i, \boldsymbol{d}) - \boldsymbol{I}) \tag{Eq.22}$$

The gradient w.r.t. opacity $\alpha_i$ requires accounting for the effect on all subsequent Gaussians $j > i$ through the blending weights:

$$\frac{\partial \mathcal{L}_{\text{color-var}}}{\partial \alpha_i} = \|\text{sh}(\boldsymbol{\theta}_i, \boldsymbol{d}) - \boldsymbol{I}\|_2^2 \prod_{j=1}^{i-1}(1-\alpha_j(\boldsymbol{r})) - \frac{1}{1-\alpha_i}\sum_{j=i+1}^{N-1} w_j(\boldsymbol{r})\|\text{sh}(\boldsymbol{\theta}_j, \boldsymbol{d}) - \boldsymbol{I}\|_2^2 \tag{Eq.27}$$

**Implementation trick**: The suffix sum $\sum_{j=i+1}^{N-1} w_j(\boldsymbol{r})\|\text{sh}(\boldsymbol{\theta}_j, \boldsymbol{d}) - \boldsymbol{I}\|_2^2$ is computed once per pixel during the forward pass, then decremented for each primitive during the backward pass (inheriting the front-to-back pass from StopThePop [49]/SOF [50]).

### Application

- Weight: $\lambda_{\text{color-var}} = 5 \times 10^{-1}$ (robust to variation: Tab.5 shows $\lambda = 0.1$ and $\lambda = 1.0$ both yield reasonable results).
- Visual effect: Fig.4 shows that first-hit Gaussians (closest to the camera) have much more consistent colors when variance losses are applied.

### Links

- -> **Alpha-Blending Rendering**: addresses the per-primitive supervision gap in Eq.1
- -> **Normal Variance Loss**: analogous loss for per-primitive normal alignment
- -> **The Geometry-Appearance Coupling Problem**: constrains the geometric free-riding enabled by alpha-blending
- -> **Total Loss Function**: appears as a weighted term in Eq.15

---

## 10. Normal Variance Loss

### Definition

Per-pixel normals used in depth-normal consistency losses [20,50,70] are also alpha-blended, so individual Gaussian normals $\boldsymbol{n}_i$ may not align with the true surface. The normal variance loss penalizes deviation of each primitive's normal from the blended pixel normal $\boldsymbol{N} = \sum_{i=0}^{N-1} w_i(\boldsymbol{r})\boldsymbol{n}_i$:

$$\mathcal{L}_{\text{normal-var}} = \sum_{i=0}^{N-1} w_i(\boldsymbol{r}) \|\boldsymbol{n}_i - \boldsymbol{N}\|_2^2 = 1 - \|\boldsymbol{N}\|_2^2. \tag{Eq.14}$$

### Properties

- The reformulation $1 - \|\boldsymbol{N}\|_2^2$ is elegant: it can be computed entirely in the forward pass (no need for a second pass to compute the mean). If all per-primitive normals are identical, $\|\boldsymbol{N}\|_2 = 1$ and the loss is 0. If they disagree, $\|\boldsymbol{N}\|_2 < 1$ and the loss increases.
- Uses the blended normal as the "ground truth" (no external supervision needed — self-supervised).
- The depth distortion loss [1] can be interpreted as a variance loss on the blended depth — analogous in spirit.
- Strongest gains on ScanNet++ (indoor, planar surfaces): Tab.3 shows +0.010 F1 on ScanNet++ vs +0.002 on T&T. Enforcing normal consensus acts as a strong regularizer for walls, tables, and floors.

### Derivation: $\mathcal{L}_{\text{normal-var}}$ Simplification (Appendix C.2, Eq.28–32)

**Step 1.** Expand the squared norm:
$$\mathcal{L}_{\text{normal-var}} = \sum_i w_i(\boldsymbol{r})(\|\boldsymbol{n}_i\|_2^2 - 2\boldsymbol{n}_i \cdot \boldsymbol{N} + \|\boldsymbol{N}\|_2^2)$$

**Step 2.** Since $\|\boldsymbol{n}_i\|_2 = 1$ (unit normals) and $\sum_i w_i = 1$:
$$= 1 - 2\boldsymbol{N} \cdot \boldsymbol{N} + \|\boldsymbol{N}\|_2^2 = 1 - \|\boldsymbol{N}\|_2^2$$

The gradient w.r.t. $\boldsymbol{n}_i$ (Eq.33):
$$\frac{\partial \mathcal{L}_{\text{normal-var}}}{\partial \boldsymbol{n}_i} = 2w_i(\boldsymbol{r})(\boldsymbol{n}_i - \boldsymbol{N})$$

The gradient w.r.t. $\alpha_i$ (Eq.34):
$$\frac{\partial \mathcal{L}_{\text{normal-var}}}{\partial \alpha_i} = \|\boldsymbol{n}_i - \boldsymbol{N}\|_2^2 \prod_{j=1}^{i-1}(1-\alpha_j(\boldsymbol{r})) - \frac{1}{1-\alpha_i}\sum_{j=i+1}^{N-1}w_j(\boldsymbol{r})\|\boldsymbol{n}_j - \boldsymbol{N}\|_2^2$$

Structurally identical to the color variance gradient (Eq.27), using the same backward pass implementation.

### Application

- Weight: $\lambda_{\text{normal-var}} = 5 \times 10^{-3}$ (robust to order-of-magnitude changes: Tab.5).
- Visual effect: Fig.4(b) and Fig.13 show substantially smoother normal maps for first-hit Gaussians.
- Fig.13 shows that the loss improves both first-hit and last-hit normal consistency, while full alpha-blended normals remain detailed.

### Links

- -> **Color Variance Loss**: analogous loss for per-primitive color alignment
- -> **Alpha-Blending Rendering**: uses the same blending weights $w_i(\boldsymbol{r})$
- -> **Total Loss Function**: appears as a weighted term in Eq.15
- -> **The Geometry-Appearance Coupling Problem**: enforces geometric consistency at the individual primitive level

---

## 11. Total Loss Function

### Definition

The final loss combines all components:

$$\mathcal{L} = \mathcal{L}_{\text{conf}} + \mathcal{L}_{\text{geom}} + \lambda_{\text{color-var}}\mathcal{L}_{\text{color-var}} + \lambda_{\text{normal-var}}\mathcal{L}_{\text{normal-var}}, \tag{Eq.15}$$

where $\mathcal{L}_{\text{geom}}$ denotes the geometry losses from SOF [50] (surface-alignment, depth-normal consistency).

### Properties

| Component | Weight | Role |
|---|---|---|
| $\mathcal{L}_{\text{conf}}$ | — | Confidence-weighted photometric loss (replaces $\mathcal{L}_{\text{rgb}}$) |
| $\mathcal{L}_{\text{geom}}$ | (from SOF) | Geometric priors (surface alignment, depth-normal consistency) |
| $\mathcal{L}_{\text{color-var}}$ | $\lambda = 0.5$ | Per-primitive color alignment |
| $\mathcal{L}_{\text{normal-var}}$ | $\lambda = 0.005$ | Per-primitive normal alignment |

- $\mathcal{L}_{\text{conf}}$ is enabled from iteration 500 (start of densification).
- Confidence learning rate: $2.5 \times 10^{-4}$.
- $\beta = 7.5 \times 10^{-2}$.
- All hyperparameters are fixed across all scenes (no per-scene tuning).

### Application

Built on top of the SOF [50] codebase. Mesh extraction uses Marching Tetrahedra with binary search [70], using the faster implementation from SOF. Optimization completes in under 20 minutes on an RTX 4090.

### Links

- -> **Confidence-Aware Gaussian Splatting**: provides $\mathcal{L}_{\text{conf}}$
- -> **Color Variance Loss**: provides $\mathcal{L}_{\text{color-var}}$
- -> **Normal Variance Loss**: provides $\mathcal{L}_{\text{normal-var}}$
- -> **Confidence-Aware Densification**: operates on gradients of this total loss

---

## 12. What Existed Before and What This Paper Changes

### 12.1 Prior Approaches and Their Limitations

**Multi-view geometric constraints** (PGSR [4], VA-GS [32], QGS [75]). These methods enforce geometric consistency across views by computing photometric or normal consistency between neighboring viewpoints. PGSR uses planar Gaussians with multi-view stereo losses; QGS extends this with second-order (quadric) geometric primitives. The problem: multi-view constraints are computationally expensive (PGSR uses full-resolution multi-view photometric loss), and they aggressively smooth fine structures and foliage because multi-view consensus penalizes thin geometry that is visible from few angles. PGSR also uses ground-truth point clouds during TSDF fusion for voxel sizing — privileged information unavailable in practice.

**Monocular depth/normal priors** (VCR-GauS [5], GeoSVR [31], UA-GS [37]). These methods use pre-trained monocular depth or normal estimators (typically DPT or Metric3Dv2) to provide per-pixel geometric supervision. This provides strong geometric guidance, especially in textureless regions. However, the pre-trained models introduce their own biases, the depth predictions are only up to scale/shift, and the computational overhead is substantial (GeoSVR takes 42 minutes vs CoMe's 18 minutes, excluding the depth preprocessing time). These methods also cannot extract unbounded meshes.

**Mesh-aware optimization** (MiLo [17]). MiLo enforces bidirectional consistency between Gaussians and an extracted mesh during training: a mesh is periodically extracted, and a loss ensures the Gaussians and mesh agree. This produces smooth, artifact-free meshes but at significant computational cost (60 minutes on T&T). The bidirectional consistency loss is most easily minimized by producing overly smooth geometry, losing fine surface details (visible in Fig.5, Fig.11).

**Post-hoc extraction baselines** (GOF [70], SOF [50]). These methods optimize 3DGS with geometric priors (surface alignment, depth-normal consistency) and extract meshes post-hoc via Marching Tetrahedra or TSDF. GOF introduced Gaussian opacity fields; SOF improved speed with sorted opacity fields and front-to-back rasterization. They are fast but struggle with view-dependent appearance: photometric losses dominate in hard regions, causing over-densification and geometric corruption.

### 12.2 What This Paper Contributes

**Contribution 1: Self-supervised confidence framework.** CoMe introduces per-primitive learnable confidence values that dynamically balance photometric and geometric losses — no external networks, no multi-view constraints. The confidence loss (Eq.9) lets the model express uncertainty where photometric supervision is unreliable, while geometric losses continue to guide surface reconstruction.

**Contribution 2: Variance-reducing losses.** Color (Eq.13) and normal (Eq.14) variance losses force individual Gaussians to align with the surface, addressing the per-primitive supervision gap inherent in alpha-blending. These losses identify geometric uncertainty through blending variance.

**Contribution 3: Decoupled D-SSIM appearance.** Analysis of D-SSIM reveals that its luminance component is illumination-sensitive while contrast and structure are not. Selectively applying the appearance model only to luminance (Eq.8) improves 3D consistency.

### 12.3 Side-by-Side Comparison

| Dimension | PGSR/QGS (bounded) | MiLo (unbounded) | GeoSVR (prior-dep.) | **CoMe (Ours)** |
|---|---|---|---|---|
| External priors | Multi-view stereo | None | Monocular depth | **None** |
| Mesh type | Bounded (TSDF) | Unbounded (MT) | Bounded (TSDF) | **Unbounded (MT)** |
| Handles illumination | No | No | No | **Yes (decoupled app.)** |
| Handles view-dep. | Multi-view consensus | Bidirectional mesh | Monocular prior | **Confidence + variance** |
| Runtime (T&T) | 28 min | 60 min | 42 min+ | **18 min** |
| F1 (T&T avg) | 0.496 | 0.485 | 0.546 | **0.521** |
| Scene-agnostic voxel | No (GT point cloud) | Yes | No (GT point cloud) | **Yes** |
| Unbounded F1 advantage | N/A (bounded only) | 0.485 | N/A (bounded only) | **0.521** |

| Dimension | SOF [50] (direct baseline) | **CoMe** |
|---|---|---|
| Confidence weighting | No | Yes (per-primitive, self-supervised) |
| Per-primitive supervision | No | Yes (color + normal variance) |
| Appearance model | VastGaussian (standard) | Improved + SSIM-decoupled |
| Adapted densification | No | Confidence-steered |
| F1 T&T | 0.474 | **0.521 (+0.047)** |
| F1 ScanNet++ | 0.615 | **0.668 (+0.053)** |
| Runtime | 17 min | 18 min |

### 12.4 The Core Shift in Thinking

Prior mesh extraction methods treat the geometry-appearance coupling problem by adding external constraints: either multi-view consistency checks, monocular depth priors, or periodic mesh-Gaussian alignment. These approaches impose a single, fixed weighting between photometric and geometric objectives across the entire scene.

CoMe shifts the paradigm from *externally constraining* the geometry to *internally acknowledging uncertainty*. The confidence framework lets each region of the scene determine its own balance between photometric and geometric supervision. Reflective surfaces, thin foliage, and rarely observed areas get low confidence — the optimizer stops trying to perfectly reproduce their appearance and instead lets geometric losses shape the surface. This per-region adaptive weighting, combined with per-primitive variance losses that remove the need for external priors, is what allows CoMe to achieve state-of-the-art results with purely self-supervised signals and competitive runtime.

The key insight is that surface extraction does not require perfect photometric reconstruction everywhere. By learning *where* to trust photometric supervision and *where* to rely on geometric priors, CoMe sidesteps the computational overhead of external constraints entirely.

---

## 13. Quick Reference Card

| # | Insight | Design Choice | Evidence |
|---|---|---|---|
| 1 | D-SSIM luminance is illumination-sensitive, contrast/structure are not | Apply $\hat{\boldsymbol{I}}^{\text{app}}$ only to luminance in D-SSIM | Fig.2: 9x luminance error reduction; Tab.4: 0.490 vs 0.484 |
| 2 | Photometric loss dominates in regions with complex appearance | Confidence-weight $\mathcal{L}_{\text{rgb}}$ with learned $\hat{C}$; log-penalty prevents collapse | Tab.3: +0.016/+0.030 F1 on T&T/ScanNet++ |
| 3 | Over-densification in low-confidence regions wastes primitives | Scale densification threshold by $1/\min(\tilde{\gamma}_i, 1)$ | Tab.5: 20% fewer primitives with adapted densification |
| 4 | Alpha-blending hides per-primitive color/normal errors | Weighted per-primitive $\mathcal{L}_2$ variance losses | Fig.4, Fig.13: consistent first-hit colors and normals |
| 5 | Color variance loss helps most on view-dependent outdoor scenes | $\lambda_{\text{color-var}} = 0.5$ | Tab.3: +0.010 F1 on T&T |
| 6 | Normal variance loss helps most on planar indoor scenes | $\lambda_{\text{normal-var}} = 0.005$ | Tab.3: +0.010 F1 on ScanNet++ |
| 7 | $\beta$ controls quality-complexity trade-off | $\beta = 0.075$ optimal; $\beta = 0.05$ for a "light" variant | Fig.7: F1 peak at 0.075; light variant still beats MiLo |
| 8 | Indoor scenes have higher learned confidence than outdoor | Confidence adapts per scene without tuning | Fig.12: indoor avg 3.22, outdoor avg 1.66 |
| 9 | Data preprocessing discrepancies inflate reported metrics | Standardize on GOF's publicly available data | Sec. B.1: +0.082 F1 swing from preprocessing alone |

---

## 14. Open Questions

1. **Sparse-view settings.** The confidence framework assumes moderately dense capture. In sparse-view scenarios, integrating multi-view constraints or monocular priors may still be necessary (Sec.5).
2. **Distant background geometry.** Insufficient view density and lack of close-up captures cause under-densification in far background regions. Pre-trained multi-view diffusion models [13,62] or tailored densification could help.
3. **Heuristic-free densification.** The $\beta$ parameter already functions as a quality-complexity dial (Fig.7). The confidence framework could replace hand-tuned gradient thresholds entirely.
4. **Downstream applications.** The confidence mechanism may benefit other 3DGS tasks: 3D inpainting, super-resolution training, or any task where selective photometric supervision is useful.

---

## 15. Concept Dependency Graph

```
  ┌──────────────────────────────────┐
  │  3DGS Preliminaries (Eq.1–3)     │
  │  Alpha-Blending Rendering        │
  └──────────┬───────────────────────┘
             │
             ├──────────────────────────────────────────┐
             │                                          │
             ▼                                          ▼
  ┌──────────────────────┐               ┌────────────────────────────┐
  │  Photometric Loss     │               │  Geometry-Appearance       │
  │  L_rgb (Eq.4–5)       │               │  Coupling Problem          │
  └──────────┬────────────┘               └────────────┬───────────────┘
             │                                         │
     ┌───────┼─────────┐                    ┌──────────┼──────────┐
     │       │         │                    │          │          │
     ▼       ▼         ▼                    ▼          ▼          ▼
  ┌──────┐ ┌──────┐ ┌──────────────┐  ┌─────────┐ ┌────────┐ ┌──────────┐
  │ App. │ │Decoup│ │ Confidence-  │  │ Color   │ │Normal  │ │Confidence│
  │Model │→│D-SSIM│ │ Aware 3DGS   │  │Variance │ │Variance│ │-Aware    │
  │(Eq.6 │ │(Eq.8)│ │ (Eq.9)       │  │Loss     │ │Loss    │ │Densif.   │
  │Eq.7) │ │      │ │              │  │(Eq.13)  │ │(Eq.14) │ │(Eq.12)   │
  └──────┘ └──────┘ └──────┬───────┘  └────┬────┘ └───┬────┘ └────┬─────┘
                           │               │          │           │
                           │               │          │           │
                    ┌──────▼───────┐       │          │           │
                    │  Confidence   │       │          │           │
                    │  Rendering    │───────┼──────────┼───────────┘
                    │  (Eq.10–11)   │       │          │
                    └──────────────┘       │          │
                           │               │          │
                           └───────┬───────┘──────────┘
                                   ▼
                        ┌────────────────────┐
                        │  Total Loss (Eq.15) │
                        └────────┬───────────┘
                                 │
                                 ▼
                        ┌────────────────────┐
                        │ Marching Tetrahedra │
                        │ → Unbounded Mesh    │
                        └─────────────────────┘
```

---

## 16. Key Equations

| # | Equation | Description | Section |
|---|---|---|---|
| Eq.1 | $\hat{\boldsymbol{I}}(\boldsymbol{r}) = \sum_i w_i(\boldsymbol{r})\,\text{sh}(\boldsymbol{\theta}_i, \boldsymbol{d})$ | Alpha-blending rendering | 3.1 |
| Eq.2 | $\alpha_i = o_i \exp(-\frac{1}{2}(\boldsymbol{x}_i^* - \boldsymbol{\mu}_i)^T\boldsymbol{\Sigma}_i^{-1}(\boldsymbol{x}_i^* - \boldsymbol{\mu}_i))$ | 3D opacity evaluation | 3.1 |
| Eq.3 | $\boldsymbol{x}_i^* = \boldsymbol{o} + \frac{\boldsymbol{d}^T\boldsymbol{\Sigma}_i^{-1}(\boldsymbol{\mu}_i - \boldsymbol{o})}{\boldsymbol{d}^T\boldsymbol{\Sigma}_i^{-1}\boldsymbol{d}}\boldsymbol{d}$ | Closest point on ray | 3.1 |
| Eq.4 | $\mathcal{L}_{\text{rgb}} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{\text{D-SSIM}}$ | Photometric reconstruction loss | 3.1 |
| Eq.5 | $\mathcal{L}_{\text{D-SSIM}} = 1 - l \cdot c \cdot s$ | D-SSIM decomposition | 3.1 |
| Eq.6 | $\boldsymbol{M}_i = F_\Theta([\text{ds}_{32}(\hat{\boldsymbol{I}}), \boldsymbol{\rho}_i])$ | Appearance CNN mapping | 3.1 |
| Eq.7 | $\hat{\boldsymbol{I}}^{\text{app}} = \hat{\boldsymbol{I}} \odot \sigma(\boldsymbol{M}_i)$ | Appearance-corrected image | 3.1 |
| Eq.8 | $\mathcal{L}_{\text{D-SSIM}}^{\text{dec}} = 1 - l(\boldsymbol{I}, \hat{\boldsymbol{I}}^{\text{app}}) \cdot c(\boldsymbol{I}, \hat{\boldsymbol{I}}) \cdot s(\boldsymbol{I}, \hat{\boldsymbol{I}})$ | **Decoupled D-SSIM** | 3.2 |
| Eq.9 | $\mathcal{L}_{\text{conf}} = \mathcal{L}_{\text{rgb}} \cdot \hat{C} - \beta\log\hat{C}$ | **Confidence loss** | 3.3 |
| Eq.10 | $\tilde{\gamma}_i = \exp(\gamma_i)$ | Per-primitive confidence | 3.3 |
| Eq.11 | $\hat{C}(\boldsymbol{r}) = \sum_i w_i(\boldsymbol{r})\tilde{\gamma}_i$ | Rendered confidence map | 3.3 |
| Eq.12 | $\tilde{\tau}_{\text{grad}} = \tau_{\text{grad}} / \min(\tilde{\gamma}_i, 1)$ | **Confidence-aware densification** | 3.3 |
| Eq.13 | $\mathcal{L}_{\text{color-var}} = \sum_i w_i(\boldsymbol{r})\|\text{sh}(\boldsymbol{\theta}_i, \boldsymbol{d}) - \boldsymbol{I}\|_2^2$ | **Color variance loss** | 3.4 |
| Eq.14 | $\mathcal{L}_{\text{normal-var}} = 1 - \|\boldsymbol{N}\|_2^2$ | **Normal variance loss** | 3.4 |
| Eq.15 | $\mathcal{L} = \mathcal{L}_{\text{conf}} + \mathcal{L}_{\text{geom}} + \lambda_c\mathcal{L}_{\text{color-var}} + \lambda_n\mathcal{L}_{\text{normal-var}}$ | **Total loss** | 4 |

---

## 17. Experimental Results Summary

### Tab.1: Tanks & Temples — Full Geometry Reconstruction (F1-score, higher is better)

| Method | Barn | Caterpillar | Courthouse | Ignatius | Meetingroom | Truck | **Average** | Runtime |
|---|---|---|---|---|---|---|---|---|
| PGSR | 0.548 | **0.437** | 0.238 | 0.728 | 0.367 | **0.658** | 0.496 | 28min |
| QGS | 0.536 | 0.374 | 0.183 | 0.733 | **0.374** | 0.645 | 0.474 | 41min |
| GOF | 0.484 | 0.402 | 0.288 | 0.674 | 0.275 | 0.596 | 0.453 | 40min |
| SOF | 0.535 | 0.408 | 0.297 | 0.736 | 0.309 | 0.558 | 0.474 | 17min |
| MiLo | **0.541** | 0.389 | 0.322 | 0.757 | 0.281 | 0.617 | 0.485 | 60min |
| **Ours** | 0.534 | **0.472** | **0.333** | **0.782** | **0.372** | 0.634 | **0.521** | **18min** |

### Tab.2: ScanNet++-v2 — Full Geometry Reconstruction (F1-score)

| Method | 5a269ba6fe | 08bbbdcc3d | 39f36da05b | c263dfbf0 | ef18cf0708 | fb564c935d | **Average** |
|---|---|---|---|---|---|---|---|
| PGSR | **0.668** | 0.666 | 0.647 | 0.630 | 0.542 | 0.632 | 0.631 |
| QGS | 0.646 | 0.622 | 0.571 | 0.522 | 0.504 | 0.572 | 0.573 |
| GOF | 0.620 | 0.670 | 0.611 | 0.672 | 0.504 | 0.661 | 0.623 |
| SOF | 0.599 | 0.681 | 0.616 | 0.675 | 0.497 | 0.621 | 0.615 |
| MiLo | 0.622 | 0.675 | 0.621 | 0.626 | 0.542 | 0.660 | 0.624 |
| **Ours** | **0.670** | **0.729** | **0.657** | **0.715** | **0.551** | **0.684** | **0.668** |

### Tab.3: Ablation Study — Cumulative Component Addition

| Method | T&T Prec. | T&T Rec. | T&T F1 | SN++ Prec. | SN++ Rec. | SN++ F1 |
|---|---|---|---|---|---|---|
| SOF [50] (baseline) | 0.542 | 0.438 | 0.474 | 0.565 | 0.677 | 0.615 |
| + Improved Appearance | 0.557 | 0.459 | 0.493 | 0.577 | 0.685 | 0.625 |
| + $\mathcal{L}_{\text{conf}}$ | 0.568 | 0.479 | 0.509 | 0.605 | 0.715 | 0.655 |
| + $\mathcal{L}_{\text{color-var}}$ | 0.571 | 0.491 | 0.519 | 0.606 | 0.722 | 0.658 |
| + $\mathcal{L}_{\text{normal-var}}$ | **0.580** | **0.491** | **0.521** | **0.613** | **0.734** | **0.668** |

Total F1 improvement: **+0.047** (T&T), **+0.053** (ScanNet++).

### Tab.4: Ablation on Decoupled Appearance Module (Tanks & Temples)

| **Baselines (Standard)** | Prec. | Rec. | F1 | **Ours (Ablations)** | Prec. | Rec. | F1 |
|---|---|---|---|---|---|---|---|
| VastG. [35] | 0.541 | 0.440 | 0.475 | VastG. (Imp.) | 0.557 | 0.445 | 0.484 |
| PGSR [4] | 0.545 | 0.441 | 0.478 | w/o $l(\cdot)$ | 0.559 | 0.441 | 0.483 |
| H3DGS [27] | 0.553 | 0.448 | 0.484 | w/o SSIM Dec. | 0.559 | 0.453 | 0.490 |
| PPISP [9] | 0.556 | **0.456** | **0.489** | **Ours** | 0.557 | **0.459** | **0.493** |

### Tab.6: Comparison with Prior-Dependent GeoSVR (Tanks & Temples)

| Method | Barn | Caterpillar | Courthouse | Ignatius | Meetingroom | Truck | Average | Runtime |
|---|---|---|---|---|---|---|---|---|
| Ours | 0.534 | 0.472 | 0.333 | 0.782 | 0.372 | 0.634 | 0.521 | 18min |
| GeoSVR | **0.557** | **0.495** | **0.347** | **0.804** | **0.382** | **0.688** | **0.546** | 42min |
| GeoSVR (w/ QGS Eval) | 0.544 | 0.432 | 0.328 | 0.787 | 0.408 | 0.651 | 0.525 | 42min |

GeoSVR achieves higher F1 but uses multi-view constraints + monocular depth prior, takes 2.3x longer (excluding depth preprocessing), relies on ground-truth point clouds for voxel sizing, and cannot extract unbounded meshes.

### Tab.7: DTU Dataset — Full Geometry Reconstruction (Chamfer Distance, lower is better)

| Method | 24 | 37 | 40 | 55 | 63 | 65 | 69 | 83 | 97 | 105 | 106 | 110 | 114 | 118 | 122 | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PGSR | 0.38 | 0.69 | 0.42 | 0.35 | 0.76 | 0.88 | 0.53 | 1.08 | 0.70 | 0.61 | 0.45 | 0.55 | 0.34 | 0.42 | 0.40 | 0.57 |
| QGS | 0.45 | 0.77 | 0.37 | 0.37 | 0.54 | 0.91 | 0.55 | 1.15 | 1.00 | 0.62 | 0.46 | 0.70 | 0.36 | 0.45 | 0.41 | 0.61 |
| GOF | 0.53 | 0.89 | 0.43 | 0.38 | 1.33 | 0.87 | 0.77 | 1.28 | 1.29 | 0.79 | 0.77 | 1.15 | 0.46 | 0.70 | 0.54 | 0.81 |
| SOF | 0.55 | 0.70 | 0.41 | 0.39 | 1.12 | 0.73 | 0.66 | 1.11 | 1.47 | 0.60 | 0.63 | 1.05 | 0.58 | 0.62 | 0.48 | 0.74 |
| MiLo | 0.46 | 0.77 | 0.35 | 0.39 | 0.80 | 0.86 | 0.72 | 1.20 | 1.22 | 0.66 | 0.43 | 0.93 | 0.38 | 0.79 | 0.49 | 0.71 |
| **Ours** | **0.43** | **0.70** | **0.34** | **0.39** | 0.92 | **0.70** | **0.67** | **1.03** | **1.19** | **0.50** | **0.49** | 1.05 | **0.47** | **0.41** | **0.40** | **0.65** |

Lowest average Chamfer distance among all unbounded methods. For bounded methods, PGSR (0.57) and QGS (0.61) benefit from multi-view constraints and TSDF fusion on small, single-object scenes.

### Tab.8: Novel View Synthesis on Mip-NeRF 360 (PSNR/SSIM/LPIPS)

| Method | Indoor PSNR | Indoor SSIM | Indoor LPIPS | Outdoor PSNR | Outdoor SSIM | Outdoor LPIPS | # Gaussians |
|---|---|---|---|---|---|---|---|
| PGSR | **30.35** | **0.924** | **0.176** | 24.29 | 0.718 | 0.236 | — |
| QGS | 30.45 | 0.919 | 0.184 | 24.32 | 0.706 | 0.242 | — |
| GOF | 30.47 | 0.918 | 0.189 | **24.79** | **0.745** | **0.208** | — |
| SOF | 30.13 | 0.916 | 0.188 | 24.78 | **0.745** | **0.208** | — |
| MiLo | 29.96 | 0.920 | 0.191 | 24.47 | 0.718 | 0.290 | — |
| Ours | 29.92 | 0.911 | 0.193 | 24.69 | 0.742 | 0.211 | 2.89M |
| Ours w/ conf | 30.24 | 0.919 | 0.179 | 24.41 | 0.719 | 0.258 | **2.12M** |

Adding $\mathcal{L}_{\text{conf}}$ reduces primitive count by 26% (2.89M -> 2.12M) with only 0.02 dB PSNR drop. Indoor confidence avg = 3.22, outdoor avg = 1.66.

---

## 18. Appearance Embedding Comparison (Appendix B.3)

For completeness, the alternative appearance models compared in Tab.4:

**PGSR** [4] uses per-image learnable scalars $\{a_i, b_i\} \in \mathbb{R}^2$:

$$\hat{\boldsymbol{I}}^{\text{app}} = \exp(a_i)\hat{\boldsymbol{I}} + b_i \tag{Eq.19}$$

This is efficient but cannot compensate for spatially-varying effects (e.g., vignetting).

**Hierarchical 3DGS** [27] uses a per-image affine mapping $\boldsymbol{A}_i \in \mathbb{R}^{3\times 3}, \boldsymbol{b}_i \in \mathbb{R}^3$:

$$\hat{\boldsymbol{I}}^{\text{app}} = \boldsymbol{A}_i\hat{\boldsymbol{I}} + \boldsymbol{b}_i \tag{Eq.20}$$

The full $3\times3$ matrix captures chromaticity shifts but is still a global (not spatially-varying) transform.

**PPISP** [9] uses a physically-motivated model with explicit exposure, vignetting, color correction, and camera response parameters. More expressive than PGSR/H3DGS, but designed for inference-time appearance control rather than 3D coherence.

---

## 19. Reference Map (Citations by Topic)

**3D Gaussian Splatting (core):**
[26] Kerbl et al. (2023) — original 3DGS; [49] Radl et al. (2024) — StopThePop sorted rendering; [50] Radl et al. (2025) — SOF (direct baseline)

**Surface extraction from 3DGS:**
[70] Yu et al. (2024) — GOF (Gaussian Opacity Fields); [17] Guedon et al. (2025) — MiLo; [4] Chen et al. (2025) — PGSR; [75] Zhang et al. (2025) — QGS; [18] Guedon & Lepetit (2024) — SuGaR; [20] Huang et al. (2024) — 2DGS

**Monocular prior methods:**
[5] Chen et al. (2024) — VCR-GauS; [31] Li et al. (2025) — GeoSVR; [37] Liu et al. (2025) — UA-GS

**Appearance modeling:**
[35] Lin et al. (2024) — VastGaussian; [27] Kerbl et al. (2024) — Hierarchical 3DGS; [9] Deutsch et al. (2026) — PPISP

**Uncertainty in radiance fields:**
[25] Kendall & Gal (2017) — Bayesian deep learning uncertainty; [59] Wang et al. (2024) — DUSt3R; [33] Li et al. (2024) — variational multi-scale uncertainty

**Benchmarks:**
[29] Knapitsch et al. (2017) — Tanks and Temples; [67] Yeshwanth et al. (2023) — ScanNet++; [1] Barron et al. (2022) — Mip-NeRF 360; [21] Jensen et al. (2014) — DTU
