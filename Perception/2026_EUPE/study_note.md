# Efficient Universal Perception Encoder (EUPE) Study Note — A Concept Mind-Map

> Paper: Efficient Universal Perception Encoder (arXiv:2603.22387v2)
> Authors: Chenchen Zhu, Saksham Suri, Cijo Jose, Maxime Oquab, Marc Szafraniec, Wei Wen, Yunyang Xiong, Patrick Labatut, Piotr Bojanowski, Raghuraman Krishnamoorthi, Vikas Chandra
> Affiliations: Meta Reality Labs, FAIR at Meta
> Code: https://github.com/facebookresearch/eupe
> Models: https://huggingface.co/collections/facebook/eupe

This note organizes every key concept in the paper as a mind-map.
Each concept is broken down into four facets:

- **Definition** — what it is, stated plainly
- **Properties** — its mathematical or behavioral characteristics
- **Application** — how the paper (or the field) uses it
- **Links** — connections to other concepts in this map

---

## 0. The Big Picture

```
              Foundation Vision Encoders
              (each excels in ONE domain)
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
    Image Under.   Dense Pred.    VLM Tasks
    (PEcore,       (DINOv3,       (SigLIP2,
     SigLIP2)       SAM)          PElang)
         │              │              │
         └──────────────┼──────────────┘
                        │
        "Can we merge all into ONE
         small efficient encoder?"
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
      Direct         Agglomerative   EUPE
      Multi-Teacher  (RADIO, DUNE)   (This Paper)
      Distillation        │              │
      (fails for     "Works for       "Scale UP
       small models)  large models,    then scale
                       not efficient    DOWN"
                       ones"            │
                                  ┌─────┼─────┐
                                  ▼     ▼     ▼
                              Stage 1  Stage 2  Stage 3
                              Multi-T  Fixed-   Multi-
                              →Proxy   Res→     Res
                              (1.9B)   Student  Finetune
                                        │
                                        ▼
                              EUPE Family: ViT-T/S/B
                              ConvNeXt-T/S/B (6M–89M)
                                        │
                              On-par or better than
                              domain experts on ALL
                              3 task domains
```

---

## 1. The Domain Expert Problem

### Definition

Foundation vision encoders are pretrained models that produce general-purpose image representations. Each encoder is trained with a specific objective — self-supervision (DINOv2, MAE), contrastive text-image alignment (CLIP, SigLIP2), or segmentation (SAM) — and as a result, each excels in its own domain but underperforms in others.

- **Image understanding** experts: PEcore, SigLIP2 — strong on ImageNet classification (zero-shot and KNN) but weak on dense prediction.
- **Dense prediction** experts: DINOv3, SAM — strong on segmentation (ADE20k), depth (NYUv2), keypoint matching (SPair) but lack text-alignment for VLM tasks.
- **Vision-language modeling** experts: PElang, SigLIP2 — strong on OCR (TextVQA), visual QA (GQA, POPE, SQA) but sacrifice spatial granularity.

The core tension: a single encoder cannot cover all three domains well, yet edge devices need one compact model that does.

### Properties

- The root cause is that different pretraining objectives optimize different feature properties: text-alignment produces semantically coherent class tokens, self-supervised objectives produce spatially precise patch tokens.
- Selecting the wrong encoder for a task causes large performance drops (e.g., DINOv3 has no text encoder → 0.0 on IN1k-ZS).
- Running multiple encoders simultaneously is prohibitive on edge devices with limited compute.

### Application

This problem motivates the entire EUPE pipeline. The paper frames the challenge as: "How do we produce a single small encoder whose frozen representations transfer well across image understanding, dense prediction, and vision-language modeling without task-specific finetuning?"

### Links

- → **Scale-Up-Then-Scale-Down Principle**: the proposed solution to this problem
- → **Teacher Selection**: choosing which domain experts to distill from
- → **Evaluation Protocol**: designed to cover all three task domains
- → **Feature Visualization Analysis**: PCA visualizations reveal each expert's feature characteristics

---

## 2. Scale-Up-Then-Scale-Down Principle

### Definition

The central insight of EUPE: to build an efficient universal encoder, first **scale up** knowledge from multiple domain-expert teachers into a single large proxy model, then **scale down** from that proxy to compact student models. Direct multi-teacher distillation into a small student fails because the small model lacks capacity to simultaneously absorb diverse feature representations from multiple teachers.

### Properties

- **Why direct distillation fails**: efficient encoders (< 100M params) have insufficient capacity to reconcile the conflicting gradient signals from teachers with different feature statistics (different mean/std per token type, different spatial granularities).
- **Why scaling up first works**: a large proxy (1.9B params) has enough capacity to learn a unified representation that balances all domains. This single universal teacher then provides a coherent learning signal to the student.
- **Empirical validation**: Table 2 shows that "Stage 2 only" (direct multi-teacher → student) yields 35.1 SPair, 0.616 NYUv2, 41.9 ADE20k — all significantly worse than the full pipeline (51.3 SPair, 0.391 NYUv2, 52.4 ADE20k).

### Application

This principle is operationalized as a three-stage pipeline: Stage 1 (scale up), Stage 2 + Stage 3 (scale down). The proxy model acts as an information bottleneck that compresses and unifies knowledge before transferring it.

### Links

- → **Multi-Stage Distillation Pipeline**: the three-stage instantiation of this principle
- → **Proxy Teacher Model**: the large intermediate model that enables scale-up
- → **The Domain Expert Problem**: the problem this principle addresses

---

## 3. Multi-Stage Distillation Pipeline

### Definition

The EUPE training recipe consists of three sequential stages, each performing knowledge distillation with the same loss formulation but different teacher-student configurations and resolution strategies.

```
Stage 1: Multi-Teacher → Proxy          Stage 2: Proxy → Student      Stage 3: Proxy → Student
                                         (Fixed Resolution)            (Multi-Resolution)
┌─────────────┐                         ┌─────────────┐               ┌─────────────┐
│ PEcore-G    │──┐                      │ Proxy 1.9B  │               │ Proxy 1.9B  │
│ (1.9B)      │  │                      │ (frozen)    │               │ (frozen)    │
├─────────────┤  │  ┌──────────┐        └──────┬──────┘               └──────┬──────┘
│ PElang-G    │──┼─▶│ Proxy    │               │                            │
│ (1.7B)      │  │  │ 1.9B     │        ┌──────▼──────┐               ┌─────▼───────┐
├─────────────┤  │  │ (student)│        │  Efficient  │               │  Efficient  │
│ DINOv3-H+   │──┘  └──────────┘        │  Student    │               │  Student    │
│ (840M)      │                         │  (ViT-B,    │               │ (init from  │
└─────────────┘                         │   86M)      │               │  Stage 2)   │
                                        └─────────────┘               └─────────────┘
 Teachers at native res                  256×256 fixed                  Pyramid: 256,
 (448 for PE, 256 for DINOv3)           390k iterations                384, 512
                                                                       100k iterations
```

### Properties

- All three stages share the same distillation loss formulation (Eq. 4–6), the same adapter head design, and the same feature normalization method. Only the teacher, student, resolution, and schedule differ.
- **Stage 1** trains for the full AM-RADIO recipe duration. Proxy model is 1.9B params (ViT-G backbone with 4 register tokens).
- **Stage 2** uses a fixed 256×256 resolution for computational efficiency, enabling a longer training schedule (390k iterations, batch 8192, cosine LR 2e-5, weight decay 1e-4).
- **Stage 3** uses a multi-resolution pyramid {256, 384, 512} with shorter schedule (100k iterations, batch 4096, LR 1e-5). One iteration at multi-resolution costs 2× one at fixed resolution.
- The student in Stage 3 is initialized from the Stage 2 checkpoint.

### Application

The full pipeline (Stage 1&2&3) achieves the best overall balance across all benchmarks (Table 2). Dropping any stage degrades performance:

| Configuration | IN1k-ZS | TextVQA | SPair | ADE20k |
|---|---|---|---|---|
| Stage 2 only | 79.6 | 46.8 | 35.1 | 41.9 |
| Stage 1&2 | 79.5 | 48.3 | 41.0 | 43.3 |
| Stage 1&3 | **80.0** | 49.5 | **53.3** | 52.0 |
| **Stage 1&2&3** | 79.7 | **50.4** | 51.3 | **52.4** |

Stage 1&3 (skipping Stage 2) gives the best dense prediction but worse VLM metrics. The long fixed-resolution Stage 2 followed by short multi-resolution Stage 3 produces the best overall balance.

### Links

- → **Scale-Up-Then-Scale-Down Principle**: the conceptual basis
- → **Stage 1: Multi-Teacher Distillation**: first stage details
- → **Stage 2: Fixed-Resolution Distillation**: second stage details
- → **Stage 3: Multi-Resolution Finetuning**: third stage details
- → **Distillation Loss**: shared across all stages
- → **Training Data**: all stages train on LVD-1689M
- → **EUPE Model Family**: the output of this pipeline

---

## 4. Stage 1: Multi-Teacher Distillation to Proxy Model

### Definition

In Stage 1, multiple domain-expert foundation encoders (teachers) simultaneously distill their knowledge into a single large proxy model (student). Each teacher processes the same input image at its native resolution in parallel. The proxy model is a heavy-weight backbone (1.9B parameters) chosen to have enough capacity to absorb the diverse representations.

### Properties

- Teachers run at their native resolutions: 448 for PEcore-G and PElang-G, 256 for DINOv3-H+.
- The input images are **label-free** — no supervised labels are needed.
- The student encoder $S(\cdot;\theta)$ and each teacher $T_i(\cdot;\phi_i)$ output a class token and patch tokens:

$$(y^c_S, y^p_S) = S(x_S; \theta), \quad y^c_S \in \mathbb{R}^{d_S}, \; y^p_S \in \mathbb{R}^{N_S \times d_S} \tag{Eq.1}$$

$$(y^c_{T_i}, y^p_{T_i}) = T_i(x_{T_i}; \phi_i), \quad y^c_{T_i} \in \mathbb{R}^{d_{T_i}}, \; y^p_{T_i} \in \mathbb{R}^{N_{T_i} \times d_{T_i}} \tag{Eq.2}$$

  where $d_S, N_S$ are the student's feature dimension and patch count, and $d_{T_i}, N_{T_i}$ are the $i$-th teacher's.
- The proxy model's tokens are mapped to each teacher's feature space through per-teacher adapter heads.
- Teacher parameters $\phi_i$ are frozen; only the proxy encoder $\theta$ and adapter head parameters $\psi^c_i, \psi^p_i$ are trained.
- The distillation loss is summed over all teachers (Eq. 6).
- A crude centering of teacher outputs is performed by measuring per-coordinate mean and variance during 500 iterations before training.

### Application

The teacher combination matters significantly. Table 3 shows the ablation:

| Teachers | IN1k-ZS | TextVQA | SPair | ADE20k |
|---|---|---|---|---|
| PEc & Dv3 | 79.9 | 48.6 | 51.5 | **52.5** |
| PEc & Dv3 & S2 | 78.8 | 44.8 | **52.1** | **52.5** |
| **PEc & Dv3 & PEl** | 79.7 | **50.4** | 51.3 | 52.4 |

Adding SigLIP2-G degrades OCR (TextVQA drops from 48.6 to 44.8) because two CLIP-style teachers (PEcore-G and SigLIP2-G) conflict. PElang-G, being derived from PEcore-G through language alignment, complements the set without causing interference.

### Links

- → **Multi-Stage Distillation Pipeline**: Stage 1 is the first of the three stages
- → **Proxy Teacher Model**: the student in this stage
- → **Teacher Selection**: which foundation encoders to use
- → **Adapter Heads**: per-teacher projection modules
- → **Distillation Loss**: the training objective
- → **Feature Normalization**: stabilizes multi-teacher training
- → **Training Data**: label-free images from LVD-1689M
- → **Stage 2: Fixed-Resolution Distillation**: receives the trained proxy

---

## 5. Proxy Teacher Model

### Definition

The proxy teacher is a large intermediate model that serves as a universal knowledge repository. It is the student in Stage 1 (learning from multiple domain experts) and becomes the teacher in Stages 2 and 3 (distilling to efficient students). The default configuration is a ViT-G backbone with 1.9B parameters and 4 register tokens.

### Properties

- **Capacity requirement**: must be large enough to absorb diverse representations from all domain-expert teachers simultaneously.
- **Architecture**: ViT-G (1.9B) is the default. The paper also tests ViT-7B in the appendix.
- Table 4 shows the proxy model's own performance before distillation:

| Proxy Teachers | IN1k-ZS | TextVQA | SPair | ADE20k |
|---|---|---|---|---|
| PEc only | **85.4** | 54.7 | 20.3 | 54.8 |
| PEc & Dv3 | 85.0 | 56.2 | 52.9 | **56.0** |
| PEc & Dv3 & PEl | 84.8 | **58.6** | 53.8 | 55.9 |

- Scaling the proxy to 7B (Table 7) improves proxy-level metrics across the board:

  **Table 7 — Proxy model performance at different scales** (PEc = PEcore-G, PEl = PElang-G, Dv3 = DINOv3-H+, Dv3-7B = DINOv3-7B):

  | Proxy Teachers (proxy size) | IN1k-ZS | IN1k-KNN | TextVQA | SPair | NYUv2↓ | ADE20k |
  |---|---|---|---|---|---|---|
  | PEc & Dv3 (1.9B) | 85.0 | 87.0 | 56.2 | 52.9 | 0.332 | 56.0 |
  | PEc & Dv3 & S2 (1.9B) | **85.3** | **87.2** | 53.2 | 54.4 | **0.321** | **56.0** |
  | PEc & Dv3 & PEl (1.9B) | 84.8 | 87.0 | 58.6 | 53.8 | 0.390 | 55.9 |
  | **PEc & Dv3-7B & PEl (7B)** | 85.1 | **87.2** | **59.7** | **56.2** | 0.305 | **56.9** |

- But these gains do not fully transfer to the ViT-B student (Table 8):

  **Table 8 — Final ViT-B student performance with scaled proxy**:

  | DINOv3 size | Proxy size | IN1k-ZS | TextVQA | Realworld | SPair | ADE20k |
  |---|---|---|---|---|---|---|
  | ViT-H+ | ViT-G (1.9B) | **79.7** | **50.4** | **55.5** | 51.3 | **52.4** |
  | ViT-7B | ViT-7B (7B) | 80.2 | 48.5 | 53.9 | **52.0** | 52.5 |

  The 7B proxy's knowledge is not fully distilled into the 86M student — VLM metrics (TextVQA, Realworld) degrade, likely due to the large capacity gap between teacher (7B) and student (86M).

- The proxy's feature statistics are computed once before Stages 2&3 and fixed for normalization.

### Application

The proxy model converts a hard problem (multi-teacher → small student) into two easier problems: (1) multi-teacher → large student, and (2) single-teacher → small student. This is conceptually related to Teacher Assistant Knowledge Distillation (TAKD) [Mirzadeh et al., 2020], which bridges capacity gaps through intermediate-sized models.

### Links

- → **Scale-Up-Then-Scale-Down Principle**: the proxy is the "scale-up" component
- → **Stage 1: Multi-Teacher Distillation**: where the proxy is trained
- → **Stage 2: Fixed-Resolution Distillation**: where the proxy becomes teacher
- → **Stage 3: Multi-Resolution Finetuning**: where the proxy is also the teacher
- → **Teacher Selection**: determines what the proxy learns

---

## 6. Adapter Heads

### Definition

Adapter heads are per-teacher projection modules that map the student's feature space to each teacher's feature space (Fig. 3). There are two types per teacher: a class-token adapter $H^c_i(\cdot;\psi^c_i)$ and a patch-token adapter $H^p_i(\cdot;\psi^p_i)$.

**Per-teacher distillation flow** (Fig. 3):
```
  Student S(·;θ)                              Teacher T_i(·;ϕ_i) [frozen]
  ┌────────────┐                              ┌────────────────┐
  │   Encoder   │                              │    Encoder      │
  └──┬─────┬───┘                              └──┬─────────┬───┘
     │     │                                     │         │
   y^c_S  y^p_S                               y^c_T_i   y^p_T_i
     │     │                                     │         │
     │     │                                  Normalize  Normalize
     │     │                                     │         │
     ▼     ▼                                  ȳ^c_T_i   ȳ^p_T_i
  ┌──────┐ ┌──────┐                              │         │
  │H^c_i │ │H^p_i │  ←── 2D Interpolate         │         │
  │(ψ^c) │ │(ψ^p) │      if spatial mismatch    │         │
  └──┬───┘ └──┬───┘                              │         │
     │        │                                  │         │
   z^c_T_i  z^p_T_i                              │         │
     │        │                                  │         │
     ▼        ▼                                  ▼         ▼
  ┌─────────────┐                           ┌─────────────────┐
  │  CLS Loss   │                           │   Patch Loss     │
  │  L^c_i(cos) │                           │ L^p_i(cos+sL1)  │
  └──────┬──────┘                           └────────┬────────┘
         └──────────────┬────────────────────────────┘
                        ▼
                  L_i = L^c_i + L^p_i
```

$$z^c_{T_i} = H^c_i(y^c_S; \psi^c_i), \quad z^c_{T_i} \in \mathbb{R}^{d_{T_i}} \tag{Eq.3a}$$

$$z^p_{T_i} = H^p_i(y^p_S; \psi^p_i), \quad z^p_{T_i} \in \mathbb{R}^{N_S \times d_{T_i}} \tag{Eq.3b}$$

### Properties

- **Architecture**: a 2-layer MLP — linear projection (no bias) → LayerNorm → GELU → linear projection (no bias).
- **Hidden dimension**: 1536 in Stage 1, 3072 in Stages 2&3.
- **Spatial alignment**: when the student's spatial resolution ($N_S$) differs from the teacher's ($N_{T_i}$), PyTorch's bicubic 2D interpolation is applied to the smaller set to match $\max(N_S, N_{T_i})$ resolution.
- The adapter heads are **trainable** parameters, learned jointly with the student encoder.
- In Stage 1, there is one adapter pair per teacher (3 teachers = 6 adapters). In Stages 2&3, there is one adapter pair (proxy is the sole teacher).

### Application

The adapter heads decouple the student's internal representation from any specific teacher's feature space. This allows the student to maintain its own representational structure while satisfying the distillation objective for each teacher. At inference time, adapter heads can be used to project student features into a teacher's space (e.g., for zero-shot classification using PEcore's text tower).

### Links

- → **Distillation Loss**: operates on adapter outputs $z^c_{T_i}, z^p_{T_i}$
- → **Stage 1: Multi-Teacher Distillation**: uses multiple adapter pairs
- → **Stage 2: Fixed-Resolution Distillation**: uses one adapter pair
- → **Feature Normalization**: applied to teacher outputs before loss computation

---

## 7. Distillation Loss

### Definition

The distillation loss measures the discrepancy between the student's adapted features and the teacher's normalized features. The same loss formulation is used across all three stages. For each teacher $T_i$, the per-teacher loss combines a class token component and a patch token component:

$$L_i = L^c_i(z^c_{T_i}, \bar{y}^c_{T_i}) + L^p_i(z^p_{T_i}, \bar{y}^p_{T_i}) \tag{Eq.4}$$

where $z^c_{T_i}, z^p_{T_i}$ are the student's adapted tokens (via adapter heads) and $\bar{y}^c_{T_i}, \bar{y}^p_{T_i}$ are the teacher's normalized tokens. The two components are defined as:

**Class token loss** (cosine similarity):

$$L^c_i(z^c_{T_i}, \bar{y}^c_{T_i}) = L_{cos}(z^c_{T_i}, \bar{y}^c_{T_i}) \tag{Eq.5a}$$

**Patch token loss** (weighted combination of cosine similarity and smooth L1):

$$L^p_i(z^p_{T_i}, \bar{y}^p_{T_i}) = \alpha \, L_{cos}(z^p_{T_i}, \bar{y}^p_{T_i}) + \beta \, L_{smooth\text{-}L1}(z^p_{T_i}, \bar{y}^p_{T_i}) \tag{Eq.5b}$$

where $\alpha = 0.9$ and $\beta = 0.1$.

The total loss sums over all teachers:

$$L = \sum_i \Big[ L^c_i(z^c_{T_i}, \bar{y}^c_{T_i}) + L^p_i(z^p_{T_i}, \bar{y}^p_{T_i}) \Big] \tag{Eq.6}$$

### Properties

- **Cosine similarity loss** measures angular alignment between feature vectors, ignoring magnitude. This is appropriate because teacher features have been normalized.
- **Smooth L1 loss** on patch tokens adds a magnitude-matching component that helps preserve spatial structure.
- The 90/10 split ($\alpha=0.9, \beta=0.1$) puts most weight on angular alignment while using smooth L1 as a regularizer for spatial precision.
- For Stage 1, $i$ ranges over all teachers (PEcore-G, PElang-G, DINOv3-H+). For Stages 2&3, $i = 1$ (proxy model only).

### DINOv3 Patch Loss Weight (Appendix)

The DINOv3 teacher's patch tokens behave differently from other teachers. An optional per-teacher weight $\gamma$ is introduced:

$$L_{Dv3} = L^c(z^c_{Dv3}, \bar{y}^c_{Dv3}) + \gamma \, L^p(z^p_{Dv3}, \bar{y}^p_{Dv3}) \tag{Eq.7}$$

Table 10 ablation (Stage 2 only, direct multi-teacher):

| $\gamma$ | IN1k-KNN | TextVQA | SPair@224 | ADE20k |
|---|---|---|---|---|
| 0.0 | 79.7 | **51.8** | 23.4 | 28.7 |
| **1.0** | 80.2 | 50.9 | 29.0 | 31.9 |
| 2.0 | **80.3** | 50.1 | **31.3** | **32.9** |

Higher $\gamma$ improves image understanding and dense prediction but hurts VLM (especially TextVQA). The default $\gamma = 1.0$ gives balanced performance.

### Application

Following AM-RADIO [Ranzinger et al., 2024], the paper keeps the loss formulation simple and uniform across stages. The key design choice is that the same loss works for both multi-teacher (Stage 1) and single-teacher (Stages 2&3) settings.

### Links

- → **Adapter Heads**: produce $z^c_{T_i}, z^p_{T_i}$ that enter the loss
- → **Feature Normalization**: produces $\bar{y}^c_{T_i}, \bar{y}^p_{T_i}$ that enter the loss
- → **Teacher Selection**: determines how many loss terms are summed
- → **Multi-Stage Distillation Pipeline**: same loss used across all stages

---

## 8. Feature Normalization

### Definition

Feature normalization standardizes each teacher's output features before computing the distillation loss:

$$y^c_{T_i} \rightarrow \bar{y}^c_{T_i}, \quad y^p_{T_i} \rightarrow \bar{y}^p_{T_i}$$

The normalization is **mean-std normalization**: subtract the mean and divide by the standard deviation, computed per-coordinate across a small batch of training data.

### Properties

- **Why it is needed**: different teachers have very different feature mean/norm and standard deviation. Without normalization, the loss is dominated by whichever teacher has the largest feature magnitude — typically the class token from one teacher overwhelms the patch tokens from others.
- **Computation**: statistics are measured by running each teacher through a small batch of training data before training begins. The mean and std are then **fixed** for the rest of training.
- **Simplicity advantage**: unlike RADIOv2.5 [Heinrich et al., 2025] which uses complex PHI-S normalization, and unlike UNIC [Sariyildiz et al., 2024] which computes statistics on-the-fly with exponential moving average (requiring all-gather across GPUs every step), EUPE's pre-computed mean-std normalization is simple and scalable.

### Application

The pre-computation step runs for 500 iterations at the start of Stage 1 to estimate per-teacher, per-coordinate mean and variance. Standard ImageNet constants are used for input normalization.

### Links

- → **Distillation Loss**: normalization is applied before loss computation
- → **Stage 1: Multi-Teacher Distillation**: especially important here due to multiple diverse teachers
- → **Adapter Heads**: the adapted student features are compared against normalized teacher features

---

## 9. Stage 2: Fixed-Resolution Distillation

### Definition

Stage 2 transfers knowledge from the trained proxy model (from Stage 1) to the target efficient student encoder. The image resolution is fixed at 256×256 throughout this stage to keep training computationally efficient, allowing a longer training schedule.

### Properties

- **Teacher**: proxy model (1.9B) from Stage 1, frozen.
- **Student**: efficient backbone — ViT-B (86M), ViT-S (21M), ViT-T (6M), ConvNeXt-Base (89M), ConvNeXt-Small (50M), or ConvNeXt-Tiny (29M).
- **Resolution**: fixed 256×256.
- **Schedule**: 390k iterations, batch size 8192, cosine learning rate schedule, base LR 2e-5, weight decay 1e-4.
- **Data augmentation**: random resized cropping, random horizontal flipping, color jittering, Gaussian blur, random solarization.
- **Rationale**: the fixed resolution enables a long training schedule because each iteration is cheap. The student needs extensive exposure to the proxy teacher's universal features at a single scale before learning multi-resolution adaptation.

### Application

Table 2 shows that adding Stage 2 (Stage 1&2&3 vs Stage 1&3) improves VLM metrics:

| Config | TextVQA | SQA | Realworld | POPE | GQA |
|---|---|---|---|---|---|
| Stage 1&3 | 49.5 | 69.2 | 55.1 | 84.6 | **67.3** |
| Stage 1&2&3 | **50.4** | **69.7** | **55.5** | **85.9** | **67.3** |

The long fixed-resolution training provides a strong foundation that the shorter multi-resolution stage (Stage 3) can refine.

### Links

- → **Proxy Teacher Model**: serves as the sole teacher here
- → **Stage 3: Multi-Resolution Finetuning**: initializes from Stage 2 checkpoint
- → **Multi-Stage Distillation Pipeline**: the second stage
- → **EUPE Model Family**: the student architectures trained here

---

## 10. Stage 3: Multi-Resolution Finetuning

### Definition

Stage 3 finetunes the efficient student encoder (initialized from Stage 2) using multi-resolution inputs from the proxy teacher. Instead of a fixed resolution, images are resized into a pyramid of scales {256, 384, 512}. For each training iteration, the teacher and student independently and randomly select one scale from the pyramid.

### Properties

- **Teacher**: proxy model (1.9B) from Stage 1, frozen.
- **Student**: initialized from Stage 2 checkpoint.
- **Resolution pyramid**: {256, 384, 512}.
- **Scale selection**: teacher and student each randomly pick one scale per iteration, independently. This means teacher and student may process different-resolution versions of the same image.
- **Schedule**: 100k iterations, batch size 4096, base LR 1e-5 (shorter than Stage 2 because multi-resolution training is computationally expensive — one multi-res iteration costs ~2× one fixed-res iteration).
- **Purpose**: downstream tasks operate at various resolutions. Without multi-resolution training, the encoder's features break into local patches at resolutions different from the training resolution (visible in Fig. 5: Stage 1&2 PCA components 1&4 show localized artifacts from resolution mismatch). Stage 3 fixes this.

### Application

Fig. 5 visualizes the effect of each stage on dense feature PCA components:
- **Stage 2 only**: noisy feature maps, no clear semantic coherence.
- **Stage 1&2**: clear semantic structure, but components 1&4 break into local regions at inference resolution (resolution mismatch artifact).
- **Stage 1&2&3**: sharp, globally coherent features with fine spatial detail.

Stage 3's multi-resolution training enables the student to produce spatially coherent features at any input resolution within the training range.

### Links

- → **Stage 2: Fixed-Resolution Distillation**: provides the initialization
- → **Proxy Teacher Model**: serves as the teacher
- → **Multi-Stage Distillation Pipeline**: the final stage
- → **Feature Visualization Analysis**: shows the visual impact of each stage

---

## 11. Teacher Selection

### Definition

The choice of which domain-expert foundation encoders to use as teachers in Stage 1. The paper selects one representative encoder from each of the three task domains to maximize coverage.

### Properties

**Default teacher set**: PEcore-G (1.9B), PElang-G (1.7B), DINOv3-H+ (840M).

- **PEcore-G** [Bolya et al., 2025]: Perception Encoder trained with multi-layer feature alignment. Expert for zero-shot image classification and retrieval. Provides strong image understanding and forms the base for VLM capabilities.
- **PElang-G** [Bolya et al., 2025]: PEcore's language-aligned variant. Derived from PEcore-G through alignment with language models. Provides OCR and general VLM capabilities. Complements PEcore-G without conflicting because it is derived from the same base model.
- **DINOv3-H+** (840M) [Simeoni et al., 2025]: Self-supervised model with Gram anchoring for dense feature locality. Expert for dense prediction (segmentation, depth, keypoints). Its patch tokens are critical for spatial tasks.

**Why not SigLIP2-G?** Table 3 shows that adding SigLIP2-G to PEcore-G + DINOv3-H+ degrades TextVQA from 48.6 to 44.8. The hypothesis: having two CLIP-style text-aligned models (PEcore-G and SigLIP2-G) in the same teacher set causes conflicting gradient signals. PElang-G avoids this because it is derived from PEcore-G (same feature space family).

### Application

The teacher set composition directly determines the proxy model's capability profile, which then propagates to the efficient students. Table 4 confirms that the proxy's strengths align with the final student's strengths (e.g., adding PElang-G improves proxy VLM from 56.2 to 58.6 TextVQA, and student VLM from 48.3 to 50.4 TextVQA in Table 3).

### Links

- → **Stage 1: Multi-Teacher Distillation**: where the teachers are used
- → **Proxy Teacher Model**: absorbs the teachers' knowledge
- → **The Domain Expert Problem**: the teachers represent domain specialization
- → **Distillation Loss**: one loss term per teacher

---

## 12. EUPE Model Family

### Definition

The full family of efficient universal perception encoders released by this work. Models span two architecture families — Vision Transformer (ViT) and ConvNeXt — at multiple scales, targeting different computational budgets for on-device deployment.

### Properties

**ViT family**:

| Model | #Params | GFLOPs (256) | GFLOPs (512) | Latency 256 (ms) | Latency 512 (ms) |
|---|---|---|---|---|---|
| ViT-T | 6M | 4 | 17 | 6.8 | 38.3 |
| ViT-S | 21M | 12 | 63 | 17.1 | 97.9 |
| ViT-B | 86M | 47 | 216 | 55.2 | 305.2 |

**ConvNeXt family**:

| Model | #Params | GFLOPs (256) | GFLOPs (512) | Latency 256 (ms) | Latency 512 (ms) |
|---|---|---|---|---|---|
| ConvNeXt-Tiny | 29M | 5 | 20 | 22.4 | 82.4 |
| ConvNeXt-Small | 50M | 11 | 46 | 38.5 | 141.9 |
| ConvNeXt-Base | 89M | 20 | 81 | 59.3 | 222.7 |

Latency measured on iPhone 15 Pro CPU via ExecuTorch. Models under 100M params show acceptable latency even at 512 resolution. ConvNeXt has more FLOPs-efficient operations but convolutional ops are less optimized on CPU compared to ViT's GEMM operations — ConvNeXt's lower FLOPs do not always translate to lower latency.

### Application

**ViT-B results** (Table 1, flagship model):

| Model | IN1k-ZS | IN1k-KNN | TextVQA | SQA | Realworld | POPE | GQA | MMEp | SPair | NYUv2↓ | ADE20k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| *Domain Experts* | | | | | | | | | | | |
| PEcore-B | 78.4 | 79.7 | 50.8 | 70.0 | 52.9 | 85.8 | 65.6 | 1375.5 | 25.9 | 0.641 | 37.4 |
| PEspatial-B | no cls | no cls | 42.8 | 69.0 | 51.6 | 84.4 | 63.3 | 1279.6 | 42.2 | 0.389 | 45.5 |
| SigLIP2-B | 78.2 | 83.2 | **51.6** | 69.8 | 52.5 | 85.0 | 65.2 | **1389.5** | 32.8 | 0.512 | 41.6 |
| DINOv3-ViT-B | no txt | 83.0 | 42.7 | 69.3 | 52.6 | 85.7 | 65.9 | 1368.0 | **51.3** | **0.373** | 51.8 |
| *Agglomerative* | | | | | | | | | | | |
| RADIOv2.5-B | 74.6 | 81.9 | 47.0 | 69.3 | 54.3 | 84.7 | 65.8 | 1349.8 | 48.7 | 0.435 | 49.0 |
| DUNE-B | no txt | 42.5 | 41.7 | 69.2 | 52.0 | 84.5 | 64.1 | 1294.8 | 39.4 | 0.375 | 40.9 |
| **EUPE-ViT-B** | **79.7** | **84.1** | 50.4 | **69.7** | **55.5** | **85.9** | **67.3** | 1374.5 | 51.3 | 0.391 | **52.4** |

EUPE-ViT-B matches or exceeds domain experts across all three task domains — a first for efficient encoders. Numbers in the paper's Table 1 show the gap with the best domain expert in brackets (e.g., TextVQA 50.4 is 1.2 below SigLIP2-B's 51.6).

**ViT-T/S results** (Table 5):

| Model | IN1k-ZS | IN1k-KNN | TextVQA | SQA | Realworld | POPE | GQA | MMEp | SPair | NYUv2↓ | ADE20k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PEcore-T | 52.9 | 61.7 | 44.3 | 68.6 | 47.4 | 82.2 | 60.9 | 1221.1 | 20.2 | 0.756 | 22.0 |
| PEspatial-T | no cls | no cls | 40.7 | 68.0 | 48.7 | 80.7 | 59.0 | 1128.0 | 18.4 | 0.589 | 28.5 |
| **EUPE-ViT-T** | 50.5 | 66.3 | 42.0 | 69.5 | 50.0 | 82.4 | 61.4 | 1258.0 | **37.2** | 0.571 | **36.7** |
| PEcore-S | 62.8 | 71.7 | 47.0 | 69.4 | 48.4 | 83.2 | 63.5 | 1339.2 | 28.6 | 0.599 | 33.7 |
| PEspatial-S | no cls | no cls | 42.5 | 67.6 | 49.0 | 82.8 | 62.9 | 1245.4 | 30.8 | 0.464 | 38.6 |
| DINOv3-ViT-S | no txt | 78.6 | 41.7 | 68.5 | 50.3 | 84.5 | 64.2 | 1263.1 | 47.9 | 0.404 | 46.9 |
| **EUPE-ViT-S** | **69.8** | **78.2** | **44.1** | **69.3** | **51.7** | 84.5 | **65.0** | 1304.9 | 46.5 | 0.455 | 46.6 |

At Tiny scale, EUPE achieves massive gains on dense prediction (+17 SPair, +14.7 ADE20k over PEcore-T). At Small scale, EUPE approaches DINOv3-level dense prediction (46.5 vs 47.9 SPair, 46.6 vs 46.9 ADE20k) while preserving VLM capability that DINOv3 lacks entirely.

**ConvNeXt results** (Table 6):

| Model | TextVQA | SPair | ADE20k |
|---|---|---|---|
| DINOv3-ConvNeXt-T | 41.6 | 35.7 | 42.7 |
| **EUPE-ConvNeXt-T** | **43.7** | **41.3** | **43.5** |
| DINOv3-ConvNeXt-S | 42.6 | 34.7 | 44.8 |
| **EUPE-ConvNeXt-S** | **45.0** | **40.1** | **46.8** |
| DINOv3-ConvNeXt-B | 42.7 | 35.0 | 46.3 |
| **EUPE-ConvNeXt-B** | **46.4** | **37.7** | **48.9** |

EUPE consistently outperforms DINOv3 across all ConvNeXt scales on both VLM and dense prediction tasks.

### Links

- → **Multi-Stage Distillation Pipeline**: the training recipe that produces these models
- → **Stage 2: Fixed-Resolution Distillation**: where student architecture is introduced
- → **Evaluation Protocol**: benchmark setup for these comparisons
- → **Feature Visualization Analysis**: qualitative comparison of EUPE features vs. others

---

## 13. Evaluation Protocol and Benchmark Design

### Definition

The evaluation tests the encoder's **frozen** representations — all encoders are kept frozen and only lightweight heads are trained (or zero-shot protocols are used). This isolates the quality of the pretrained features from task-specific finetuning effects. Three task domains are covered.

### Properties

**Task Domain 1: Image Understanding**
- **IN1k-ZS** (ImageNet zero-shot): use the teacher's text tower to build classifier weights, project student class token into teacher space via adapter head, classify by softmax over dot products. Resolution 224×224.
- **IN1k-KNN** (ImageNet KNN): $k=10$ nearest neighbors using class token L2 distances. Resolution 224×224.

**Task Domain 2: Dense Prediction**
- **ADE20k** (semantic segmentation): linear probe on frozen patch tokens (after LayerNorm), 512×512, mIoU metric, 40k iterations.
- **NYUv2** (monocular depth): linear probe on frozen patch tokens + BatchNorm, RMSE metric (lower is better), 38k iterations.
- **SPair-71k** (semantic keypoint correspondence): training-free, extract dense features from frozen encoder, compute cosine similarity maps, PCK@0.1 metric. Images resized to 448/512 for patch size 14/16.

**Task Domain 3: Vision-Language Modeling**
- Train a LLaVA-1.5 model by swapping the vision encoder. Two-stage training: (1) projector-only on 558K pairs, (2) full finetune on 665K instruction data.
- 6 benchmarks following Cambrian-1 taxonomy: TextVQA (OCR), SQA (knowledge), Realworld and POPE (vision), GQA and MME (general).
- Input resolution: 336×336 for patch-14, 384×384 for patch-16 (to keep visual token count constant).

### Application

This three-domain test bed is designed so that no single encoder architecture can trivially ace all benchmarks. The paper identifies this evaluation suite as a meaningful stress test for universality.

### Links

- → **The Domain Expert Problem**: the evaluation is designed to expose domain specialization
- → **EUPE Model Family**: the models being evaluated
- → **Feature Visualization Analysis**: qualitative complement to quantitative benchmarks

---

## 14. Feature Visualization Analysis

### Definition

Dense patch tokens from each encoder are projected to a 3-dimensional space using Principal Component Analysis (PCA), then mapped to RGB for visualization. This reveals what kind of spatial and semantic information each encoder captures.

### Properties

The paper compares EUPE-ViT-B16 against DINOv3-ViT-B16, DUNE-B14, RADIOv2.5-B16, SigLIP2-B16, and PEcore-B16 (Fig. 4):

- **PEcore / SigLIP2**: patch tokens contain semantic information but are spatially inconsistent → noisy visualizations.
- **DINOv3**: high semantic coherence but lacks discrimination for fine-grained details (e.g., food vs. plates get similar representations).
- **DUNE**: similar to DINOv3 (distilled from dense prediction experts) — coherent but not fine-grained.
- **RADIO**: overly sensitive features — black fur merges with dark backgrounds, breaking semantic coherence.
- **EUPE**: combines **semantically sharp features** with **sensitivity to fine-grained details**. Captures semantic coherence (rows 1&2), fine granularity (row 3), complex spatial structure (row 4), and text awareness (rows 5&6) simultaneously.

### Application

The PCA visualization across training stages (Fig. 5) demonstrates each stage's contribution:
- Stage 2 only → noisy, incoherent features.
- Stage 1&2 → semantically coherent but locally fragmented (resolution mismatch).
- Stage 1&2&3 → sharp, globally coherent features.

This provides visual evidence for why all three stages are necessary and why the scale-up-then-scale-down approach works.

### Links

- → **Multi-Stage Distillation Pipeline**: visualizes the effect of each stage
- → **Stage 3: Multi-Resolution Finetuning**: fixes the local fragmentation visible after Stage 2
- → **The Domain Expert Problem**: shows how different experts have different feature characteristics
- → **EUPE Model Family**: the final model achieves the best visual quality
- → **Evaluation Protocol**: qualitative complement to the quantitative benchmarks

---

## 15. Training Data

### Definition

All three stages train on the same dataset: LVD-1689M [Simeoni et al., 2025], the DINOv3 training set. It consists of a balanced mix of web-crawled images covering all visual concepts and high-quality public datasets such as ImageNet-1k.

### Properties

- **Sampling strategy**: following DINOv3, both homogeneous batches (all from ImageNet-1k) and heterogeneous batches (mix of ImageNet-1k and LVD) are used.
- **ImageNet sampling probability**: 10% of data comes from ImageNet-1k.
- **Comparison with MetaCLIP** (Table 9): despite MetaCLIP having 2.5B images (vs. LVD's 1.689B), training on LVD yields better performance on almost all benchmarks. This indicates LVD's higher data quality compensates for MetaCLIP's larger quantity.

| Training data | IN1k-ZS | IN1k-KNN | TextVQA | SPair | ADE20k |
|---|---|---|---|---|---|
| MetaCLIP + 10% IN1k | 79.3 | 83.7 | 48.5 | 49.0 | 52.6 |
| **LVD + 10% IN1k** | **79.9** | **84.3** | **48.6** | **51.5** | 52.5 |

### Application

Using the same dataset across all stages simplifies the training recipe and ensures that the proxy and student models see the same data distribution, avoiding distribution shift between stages.

### Links

- → **Multi-Stage Distillation Pipeline**: all stages use this data
- → **Stage 1: Multi-Teacher Distillation**: label-free training on this dataset
- → **Feature Normalization**: statistics computed on this dataset

---

## 16. What Existed Before and What This Paper Changes

### 16.1 Prior Approaches and Their Limitations

**Domain-Expert Foundation Encoders.** Models like PEcore [Bolya et al., 2025], DINOv3 [Simeoni et al., 2025], SigLIP2 [Tschannen et al., 2025], and SAM [Kirillov et al., 2023] each push the frontier in their respective domains. PEcore shows that high-quality general features exist in intermediate layers of a contrastively-trained network. DINOv3 uses Gram anchoring to maintain dense feature locality during large-scale self-supervised training. SigLIP2 achieves strong multilingual vision-language understanding. But each is a specialist — deploying all three on an edge device is infeasible, and picking one means accepting gaps in the other domains.

**Direct Multi-Teacher Distillation (AM-RADIO, UNIC, DUNE).** AM-RADIO [Ranzinger et al., 2024] creates a unified student from CLIP, DINOv2, and SAM by progressively merging similar tokens in deeper layers. RADIOv2.5 [Heinrich et al., 2025] addresses resolution mode shifts and teacher imbalance. UNIC [Sariyildiz et al., 2024] uses a "ladder of projectors" and teacher dropping to prevent gradient domination. DUNE [Sariyildiz et al., 2025] merges 2D vision and 3D perception teachers through heterogeneous co-distillation. These methods work well for large backbones (> 300M params) but break down for efficient models (< 100M params). RADIOv2.5-B at ViT-B scale shows significant gaps vs. domain experts on dense prediction and VLM tasks (Fig. 1). The reason: small models cannot simultaneously absorb conflicting feature representations from multiple teachers.

**Single-Teacher Distillation (EfficientSAM, PEspatial).** EfficientSAM [Xiong et al., 2024] distills SAM into a smaller encoder using masked image pretraining. PEspatial [Bolya et al., 2025] distills PEcore's spatial features into intermediate layers. These preserve specific capabilities well but are inherently single-domain — they produce specialists, not generalists.

### 16.2 What This Paper Contributes

**Contribution 1: The scale-up-then-scale-down recipe.** The core insight that efficient universal encoders need a proxy-mediated distillation path. Rather than forcing a small model to learn from multiple diverse teachers directly, first unify the knowledge in a large proxy, then distill from this single coherent source. This fills the gap between "large agglomerative models that work" and "small efficient models that cannot absorb multi-teacher signals."

**Contribution 2: A family of efficient universal checkpoints.** EUPE releases ViT-T/S/B and ConvNeXt-T/S/B models that achieve on-par or better performance than domain experts across image understanding, VLM, and dense prediction — without any task-specific finetuning of the encoder.

**Contribution 3: Empirical insights on the distillation recipe.** Systematic ablations on training stages (Table 2), teacher combinations (Table 3), proxy model capacity (Tables 4, 7, 8), data choices (Table 9), and loss weights (Table 10) provide a practical guide for building efficient universal encoders.

### 16.3 Side-by-Side: Prior Art vs. This Paper

| Dimension | AM-RADIO / RADIOv2.5 | UNIC / DUNE | EUPE |
|---|---|---|---|
| Distillation approach | Direct multi-teacher | Direct multi-teacher | Proxy-mediated (scale up, then down) |
| Works for large models (>300M) | Yes | Yes | Yes |
| Works for efficient models (<100M) | Degrades significantly | Degrades significantly | On-par with domain experts |
| Teacher normalization | PHI-S (complex) | EMA-based (costly, all-gather) | Pre-computed mean-std (simple) |
| Multi-resolution support | RADIOv2.5 adds resolution handling | Not explicit | Dedicated Stage 3 |
| Loss formulation | Per-teacher specific | Teacher dropping | Uniform across all stages |
| Number of distillation stages | 1 | 1 | 3 (proxy + fixed-res + multi-res) |

| Dimension | Best Domain Expert (per task) | EUPE-ViT-B |
|---|---|---|
| Image understanding (IN1k-ZS) | 78.4 (PEcore-B) | **79.7** |
| Dense prediction (ADE20k mIoU) | 51.8 (DINOv3-ViT-B) | **52.4** |
| VLM (Realworld) | 54.3 (RADIOv2.5-B) | **55.5** |
| VLM (GQA) | 65.9 (DINOv3-ViT-B) | **67.3** |
| #Models needed | 3+ (one per domain) | **1** |

### 16.4 The Core Shift in Thinking

Prior agglomerative methods treated multi-teacher distillation as a single-step problem: take $N$ teachers and one student, define a combined loss, and train end-to-end. The assumption was that a good loss formulation and training recipe could handle the knowledge unification and compression simultaneously. This works when the student is large enough to absorb all signals, but fails for efficient models because the unification task and the compression task compete for the student's limited capacity.

EUPE separates these two tasks. Stage 1 handles **unification**: combining diverse domain-expert knowledge into a single coherent representation, using a model large enough that capacity is not a bottleneck. Stages 2&3 handle **compression**: transferring the already-unified knowledge to a small model, which is now a standard single-teacher distillation problem — a well-understood setting where small models can learn effectively.

This separation is simple in hindsight but represents a genuine shift from "make the loss smarter" to "make the problem easier." The paper shows that a simple loss (cosine + smooth L1) applied through a proxy achieves what sophisticated multi-teacher techniques (teacher dropping, ladder of projectors, progressive token merging) cannot: efficient universal encoders that match domain experts.

---

## 17. Quick Reference Card

| # | Finding | Design Choice | Evidence |
|---|---|---|---|
| 1 | Direct multi-teacher distillation fails for small models | Use proxy-mediated approach: scale up first, then down | Table 2: Stage 2 only = 35.1 SPair vs. full pipeline = 51.3 |
| 2 | All three stages contribute complementary gains | Use Stage 1 → Stage 2 → Stage 3 sequentially | Table 2: each removal degrades some task domain |
| 3 | Long fixed-res before short multi-res beats the reverse | Stage 2 (390k iters, 256px) then Stage 3 (100k iters, multi-res) | Table 2: 1&2&3 best overall balance |
| 4 | Two CLIP-style teachers conflict | Avoid PEcore-G + SigLIP2-G together; use PElang-G instead | Table 3: SigLIP2 drops TextVQA by 3.8 points |
| 5 | PElang-G complements PEcore-G for VLM | Include PElang-G for OCR and general VLM | Table 3: TextVQA improves from 48.6 to 50.4 |
| 6 | DINOv3 patch tokens are critical for dense prediction | Include DINOv3-H+ as a teacher | Table 10: $\gamma=0$ drops SPair from 29.0 to 23.4 |
| 7 | Pre-computed mean-std normalization is sufficient | No need for complex on-the-fly normalization schemes | Proven effective across all experiments |
| 8 | LVD data quality beats MetaCLIP quantity | Train on LVD-1689M, not MetaCLIP-2.5B | Table 9: LVD wins on nearly all benchmarks |
| 9 | 7B proxy does not fully transfer to ViT-B student | 1.9B proxy is the sweet spot for ViT-B students | Table 8: 7B proxy degrades VLM transfer |
| 10 | The recipe generalizes across architectures | Works for both ViT (T/S/B) and ConvNeXt (T/S/B) families | Tables 1, 5, 6 |

---

## 18. Open Questions

1. **Progressive distillation for large capacity gaps.** The 7B proxy model improves proxy-level metrics but does not fully transfer to ViT-B (Table 8). The paper suggests Teacher Assistant Knowledge Distillation (TAKD) [Mirzadeh et al., 2020] — cascading through intermediate-sized models (7B → 1.9B → 86M) — as future work.

2. **Scaling proxy teachers further.** While scaling from 1.9B to 7B shows mixed results on the student, the proxy itself improves. Developing better distillation techniques to bridge the capacity gap could unlock larger-proxy benefits.

3. **Improving universal representations for edge and multi-task settings.** The paper positions EUPE as a baseline for future work on deploying versatile vision systems under tight computational budgets.

4. **Interaction between teacher compatibility and loss design.** The SigLIP2 incompatibility (Table 3) suggests that teacher selection and loss design are intertwined. Future work could explore per-teacher loss weights or adaptive loss scheduling to accommodate less-compatible teacher combinations.

5. **ConvNeXt latency vs. ViT.** Despite lower FLOPs, ConvNeXt models have higher latency on CPU due to less-optimized convolutional operations (Table 11). Optimizing ConvNeXt inference on edge hardware could shift the architecture recommendation.

---

## 19. Concept Dependency Graph

```
          The Domain Expert Problem
                    │
                    ▼
    Scale-Up-Then-Scale-Down Principle
                    │
                    ▼
     Multi-Stage Distillation Pipeline
         ┌──────────┼──────────┐
         ▼          ▼          ▼
      Stage 1     Stage 2     Stage 3
      Multi-T     Fixed-Res   Multi-Res
      →Proxy      →Student    Finetune
         │          │          │
         │          │          └──→ Feature Visualization
         │          │                    Analysis
         ▼          ▼
    Proxy Teacher  EUPE Model
      Model        Family
         │
    ┌────┼────┐
    ▼    ▼    ▼
Teacher  Adapter  Distillation
Select.  Heads    Loss
    │              │
    │              ▼
    │         Feature
    │         Normalization
    │              │
    └──────────────┘
              │
              ▼
        Training Data
        (LVD-1689M)
              │
              ▼
        Evaluation Protocol
        & Benchmark Design
```

---

## 20. Key Equations

| # | Equation | Description | Section |
|---|---|---|---|
| Eq. 1 | $(y^c_S, y^p_S) = S(x_S; \theta)$ | Student encoder output: class token + patch tokens | §3.1 |
| Eq. 2 | $(y^c_{T_i}, y^p_{T_i}) = T_i(x_{T_i}; \phi_i)$ | $i$-th teacher encoder output | §3.1 |
| Eq. 3 | $z^c_{T_i} = H^c_i(y^c_S; \psi^c_i), \; z^p_{T_i} = H^p_i(y^p_S; \psi^p_i)$ | Adapter head projections (class + patch) | §3.1 |
| Eq. 4 | $L_i = L^c_i(z^c_{T_i}, \bar{y}^c_{T_i}) + L^p_i(z^p_{T_i}, \bar{y}^p_{T_i})$ | Per-teacher distillation loss | §3.1 |
| Eq. 5 | $L^c_i = L_{cos}(\cdot), \; L^p_i = 0.9 \, L_{cos}(\cdot) + 0.1 \, L_{smooth\text{-}L1}(\cdot)$ | Class token loss (cosine) + patch token loss (cosine + smooth L1) | §3.2 |
| Eq. 6 | $L = \sum_i [L^c_i + L^p_i]$ | Total distillation loss over all teachers | §3.2 |
| Eq. 7 | $L_{Dv3} = L^c_{Dv3} + \gamma \, L^p_{Dv3}$ | DINOv3 patch loss weight ablation | Appendix A |

---

## 21. Reference Map

### Foundation Vision Encoders
- PEcore / PEspatial / PElang [Bolya et al., 2025] — Perception Encoder family
- DINOv2 [Oquab et al., 2024] — Self-supervised ViT with dense features
- DINOv3 [Simeoni et al., 2025] — 7B-param self-supervised model with Gram anchoring
- SigLIP2 [Tschannen et al., 2025] — Multilingual VL encoder with dense features
- CLIP [Radford et al., 2021] — Contrastive language-image pretraining
- SAM [Kirillov et al., 2023] — Segment Anything Model
- MAE [He et al., 2022] — Masked autoencoder
- AIMv2 [Fini et al., 2025] — Multimodal autoregressive pretraining
- SILC [Naeem et al., 2024] — Contrastive learning with local self-distillation

### Knowledge Distillation
- Hinton et al. (2015) — Original knowledge distillation framework
- TAKD [Mirzadeh et al., 2020] — Teacher Assistant KD for bridging capacity gaps
- EfficientSAM [Xiong et al., 2024] — Distilling SAM to efficient encoders
- Heo et al. (2019) — Feature distillation stabilization

### Agglomerative / Multi-Teacher Methods
- AM-RADIO [Ranzinger et al., 2024] — Agglomerative vision foundation model
- RADIOv2.5 [Heinrich et al., 2025] — Improved baselines for agglomerative VFMs
- UNIC [Sariyildiz et al., 2024] — Universal classification via multi-teacher distillation
- DUNE [Sariyildiz et al., 2025] — Distilling universal encoder from 2D and 3D teachers
- ComBo [Ramtoula et al., 2025] — Probing to combine features from multiple FMs
- Formont et al. (2025) — Task-agnostic multi-teacher distillation framework

### Architectures
- ViT [Dosovitskiy et al., 2021] — Vision Transformer
- ResNet [He et al., 2016] — Residual networks
- Swin Transformer [Liu et al., 2021] — Hierarchical ViT with shifted windows
- ConvNeXt [Liu et al., 2022] — Modernized ConvNet
- DenseNet [Huang et al., 2017] — Dense connections
- ResNeXt [Xie et al., 2017] — Aggregated residual transformations

### Benchmarks and Datasets
- ImageNet-1k [Deng et al., 2009] — Image classification
- ADE20k [Zhou et al., 2017] — Semantic segmentation
- NYUv2 [Silberman et al., 2012] — Depth estimation
- SPair-71k [Min et al., 2019] — Semantic keypoint correspondence
- LVD-1689M [Simeoni et al., 2025] — DINOv3 training data
- MetaCLIP [Xu et al., 2023] — Curated CLIP training data

### Vision-Language Modeling
- LLaVA-1.5 [Liu et al., 2023] — Visual instruction tuning
- Cambrian-1 [Tong et al., 2024] — Vision-centric multimodal LLM exploration
- TextVQA [Singh et al., 2019], SQA [Lu et al., 2022], GQA [Hudson & Manning, 2019], POPE [Li et al., 2023], MME [Fu et al., 2023], RealworldQA [xAI, 2024]

### Dense Prediction Evaluation
- LIFT [Suri et al., 2024] — Feature transform for dense ViT descriptors
- Walmer et al. (2023) — Investigating supervision in vision transformers
