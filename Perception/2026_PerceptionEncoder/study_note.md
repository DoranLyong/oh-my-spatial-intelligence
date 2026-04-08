# Perception Encoder Study Note — A Concept Mind-Map

> Paper: Perception Encoder: The best visual embeddings are not at the output of the network (arXiv:2504.13181v2)
> Authors: Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Rasheed, Junke Wang, Marco Monteiro, Hu Xu, Shiyu Dong, Nikhila Ravi, Daniel Li, Piotr Dollár, Christoph Feichtenhofer
> Affiliations: Meta FAIR, UT Austin, MBZUAI, Fudan University, Meta Reality Labs
> Code: https://github.com/facebookresearch/perception_models
> Dataset: https://ai.meta.com/datasets/pe-video/

This note organizes every key concept in the paper as a mind-map.
Each concept is broken down into four facets:

- **Definition** — what it is, stated plainly
- **Properties** — its mathematical or behavioral characteristics
- **Application** — how the paper (or the field) uses it
- **Links** — connections to other concepts in this map

---

## 0. The Big Picture

```
            Pretraining Objectives for Vision Encoders
            (each tailored to specific downstream tasks)
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   Contrastive            Captioning          Self-Supervised
   (CLIP, SigLIP)        (AIMv2, CoCa)       (DINOv2, MAE)
   Good at: classif.     Good at: MLLM       Good at: spatial
   Bad at: spatial        Bad at: spatial      Bad at: language
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
              "Can ONE simple pretraining
               objective do it ALL?"
                              │
                              ▼
              ┌───────────────────────────┐
              │  Robust Contrastive       │
              │  Pretraining Recipe       │
              │  (9 cumulative changes)   │
              └─────────────┬─────────────┘
                            │
                            ▼
                  SURPRISING DISCOVERY:
            The best features for ALL tasks
            exist in INTERMEDIATE layers,
            not at the output (layer 50)
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
            PE_core      PE_lang    PE_spatial
            (§2)         (§4)       (§5)
            CLIP loss    Language   Self-distill
            Image+Video  alignment  + SAM logits
            Classif.     MLLM       Detection
            & Retrieval  tasks      Depth, Track
                │           │           │
                └───────────┼───────────┘
                            ▼
                  PE Family: B (0.1B), L (0.3B), G (1.9B)
                  SOTA across classification, retrieval,
                  MLLM, detection, depth, tracking
```

---

## 1. The Central Discovery: Best Features Are Hidden in Intermediate Layers

### Definition

The paper's core finding: a well-tuned large-scale contrastive vision encoder (PE_core) produces strong, general features for *all* downstream tasks — classification, language modeling, grounding, detection, depth estimation, and tracking — but these features are **hidden in the intermediate layers** of the network, not at the final output. The output layer optimizes for the CLIP contrastive objective, which obfuscates the general features learned internally.

### Properties

- **AIMv2-3B** [37]: best layer for language tasks peaks at layer 24 (of 24), best for spatial tasks also peaks in the middle. Intermediate layers of PE_core match or exceed AIMv2 on language tasks.
- **DINOv2-g** [98]: best layer for spatial tasks peaks at layer 34 (of 40), best for classification peaks in the middle. Intermediate layers of PE_core match or exceed DINOv2 on spatial tasks.
- **PE_core G**: best layer for classification is the last (layer 50), but for spatial tasks it peaks around layer 32–39. For language modeling tasks, intermediate layers (around layer 38–47) perform best. The performance *drops sharply* in the final ~6 layers for all non-CLIP-aligned tasks.
- The last ~18 layers of PE_core function as a "decoder" for the global CLIP token — they progressively aggregate information into global tokens, destroying spatial locality in the process.
- Fig. 8 shows this phenomenon across 7 tasks (classification, OCR Q&A, visual Q&A, grounding, detection, depth, tracking).

### Application

This discovery motivates the two alignment methods (PE_lang and PE_spatial), which "lift" these hidden intermediate features to the end of the network so downstream systems can use them directly without hunting for the optimal layer.

### Links

- → **Robust Image Pretraining Recipe**: the recipe that enables general features to form in intermediate layers
- → **The 18-Layer Decoder and Global Tokens**: explains *why* features degrade in the final layers
- → **Downstream Scalability**: shows the phenomenon scales with model size
- → **Language Alignment (PE_lang)**: one method to extract the hidden features
- → **Spatial Alignment (PE_spatial)**: the other method to extract the hidden features

---

## 2. Robust Image Pretraining Recipe

### Definition

A series of 9 cumulative modifications to vanilla CLIP training that collectively improve ImageNet val accuracy from 78.9% to 81.3% and average robustness from 75.3 to 80.9, while keeping training FLOPs constant at ~1 ZFLOP (1T GFLOPs). The recipe is evaluated on an OpenCLIP ViT-L/14 model at 224 resolution, trained on MetaCLIP 2.3B image-text data.

### Properties

**The 9 changes** (Fig. 2, cumulative results):

| # | Change | ImageNet val | Robustness (avg 6) | Training ZFLOPs |
|---|---|---|---|---|
| 1 | Baseline (vanilla CLIP) | 78.9 | 75.3 | 1.0 |
| 2 | Progressive resolution | 78.9 | 75.1 | 0.5 |
| 3 | Batch size 32K → 64K | 79.5 | 76.2 | 1.1 |
| 4 | LAMB optimizer (LR 2e-3) | 79.9 | 76.9 | 1.1 |
| 5 | High resolution (336 stage) | 80.4 | 78.3 | 1.2 |
| 6 | 2D RoPE | 80.7 | 79.2 | 1.2 |
| 7 | Attention pooling (with CLS) | 81.0 | 80.1 | 1.2 |
| 8 | Data augmentation | 81.1 | 80.8 | 1.2 |
| 9 | Mask regularization | 81.3 | 80.9 | 1.2 |

**Key insights per change:**

- **Progressive resolution** (step 2): split 12B-sample run into 98→154→224 stages (4B each). Halves FLOPs while maintaining performance. Enables longer schedules.
- **Batch size** (step 3): 32K→64K (24B total samples). Larger batches increase "task difficulty" — harder negatives improve contrastive learning by +0.6% IN val and +1.1% robustness.
- **LAMB optimizer** (step 4): replaces AdamW. Stabilizes large-batch training and enables higher LR (2e-3 vs 5e-4), which helps adapt to progressive resolution changes.
- **High resolution** (step 5): adds 4th stage at 336 resolution. Robustness improves 3× more than IN val (+1.4% vs +0.5%).
- **2D RoPE** (step 6): improves extrapolation to different resolutions. Robustness +0.9%, IN val +0.3%.
- **Attention pooling** (step 7): constructs CLIP embedding via attention probing block. Keeping the class token as input to this block is important for small models. +0.3% IN val, +0.9% robustness.
- **Data augmentation** (step 8): heavy random cropping, brightness/saturation jitter, horizontal flip. Random cropping covers full caption context. Horizontal flip helps natural images without hurting OCR. +0.7% robustness, +2.4% ObjectNet.
- **Mask regularization** (step 9): converts MaskFeat [147] into a regularization loss. Duplicates and masks 1/16 of the batch, aligns masked tokens to unmasked counterparts via cosine similarity. CLIP and masked gradients are kept disjoint.

**Scaling behavior** (Figs. 3, 4): The recipe scales well across S/14, B/14, L/14 model sizes and 120K/240K/360K training steps. On difficult datasets (ObjectNet, ImageNet-A), the recipe shows *better* scaling than vanilla CLIP, meaning improvements are not at the cost of scalability.

### Application

This recipe forms the foundation for PE_core. All subsequent stages (video finetuning, distillation, alignment) build on these improvements. The downstream impact is measured via frozen COCO detection (Fig. 9): each change improves the best layer's performance by up to +10 mAP over vanilla CLIP, with progressive resolution and attention pooling contributing the most.

### Links

- → **The Central Discovery**: the recipe enables general features in intermediate layers
- → **PE_core**: the recipe is scaled up to produce the final PE_core models
- → **Downstream Scalability**: the recipe's changes each contribute to downstream performance
- → **Video Data Engine**: the image-pretrained model bootstraps the video captioner

---

## 3. Video Data Engine

### Definition

A 3-phase pipeline that generates well-aligned synthetic video captions for contrastive video finetuning. Web videos lack the descriptive alt-text that web images have, so the engine synthesizes aligned captions from multiple sources: a video captioner, frame-level image captions, and video metadata (titles, descriptions).

### Properties

**Phase 1 — Base Video Captioner (PLM)**: train a Perception Language Model [21] using PE as the vision encoder and Llama [82] as the language decoder, on 64.7M images and videos (natural images, charts, documents, egocentric).

**Phase 2 — PLM + Human-Refined Data**: collect 265K videos (105K from PVD), caption with PLM, have 200 human raters refine captions (correct hallucinations, add missing details). Finetuning PLM on this data greatly boosts captioning quality (Table 1):

| Captioner | AuroraCap [13] Score | VCG Diverse [87] Acc | VCG Bench [86] Score |
|---|---|---|---|
| PLM | 2.2 / 51.9 | 3.1 / 65.1 | 34.3 |
| PLM + Human-Refined | **3.4 / 71.1** | **3.6 / 79.4** | **35.2** |

**Phase 3 — LLM Summarization**: combine PLM video captions, Llama 3.2 [82] frame captions, and video metadata (titles, descriptions) using Llama 3.3 70B to produce the final aligned caption. Video metadata is the most critical component — it contains knowledge not covered by captioning models.

**Video Data Engine ablation** (Table 2): each component contributes. The most important are video metadata and the video caption. Frame captions provide the largest single-component boost to video zero-shot performance.

### Application

The engine produces 22M recaptioned video-text pairs for contrastive finetuning. Despite being a small fraction of the image training data (22M vs billions), this video data substantially improves *both* image and video performance (Fig. 6):
- Image: +1.1% ObjectNet, +1.6% ImageNet-A (plateaus around 7M)
- Video classification: +5.6% to +11.7% (no plateau at 17M)
- Video retrieval: +7.7 to +15.3 top-1 recall (no plateau)

### Links

- → **Robust Image Pretraining Recipe**: provides the image model that bootstraps the video captioner
- → **PE Video Dataset (PVD)**: provides human-refined data for Phase 2
- → **PE_core**: finetuned on the engine's output
- → **Language Alignment (PE_lang)**: PE_lang inherits the video capability through PE_core

---

## 4. PE Video Dataset (PVD)

### Definition

A new video dataset released with this paper. PVD comprises 1M high-quality and diverse videos with accompanying tags and descriptions. The videos are motion-centered, covering both first-person and third-person views. A curated subset of 120K videos has human-refined captions.

### Properties

**Dataset statistics** (Table 3):

| Metric | Value |
|---|---|
| Videos | 998,862 |
| Human captions | 118,862 |
| Total duration | 4625 hrs |
| Duration (s) | 16.7 ± 9.8 |
| Human caption length | 57.1 ± 25.4 words |
| Model caption length | 111.7 ± 43.2 words |

Two annotation types for the 120K subset:
1. **Human-verified captions**: extended summaries (avg 57.1 words), suitable for CLIP-style training.
2. **Long automated captions**: detailed descriptions (avg 111.7 words), ideal for fine-grained video understanding.

**PVD Benchmark**: 15K human-refined video-caption pairs held out as a benchmark for fine-grained video-caption alignment. 10 categories (hand actions, object interactions, food prep, work activities, outdoor scenes, animals, water scenes, object handling, close-ups, nature).

### Application

PVD serves three roles: (1) 105K refined samples improve the video data engine (Phase 2), (2) 15K samples form a new video retrieval benchmark, and (3) the full 1M videos contribute to the video finetuning data. On PVD Benchmark (Table 7), PE_core G outperforms InternVL [19] by +13.6% on text→video and +9.5% on video→text vs SigLIP2 [138].

### Links

- → **Video Data Engine**: provides human-refined data for Phase 2
- → **PE_core**: evaluated on PVD benchmark
- → **Robust Image Pretraining Recipe**: the image model is used as a frame encoder

---

## 5. PE_core: Unified Image-Video Encoder

### Definition

PE_core is a family of large-scale contrastively pretrained vision encoders for both images and videos. It is built by scaling the robust image pretraining recipe to 2B parameters (G scale) and finetuning on recaptioned video data. PE_core is the first open vision encoder to outperform models trained on proprietary JFT-3B and WebLI data on zero-shot image classification.

### Properties

**Architecture** (Table 4, Table 17):

| Scale | Vision Params | Text Params | Width | Depth | MLP | Heads | CLIP Dim | Resolution | Patch |
|---|---|---|---|---|---|---|---|---|---|
| B | 0.09B | 0.31B | 768 | 12 | 3072 | 12 | 1024 | 224 | 16 |
| L | 0.32B | 0.31B | 1024 | 24 | 4096 | 16 | 1024 | 336 | 14 |
| G | 1.88B | 0.47B | 1536 | 50 | 8960 | 16 | 1280 | 448 | 14 |

- Pooling: attention probing block with 8 heads from the last-layer feature (with class token for B, L; without for G).
- Positional embedding: 2D RoPE + 2D learnable absolute positional embeddings (abs). Both are interpolated for variable resolutions.
- Video encoding: average-pool N=8 frame embeddings into one video CLIP embedding. This simple approach outperforms attention-based pooling [19, 84].

**Three-stage training**:
1. **Image pretraining**: 5.4B MetaCLIP image-text pairs, 86B samples seen (B, L) or 58B (G), batch 131K, progressive resolution from 98 up to 448.
2. **Image+video finetuning**: 50M image samples (cooldown) + 22M recaptioned video samples. Videos encoded as N=8 frames, average-pooled, aligned to captions with the same contrastive objective.
3. **Distillation** (B, L only): distill from PE_core G with KL-divergence on image-to-text and text-to-image similarity distributions. Teacher temperature = 0.5× student temperature. 4B samples, ~8% of pretraining schedule.

### Application

**Zero-shot image results** (Table 5, selected rows — all ViT-based, at native resolution):

| Model | Params | Data | IN val | Avg Robust | Avg Fine | Avg Retrieval |
|---|---|---|---|---|---|---|
| SigLIP2-B/16 [138] | 0.1B | 10B | 73.1 | 69.2 | 85.7 | 77.9 |
| **PE_core B** | 0.1B | 5.4B | **73.2** | **78.4** | **71.7** | **74.0** |
| SigLIP2-L/16 [138] | 0.3B | 10B | 81.7 | 84.3 | 95.7 | 82.6 |
| **PE_core L** | 0.3B | 5.4B | **83.9** | **85.3** | **77.9** | **85.5** |
| SigLIP2-g-opt [138] | 1.1B | 10B | 86.2 | 85.0 | 79.8 | 85.2 |
| **PE_core G** | 1.9B | 5.4B | **86.6** | **85.4** | 86.0 | **85.2** |

PE_core G is the first open encoder to outperform proprietary JFT-3B [29] and WebLI [17] models on general classification. It simultaneously obtains SOTA on retrieval and fine-grained classification.

**Zero-shot video results** (Table 6, selected rows):

| Model | Frames | Video Data | Avg Video Classif. | Avg Video Retrieval |
|---|---|---|---|---|
| SigLIP2-B/16 [138] | 8 | n/a | 48.4 | 28.6 |
| **PE_core B** | 8 | 22M | **58.4** | **39.0** |
| InternVideo2 [146] | 8 | 102M | 53.9 | **60.4** |
| **PE_core G** | 8 | 22M | **58.7** | 51.2 |

PE_core outperforms all image-only encoders on video, and matches or exceeds native video models on classification despite using only 22M videos (vs 102M+ for others). Retrieval gap vs InternVideo2 narrows from G scale.

### Links

- → **Robust Image Pretraining Recipe**: foundation for image pretraining
- → **Video Data Engine**: provides recaptioned videos for finetuning
- → **Smaller Model Distillation**: B and L scales distilled from G
- → **The Central Discovery**: PE_core's intermediate layers contain general features
- → **Downstream Scalability**: PE_core's robust recipe enables scaling on downstream tasks
- → **Language Alignment (PE_lang)**: initialized from PE_core
- → **Spatial Alignment (PE_spatial)**: initialized from PE_core
- → **PE Model Family**: PE_core defines the B/L/G architecture

---

## 6. Smaller Model Distillation

### Definition

A distillation finetuning approach [49] to transfer knowledge from PE_core G (teacher) to smaller B and L scale models (students). Both teacher and student encode image and text inputs separately to compute similarity distributions, and the student is optimized via KL-divergence to match the teacher.

### Properties

- **Teacher temperature**: using a smaller softmax temperature for the teacher (0.5× the student's) produces a sharper distribution, which significantly improves distillation. Table 28 shows the ablation: 0.5× beats 1.0× by +0.3 avg class, and beats 2.0× by +2.1%.
- **Schedule**: short — approximately 4B samples seen (~8% of pretraining), lower LR, no weight decay.
- **Building strong L-scale models** (Table 29): starting from PE_core L trained with image pretraining, adding distillation from PE_core G and then video finetuning progressively improves both image and video benchmarks. Image pretraining alone: 76.0 avg. + distillation: 78.1. + video finetuning: **78.0** avg image, **75.4** avg video.

### Application

Distillation enables PE_core B and L to significantly outperform models trained from scratch at those scales. The short schedule (~8% cost) provides a large quality boost.

### Links

- → **PE_core**: provides the G-scale teacher
- → **PE Model Family**: defines the B/L student architectures
- → **Robust Image Pretraining Recipe**: students start from pretrained checkpoints

---

## 7. Layerwise Feature Analysis and The Alignment Problem

### Definition

A systematic analysis of intermediate layers across multiple encoders (AIMv2-3B [37], DINOv2-g [98], PE_core G) on diverse downstream tasks. The key finding: **PE_core has intermediate layers that match or exceed the best specialized models on their own tasks** — but the final output layer fails to express these features.

### Properties

**Experimental setup** (Fig. 8): For each encoder and task, probe every layer with a randomly initialized cross-attention transformer block. For language tasks, use PLM [21] with a frozen encoder and finetuned decoder. For spatial tasks, use ViTDet [72] / Mask R-CNN [43] for detection, DPT [109] for depth, and zero-shot feature correspondence for tracking. Window size = 32×32 tokens (equivalent image size).

**Findings** (Fig. 8):
- *Classification*: all three models peak near the end. PE_core is best.
- *Language modeling* (OCR Q&A, Visual Q&A): AIMv2 peaks at its last layer (24). PE_core's intermediate layers (around 38–47) match AIMv2, but the final layer is significantly worse.
- *Grounding*: DINOv2 performs best overall. PE_core's best layer rivals DINOv2, but the last layer drops sharply.
- *Spatial tasks* (detection, depth, tracking): DINOv2 peaks at layer 34/40. PE_core's best layer matches DINOv2 on detection and depth, and exceeds it on tracking, but final layer is much worse.

**The Alignment Problem**: a well-tuned contrastive model learns general features during pretraining, but the contrastive loss forces the final layers to construct a global CLIP embedding, which obfuscates the general features. The model *fails to output* these features even though it has learned them.

**Encoder probing results** (Table 8): PE_core G's frozen features evaluated with KNN [98], linear probing [19], and attention probing [37] on ImageNet-1k:

| Model | Params | Resolution | Data | KNN | Linear | Attn Probe |
|---|---|---|---|---|---|---|
| DINOv2-g [98] | 1.1B | 224 | 145M | 83.5 | 86.5 | 87.2 |
| RADIOv2.5-g [45] | 1.1B | 518 | — | 85.3 | — | — |
| AIMv2-3B [37] | 2.7B | 448 | 7.2B | — | — | 89.5 |
| EVA 18B [130] | 17.5B | 224 | 2B | — | 88.9 | — |
| **PE_core G** | 1.9B | 448 | 5.4B | **86.8** | **89.5** | **89.8** |

PE_core G outperforms all open encoders on all probing methods, including models with significantly more parameters.

**Impact of robust pretraining on downstream** (Fig. 9): Each of the 9 recipe changes improves the best layer's COCO detection mAP. Total improvement: ~+10 mAP over vanilla CLIP. Progressive resolution (step 2) and attention pooling (step 7) push the best layer deeper into the network (from layer 14/24 to 18/24).

### Application

This analysis motivates the entire PE_lang and PE_spatial lines of work. Without alignment tuning, PE_core's intermediate features are inaccessible to standard downstream pipelines that only use the last layer.

### Links

- → **The Central Discovery**: this analysis is the evidence for the discovery
- → **Robust Image Pretraining Recipe**: the recipe's downstream effects are measured here
- → **Language Alignment (PE_lang)**: lifts language features to the last layer
- → **Spatial Alignment (PE_spatial)**: lifts spatial features to the last layer
- → **The 18-Layer Decoder and Global Tokens**: explains the mechanism behind performance degradation
- → **Downstream Scalability**: extends this analysis across model sizes

---

## 8. Downstream Scalability of Robust Pretraining

### Definition

An analysis showing that the robust pretraining recipe enables contrastive models to produce general features that **scale with model size** across diverse downstream tasks — detection, OCR Q&A, visual Q&A, captioning, grounding, video Q&A (Figs. 11, 12). Prior work [31] found that CLIP models fail to scale on downstream tasks; this paper shows the robust recipe fixes that.

### Properties

**Spatial task scaling** (Fig. 11): Frozen feature layer analysis on S/14, B/14, L/14, and G-scale models for COCO detection. Vanilla CLIP quickly plateaus at L scale (300M). The robust recipe continues scaling to G (2B) and beyond — the best layer's performance increases monotonically with model size.

**Language task scaling** (Fig. 12): Similar analysis for OCR Q&A, Visual Q&A, Captioning, Grounding, Video Q&A. The robust recipe improves scaling for all tasks, including grounding (RefCOCO), which the vanilla recipe fails to scale on at all. The *last layer* still stagnates for both recipes, but the *best layer* scales well with the robust recipe.

**Implication**: contrastive pretraining can produce features as general as those from captioning (AIMv2) or self-supervised (DINOv2) methods — the problem was always the training recipe, not the objective.

### Application

This scalability evidence justifies scaling PE_core to 2B parameters (G scale). It also validates the claim that a single pretraining objective (contrastive) is sufficient — no need to combine multiple objectives.

### Links

- → **Robust Image Pretraining Recipe**: the recipe changes that enable scaling
- → **The Central Discovery**: the scaling analysis supports the central claim
- → **Layerwise Feature Analysis**: the analysis methodology
- → **PE_core**: the scaled-up result of this finding

---

## 9. Language Alignment (PE_lang)

### Definition

A method to "lift" the strong intermediate features in PE_core to the end of the network for multimodal language model (MLLM) tasks. The idea is to adapt PE_core to a pretrained decoder-only LLM via *midtraining* — using autoregressive next-token prediction loss on a mix of image-text tasks — to produce PE_lang, a vision encoder optimized for MLLM use.

### Properties

**Architecture**: PE_core (frozen initially) connected to a pretrained Llama 3.2 3B [82] via a 2-layer MLP vision projector. Output layer of PE_core G: **layer 47** (not the last layer 50 — discarding the last 3 layers improves performance, consistent with the general features being hidden inside).

**Training** (Table 9 ablation):
- **LLM scale**: 1B → 3B → 3B unfrozen. Going to 3B: +1.6 avg. Unfreezing: +78.1→78.4 avg.
- **Vision projector**: 2-layer MLP beats linear (77.2 → 78.1 avg) with only +13.5M parameters.
- **PE output layer**: layer 47 is optimal (vs 41 and 50/last). Layer 50 is worst.
- **Regularization**: LayerScale [135] + DropPath [50] on the vision encoder during alignment boosts average score from 78.1 to 79.9 (+1.8 points). Further unfreezing the LLM: 80.1 avg.

**Final recipe**: Llama 3.2 3B, unfrozen, 2-layer MLP projector on PE_core G layer 47, with LayerScale and DropPath. Two-stage training: (1) warmup on 1M image-text pairs (projector only), (2) finetune all parameters on 70M diverse data samples [21].

**Effect on layer features** (Fig. 13): After language alignment, the best-performing layer moves from intermediate to the *last* layer across all task categories (OCR Q&A, Visual Q&A, Captioning, Grounding). The alignment successfully lifts the hidden features to the output.

### Application

**MLLM comparison** (Table 10, Llama 3.1 8B, native resolution, selected):

| Model | Params | Tokens/img | Avg OCR QA | Avg VQA | Avg Cap | Avg Ground | Avg Video |
|---|---|---|---|---|---|---|---|
| AIMv2-3B [37] | 2.7B | 448/14 | 53.9 | 67.2 | 73.0 | 64.1 | 54.6 |
| InternViT2.5 6B [18] | 5.5B | 448/14 | 72.5 | 75.5 | 68.4 | 69.4 | 64.9 |
| SigLIP2-so [138] | 0.4B | 384/16 | 53.3 | 72.2 | 72.2 | 73.6 | 51.5 |
| **PE_lang L** | 0.3B | 448/14 | 70.5 | 71.8 | 70.2 | 73.0 | 63.7 |
| **PE_lang G** | 1.7B* | 448/14 | **72.9** | **81.8** | **82.8** | **89.8** | **68.8** |

PE_lang G outperforms all vision encoders across all benchmarks, including InternViT2.5 6B which is 3× larger and specifically trained for MLLM midtraining.

**Transferability** (Table 11): PE_lang also transfers to QwenLM 2.5 7B — outperforming all encoders including InternViT2.5 which is specifically aligned to QwenLM.

**System-level MLLM** (Table 12): PLM-8B (PE_lang G + Llama 3.1 8B) outperforms LLaVA-OneVision 7B, Gemma 12B, Qwen2 VL 7B, InternVL 2.5 8B, and InternVL 3 8B across OCR, Q&A, captioning, grounding, and video benchmarks.

### Links

- → **The Central Discovery**: language alignment lifts hidden features to the output
- → **Layerwise Feature Analysis**: identifies the alignment problem that PE_lang solves
- → **PE_core**: provides the base encoder
- → **Spatial Alignment (PE_spatial)**: the other alignment method (for spatial tasks)
- → **PE Model Family**: PE_lang is produced at L and G scales

---

## 10. The 18-Layer Decoder and Global Tokens

### Definition

An analysis of *why* PE_core's last-layer features are poor for spatial tasks. In the final ~18 layers (layers 33–50 of the G model), the attention patterns shift abruptly: a few "background" tokens become **global tokens** that every other token attends to, effectively acting as a built-in decoder that aggregates information for the global CLIP embedding.

### Properties

**Feature analysis** (Fig. 14): At PE_core G layers 30–32, attention maps remain local — each token attends to its spatial neighborhood. At layer 33, this changes abruptly: several tokens (typically in the background) become "global tokens" visible as vertical lines in the full attention matrix. From layer 33 onward, *every* token attends to these global tokens.

**Consequence**: layers 33+ function as a "decoder" for the CLIP contrastive objective. This decoder:
- Destroys spatial correspondences (tracking performance drops, detection degrades)
- Aggregates useful information into the global tokens (the CLIP embedding improves)
- Is beneficial for tasks that rely on semantic understanding (Visual Q&A, as seen in Fig. 8) when accessed through an LLM decoder
- Is harmful for tasks requiring spatial precision (tracking peaks at layer 32, detection at ~39)

**The dichotomy**: the optimal layer for higher-level spatial tasks (detection: ~layer 40, including global tokens) differs from the optimal layer for pure spatial tasks (tracking: ~layer 32, before global tokens).

**Register tokens** [23]: this global token phenomenon is not new — recent work shows it happens in all modern ViTs above L scale. The global tokens are "not necessarily harmful" — for most tasks in Fig. 8, the optimal layer falls within the global token region.

### Application

This analysis directly informs the spatial alignment method (§5.2). The two spatial alignment objectives address the two halves of the dichotomy:
1. Self-distillation from layer 41 → retains semantic information (including beneficial global tokens)
2. SAM mask logit alignment → enforces spatial locality without relying on global tokens

### Links

- → **The Central Discovery**: explains the mechanism behind the discovery
- → **Layerwise Feature Analysis**: this is a deeper dive into the layer analysis
- → **Spatial Alignment (PE_spatial)**: directly informs the spatial alignment design
- → **PE_core**: the encoder being analyzed

---

## 11. Spatial Alignment (PE_spatial)

### Definition

A method to lift PE_core's internal spatial features to the output layer for dense prediction tasks (detection, segmentation, depth, tracking). Unlike language alignment which uses an LLM, spatial alignment uses **self-distillation** from the model's own intermediate layer plus a **novel use of SAM 2.1 mask logits** as a spatial correspondence teacher.

### Properties

**Two objectives**, combined as $L_{\text{spatial}} = L_{\text{core}} + L_{\text{loc}}$:

**1. Self-distillation loss** (retaining semantics):

$$L_{\text{core}} = \frac{1}{n_{\text{tok}}} \sum \left( \frac{(S_{50})(T_{41})^T}{||S_{50}|| \cdot ||T_{41}||} \right) \tag{Eq.2}$$

where $S_{50}$ = last layer features of the student, $T_{41}$ = frozen layer 41 features of the original PE_core G. The model trains its last layer to reproduce its own intermediate layer 41, with heavy regularization (DropPath 0.4, LayerScale 0.1, MaskFeat 75% masking) to prevent trivial solutions.

**2. Locality loss** (encouraging spatial correspondences via SAM):

$$L_{\text{loc}} = \frac{1}{n_{\text{tok}}^2} \sum \left( \frac{(S_{50})(S_{50})^T}{||S_{50}||^2} - \frac{(T_{\text{SAM}})(T_{\text{SAM}})^T}{||T_{\text{SAM}}||^2} \right)^2 \tag{Eq.3}$$

where $T_{\text{SAM}}$ are **SAM 2.1 mask logits** (not raw features). The key insight: SAM's raw features also suffer from global tokens (Fig. 15), making them poor alignment targets. But SAM's mask logits — the $H \times W$ output per query point — are inherently spatially well-defined and free of global token artifacts. These logits are concatenated across all 1024 query points into an $H \times W \times 1024$ tensor and used as the teacher feature map.

A temperature parameter $t$ with exponential scaling $e^{t(x-1)}$ is applied to the SAM logits.

**Effect** (Fig. 16): Self-distillation to layer 41 improves detection, segmentation, and depth, but hurts tracking. SAM logit alignment improves tracking but hurts detection. Combining both achieves the best overall balance — PE_spatial lifts all tasks to the last layer while preserving strong tracking performance.

**Feature visualization** (Fig. 17): PE_core last layer = poor spatial coherence. Self-aligned to layer 41 = better semantics but lacking spatial quality. SAM-aligned = locally clear but without semantics. PE_spatial (both) = semantics + spatial quality.

### Application

**Frozen feature dense prediction** (Table 13, best/last layer):

| Model | Params | DAVIS (tracking) | ADE20k (seg) | NYUv2 (depth) |
|---|---|---|---|---|
| DINOv2-g [98] | 1.1B | 58.5/58.5 | 48.7/47.3 | .297/.308 |
| SigLIP2-g-opt [138] | 1.1B | 51.4/45.3 | 44.0/42.9 | .306/.324 |
| PE_core G | 1.9B | **56.8**/42.8 | 48.4/48.9 | **.262**/.275 |
| **PE_spatial G** | 1.9B | 61.5/**61.5** | **49.3**/**48.9** | .262/.275 |

PE_spatial outperforms all models on all tasks, with the best and last layer features now aligned.

**End-to-end detection** (Table 14, COCO + LVIS):

| Encoder | COCO AP_box | COCO AP_mask |
|---|---|---|
| DINOv2-g [98] | 51.5 | 47.3 |
| SigLIP2-g-opt [138] | 51.6 | 47.3 |
| PE_core G | 51.9 | 47.9 |
| **PE_spatial G** | **54.2** | **49.3** |

**System-level detection** (Table 15, COCO val2017):
PE_spatial G achieves **66.0** box mAP — a new absolute state-of-the-art — using only Object365 as extra data and a simple DETR-style decoder (DETA [99]), outperforming models with much more complex decoders (CoDETR 65.9, HTC++ 62.5).

### Links

- → **The Central Discovery**: spatial alignment lifts hidden spatial features to the output
- → **The 18-Layer Decoder and Global Tokens**: informs the two-teacher design
- → **Layerwise Feature Analysis**: identifies the spatial feature dichotomy
- → **PE_core**: provides the base encoder and self-distillation teacher
- → **Language Alignment (PE_lang)**: the other alignment method (for language tasks)
- → **PE Model Family**: PE_spatial is produced at G scale

---

## 12. PE Model Family and Architecture

### Definition

The complete family of Perception Encoder checkpoints released by this work, spanning three model scales (B, L, G) and three alignment variants (core, lang, spatial).

### Properties

**Full model matrix**:

| Variant | B (0.1B) | L (0.3B) | G (1.9B) |
|---|---|---|---|
| PE_core | ✓ | ✓ | ✓ |
| PE_lang | — | ✓ | ✓ |
| PE_spatial | — | — | ✓ |

**Architecture details** (Table 17):
- All models use ViT backbone with attention pooling (8 heads) for the CLIP embedding.
- Positional encoding: 2D RoPE + absolute learnable embeddings.
- Class token register: used for B and L (not G).
- Text tower: separate transformer, shared CLIP dimension (1024 for B/L, 1280 for G).

**Training pipeline summary**:
1. Image pretraining → PE_core (all scales)
2. Distillation from G → B, L
3. Video finetuning → PE_core (all scales)
4. Language alignment → PE_lang (L, G)
5. Spatial alignment → PE_spatial (G)

### Application

Each variant serves different downstream needs:
- **PE_core**: zero-shot image/video classification and retrieval
- **PE_lang**: plug into MLLMs for OCR, VQA, captioning, grounding, video understanding
- **PE_spatial**: detection, segmentation, depth estimation, tracking

### Links

- → **PE_core**: the base variant
- → **Language Alignment (PE_lang)**: produces the lang variant
- → **Spatial Alignment (PE_spatial)**: produces the spatial variant
- → **Smaller Model Distillation**: produces B and L from G
- → **Robust Image Pretraining Recipe**: defines the training recipe

---

## 13. What Existed Before and What This Paper Changes

### 13.1 Prior Approaches and Their Limitations

**Contrastive encoders (CLIP, SigLIP, SigLIP2).** These learn a global vision-language embedding space via contrastive loss. They excel at zero-shot classification and retrieval. CLIP-style encoders have been the go-to for classification for years, with SigLIP [160] replacing softmax with sigmoid loss and SigLIP2 [138] adding dense feature capabilities. The limitation: conventional wisdom (and recent empirical evidence [31]) holds that CLIP models *fail to scale* on downstream tasks like detection, depth, and VQA. This paper shows the failure is in the training recipe, not the objective.

**Captioning encoders (AIMv2, CoCa).** AIMv2-3B [37] uses multimodal autoregressive pretraining and produces the best features for language modeling tasks (OCR Q&A, VQA). CoCa [158] combines contrastive and captioning losses. These models scale well for language tasks but their spatial features are weak — AIMv2 has no spatial capability worth mentioning.

**Self-supervised spatial encoders (DINOv2, MAE).** DINOv2-g [98] learns dense spatial correspondences without language supervision, producing the best features for detection, tracking, and depth. MAE [44] learns through masked image reconstruction. These models dominate spatial tasks but have no language alignment — they cannot do zero-shot classification or serve as MLLM vision encoders.

**Multi-objective / agglomerative approaches (RADIO, DUNE, UNIC).** These attempt to unify multiple capabilities by combining objectives or distilling from multiple teachers. While they achieve breadth, they add complexity that grows exponentially with the number of objectives [19, 34, 35, 37, 45, 90, 110, 158]. No single, simple, scalable method has produced SOTA features across all tasks.

### 13.2 What This Paper Contributes

**Contribution 1: The discovery that robust contrastive pretraining produces general features hidden in intermediate layers.** This overturns the prior assumption that contrastive training is fundamentally limited to classification. The evidence: PE_core's intermediate layers match AIMv2 on language tasks and DINOv2 on spatial tasks — using only contrastive loss.

**Contribution 2: A robust, scalable image pretraining recipe** that enables contrastive training to produce general features that scale with model size. Nine specific, validated changes (Fig. 2) that collectively add ~+10 mAP on frozen COCO detection (Fig. 9).

**Contribution 3: A video data engine and PE Video Dataset** that turn an image-only encoder into a unified image-video encoder with only 22M synthetic videos, boosting both image and video performance.

**Contribution 4: Two alignment methods** — language alignment (PE_lang) and spatial alignment (PE_spatial) — that extract the hidden general features to produce SOTA encoders for MLLM tasks and spatial tasks, respectively.

**Contribution 5: A new COCO detection state-of-the-art** (66.0 mAP) using a contrastively pretrained encoder and a simple DETR-style decoder, achieved without spatial pretraining data.

### 13.3 Side-by-Side: Prior Art vs. This Paper

| Dimension | CLIP/SigLIP2 | AIMv2 | DINOv2 | PE Family |
|---|---|---|---|---|
| Pretraining objective | Contrastive | Autoregressive | Self-supervised | Contrastive |
| Classification / Retrieval | SOTA | Weak | No zero-shot | SOTA (PE_core) |
| Language modeling (MLLM) | Moderate | SOTA | Poor | SOTA (PE_lang) |
| Spatial tasks (detect/depth) | Poor | Poor | SOTA | SOTA (PE_spatial) |
| Video support | Limited | No | No | Yes (22M videos) |
| Single objective | ✓ | ✓ | ✓ | ✓ |
| Scales to downstream tasks | ✗ [31] | ✓ (lang only) | ✓ (spatial only) | ✓ (all tasks) |

### 13.4 The Core Shift in Thinking

The vision community has operated under the assumption that different pretraining objectives produce fundamentally different feature types — contrastive for classification, captioning for language, self-supervised for spatial. This led to an arms race of multi-objective pretraining methods, each trying to combine the best of multiple worlds at increasing complexity.

PE shows that this assumption was wrong. A single contrastive objective, when trained with the right recipe, already learns all the feature types internally. The problem was never the objective — it was the training recipe (not robust enough to produce general features) and the extraction method (relying on the last layer, which is optimized for the contrastive objective rather than generality).

The shift is from "we need multiple objectives" to "we need better recipes and better ways to extract what one objective already learns." This is a simplification of the paradigm, not a complication — and that is what makes it likely to scale.

---

## 14. Quick Reference Card

| # | Finding | Design Choice | Evidence |
|---|---|---|---|
| 1 | Contrastive pretraining produces general features hidden in intermediate layers | Use alignment tuning to extract them | Fig. 8: PE_core matches AIMv2 on language, DINOv2 on spatial |
| 2 | Vanilla CLIP recipe prevents downstream scaling | Use robust recipe (9 changes) | Figs. 3, 4, 11: robust recipe scales to G, vanilla plateaus at L |
| 3 | Progressive resolution halves training FLOPs | Split training into 98→154→224→336 stages | Fig. 2 step 2: same performance at 0.5× FLOPs |
| 4 | LAMB + high LR stabilizes large-batch training | Use LAMB with LR=2e-3 instead of AdamW 5e-4 | Fig. 2 step 4: +0.4% IN val, +0.7% robustness |
| 5 | Mask regularization is the single biggest robustness gain | MaskFeat as reg loss (1/16 batch, disjoint gradients) | Fig. 2 step 9: +0.1% IN val, cumulative robustness 80.9 |
| 6 | Video metadata is the most critical caption component | Include titles/descriptions in LLM summarization | Table 2: metadata provides the largest single-component boost |
| 7 | Simple frame average pooling beats attention pooling for video | Average 8 frame embeddings | Consistent finding with [19, 84] |
| 8 | Teacher temperature 0.5× is optimal for distillation | Use sharper teacher distribution | Table 28: 0.5× beats 1.0× by +0.3 avg |
| 9 | Layer 47 (not last) is optimal for language alignment | Discard last 3 layers of PE_core G | Table 9: layer 47 > layer 50 |
| 10 | SAM mask logits (not features) are the right spatial teacher | Concatenate 1024-point mask logits as H×W×1024 tensor | Fig. 15: SAM features have global tokens, logits don't |
| 11 | PE_spatial sets new COCO detection SOTA | 66.0 mAP with simple DETA decoder | Table 15: first contrastive encoder to hold detection SOTA |

---

## 15. Open Questions

1. **Unifying PE_lang and PE_spatial.** Currently, language and spatial alignment are separate variants. Can a single alignment procedure produce one encoder that is SOTA on both MLLM and spatial tasks simultaneously?

2. **Scaling video data beyond 22M.** Fig. 6 shows no plateau for video classification or retrieval scaling. How far can synthetic video data take the model?

3. **Improving tracking performance of PE_spatial.** PE_spatial's tracking is lower than the SAM-only aligned model (Fig. 16) — the self-distillation to layer 41 partially counters the locality gains. Can the balance be improved?

4. **Extending to even larger models.** The scaling analysis (Figs. 11, 12) shows no signs of plateauing at G scale. Does the recipe continue to scale at 7B+ parameters?

5. **Closing the gap on video retrieval.** PE_core G trails InternVideo2 on video retrieval (51.2 vs 60.4) despite using much less video data (22M vs 102M+). Can more video data or temporal modeling close this gap?

---

## 16. Concept Dependency Graph

```
     Robust Image Pretraining Recipe
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
   Video Data    PE_core    Downstream
   Engine       (B, L, G)   Scalability
      │             │              │
      ▼             │              ▼
     PVD            │    Layerwise Feature
   Dataset          │      Analysis
      │             │         │
      └──────┐      │    ┌────┘
             ▼      ▼    ▼
         The Central Discovery
         (features hidden in
          intermediate layers)
                    │
         ┌──────────┼──────────┐
         ▼                     ▼
  18-Layer Decoder        Alignment Problem
  & Global Tokens              │
         │          ┌──────────┼──────────┐
         │          ▼                     ▼
         └──→ Language Alignment    Spatial Alignment
              (PE_lang)            (PE_spatial)
                    │                     │
                    └──────────┼──────────┘
                               ▼
                      Smaller Model
                       Distillation
                               │
                               ▼
                      PE Model Family
                      (B, L, G × core/lang/spatial)
```

---

## 17. Key Equations

| # | Equation | Description | Section |
|---|---|---|---|
| Eq. 1 | $\text{scores} = \text{scores} \times \text{softmax}(\text{scores}, \text{dim}=0)$ | Retrieval reweighting (DSL) for fair evaluation | Appendix B.1.2 |
| Eq. 2 | $L_{\text{core}} = \frac{1}{n_{\text{tok}}} \sum \cos(S_{50}, T_{41})$ | Self-distillation loss: last layer to frozen layer 41 | §5.2 |
| Eq. 3 | $L_{\text{loc}} = \frac{1}{n_{\text{tok}}^2} \sum \left(\text{pwcos}(S_{50}) - \text{pwcos}(T_{\text{SAM}})\right)^2$ | Locality loss: pairwise cosine similarity alignment with SAM mask logits | §5.2 |
| Eq. 4 | $0.5x + 0.5g(x, k{=}3, \sigma{=}1)$ | Feature visualization: blend raw features with Gaussian-blurred features to reveal spatial info hidden by global tokens | Appendix B.3.2 |

---

## 18. Reference Map

### Contrastive Vision-Language Pretraining
- CLIP [106], OpenCLIP [51] — foundational contrastive framework
- SigLIP [160], SigLIP2 [138] — sigmoid loss, dense features
- ALIGN [54], BASIC [102], EVA-CLIP [129, 130] — scaling contrastive training
- MetaCLIP [152] — data curation for CLIP
- FLIP [74] — masking for training efficiency

### Captioning / Autoregressive Pretraining
- AIMv1 [30], AIMv2 [37] — autoregressive image models
- CoCa [158] — contrastive + captioning combined
- Image captioners [96, 151] — recaptioning data engines

### Self-Supervised Spatial Encoders
- DINOv2 [98] — self-supervised ViT with dense features
- MAE [44] — masked autoencoder
- MaskFeat [147] — masked feature prediction
- SAM [58], SAM 2 [111] — Segment Anything Model

### Multi-Teacher / Agglomerative Methods
- AM-RADIO [110], RADIOv2.5 [45] — multi-teacher distillation
- UNIC [116], DUNE — multi-teacher for universal encoders
- Theia [119] — distilling for robotics

### Multimodal Language Models
- PLM [21] — Perception Language Model
- Llama 3 [82] — language model backbone
- InternVL [19], InternVL 2.5 [18], InternVL 3 [168] — open MLLMs
- LLaVA-OneVision [66], Qwen-VL [3, 144, 155] — competing MLLMs

### Detection and Dense Prediction
- ViTDet [72] — ViT backbone for detection
- Mask R-CNN [43] — instance segmentation
- DPT [109] — dense prediction transformer
- DETA [99] — DETR with collaborative assignment
- Object365 [120] — detection dataset

### Video Understanding
- InternVideo2 [146], VideoPrism [164] — video foundation models
- CLIP4Clip [84] — CLIP for video retrieval
- Kinetics [55, 62, 126] — video classification benchmarks
- MSR-VTT [153], MSVD — video retrieval benchmarks

### Architectures and Training Techniques
- ViT [29] — Vision Transformer
- RoPE [127] — Rotary Position Embedding
- LAMB [156] — Large batch optimizer
- LayerScale [135], DropPath [50] — regularization
- Register tokens [23] — explaining global tokens in ViTs

### Benchmarks
- ImageNet [26] and robustness variants [4, 46, 47, 112, 143]
- COCO [76] — detection/segmentation
- ADE20K [167] — semantic segmentation
- NYUv2 [123] — depth estimation
- DAVIS [104] — video object segmentation / tracking
- RefCOCO [56] — visual grounding
- TextVQA [125], DocVQA [91], ChartQA [165], InfoVQA [92] — OCR Q&A
- VQAv2 [40], OK-VQA [118], AI2D [57] — Visual Q&A
- MVBench [68], STAR [148], EgoSchema [89], PerceptionTest [105] — Video understanding
