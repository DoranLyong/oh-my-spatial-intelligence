# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **curated taxonomy of spatial intelligence and 3D computer vision research**, organized by category. Each subdirectory contains an official or reference implementation of a published paper. The repository is research-focused — it collects, adapts, and experiments with state-of-the-art methods rather than shipping a unified product.

The taxonomy has 6 categories (directories map to these):

| Category | Directory | Sub-categories | Examples |
|---|---|---|---|
| 1. Rendering & Representation | `Rendering_and_Representation/` | — | NeRF, 3DGS, INR Dictionaries |
| 2. Geometry & Structure | `Geometry_and_Structure/` | — | DUSt3R, Test3R, CoMe, UFO-4D, LoGeR, TTT3R |
| 3. Generative 3D | `Generative_3D/` | — | SAM 3D, GaussianGPT, DreamFusion |
| 4. Perception & Understanding | `Perception/` | 00 Universal Encoders, 4-1 Point Cloud Analysis, 4-2 Scene Graphs | UNIC, DUNE, EUPE, Perception Encoder, PointMamba, Hydra |
| 5. 6D Pose Estimation | `6DoF_Pose/` | — | FoundationPose, Any6D |
| 6. Physical AI & Interaction | *(planned)* | 6-1 Dynamic 4D, 6-2 Physics, 6-3 Affordance | 4D GS, Contact-GraspNet |

The root `README.md` defines the full taxonomy in Korean with rationale for each category.

## Architecture

- **No shared codebase.** Each project (e.g. `6DoF_Pose/2024_FoundationPose/`) is self-contained with its own dependencies, configs, and entry points.
- Projects are named `YYYY_ProjectName/` by publication year.
- Common internal layout per project: `models/`, `datasets/`, `cfgs/` (YAML configs), `demo/`, `requirements.txt`, conda `environments/` YAML.
- Some projects include C++/CUDA extensions built via CMake + pybind11 (FoundationPose) or compiled in-place (CroCo RoPE kernels in Test3R).

## Study Notes

Each studied paper has a `study_note.md` in its directory, created via the `/paper-to-note` skill. These follow a standardized concept mind-map format with 4-facet breakdowns (Definition, Properties, Application, Links), key equations, tables, and cross-references.

| Paper | Location |
|---|---|
| CoMe (2026) | `Geometry_and_Structure/2026_CoMe/study_note.md` |
| GaussianGPT (2026) | `Generative_3D/2026_GaussianGPT/study_note.md` |
| Test3R (2025) | `Geometry_and_Structure/2025_Test3R/study_note.md` |
| SAM 3D (2025) | `Generative_3D/2025_SAM3D-objects/study_note.md` |
| UFO-4D (2026) | `Geometry_and_Structure/2026_UFO-4D/study_note.md` |
| LoGeR (2026) | `Geometry_and_Structure/2026_LoGeR/study_note.md` |
| TTT3R (2026) | `Geometry_and_Structure/2026_TTT3R/study_note.md` |
| EUPE (2026) | `Perception/00_UniversalEncoders/2026_EUPE/study_note.md` |
| Perception Encoder (2025) | `Perception/00_UniversalEncoders/2026_PerceptionEncoder/study_note.md` |
| SegAnyGAussians (2025) | `Perception/2025_SegAnyGAussians/study_note.md` |

## Environment & Dependencies

- **Python 3.9–3.12** across projects.
- **PyTorch 2.7.0 + CUDA 12.8** is the standardized stack.
- Package managers: pip, conda/mamba. Each project has its own `requirements.txt` and/or conda YAML.
- Docker support exists for FoundationPose.

## Per-Project Setup

Each project must be set up independently. The general pattern:

```bash
cd <Category>/<YYYY_ProjectName>
# Create env from conda YAML if present:
mamba env create -f environments/<env>.yml
# Or install via pip:
pip install -r requirements.txt
```

Check each project's README.md for specific instructions — dataset downloads, model checkpoint links, and build steps vary significantly.

## Adding a New Paper

1. Place it under the correct taxonomy category directory.
2. Name the directory `YYYY_PaperName/`.
3. Include the paper PDF and a `README.md` with setup/citation.
4. Use `/paper-to-note` to generate a `study_note.md`.
