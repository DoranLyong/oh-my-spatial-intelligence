# SAM3D Objects - Architecture Flow

## Overview

`demo.py` 실행 시 2D 이미지에서 3D 모델(GLB)을 생성하는 전체 파이프라인 흐름.

## High-Level Flow

> [!abstract]- Architecture Flow
> ```mermaid
> flowchart TD
>     subgraph Input
>         A[Image PNG] --> C[demo.py]
>         B[Mask PNG] --> C
>     end
>
>     subgraph Inference["Inference"]
>         C --> D[notebook/inference.py]
>         D --> E[Load Config<br/>pipeline.yaml]
>         E --> F[Instantiate<br/>InferencePipelinePointMap]
>         F --> G[merge_mask_to_rgba]
>         G --> H2[pipeline.run]
>     end
>
>     subgraph Pipeline["InferencePipelinePointMap"]
>         H2 --> I[compute_pointmap]
>         I --> J[preprocess_image]
>         J --> K[sample_sparse_structure]
>         K --> L[pose_decoder]
>         L --> M[sample_slat]
>         M --> N[decode_slat]
>         N --> O[postprocess_slat_output]
>         O --> P[run_post_optimization]
>     end
>
>     subgraph Output
>         P --> Q[Gaussian Splat]
>         P --> R[GLB Mesh]
>     end
>
>     R --> S[model.glb]
> ```

## Detailed Pipeline

> [!abstract]- Detailed Pipeline (Stage 1 & 2)
> ```mermaid
> flowchart TB
>     subgraph Stage1["Stage 1: Sparse Structure Generation"]
>         A1[RGBA Image] --> A2[Depth Model<br/>MoGe]
>         A2 --> A3[PointMap<br/>3D Point Cloud]
>         A3 --> A4[SS Preprocessor]
>         A4 --> A5[DiT Model<br/>Diffusion Transformer]
>         A5 --> A6[Sparse Structure<br/>Voxel Coordinates]
>     end
>
>     subgraph PoseEstimation["Pose Estimation"]
>         A6 --> B1[Pose Decoder]
>         B1 --> B2[Rotation<br/>Quaternion]
>         B1 --> B3[Translation]
>         B1 --> B4[Scale]
>     end
>
>     subgraph Stage2["Stage 2: Structured Latent Generation"]
>         A6 --> C1[SLAT Preprocessor]
>         C1 --> C2[SLAT Model<br/>Structured Latent]
>         C2 --> C3[Latent Features]
>     end
>
>     subgraph Decoding["Decoding & Output"]
>         C3 --> D1[Gaussian Decoder]
>         C3 --> D2[Mesh Decoder]
>         D1 --> D3[3D Gaussian Splat<br/>.ply]
>         D2 --> D4[Trimesh GLB<br/>.glb]
>     end
>
>     subgraph PostProcess["Post Processing"]
>         D4 --> E1[Layout Post Optimization]
>         B2 --> E1
>         B3 --> E1
>         B4 --> E1
>         E1 --> E2[Final GLB]
>     end
> ```

## Key Components

> [!abstract]- Class Diagram
> ```mermaid
> classDiagram
>     class demo_py {
>         +load_image()
>         +load_single_mask()
>         +inference()
>         +export glb
>     }
>
>     class Inference {
>         -_pipeline: InferencePipelinePointMap
>         +__init__(config_file, compile)
>         +merge_mask_to_rgba(image, mask)
>         +__call__(image, mask, seed)
>     }
>
>     class InferencePipelinePointMap {
>         -depth_model
>         -ss_preprocessor
>         -slat_preprocessor
>         -models
>         +compute_pointmap(image)
>         +preprocess_image(image, preprocessor)
>         +sample_sparse_structure(input_dict)
>         +sample_slat(input_dict, coords)
>         +decode_slat(slat, formats)
>         +run(image, mask, seed)
>     }
>
>     demo_py --> Inference : creates
>     Inference --> InferencePipelinePointMap : contains
> ```

## Data Flow

> [!abstract]- Data Flow
> ```mermaid
> flowchart LR
>     subgraph Input["Input Data"]
>         I1["image.png<br/>(RGB)"]
>         I2["mask.png<br/>(Binary)"]
>     end
>
>     subgraph Transform["Data Transform"]
>         T1["RGBA<br/>(H,W,4)"]
>         T2["PointMap<br/>(3,H,W)"]
>         T3["Preprocessed<br/>Tensor"]
>     end
>
>     subgraph Model["Model Output"]
>         M1["Sparse Structure<br/>coords"]
>         M2["Pose<br/>R,T,S"]
>         M3["SLAT<br/>latent"]
>     end
>
>     subgraph Output["Final Output"]
>         O1["gaussian<br/>GaussianSplat"]
>         O2["glb<br/>Trimesh"]
>     end
>
>     I1 --> T1
>     I2 --> T1
>     T1 --> T2
>     T2 --> T3
>     T3 --> M1
>     M1 --> M2
>     M1 --> M3
>     M3 --> O1
>     M3 --> O2
> ```

## File Structure

| File | Role |
|------|------|
| `demo.py` | Entry point, 사용자 인터페이스 |
| `notebook/inference.py` | Inference wrapper, 설정 로드 |
| `sam3d_objects/pipeline/inference_pipeline_pointmap.py` | 핵심 파이프라인 로직 |
| `checkpoints/hf/pipeline.yaml` | 모델 설정 파일 |

## Execution Sequence

> [!abstract]- Sequence Diagram
> ```mermaid
> sequenceDiagram
>     participant User
>     participant demo.py
>     participant Inference
>     participant Pipeline as InferencePipelinePointMap
>     participant DepthModel
>     participant DiT
>     participant Decoder
>
>     User->>demo.py: python demo.py
>     demo.py->>Inference: __init__(config_path)
>     Inference->>Pipeline: instantiate(config)
>
>     demo.py->>demo.py: load_image(), load_single_mask()
>     demo.py->>Inference: __call__(image, mask, seed)
>     Inference->>Inference: merge_mask_to_rgba()
>     Inference->>Pipeline: run(image)
>
>     Pipeline->>DepthModel: compute_pointmap()
>     DepthModel-->>Pipeline: pointmap (3D)
>
>     Pipeline->>DiT: sample_sparse_structure()
>     DiT-->>Pipeline: coords, features
>
>     Pipeline->>Pipeline: pose_decoder()
>     Pipeline->>DiT: sample_slat()
>     DiT-->>Pipeline: structured latent
>
>     Pipeline->>Decoder: decode_slat()
>     Decoder-->>Pipeline: gaussian, glb
>
>     Pipeline-->>Inference: output dict
>     Inference-->>demo.py: output
>     demo.py->>User: model.glb saved
> ```
