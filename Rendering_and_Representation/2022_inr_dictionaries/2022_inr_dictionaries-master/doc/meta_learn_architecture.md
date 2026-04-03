# Meta-Learning Architecture Flow

  
`meta_learn.py` 스크립트의 아키텍처와 실행 흐름을 설명합니다.

---
## 1. 전체 아키텍처 개요

  

```mermaid

flowchart TB

subgraph Input["🔹 입력 단계"]

A[CelebA Dataset<br/>얼굴 이미지 데이터셋] --> B[Image Preprocessing<br/>64x64 Grayscale 변환]

C[SIREN Model<br/>w0=30, width=256, depth=5] --> D[Initial Parameters<br/>랜덤 초기화]

end

  

subgraph Meta["🔸 Meta-Learning Loop"]

E[Outer Loop<br/>5000 iterations] --> F[Inner Loop<br/>2 steps per image]

F --> G[Parameter Update<br/>MAML or REPTILE]

G --> E

end

  

subgraph Output["🔹 출력"]

H[Meta-Learned Parameters<br/>maml_celebA_5000.pickle]

end

  

B --> Meta

D --> Meta

Meta --> H

```

  

---
## 2. SIREN 모델 구조

SIREN (Sinusoidal Representation Network)은 좌표 `(x, y)`를 입력받아 픽셀 값을 출력하는 INR입니다.
  
```mermaid

flowchart LR

subgraph SIREN["SIREN Architecture"]

I["Input<br/>(x, y) ∈ [0,1]²"] --> L1["Linear Layer<br/>2 → 256"]

L1 --> S1["sin(ω₀ · x)<br/>ω₀ = 30"]

S1 --> L2["Linear Layer<br/>256 → 256"]

L2 --> S2["sin(ω · x)<br/>ω = 30"]

S2 --> L3["...×3 더 반복"]

L3 --> L4["Linear Layer<br/>256 → 1"]

L4 --> O["Output<br/>Pixel Value"]

end

  

style I fill:#e1f5fe

style O fill:#e8f5e9

```

  
### 하이퍼파라미터

| 파라미터        | 값   | 설명                |
| ----------- | --- | ----------------- |
| `w0`        | 30  | 첫 번째 레이어의 주파수 스케일 |
| `hidden_w0` | 30  | 히든 레이어들의 주파수 스케일  |
| `width`     | 256 | 각 레이어의 뉴런 수       |
| `depth`     | 5   | 총 레이어 수           |
 
---
## 3. Meta-Learning 알고리즘 흐름

  
### 3.1 MAML (Model-Agnostic Meta-Learning) 방식
  

```mermaid

flowchart TB

subgraph Outer["Outer Loop (Meta Update)"]

direction TB

P0["θ: Meta Parameters"] --> BATCH["Sample Batch<br/>3 images"]

  

subgraph Inner["Inner Loop (Task Adaptation)"]

direction LR

IMG1["Image 1"] --> ADAPT1["θ → θ'₁<br/>2 SGD steps"]

IMG2["Image 2"] --> ADAPT2["θ → θ'₂<br/>2 SGD steps"]

IMG3["Image 3"] --> ADAPT3["θ → θ'₃<br/>2 SGD steps"]

end

  

BATCH --> Inner

  

ADAPT1 --> LOSS["Loss = Σ MSE(f_θ'ᵢ, Imageᵢ)"]

ADAPT2 --> LOSS

ADAPT3 --> LOSS

  

LOSS --> GRAD["∇_θ Loss<br/>Backprop through Inner Loop"]

GRAD --> UPDATE["θ ← θ - α·∇_θ Loss<br/>Adam optimizer"]

UPDATE --> P0

end

  

style P0 fill:#fff3e0

style UPDATE fill:#e8f5e9

```

  

### 3.2 Inner Loop vs Outer Loop

  

```mermaid

sequenceDiagram

participant θ as Meta Params (θ)

participant Inner as Inner Loop

participant Outer as Outer Loop

  

Note over θ: 초기 랜덤 파라미터

  

loop 5000 iterations

θ->>Inner: Copy parameters

  

loop 2 inner steps (per image)

Inner->>Inner: loss = MSE(model(coords), image)

Inner->>Inner: θ' ← θ' - 0.01·∇loss (SGD)

end

  

Inner->>Outer: Adapted params θ'

Outer->>Outer: Meta-loss = MSE(f_θ', image)

Outer->>θ: θ ← θ - 1e-5·∇meta_loss (Adam)

end

  

Note over θ: Meta-learned params 저장

```

  

---
## 4. 데이터 처리 파이프라인
  

```mermaid

flowchart LR

subgraph Dataset["CelebA Dataset"]

RAW["Raw Image<br/>218×178 RGB"]

end

  

subgraph Process["Preprocessing"]

CROP["Center Crop<br/>178×178"]

GRAY["Grayscale<br/>변환"]

RESIZE["Resize<br/>64×64"]

end

  

subgraph Coord["Coordinate Grid"]

GRID["(x,y) Grid<br/>64×64 = 4096 points<br/>x,y ∈ [0,1]"]

end

  

RAW --> CROP --> GRAY --> RESIZE

RESIZE --> TARGET["Target Pixels<br/>4096×1"]

GRID --> INPUT["Input Coords<br/>4096×2"]

  

INPUT --> MODEL["SIREN"]

TARGET --> LOSS["MSE Loss"]

MODEL --> PRED["Predictions<br/>4096×1"]

PRED --> LOSS

  

style TARGET fill:#ffebee

style PRED fill:#e8f5e9

```

  

---
## 5. 핵심 개념: Dictionary Learning 관점
  
논문의 핵심 통찰은 **Meta-Learning이 Dictionary Learning과 유사하다**는 것입니다.
  

```mermaid

flowchart TB

subgraph Before["🔴 Meta-Learning 전"]

NTK1["NTK Eigenfunctions<br/>(Random patterns)"]

IMG1["Target Image"]

ENC1["❌ 비효율적 인코딩<br/>많은 eigenfunction 필요"]

end

  

subgraph After["🟢 Meta-Learning 후"]

NTK2["NTK Eigenfunctions<br/>(Face-like patterns)"]

IMG2["Target Image"]

ENC2["✅ 효율적 인코딩<br/>적은 eigenfunction으로 표현"]

end

  

NTK1 --> ENC1

IMG1 --> ENC1

  

NTK2 --> ENC2

IMG2 --> ENC2

  

Before -->|"MAML Training<br/>on CelebA"| After

```

  

### 논문 인용 (Section 5.3)

> "Meta-learning has a reshaping effect on the NTK analogous to dictionary learning, building dictionary atoms as a combination of the examples seen during meta-training."
  

**해석**: MAML로 학습하면 NTK의 eigenfunctions이 얼굴 모양으로 reshape 됩니다. 이로 인해:

- 새로운 얼굴 이미지를 더 빠르게 학습
- 더 적은 gradient step으로 수렴
- 더 좋은 일반화 성능

  

---
## 6. 실행 흐름 상세
  
```mermaid

stateDiagram-v2

[*] --> Initialize: 스크립트 시작

  

Initialize --> LoadData: SIREN 모델 초기화

LoadData --> MetaTrain: CelebA 데이터 로드

  

state MetaTrain {

[*] --> SampleBatch

SampleBatch --> InnerLoop: 3개 이미지 샘플링

  

state InnerLoop {

[*] --> Forward

Forward --> ComputeLoss: 좌표 → 픽셀 예측

ComputeLoss --> Backward: MSE 계산

Backward --> UpdateInner: Gradient 계산

UpdateInner --> CheckSteps: SGD 업데이트

CheckSteps --> Forward: steps < 2

CheckSteps --> [*]: steps = 2

}

  

InnerLoop --> OuterUpdate: 적응된 파라미터

OuterUpdate --> Validate: Meta gradient로 업데이트

Validate --> SampleBatch: iter < 5000

Validate --> [*]: iter = 5000

}

  

MetaTrain --> SaveParams: 학습 완료

SaveParams --> [*]: pickle 파일 저장

```

  
---
## 7. 코드-개념 매핑
  
| 코드 위치                       | 개념               | 설명                       |
| --------------------------- | ---------------- | ------------------------ |
| `meta_learn.py:20-23`       | Model Init       | SIREN 모델 생성 및 파라미터 초기화   |
| `meta_learn.py:25-33`       | Data Load        | CelebA train/val 데이터셋 준비 |
| `train/meta_learn.py:23-38` | Inner Loop       | 태스크별 적응 (2 SGD steps)    |
| `train/meta_learn.py:42-71` | Outer Loop       | 메타 파라미터 업데이트             |
| `train/meta_learn.py:57-67` | MAML Gradient    | 내부 루프를 통한 역전파            |
| `train/meta_learn.py:44-54` | REPTILE Gradient | 파라미터 차이 기반 업데이트          |
| `meta_learn.py:49-50`       | Save Output      | 학습된 파라미터 저장              |
 
---

## 8. 하이퍼파라미터 요약

```mermaid
flowchart TB
    subgraph HP["Meta-Learn Hyperparams"]
        subgraph TR["Training"]
            TR1["BATCH_SIZE: 3"]
            TR2["MAX_ITERS: 5000"]
            TR3["META_METHOD: MAML"]
        end
        subgraph IN["Inner Loop"]
            IN1["INNER_LR: 0.01"]
            IN2["INNER_STEPS: 2"]
            IN3["Optimizer: SGD"]
        end
        subgraph OU["Outer Loop"]
            OU1["OUTER_LR: 1e-5"]
            OU2["Optimizer: Adam"]
        end
        subgraph MO["Model"]
            MO1["w0: 30"]
            MO2["width: 256"]
            MO3["depth: 5"]
        end
        subgraph DA["Data"]
            DA1["Resolution: 64x64"]
            DA2["Grayscale: Yes"]
            DA3["Val Examples: 5"]
        end
    end

    style TR fill:#e3f2fd
    style IN fill:#fff3e0
    style OU fill:#e8f5e9
    style MO fill:#fce4ec
    style DA fill:#f3e5f5
```

  
---
## 9. 출력 파일 활용

`maml_celebA_5000.pickle`에 저장된 meta-learned parameters는 다른 실험에서 사용됩니다:
  

```mermaid

flowchart LR

META["meta_learn.py"] -->|생성| PICKLE["maml_celebA_5000.pickle"]

PICKLE -->|로드| FIG4["figure_4.py<br/>NTK Eigenfunction 분석"]

PICKLE -->|로드| FIG5["figure_5.py<br/>Energy Concentration 분석"]

  

style PICKLE fill:#fff9c4

```

  
이 파라미터로 초기화된 SIREN은:

1. **빠른 수렴**: 적은 gradient step으로 새 이미지 학습
2. **NTK 변형**: Eigenfunctions이 얼굴 형태로 reshape
3. **효율적 인코딩**: 적은 eigenfunction으로 얼굴 신호 표현

  

---
## 참고 문헌

- Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (MAML)
- Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions" (SIREN)
- Tancik et al., "Meta-learned Neural Neural Representations" (Meta-SDF)