# Figure 2 Architecture Flow

  
`figure_2.py` 스크립트의 아키텍처와 실행 흐름을 설명합니다.

이 실험은 **Fourier Feature Networks (FFN)**의 주파수 매핑이 이미지 재구성에 미치는 영향을 시각화합니다.
  
---

  

## 1. 전체 아키텍처 개요

  
```mermaid

flowchart TB

subgraph Input["🔹 입력 단계"]

A["input.jpeg<br/>원본 이미지"] --> B["Image Preprocessing<br/>512×512 RGB로 리사이즈"]

B --> C["Coordinate Grid<br/>좌표 생성 x,y ∈ &#40;-1,1&#41;"]

end

  

subgraph Models["🔸 3가지 FFN 모델 비교"]

D["FFN &#40;σ=10&#41;<br/>Random Fourier Features"]

E["FFN &#40;f₀=1&#41;<br/>Single Frequency"]

F["FFN &#40;f₀=0.5&#41;<br/>Single Frequency"]

end

  

subgraph Training["🔹 학습"]

G["fit_image&#40;&#41;<br/>2000 iterations<br/>Adam optimizer"]

end

  

subgraph Output["🔸 출력"]

H["Reconstructed Images<br/>+ DFT 시각화"]

I["figures/figure_2/<br/>PDF 저장"]

end

  

C --> Models

Models --> G

G --> H --> I

```

---
## 2. 논문의 핵심 메시지 (Section 4.1)
  
> **"The set of frequencies that define the base embedding γ(r) completely determines the frequency support of the reconstruction f_θ(r)."**

  
이 실험은 **입력 매핑 주파수 선택**이 재구성 품질에 결정적 영향을 미친다는 것을 보여줍니다.

  
```mermaid

flowchart LR

subgraph Theory["이론적 배경"]

T1["Theorem 1:<br/>FFN의 표현력 = <br/>γ(r) 주파수의 정수 배음"]

end

  

subgraph Problem["문제 상황"]

P1["f₀ = 1 선택 시<br/>짝수 배수만 표현 가능"]

P2["홀수 주파수 성분 손실<br/>→ 체크보드 아티팩트"]

end

  

subgraph Solution["해결책"]

S1["f₀ = 0.5 선택<br/>모든 정수 주파수 커버"]

S2["Random Fourier (σ=10)<br/>넓은 스펙트럼 커버"]

end

  

Theory --> Problem

Problem --> Solution

```

  
---
## 3. FFN (Fourier Feature Network) 모델 구조
 
```mermaid

flowchart LR

subgraph FFN["FFN Architecture"]

I["Input<br/>(x, y) ∈ [-1,1]²"] --> FM["Fourier Mapping<br/>γ(r) = [sin(2πBr), cos(2πBr)]"]

FM --> L1["Dense(256)<br/>+ ReLU"]

L1 --> L2["Dense(256)<br/>+ ReLU"]

L2 --> L3["Dense(256)<br/>+ ReLU"]

L3 --> L4["Dense(3)<br/>RGB output"]

L4 --> O["Output<br/>RGB Pixel"]

end

  

style I fill:#e1f5fe

style FM fill:#fff3e0

style O fill:#e8f5e9

```

  
### Fourier Mapping 수식

$$\gamma(r) = \begin{bmatrix} \sin(2\pi B \cdot r) \\ \cos(2\pi B \cdot r) \end{bmatrix}$$

여기서 **B 행렬**이 주파수 특성을 결정합니다:

  
| 실험           | B 행렬                      | 주파수 특성                       |
| ------------ | ------------------------- | ---------------------------- |
| RFF (σ=10)   | `10 × N(0,1)` (256×2)     | 랜덤 Fourier Features, 넓은 스펙트럼 |
| BFF (f₀=1)   | `[[1,0],[0,1]]` (2×2)     | 단일 주파수, 짝수 배수만 표현            |
| BFF (f₀=0.5) | `[[0.5,0],[0,0.5]]` (2×2) | 단일 주파수, 모든 정수 주파수 커버         |

---
## 4. 3가지 실험 비교

```mermaid

flowchart TB

subgraph Exp1["실험 1: Random Fourier Features (σ=10)"]

B1["B = 10 × N(0,1)<br/>256개 랜덤 주파수"]

R1["✅ 고주파 디테일 복원<br/>넓은 스펙트럼 커버"]

end

  

subgraph Exp2["실험 2: Single Frequency (f₀=1)"]

B2["B = I₂ (단위행렬)<br/>주파수 = 1"]

R2["❌ 체크보드 아티팩트<br/>H(Ω) ⊆ {2k·π | k∈ℤ}"]

end

  

subgraph Exp3["실험 3: Single Frequency (f₀=0.5)"]

B3["B = 0.5·I₂<br/>주파수 = 0.5"]

R3["✅ 아티팩트 없음<br/>H(Ω) ⊆ {k·π | k∈ℤ}"]

end

  

B1 --> R1

B2 --> R2

B3 --> R3

  

style R1 fill:#e8f5e9

style R2 fill:#ffebee

style R3 fill:#e8f5e9

```

  

---
## 5. 데이터 처리 파이프라인

  
```mermaid

flowchart TB

subgraph Load["이미지 로드"]

RAW["data/input.jpeg<br/>원본 이미지"]

end

  

subgraph Preprocess["전처리"]

NORM["Normalize<br/>÷ 255 → [0,1]"]

CROP["crop_from_right()<br/>960px 크롭"]

RESIZE["Resize<br/>512×512"]

end

  

subgraph Dataset["데이터셋 생성"]

COORD["Coordinate Grid<br/>512×512×2<br/>x,y ∈ [-1,1]"]

PIXEL["Pixel Values<br/>512×512×3<br/>RGB"]

end

  

subgraph Split["Train/Test 분할"]

TRAIN["Train Data<br/>256×256 (1/2 샘플링)"]

TEST["Test Data<br/>512×512 (전체)"]

end

  

RAW --> NORM --> CROP --> RESIZE

RESIZE --> COORD

RESIZE --> PIXEL

COORD --> TRAIN

COORD --> TEST

PIXEL --> TRAIN

PIXEL --> TEST

```

  
### image_to_dataset() 함수 동작


```python

# 좌표 그리드 생성: [-1, 1] 범위
coords = np.linspace(-1, 1, 512)

x_test = np.meshgrid(coords, coords) # 512×512×2


# Train: 1/2 다운샘플링 (256×256)
train_data = [x_test[::2, ::2], img[::2, ::2]]


# Test: 전체 해상도 (512×512)
test_data = [x_test, img]
```


---
## 6. 학습 루프 상세
```mermaid

flowchart TB

subgraph Init["초기화"]

A["파라미터 랜덤 초기화"]

end

  

subgraph Loop["Training Loop × 2000"]

B["Train Data: 좌표 x, y"] --> C["γ(r) = sin/cos(2πBr)"]

C --> D["MLP Forward Pass"]

D --> E["예측 RGB 출력"]

E --> F["Loss = 0.5 × MSE"]

F --> G["∇Loss 계산"]

G --> H["θ ← θ - lr·∇Loss"]

H -.-> B

end

  

subgraph Output["출력"]

I["최종 파라미터로 이미지 재구성"]

end

  

Init --> Loop

Loop --> Output

  

style C fill:#fff3e0

style F fill:#ffebee

style H fill:#e8f5e9

```
### 학습 하이퍼파라미터

| 파라미터            | 값    | 설명                 |
| --------------- | ---- | ------------------ |
| `iters`         | 2000 | 총 학습 반복 횟수         |
| `learning_rate` | 1e-4 | Adam learning rate |
| `batch_size`    | None | Full batch (전체 픽셀) |
| `log_every`     | 25   | 로깅 주기              |

---
## 7. 출력 시각화

```mermaid

flowchart LR

subgraph Outputs["출력 파일들"]

direction TB

REC["rec_{name}.pdf<br/>재구성 이미지"]

GT["gt_{name}.pdf<br/>Ground Truth"]

REC_FT["rec_ft_{name}.pdf<br/>재구성 DFT"]

GT_FT["gt_ft_{name}.pdf<br/>GT DFT"]

end

  

subgraph Names["실험별 파일명"]

N1["rff_256: Random FF"]

N2["bff_1: f₀=1"]

N3["bff_05: f₀=0.5"]

end

  

Names --> Outputs

Outputs --> DIR["figures/figure_2/"]

```

  
### DFT 시각화의 의미

```mermaid

flowchart TB

subgraph DFT_Analysis["DFT 분석"]

direction LR

IMG["재구성 이미지"] --> FFT["2D FFT"]

FFT --> MAG["Magnitude<br/>|F(u,v)|"]

MAG --> LOG["Log Scale<br/>log(1 + |F|)"]

LOG --> VIS["스펙트럼 시각화"]

end

  

subgraph Interpretation["해석"]

I1["중심: 저주파 (DC)"]

I2["가장자리: 고주파"]

I3["격자 패턴: 아티팩트"]

end

  

DFT_Analysis --> Interpretation

```

  

---

## 8. 주파수 커버리지 이론

### Theorem 1 적용

논문의 Theorem 1에 따르면, FFN이 표현할 수 있는 주파수 집합 H(Ω)는:

$$\mathcal{H}(\Omega) \subseteq \left\{ \sum_{i} k_i \omega_i \mid k_i \in \mathbb{Z} \right\}$$

```mermaid

flowchart TB

subgraph Case1["f₀ = 1 인 경우"]

F1["기본 주파수: ω = 2π"]

H1["H(Ω) = {2kπ | k∈ℤ}"]

P1["DFT에서 짝수 인덱스만 가능"]

A1["❌ 홀수 주파수 손실 → 아티팩트"]

end

  

subgraph Case2["f₀ = 0.5 인 경우"]

F2["기본 주파수: ω = π"]

H2["H(Ω) = {kπ | k∈ℤ}"]

P2["DFT에서 모든 인덱스 가능"]

A2["✅ 완전한 주파수 커버리지"]

end

  

F1 --> H1 --> P1 --> A1

F2 --> H2 --> P2 --> A2

  

style A1 fill:#ffebee

style A2 fill:#e8f5e9

```

  

---
## 9. 실행 흐름 State Diagram


```mermaid

stateDiagram-v2

[*] --> LoadImage: 스크립트 시작

  

LoadImage --> Preprocess: input.jpeg 로드

Preprocess --> CreateDataset: 512×512 리사이즈

  

CreateDataset --> Experiment1: 데이터셋 생성

  

state Experiments {

Experiment1: FFN (σ=10)

Experiment2: FFN (f₀=1)

Experiment3: FFN (f₀=0.5)

  

Experiment1 --> Experiment2: 학습 완료

Experiment2 --> Experiment3: 학습 완료

}

  

state EachExperiment {

[*] --> InitModel

InitModel --> Training: 모델 초기화

Training --> Reconstruct: 2000 iterations

Reconstruct --> SaveFigures: 이미지 재구성

SaveFigures --> [*]: PDF 저장

}

  

Experiment3 --> [*]: 모든 실험 완료

```

  
---

  

## 10. 코드-개념 매핑

  
| 코드 위치                         | 개념              | 설명                         |
| ----------------------------- | --------------- | -------------------------- |
| `figure_2.py:91-95`           | Image Load      | 이미지 로드 및 전처리               |
| `figure_2.py:98`              | Dataset         | 좌표-픽셀 데이터셋 생성              |
| `figure_2.py:101-111`         | RFF Experiment  | Random Fourier Features 실험 |
| `figure_2.py:115-122`         | BFF f₀=1        | 단일 주파수 (아티팩트 발생)           |
| `figure_2.py:126-133`         | BFF f₀=0.5      | 단일 주파수 (아티팩트 없음)           |
| `models/models_flax.py:46-51` | Fourier Mapping | γ(r) 구현                    |
| `train/standard.py:82-100`    | Training Loop   | Adam 최적화 루프                |
| `utils/graphics.py`           | DFT Plot        | 스펙트럼 시각화                   |

---
## 11. 핵심 인사이트 요약

```mermaid

flowchart TB

subgraph Main["Figure 2 Key Insights"]

direction TB

subgraph FS["Frequency Selection"]

FS1["기본 주파수가 표현력 결정"]

FS2["f0=1 이면 짝수 배수만"]

FS3["f0=0.5 이면 모든 정수"]

end

subgraph AR["Artifacts"]

AR1["체크보드 패턴"]

AR2["DFT에서 격자로 나타남"]

AR3["주파수 커버리지 부족"]

end

subgraph SO["Solutions"]

SO1["Random Fourier"]

SO2["적절한 기본 주파수 선택"]

SO3["넓은 스펙트럼 커버"]

end

subgraph TH["Theory Connection"]

TH1["Theorem 1 정수 배음"]

TH2["H 집합 분석"]

TH3["Dictionary atoms"]

end

end

  

style FS fill:#e3f2fd

style AR fill:#ffebee

style SO fill:#e8f5e9

style TH fill:#fff3e0

```
---
## 12. 실험 결과 예측

| 실험           | 예상 PSNR | DFT 패턴   | 시각적 품질     |
| ------------ | ------- | -------- | ---------- |
| RFF (σ=10)   | 높음      | 연속적 스펙트럼 | 고주파 디테일 복원 |
| BFF (f₀=1)   | 낮음      | 격자 패턴    | 체크보드 아티팩트  |
| BFF (f₀=0.5) | 중간      | 연속적 스펙트럼 | 부드러운 재구성   |

---
## 참고 문헌

- Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" (FFN)
- 논문 Section 4.1: "Spatial artifacts stemming from limited frequency support"
- 논문 Figure 2: 원본 실험 결과