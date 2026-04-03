# Taxonomy of Spatial Intelligence: From 3D Perception to Physical Interaction
(공간 지능의 분류 체계: 3D 인지에서 물리적 상호작용까지)

## 1. Rendering & Representation (어떻게 표현하고 그릴 것인가?)
* **핵심 키워드:** Photometric Consistency (광도 일관성) & Neural Rendering
* **분류 근거:** "기하학적 정확성(Geometry)보다는, 렌더링했을 때 입력 이미지와 **'얼마나 똑같이 보이는가'**에 최적화된 표현 방식(Representation)이기 때문입니다."
    * Mesh가 울퉁불퉁해도 렌더링된 결과물(이미지)만 좋으면 성공으로 칩니다.
* **ex)** NeRF, 3D Gaussian Splatting (3DGS), Instant-NGP

## 2. Geometry & Structure (어떻게 형상을 복원할 것인가?)
* **핵심 키워드:** Correspondence (대응점) & Global Alignment
* **분류 근거:** "이미지 픽셀 값을 3D 좌표로 직접 매핑(Regression)하여, **'공간적 구조와 관계'**를 해석하는 것이 목표이기 때문입니다."
    * 렌더링 퀄리티보다는 3D 점들이 정확한 위치에 찍히는지가(Metric Accuracy) 중요합니다.
* **ex)** DUSt3R, MVSNet, COLMAP

## 3. Generative 3D (어떻게 보이지 않는 곳을 상상하여 채울 것인가?)
* **핵심 키워드:** Hallucination (생성적 추론) & Prior (사전 지식)
* **분류 근거:** "보이지 않는 곳을 **'계산'**하는 것이 아니라, 학습된 데이터 분포(Prior)를 기반으로 **'상상'**해서 채워 넣기 때문입니다."
    * Input 데이터가 부족해도(Single-view), 그럴듯한(Plausible) 3D 형상을 만들어냅니다.
* **ex)** SAM 3D Objects, DreamFusion, Magic3D

## 4. Perception & Understanding (무엇인지 인지하고 해석할 것인가?)
* **핵심 키워드:** Semantics (의미) & Feature Extraction (특징 추출)
* **분류 근거:** "주어진 3D 데이터(Point Cloud 등)를 입력받아, 기하학적 형상 너머의 **'의미론적 클래스(Label)나 부위(Part)'**를 식별하는 것이 목표이기 때문입니다."
    * 단순한 좌표값(XYZ)을 고차원의 특징 벡터(Feature Vector)로 변환하여 '이것이 무엇인가'를 판단합니다.

### 4-1. Point Cloud Analysis (3D 데이터를 어떻게 이해할 것인가?)
* **핵심 키워드:** Backbone Networks & 3D Feature Learning
* **분류 근거:** "포인트 클라우드를 입력받아, 분류(Classification)·분할(Segmentation) 등을 위한 **고차원 특징 표현**을 학습하는 백본 네트워크입니다."
* **ex)** PointMamba (SSM), PointGST, PointNet++, DGCNN

### 4-2. 3D Scene Graph Generation (공간의 의미론적 관계는 무엇인가?)
* **핵심 키워드:** Hierarchical Understanding (계층적 이해) & Relation Reasoning
* **분류 근거:** "단순한 3D 지도를 넘어, 객체 간의 **'위계적 관계(Relation)와 속성(Attribute)'**을 그래프 구조로 표현하여 공간에 대한 고차원 의미 추론을 수행합니다."
* **ex)** Open-Vocabulary 3D Scene Graphs, Hydra, S-Graphs

## 5. 6D Pose Estimation (객체가 공간상 어디에, 어떤 자세로 놓여있는가?)
* **핵심 키워드:** Pose Matching & Registration (자세 정합 및 등록)
* **분류 근거:** "대상의 형태를 바꾸거나 만드는 것이 아니라, 관측된 데이터(Observation)와 이미 알고 있는 모델(Reference) 사이의 **'공간적 변환 관계(Rotation, Translation)'**를 찾아내는 것이 목표이기 때문입니다."
    * 객체의 형상은 고정체(Rigid Body)로 가정하며, SE(3) 공간 상에서의 최적의 파라미터($R, t$)를 추정하는 것이 핵심입니다.
* **ex)** FoundationPose, MegaPose, PVNet

---

## 6. Physical AI & Dynamic Interaction (어떻게 상호작용하고 움직일 것인가?)
* **핵심 키워드:** Action-oriented Perception (행동 지향적 인지) & Physics (물리)
* **분류 근거:** "단순히 '보는 것'을 넘어, 시간의 흐름에 따른 변화를 추적하거나 로봇이 환경과 **'상호작용(Interaction)'**하기 위해 필요한 물리적/기능적 속성을 파악하는 것이 목표입니다."

### 6-1. Dynamic 4D Reconstruction (시간에 따라 형태가 어떻게 변하는가?)
* **핵심 키워드:** Deformation (변형) & Scene Flow (장면 흐름)
* **분류 근거:** "정지된 세상이 아니라, 움직이거나 변형되는 객체(Non-rigid body)를 **'시간 축(Time-axis)'**을 포함하여 4차원으로 복원합니다."
* **ex)** 4D Gaussian Splatting, Deformable NeRF (D-NeRF)

### 6-2. Physics-based Vision (물리적 속성은 무엇인가?)
* **핵심 키워드:** System Identification (시스템 식별) & Simulation-to-Real
* **분류 근거:** "시각 정보만으로는 알 수 없는 **'질량, 마찰력, 탄성 계수'** 등의 잠재된 물리 파라미터를 영상을 통해 역추정(Inverse Physics)합니다."
* **ex)** Video-to-Physics, Diff-Physics, PAC-NeRF

### 6-3. Affordance Learning (어디를 어떻게 조작할 수 있는가?)
* **핵심 키워드:** Interaction Hotspots (상호작용 지점) & Grasp Generation
* **분류 근거:** "객체의 이름(Semantics)이 아니라, **'기능적 가능성(Affordance)'**을 3D 공간상에 매핑합니다."
    * 컵의 손잡이, 과일의 절단 위치 등 로봇이 행동을 취해야 할 좌표와 방향을 추론합니다.
* **ex)** Contact-GraspNet, Affordance Diffusion, Where2Act
