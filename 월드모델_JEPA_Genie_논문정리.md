# 월드 모델(World Model) 논문 정리: JEPA & Genie

> 정리 기준일: 2026-10-01
> 수치와 세부 설정은 원 논문 기준으로 요약한 것입니다. 발표·인용에 쓸 때는 원문에서 한 번 더 확인하세요.

---

## 0. 월드 모델이란?

**월드 모델**은 "지금 상태와 행동이 주어졌을 때 세상이 다음에 어떻게 될지"를 예측하는 내부 모델입니다.

```
현재 상태 s_t  +  행동 a_t   ──▶  [World Model]  ──▶  다음 상태 ŝ_{t+1}
```

이렇게 하면 에이전트는 실제로 행동하기 전에 머릿속에서 결과를 시뮬레이션해 볼 수 있습니다(계획, planning). 예측을 **어디서** 하느냐에 따라 두 가지 흐름으로 나뉩니다.

| 흐름 | 대표 | 예측 대상 | 한 줄 요약 |
|---|---|---|---|
| **비생성형 (표현 공간 예측)** | Meta **JEPA** 계열 (Yann LeCun) | 추상적인 **임베딩(representation)** | "픽셀은 버리고 의미만 예측하자" |
| **생성형 (픽셀/토큰 공간 예측)** | Google DeepMind **Genie** 계열 | 실제 **영상 프레임** | "조작할 수 있는 세계를 통째로 생성하자" |

---

## Part 1. JEPA (Joint-Embedding Predictive Architecture)

### 1-1. A Path Towards Autonomous Machine Intelligence (LeCun, 2022, 포지션 페이퍼)

JEPA 계열의 철학적 출발점이 되는 글입니다.

**문제의식**
- LLM이나 픽셀 단위 생성 모델은 세상의 **예측 불가능한 세부**(나뭇잎 흔들림, 물결 질감 등)까지 맞히느라 용량을 낭비합니다.
- 사람이나 동물처럼 적은 데이터로 상식을 배우려면 **추상화된 표현 공간에서 예측**해야 한다고 봅니다.

**제안한 자율 지능 아키텍처 (6개 모듈)**

| 모듈 | 역할 |
|---|---|
| Configurator | 과제에 맞게 다른 모듈을 설정하는 총괄 |
| Perception | 현재 세상 상태를 추정 |
| **World Model** | 행동에 따른 미래 상태를 예측 (핵심) |
| Cost | 내재 비용(intrinsic cost)과 학습되는 비평가(critic)로 "불편함" 측정 |
| Short-term Memory | 상태와 비용 기록 |
| Actor | 비용을 최소화하는 행동 시퀀스 탐색 |

**핵심 개념**
- **JEPA**: x와 y를 각각 인코딩한 뒤, s_x로부터 s_y를 **임베딩 공간에서** 예측합니다. y를 픽셀로 복원하지 않습니다.
- **에너지 기반 모델(EBM)** 관점: 호환되는 (x, y) 쌍은 낮은 에너지, 호환되지 않는 쌍은 높은 에너지를 갖도록 학습합니다.
- **붕괴(collapse) 방지**: 모든 입력을 같은 벡터로 보내는 자명한 해를 막아야 합니다. 대조 학습(contrastive) 대신 **정규화 기반**(VICReg 같은 분산·공분산 제약)을 선호합니다.
- **H-JEPA**: JEPA를 계층적으로 쌓아 짧은 시간 단위의 세부 예측과 긴 시간 단위의 추상 예측을 함께 다룹니다.
- **Mode-1 / Mode-2**: 반사적 행동(System 1)과 월드 모델을 이용한 숙고형 계획(System 2)을 구분합니다.

---

### 1-2. I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (Assran et al., CVPR 2023)

JEPA 아이디어를 **이미지**에 처음 구현한 논문입니다.

**구조**
```
이미지 ─┬─ context block ──▶ Context Encoder (ViT) ──┐
        │                                            ├─▶ Predictor (좁은 ViT) ──▶ 예측 임베딩
        │                        (target 위치 mask token)┘                          │
        └─ 전체 이미지 ──▶ Target Encoder (context encoder의 EMA) ──▶ target block 임베딩 ─┘
                                                            L2 loss
```

- **Context encoder**: 보이는 부분(context block)만 인코딩합니다.
- **Target encoder**: 전체 이미지를 인코딩한 뒤 target block 위치의 임베딩을 뽑습니다. 가중치는 context encoder의 **EMA**로 갱신합니다(붕괴 방지).
- **Predictor**: context 임베딩과 target 위치 정보를 받아 target 임베딩을 예측합니다.

**Multi-block masking 전략 (핵심 설계)**
- Target: 4개 블록, 각각 이미지의 약 15~20%, 종횡비 0.75~1.5
- Context: 이미지의 약 85~100% 크기 블록 하나에서 target과 겹치는 영역을 제거
- Target 블록이 충분히 커야 **의미 수준(semantic)** 의 표현을 학습하고, context가 충분히 정보를 담고 있어야 예측이 가능하다는 점을 실험으로 보였습니다.

**특징과 결과**
- 색 변환·크롭 같은 **수작업 데이터 증강 없이** 학습합니다(DINO, SimCLR 같은 불변성 기반 방법과의 차이).
- MAE(픽셀 복원)보다 선형 평가(linear probing) 성능이 높고, 불변성 기반 방법보다 객체 개수 세기·깊이 예측 같은 **저수준 과제**에서 강합니다.
- **효율성**: ViT-H/14를 ImageNet-1K에서 A100 16장으로 72시간 이내에 학습했습니다. 비슷한 성능의 기존 방법보다 훨씬 적은 GPU 시간이 들었습니다.

---

### 1-3. V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video (Bardes et al., 2024)

JEPA를 **비디오**로 확장한 논문입니다.

**핵심 주장**: 텍스트, 사전학습된 이미지 인코더, 픽셀 복원, 음성 샘플 없이 **특징 예측(feature prediction)만으로** 강력한 비디오 표현을 학습할 수 있다.

**방법**
- 데이터: 공개 데이터셋을 합친 **VideoMix2M** (약 200만 개 영상)
- 비디오를 시공간 패치(tubelet)로 나누고, **시공간 multi-block masking**으로 대부분(약 90%)을 가립니다.
  - short-range mask(작은 블록 여러 개)와 long-range mask(큰 블록 소수)를 함께 사용합니다.
  - 마스크는 시간 축 전체에 걸쳐 확장되어, 인접 프레임에서 정보를 복사하는 지름길을 막습니다.
- 손실은 target encoder(EMA) 출력과 예측의 **L1 loss**입니다.

**평가**: 백본을 얼린 채(frozen) **attentive probe**만 학습합니다.
- ViT-H/16 기준 Kinetics-400 81.9%, Something-Something v2 72.2%, ImageNet-1K 77.9%
- 특히 **움직임 이해가 중요한 SSv2**에서 이미지 기반 모델보다 크게 앞섭니다.
- 픽셀 예측(VideoMAE 등) 대비 **적은 학습 시간과 라벨 효율성**에서 우위를 보였습니다.

---

### 1-4. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning (Meta FAIR, 2025.06)

JEPA가 **"진짜 월드 모델"로 쓰일 수 있음**을 보인 논문입니다. 이해 → 예측 → 계획까지 연결합니다.

**1단계: 행동 없는 사전학습 (V-JEPA 2)**
- 인터넷 영상 **100만 시간 이상**과 이미지 약 100만 장
- ViT-g(약 10억 파라미터)까지 확장하고, 데이터·모델·학습 길이·해상도를 점진적으로 키우는 스케일링 전략을 사용
- 결과
  - 동작 이해: SSv2 top-1 77.3
  - 행동 예측(anticipation): Epic-Kitchens-100 recall@5 39.7 (당시 SOTA)
  - LLM과 정렬 후 비디오 QA: PerceptionTest 84.0, TempCompass 76.9 (8B 규모 기준 SOTA)

**2단계: 행동 조건부 사후학습 (V-JEPA 2-AC)**
- 얼린 V-JEPA 2 인코더 위에 **action-conditioned predictor**를 학습합니다.
- 데이터: **Droid 데이터셋의 라벨 없는 로봇 영상 62시간 미만**
- 입력은 현재 표현과 로봇 행동(엔드 이펙터 상태 변화)이고, 출력은 다음 시점의 표현입니다.

**계획 (Planning)**
- 목표 이미지를 인코딩한 뒤, 예측된 미래 표현과 목표 표현 사이의 거리(에너지)를 최소화하는 행동 시퀀스를 **CEM(Cross-Entropy Method)** 으로 탐색합니다.
- MPC 방식으로 첫 행동만 실행하고 다시 계획합니다.
- 서로 다른 두 연구실의 Franka 로봇팔에 **제로샷**으로 투입했습니다. 해당 환경 데이터 수집, 과제별 학습, 보상 설계 없이 집기·놓기(pick-and-place)를 수행했습니다.
- 픽셀 생성형 월드 모델(Cosmos) 기반 계획보다 행동당 계획 시간이 훨씬 짧았습니다(수 초 vs 수 분).

**함께 공개한 벤치마크**: IntPhys 2(직관 물리), MVPBench, CausalVQA (물리·인과 이해 평가)

---

### 1-5. 최근 JEPA 계열 흐름 (간단히)

- **LeJEPA** (Balestriero & LeCun, 2025.11): EMA나 stop-gradient 같은 휴리스틱 없이, 임베딩이 **등방성 가우시안 분포**를 따르도록 하는 정규화(SIGReg)로 붕괴를 막는 이론 기반 JEPA입니다.
- 오디오(A-JEPA), 포인트 클라우드(Point-JEPA), 언어(LLM-JEPA), 비전-언어(VL-JEPA) 등 다른 모달리티로 확장되고 있습니다.
- LeCun은 2025년 말 Meta를 떠나 JEPA 기반 월드 모델을 연구하는 스타트업(AMI Labs)을 세웠습니다.

---

## Part 2. Genie (Generative Interactive Environments)

### 2-1. Genie: Generative Interactive Environments (Bruce et al., Google DeepMind, 2024.02 / ICML 2024 Best Paper)

**핵심 아이디어**: **행동 라벨이 없는 인터넷 게임 영상만으로** 사용자가 조작할 수 있는 2D 게임 세계를 생성하는 모델을 학습한다.

**데이터와 규모**
- 인터넷의 2D 플랫포머 게임 영상 약 20만 시간을 필터링해 **3만 시간** 사용
- 최종 모델 **11B 파라미터**, 160×90 해상도, 10 FPS

**3가지 구성요소** (모두 메모리 효율적인 **ST-Transformer**, 즉 공간 attention과 시간 attention을 번갈아 쓰는 구조 기반)

```
① Video Tokenizer (ST-ViViT, VQ-VAE)
     영상 프레임 → 이산 토큰

② Latent Action Model (LAM)  ← Genie의 핵심 기여
     (과거 프레임들, 다음 프레임) → 인코더 → 이산 잠재 행동 a_t (코드북 크기 8)
     디코더: (과거 프레임들, a_t) → 다음 프레임 복원
     ⇒ "두 프레임 사이에 무슨 행동이 있었나"를 라벨 없이 스스로 발견

③ Dynamics Model (MaskGIT 방식 트랜스포머)
     (과거 토큰들, 잠재 행동) → 다음 프레임 토큰 예측 (자기회귀)
```

- **추론 시**: 사용자가 이미지 한 장(게임 스크린샷, 손그림, text-to-image 결과 등)으로 시작한 뒤, 8개 중 하나의 잠재 행동을 고르면 다음 프레임이 생성됩니다. LAM은 추론 때 쓰지 않습니다.
- 학습된 잠재 행동은 **입력이 달라도 일관된 의미**(왼쪽 이동, 점프 등)를 가집니다.

**주요 발견**
- 모델 크기와 배치 크기에 따라 성능이 꾸준히 좋아지는 **스케일링 경향**을 보였습니다.
- 패럴랙스(시차 스크롤) 같은 물리적 특성도 학습했습니다.
- **로보틱스 확장**: RT-1 로봇 영상(행동 라벨 제거)으로 2.5B 모델을 학습해, 일관된 잠재 행동과 변형되는 물체의 동역학을 학습했습니다.
- **에이전트 학습 가능성**: 처음 보는 CoinRun 환경에서 전문가 영상의 잠재 행동을 LAM으로 추론하고, 소량의 데이터로 실제 행동에 매핑해 행동 복제(behavior cloning) 성능을 냈습니다.

**한계**: 낮은 해상도와 FPS(약 1 FPS 생성 속도), 16프레임 수준의 짧은 기억, 환각, 2D 환경 한정

---

### 2-2. Genie 2 (Google DeepMind, 2024.12, 블로그 발표)

**핵심**: 이미지 한 장으로 **3D로 조작 가능한 세계**를 생성하는 대규모 파운데이션 월드 모델

- 입력: 한 장의 프롬프트 이미지(주로 Imagen 3로 생성)
- 키보드·마우스 입력으로 캐릭터를 조작할 수 있습니다.
- **구조**: 자기회귀 **잠재 확산 모델(latent diffusion)**. 비디오 오토인코더로 얻은 잠재 프레임을 LLM처럼 causal mask를 쓰는 트랜스포머 동역학 모델이 처리합니다. 행동 조작성을 높이기 위해 classifier-free guidance를 씁니다.
- **나타나는 능력**: 행동에 대한 반응, 반사실(counterfactual) 생성, 화면 밖으로 나갔다 돌아온 영역의 기억(객체 영속성), 1인칭/3인칭/등각 시점, 물리(중력·물·연기), 조명·반사, NPC 상호작용
- 일관성: 최대 약 1분 (대부분 10~20초)
- **활용**: 구글의 범용 3D 에이전트 **SIMA**를 Genie 2가 만든 새 환경에서 테스트했습니다. **무한한 학습 환경을 만들어 에이전트를 훈련시키는 것**이 목표입니다.

---

### 2-3. Genie 3 (Google DeepMind, 2025.08, 블로그 발표)

**핵심**: 텍스트 프롬프트로 **실시간** 상호작용이 가능한 세계를 생성

| 항목 | Genie 2 | Genie 3 |
|---|---|---|
| 입력 | 이미지 | **텍스트** |
| 해상도·속도 | 360p급, 비실시간 | **720p, 24 FPS 실시간** |
| 일관성 유지 | 10~20초 (최대 1분) | **수 분** |
| 기억 | 제한적 | 약 1분 전 장면까지 시각적으로 기억 |

- **Promptable World Events**: 생성 도중 텍스트로 날씨 변화, 새 물체·캐릭터 등장 같은 사건을 일으킬 수 있습니다.
- 명시적 3D 표현(NeRF, Gaussian Splatting) 없이 프레임 단위 자기회귀 생성만으로 일관성이 **창발**합니다.
- SIMA 에이전트가 Genie 3 세계 안에서 목표를 수행하는 실험을 했습니다.
- **한계**: 제한된 행동 공간, 여러 에이전트 간 상호작용의 어려움, 실제 장소의 지리적 정확도 부족, 텍스트 렌더링 문제, 수 분 수준의 상호작용 길이
- 이후 2026년 초 **Project Genie**라는 이름으로 일부 사용자(미국 AI Ultra 구독자)에게 체험 버전이 공개되었습니다.

---

## Part 3. JEPA vs Genie 비교

| 비교 항목 | JEPA 계열 (Meta) | Genie 계열 (DeepMind) |
|---|---|---|
| 패러다임 | **비생성형**, 표현 공간 예측 | **생성형**, 픽셀/토큰 예측 |
| 예측 대상 | 미래의 **임베딩** | 미래의 **프레임** |
| 행동 정보 | 사전학습은 행동 없이, 사후학습에서 실제 로봇 행동 사용 (V-JEPA 2-AC) | 라벨 없이 **잠재 행동**을 스스로 발견 (Genie 1) |
| 주요 목적 | 좋은 표현 학습, 로봇 **계획(planning)** | 조작 가능한 **환경 생성**, 에이전트 학습장 |
| 장점 | 계산 효율, 불필요한 세부 무시, 빠른 계획 | 사람이 직접 보고 체험 가능, 데이터 생성기로 활용 |
| 단점 | 결과를 눈으로 보기 어려움, 붕괴 방지 설계 필요 | 계산 비용이 큼, 환각, 장기 일관성 유지가 어려움 |
| 평가 방식 | 다운스트림 과제(분류, 예측, 로봇 성공률) | 시각 품질(FVD 등), 조작성, 일관성 |
| 철학 | "세상을 이해하려면 **다 그릴 필요는 없다**" | "세상을 **그릴 수 있으면** 이해한 것이다" |

**정리하면**
- **JEPA**는 **"머릿속 시뮬레이터"** 에 가깝습니다. 에이전트가 행동을 고르기 위해 추상 공간에서 빠르게 결과를 예측합니다.
- **Genie**는 **"세계 생성기"** 에 가깝습니다. 사람이나 AI 에이전트가 들어가 상호작용할 수 있는 환경 자체를 만듭니다.
- 두 흐름은 경쟁 관계이면서도 보완 관계입니다. 생성형 모델은 에이전트를 위한 무한한 학습 환경을 제공하고, 표현 기반 모델은 그 안에서 효율적으로 계획하는 두뇌 역할을 할 수 있습니다.

---

## Part 4. 함께 보면 좋은 관련 연구

| 연구 | 한 줄 설명 |
|---|---|
| World Models (Ha & Schmidhuber, 2018) | VAE와 RNN으로 "꿈속에서" 정책을 학습한 초기 월드 모델 |
| Dreamer V1~V3 (Hafner et al.) | 잠재 공간 월드 모델로 상상 속 강화학습, V3는 마인크래프트 다이아몬드 채굴 |
| MuZero (DeepMind, 2020) | 규칙을 모른 채 학습한 모델로 계획 |
| Sora (OpenAI, 2024) | "world simulator"를 표방한 비디오 생성 모델 |
| Cosmos (NVIDIA, 2025) | 물리 AI(로봇·자율주행)용 월드 파운데이션 모델 플랫폼 |
| GameNGen (Google, 2024) | 확산 모델로 DOOM을 실시간 시뮬레이션 |
| DINO-WM (2024) | 사전학습된 DINOv2 특징 공간에서 월드 모델 학습 후 계획 (JEPA와 비슷한 철학) |

---

## 참고 문헌

1. LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*. OpenReview.
2. Assran, M. et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. CVPR 2023. arXiv:2301.08243
3. Bardes, A. et al. (2024). *Revisiting Feature Prediction for Learning Visual Representations from Video*. arXiv:2404.08471
4. Assran, M. et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*. arXiv:2506.09985
5. Balestriero, R. & LeCun, Y. (2025). *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*. arXiv:2511.08544
6. Bruce, J. et al. (2024). *Genie: Generative Interactive Environments*. ICML 2024. arXiv:2402.15391
7. Google DeepMind (2024.12). *Genie 2: A large-scale foundation world model*. (블로그)
8. Google DeepMind (2025.08). *Genie 3: A new frontier for world models*. (블로그)
