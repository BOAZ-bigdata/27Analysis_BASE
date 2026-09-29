# V-JEPA 2 & Genie 논문 정리

## 정리한 논문

### V-JEPA 2

- **논문명:** *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*
- **저자:** Mahmoud Assran 외
- **발표:** arXiv, 2025
- **링크:** [arXiv 2506.09985](https://arxiv.org/abs/2506.09985)

### Genie

- **논문명:** *Genie: Generative Interactive Environments*
- **저자:** Jake Bruce 외
- **발표:** arXiv, 2024
- **링크:** [arXiv 2402.15391](https://arxiv.org/abs/2402.15391)

두 논문은 모두 비디오로부터 세계의 변화를 학습하려는 **World Model** 연구이다. 다만 목표와 예측 방식은 다르다.

- **V-JEPA 2:** 픽셀을 생성하지 않고 영상의 표현을 예측하며, 여기에 실제 로봇 행동 정보를 추가로 학습해 행동 계획에 사용한다.
- **Genie:** 다음 영상 프레임을 생성하며, 행동 라벨이 없는 비디오에서 이산적인 잠재 행동을 발견해 사용자가 생성 세계를 조작할 수 있게 한다.

---

# 1. V-JEPA 2

## 1.1 연구 배경

World Model은 현재 상태와 행동을 바탕으로 앞으로 세계가 어떻게 변할지 예측하는 모델이다. 이런 모델이 있다면 에이전트는 실제로 모든 행동을 실행해 보지 않고도, 모델 안에서 여러 행동의 결과를 예상한 뒤 목표에 가까워지는 행동을 선택할 수 있다.

기존 World Model 연구에는 두 가지 문제가 있었다.

1. 실제 로봇의 상태와 행동이 함께 기록된 interaction data는 인터넷 비디오보다 수집하기 어렵다.
2. 픽셀 단위로 미래 영상을 생성하면 행동 계획에 중요하지 않은 세부 정보까지 예측해야 하므로 계산량이 커진다.

예를 들어 로봇이 컵을 집는 데에는 컵과 로봇 팔의 위치 변화가 중요하지만, 배경에 있는 물체의 미세한 무늬까지 정확히 생성할 필요는 없다.

V-JEPA 2는 픽셀 대신 **학습된 표현 공간에서 미래를 예측**한다. 논문의 연구 질문은 다음과 같이 정리할 수 있다.

> 대규모 인터넷 비디오를 보면서 세계의 일반적인 구조와 움직임을 먼저 학습하고, 소량의 로봇 상호작용 데이터만 추가하면 새로운 환경에서도 계획할 수 있는 World Model을 만들 수 있는가?

## 1.2 전체 학습 구조

V-JEPA 2는 Figure 1과 Figure 2에서 두 단계의 학습 과정을 제시한다.

```text
1단계: V-JEPA 2 사전학습
인터넷 비디오·이미지
-> 가려진 영상 부분의 표현을 예측
-> 일반적인 영상 표현과 움직임 학습

2단계: V-JEPA 2-AC 후속학습
고정된 V-JEPA 2 encoder + 로봇 상태·행동
-> 행동을 조건으로 다음 프레임의 표현을 예측
-> 목표 이미지에 도달할 행동을 계획
```

첫 단계에서는 행동 정보가 없는 비디오를 사용한다. 두 번째 단계에서는 encoder를 고정하고, 로봇 행동의 결과를 예측하는 새로운 predictor만 학습한다.

## 1.3 V-JEPA 2의 사전학습

### 표현 공간에서의 Mask Denoising

원본 비디오 `y`를 작은 시공간 patch로 나눈 뒤 일부 patch를 제거해 masked video `x`를 만든다.

```text
원본 비디오 y
-> 일부 patch 제거
-> masked video x
-> encoder
-> predictor가 가려진 부분의 표현을 예측
```

논문의 Equation 1은 다음 목적을 나타낸다.

```text
예측값 = P_phi(mask token, E_theta(x))
정답값 = stop-gradient(E_teacher(y))

loss = L1(예측값, 정답값)
```

- `E_theta`: 가려진 영상을 처리하는 학습 대상 encoder
- `P_phi`: 가려진 위치의 표현을 예측하는 predictor
- `E_teacher`: 원본 영상을 처리해 정답 표현을 만드는 teacher encoder
- `mask token`: 어떤 위치가 가려졌는지 predictor에 알려주는 학습 가능한 토큰
- `L1 loss`: 예측한 표현과 정답 표현 사이의 절댓값 차이

Teacher encoder의 파라미터는 직접 역전파로 학습하지 않고, 학습 중인 encoder 파라미터의 Exponential Moving Average로 갱신한다. 정답 쪽에는 stop-gradient를 적용한다. 이는 encoder와 predictor가 모든 입력에 똑같은 값을 내놓는 representation collapse를 막기 위한 장치이다.

### 픽셀 예측과의 차이

V-JEPA 2는 가려진 픽셀 자체를 복원하지 않는다. 원본 영상을 teacher encoder에 통과시켜 얻은 **추상적인 표현**을 맞힌다.

```text
픽셀 생성 방식
-> 미래 화면의 모든 색과 세부 무늬까지 예측

V-JEPA 2
-> 학습된 표현 공간에서 중요한 상태와 움직임을 예측
```

논문은 이 방식이 물체의 이동처럼 예측 가능한 정보에 집중하고, 나뭇잎 하나의 정확한 위치처럼 예측하기 어렵고 계획에 불필요한 세부 정보는 무시할 수 있다고 설명한다.

단, 모델이 실제로 어떤 정보를 버리고 어떤 정보를 보존하는지는 학습 결과에 의해 결정된다. 표현 공간 예측이라는 설계만으로 필요한 정보가 항상 보존된다고 보장되는 것은 아니다.

## 1.4 모델 구조

Encoder와 predictor는 모두 Vision Transformer를 사용한다.

| 구성 | 논문의 설정 |
|---|---|
| 비디오 단위 | `2 × 16 × 16` 크기의 tubelet `(시간 × 높이 × 너비)` |
| Encoder | ViT-L, ViT-H, ViT-g |
| 최대 Encoder 크기 | 약 10억 파라미터인 ViT-g |
| 위치 정보 | 3D Rotary Position Embedding |
| 위치 축 | 시간, 높이, 너비 |
| Mask 방식 | 기존 V-JEPA의 multi-block masking |

기존 V-JEPA가 사용한 절대적인 sin-cos 위치 임베딩 대신 3D-RoPE를 사용한다. 특징 차원을 시간·높이·너비 세 부분으로 나누고 각 축에 회전을 적용한다. 저자들은 이 변경이 가장 큰 모델의 학습을 안정화했다고 보고한다.

## 1.5 데이터와 Scaling

### VideoMix22M

V-JEPA 2는 총 2,200만 개의 비디오·이미지 표본으로 구성된 `VideoMix22M`을 사용한다. 전체 비디오는 100만 시간을 넘는다.

| 데이터 | 표본 수 | 종류 | 총 시간 | 학습 sampling weight |
|---|---:|---|---:|---:|
| Something-Something v2 | 16.8만 | Ego video | 168시간 | 0.056 |
| Kinetics | 73.3만 | Exo video | 614시간 | 0.188 |
| HowTo100M | 110만 | Exo video | 13.4만 시간 | 0.318 |
| YT-Temporal-1B | 1,900만 | Exo video | 160만 시간 | 0.188 |
| ImageNet | 100만 | Image | 해당 없음 | 0.250 |

ImageNet 이미지는 같은 이미지를 시간축으로 반복하여 16-frame video처럼 처리한다. YT-Temporal-1B는 노이즈가 많기 때문에 Kinetics, SSv2, COIN, Epic-Kitchens의 분포를 기준으로 retrieval-based curation을 수행한다.

Figure 4의 결과는 다음을 보여준다.

- VM2M에서 VM22M으로 데이터를 늘리면 6개 분류 과제의 평균 정확도가 약 `+1.0`점 상승한다.
- YT1B를 그대로 사용하는 대신 선별하면 평균 정확도가 약 `+1.4`점 상승한다.
- 단순히 데이터 양만 늘리는 것뿐 아니라 데이터의 구성과 선별도 중요하다.

### 네 가지 Scaling 요소

논문은 기존 V-JEPA에서 다음 요소를 차례로 확장한다.

| 단계 | 변경 | 6개 과제 평균 정확도 |
|---|---|---:|
| 기존 V-JEPA 설정 | ViT-L, 약 200만 비디오 | 84.2 |
| Data scaling | VM22M 사용 | 85.2 |
| Model scaling | ViT-g, 약 10억 파라미터 | 86.7 |
| Longer training | 90K에서 252K iteration | 87.5 |
| Higher resolution | 최대 64 frame, 384 × 384 | 88.2 |

Figure 3 기준으로 각 변경을 합치면 평균 정확도가 총 4.0점 높아졌다. 따라서 최종 성능을 하나의 새로운 구조 때문이라고 보기보다 데이터, 모델 크기, 학습 길이, 입력 해상도를 함께 확장한 결과로 봐야 한다.

### Progressive Resolution Training

64-frame, 384 × 384 영상을 처음부터 끝까지 학습하면 계산량이 매우 커진다. 논문은 다음과 같이 입력 크기를 마지막에만 높인다.

```text
Warmup: 12K iteration
16 frame, 256 × 256

Constant phase: 228K iteration
16 frame, 256 × 256

Cooldown: 12K iteration
최대 64 frame, 384 × 384로 확장
learning rate를 선형 감소
```

Figure 5에서 이 방식은 처음부터 전체 해상도로 학습하는 경우보다 GPU 시간을 약 `8.4배` 줄였다. 또한 pretraining의 clip 길이를 16 frame에서 64 frame으로 늘리면, 평가에서는 여전히 16 frame만 사용해도 평균 성능이 약 0.7점 올랐다.

반면 128·256 frame까지 더 늘렸을 때는 이 논문의 분류 과제에서 64 frame보다 추가 향상이 관찰되지 않았다.

## 1.6 V-JEPA 2-AC: 행동을 조건으로 한 World Model

사전학습된 V-JEPA 2는 영상에서 빠진 부분을 예측하지만, 특정 행동을 했을 때 무엇이 발생하는지는 직접 예측하지 않는다. 이를 위해 논문은 `V-JEPA 2-AC(Action-Conditioned)`를 별도로 학습한다.

### 학습 데이터

| 항목 | 설정 |
|---|---|
| 데이터셋 | Droid |
| 사용량 | 62시간 미만, 약 2.3만 trajectory |
| 로봇 | 7-DoF Franka Emika Panda + 2-finger gripper |
| 영상 | 256 × 256, 4 FPS, 4초 |
| 입력 길이 | 16 frame |
| 상태 | 위치 3 + 회전 3 + gripper 1 = 7차원 |
| 행동 | 인접 frame 사이 end-effector 상태 변화량 7차원 |

논문에서 `unlabeled robot video`라고 표현하지만, **행동 정보가 전혀 없다는 뜻은 아니다.** 보상, 과제 이름, 성공·실패 라벨은 사용하지 않지만, 각 frame의 end-effector state를 사용하고 그 차이로 행동을 계산한다. 즉, V-JEPA 2의 첫 단계는 action-free이지만 V-JEPA 2-AC의 두 번째 단계는 action-conditioned 학습이다.

### 구조

각 frame을 고정된 V-JEPA 2 ViT-g encoder로 따로 인코딩하면 `16 × 16 × 1408` 크기의 feature map이 나온다. 새 predictor는 영상 표현, 현재 로봇 상태, 행동을 시간 순서로 함께 처리한다.

| 항목 | 설정 |
|---|---|
| Predictor 크기 | 약 3억 파라미터 |
| Transformer layer | 24 |
| Attention head | 16 |
| Hidden dimension | 1024 |
| Attention | Block-causal attention |
| Encoder | 사전학습된 V-JEPA 2, 학습 중 고정 |

Block-causal attention을 사용하므로 한 시점의 token은 같은 시점과 과거 시점의 영상·행동·로봇 상태만 볼 수 있다.

### Teacher-forcing loss

Equation 2에서는 실제 현재 frame의 표현을 입력으로 주고 다음 frame의 표현을 예측한다.

```text
현재 표현 z_k + 상태 s_k + 행동 a_k
-> predictor
-> 다음 표현 예측 z_hat_(k+1)

L_teacher = 평균 L1(z_hat_(k+1), z_(k+1))
```

### Rollout loss

Teacher forcing만 사용하면 학습 중에는 항상 실제 과거 상태를 보지만, 추론할 때는 자신의 예측을 다시 입력으로 사용해야 한다. 이 차이 때문에 여러 step을 예측할수록 오차가 쌓일 수 있다.

Equation 3에서는 predictor의 출력을 다음 입력으로 다시 넣어 2-step rollout을 학습한다.

```text
L_rollout
= L1(여러 행동을 적용해 예측한 미래 표현, 실제 미래 표현)
```

최종 목적함수는 Equation 4와 같다.

```text
L_total = L_teacher + L_rollout
```

## 1.7 목표 이미지로 행동 계획하기

V-JEPA 2-AC는 행동을 바로 출력하는 policy가 아니다. 여러 행동 후보의 결과를 모델 안에서 예측한 뒤, 목표에 가장 가까운 행동을 찾는다.

Equation 5의 energy는 다음과 같다.

```text
현재 frame 표현: z_k
목표 이미지 표현: z_g
행동 후보: a_(1:T)

예측 미래 표현 = P(a_(1:T), 현재 상태, z_k)
energy = L1(예측 미래 표현, z_g)
```

Energy가 작을수록 행동 후의 예상 상태가 목표 이미지에 가깝다는 뜻이다. 실제 계획 과정은 다음과 같다.

```text
1. 여러 행동 sequence를 Gaussian distribution에서 sampling함.
2. 각 행동 sequence의 미래 표현을 V-JEPA 2-AC로 예측함.
3. 목표 표현과의 L1 distance가 작은 top-k sequence를 고름.
4. top-k의 평균과 분산으로 sampling distribution을 갱신함.
5. 위 과정을 반복한 뒤 최종 행동 sequence를 선택함.
6. 첫 행동 하나만 실제 로봇에서 실행함.
7. 새 영상을 관찰한 뒤 다시 계획함.
```

행동 후보를 개선하는 방법은 Cross-Entropy Method이며, 첫 행동만 실행하고 다시 계획하는 방식은 Model Predictive Control이다.

## 1.8 영상 이해 실험

Encoder를 고정하고 4-layer attentive probe만 과제별로 학습한다. SSv2, Diving-48, Jester는 시간에 따른 움직임이 중요한 과제이고 Kinetics-400, COIN, ImageNet은 외형 정보의 영향이 큰 과제이다.

Table 4의 주요 결과는 다음과 같다.

| 모델 | 평균 | SSv2 | Diving-48 | Jester | K400 | COIN | ImageNet |
|---|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 | 81.1 | 50.7 | 82.5 | 93.4 | 83.6 | 90.7 | 86.1 |
| InternVideo2s2-1B | 87.0 | 69.7 | 86.4 | 97.0 | 89.4 | 93.8 | 85.8 |
| 기존 V-JEPA ViT-H | 85.2 | 74.3 | 87.9 | 97.7 | 84.5 | 87.1 | 80.0 |
| V-JEPA 2 ViT-g | 87.5 | 75.3 | 90.1 | 97.7 | 86.6 | 90.7 | 84.6 |
| V-JEPA 2 ViT-g384 | **88.2** | **77.3** | **90.2** | **97.8** | 87.3 | 91.1 | 85.1 |

V-JEPA 2는 특히 motion understanding 과제에서 강했다. 반면 K400, COIN, ImageNet과 같은 appearance 과제에서는 모든 개별 지표에서 최고였던 것은 아니다.

또한 비교 모델들은 서로 다른 데이터와 방법으로 사전학습되었다. 논문도 이 결과를 완전히 동일한 사전학습 조건의 비교가 아니라, 같은 probe 평가 방식을 적용한 **system-level 비교**라고 설명한다.

## 1.9 미래 행동 예측 실험

Epic-Kitchens-100의 Action Anticipation은 현재까지의 주방 영상을 보고 1초 뒤 시작될 행동을 예측하는 과제이다. 여러 정답이 가능할 수 있어 mean-class recall-at-5를 사용한다.

V-JEPA 2는 현재 영상의 encoder 표현과 1초 뒤 frame에 해당하는 mask token을 predictor에 넣어 미래 표현을 예측한다. 이 표현을 attentive probe가 동사, 명사, 결합 행동으로 분류한다.

Table 5의 결과는 다음과 같다.

| 모델 | 파라미터 | Verb | Noun | Action recall-at-5 |
|---|---:|---:|---:|---:|
| PlausiVL | 8B | 55.6 | 54.2 | 27.6 |
| V-JEPA 2 ViT-L | 300M | 57.8 | 53.8 | 32.7 |
| V-JEPA 2 ViT-H | 600M | 59.2 | 54.6 | 36.5 |
| V-JEPA 2 ViT-g | 1B | 61.2 | 55.7 | 38.0 |
| V-JEPA 2 ViT-g384 | 1B | **63.6** | **57.1** | **39.7** |

V-JEPA 2 ViT-g384는 이전 최고 결과인 PlausiVL보다 action recall-at-5가 12.1점 높았으며, 상대적으로는 약 44% 향상되었다.

다만 이 실험은 주방이라는 제한된 환경과 고정된 행동 vocabulary를 사용한다. 논문은 예측 시간을 1초보다 길게 늘리면 성능이 감소한다고 밝힌다.

## 1.10 Video Question Answering

V-JEPA 2 자체에는 언어 모델이 없다. Video QA를 위해 encoder의 visual token을 projector로 LLM의 입력 공간에 맞춘 뒤, LLaVA 방식의 visual instruction tuning을 수행한다.

```text
비디오
-> V-JEPA 2 encoder
-> visual tokens
-> MLP projector
-> LLM
-> 자연어 답변
```

### 고정된 Encoder 비교

Table 6에서는 Qwen2-7B-Instruct, 1,800만 개 alignment sample, 고정된 vision encoder라는 조건을 동일하게 맞춘다.

| Vision encoder | 평균 | PerceptionTest | MVP | TempCompass | TemporalBench | TVBench | TOMATO | MVBench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 | 45.7 | 67.1 | 22.4 | 62.3 | 26.8 | 47.6 | 32.0 | 61.8 |
| SigLIP2 | 48.1 | 72.4 | 26.2 | 66.8 | 25.7 | 48.7 | 33.2 | 64.0 |
| Perception Encoder | 49.1 | 72.3 | 26.7 | 67.0 | 27.5 | 51.6 | 34.0 | 64.7 |
| V-JEPA 2 ViT-g512 | **52.3** | 72.0 | **31.1** | **69.2** | **33.3** | **55.9** | **37.0** | **67.7** |

언어 supervision 없이 사전학습한 V-JEPA 2가 PerceptionTest를 제외한 항목에서 비교한 image encoder보다 높았다. 특히 움직임과 시간 정보가 중요한 benchmark에서 차이가 컸다.

### 최종 8B 모델

최종 모델은 8,850만 개의 image/video-text alignment sample과 Llama 3.1 8B를 사용한다. Table 8의 주요 결과는 다음과 같다.

| 모델 | 평균 | PerceptionTest | MVP | TempCompass | TemporalBench | TOMATO | TVBench | MVBench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PerceptionLM 8B | 56.7 | 82.7 | 39.7 | 72.7 | 28.3 | 33.2 | **63.5** | **77.1** |
| V-JEPA 2 + Llama 3.1 8B | **59.5** | **84.0** | **44.5** | **76.9** | **36.7** | **40.3** | 60.6 | 73.5 |

V-JEPA 2 기반 모델은 5개 benchmark에서 당시 8B급 최고 결과를 보고했지만 TVBench와 MVBench에서는 PerceptionLM보다 낮았다. 또한 이 결과는 V-JEPA 2 encoder만의 zero-shot 성능이 아니라, 대규모 alignment data와 LLM을 이용해 별도로 학습한 MLLM의 성능이다.

## 1.11 로봇 계획 실험

V-JEPA 2-AC는 Droid에 없는 두 연구실의 Franka arm에 동일한 가중치와 코드로 적용된다. 카메라는 보정하지 않은 저해상도 monocular RGB camera를 사용한다.

### Single-goal reaching

Figure 8에서 x, y, z 방향의 목표 위치로 이동하는 세 실험 모두 end-effector가 4cm 이내까지 접근했고, step이 진행될수록 위치 오차가 감소했다. Figure 9의 energy landscape는 정답 행동 근처에서 낮은 값을 가지며 비교적 부드러운 형태를 보였다.

### 물체 조작

Table 2의 두 연구실 평균 성공률은 다음과 같다. 각 수치는 조건별 10회 실행 결과이다.

| 모델 | Reach | Grasp Cup | Grasp Box | Reach w/ Cup | Reach w/ Box | Pick & Place Cup | Pick & Place Box |
|---|---:|---:|---:|---:|---:|---:|---:|
| Octo | 100% | 15% | 0% | 15% | 70% | 15% | 10% |
| V-JEPA 2-AC | **100%** | **65%** | **25%** | **75%** | **75%** | **80%** | **65%** |

Pick-and-place에서는 한 번에 최종 목표까지 계획하지 않는다. 물체를 잡은 이미지, 목표 위치 근처로 옮긴 이미지, 최종 배치 이미지의 세 sub-goal을 순서대로 제공한다.

Table 3은 Lab 2에서 latent diffusion 기반 Cosmos와 계획 성능 및 시간을 비교한다.

| 모델 | Action sample 수 | 행동 하나 계획 시간 | Reach | Grasp Cup | Grasp Box | Pick & Place Cup | Pick & Place Box |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cosmos | 80 | 약 4분 | 80% | 0% | 20% | 0% | 0% |
| V-JEPA 2-AC | 800 | 약 16초 | **100%** | **60%** | 20% | **80%** | **50%** |

V-JEPA 2-AC는 10배 많은 행동 후보를 평가하면서도 픽셀 영상을 생성하는 Cosmos보다 행동 하나를 훨씬 빠르게 계획했다. 이는 표현 공간 예측의 계산상 장점을 보여준다. 다만 16초 역시 실시간 제어 속도는 아니다.

## 1.12 한계와 비판적으로 볼 부분

### 카메라 위치에 민감함

V-JEPA 2-AC는 카메라 보정값 없이 영상만 보고 로봇 좌표축을 추론해야 한다. 논문은 여러 카메라 위치를 수동으로 시도한 뒤 잘 작동하는 위치를 선택했다고 밝힌다. 따라서 모든 임의의 카메라 배치에 바로 일반화했다고 볼 수 없다.

### 장기 계획이 어려움

Autoregressive rollout에서는 예측 오차가 누적되고, horizon이 길어질수록 탐색해야 할 행동 sequence가 기하급수적으로 늘어난다. 논문도 약 16초 이내의 예측에 집중하며, pick-and-place에서는 사람이 정한 중간 목표 이미지를 사용한다.

### 목표를 이미지로 제공해야 함

현재 energy function은 목표 이미지를 필요로 한다. 자연어로 “컵을 상자 안에 넣어라”라고 지시하는 방식은 이 논문에서 구현되지 않았다.

### Robot 평가 규모

각 조작 조건의 평가는 10회이며, 두 연구실과 같은 Franka 계열 로봇에서 수행되었다. 새로운 연구실과 물체에 대한 zero-shot 결과는 의미가 있지만, 서로 다른 로봇 형태나 광범위한 실제 환경까지 검증한 것은 아니다.

### 성능 향상의 원인

최종 V-JEPA 2에는 더 많은 데이터, 데이터 선별, 큰 모델, 긴 학습, 높은 시공간 해상도가 모두 들어간다. 따라서 V-JEPA 2의 전체 향상을 JEPA objective 하나의 효과라고 해석하면 안 된다.

## 1.13 논문의 의미

V-JEPA 2 논문의 핵심은 하나의 비디오 encoder가 분류와 Video QA에서 높은 성능을 냈다는 것만이 아니다. **행동 없이 관찰한 대규모 비디오에서 표현을 먼저 학습하고, 적은 로봇 상호작용 데이터로 행동 조건부 predictor를 추가하여 실제 계획에 사용했다**는 데 의미가 있다.

또한 미래 픽셀 전체를 생성하지 않아도, 목표와 미래 상태의 표현 거리를 이용해 물리적 행동을 선택할 수 있음을 실험으로 보였다.

---

# 2. Genie

## 2.1 연구 배경

기존 비디오 생성 모델은 시작 이미지나 텍스트를 입력받아 영상을 만들 수 있지만, 사용자가 매 frame마다 캐릭터를 직접 조작하기는 어렵다. 반대로 기존 World Model은 행동을 조건으로 다음 상태를 예측할 수 있지만, 학습에 `영상 + 실제 행동 라벨`이 필요하다.

인터넷 게임 영상에는 화면은 많지만, 플레이어가 어떤 버튼을 눌렀는지는 대부분 기록되어 있지 않다. Genie의 연구 질문은 다음과 같다.

> 행동 라벨이 없는 인터넷 비디오만으로, 사용자가 frame 단위로 조작할 수 있는 생성 환경을 학습할 수 있는가?

논문은 Genie를 `Generative Interactive Environment`라고 부른다. 하나의 이미지에서 시작해 사용자가 잠재 행동을 선택할 때마다 다음 frame을 생성하므로, 미리 완성된 동영상을 보는 것이 아니라 생성 모델과 상호작용하게 된다.

## 2.2 논문이 구분한 모델 종류

Table 1의 구분은 다음과 같다.

| 모델 종류 | 학습 데이터 | 제어 단위 |
|---|---|---|
| 기존 World Model | Video + Action | Frame-level |
| Video Model | Video + Text | Video-level |
| Genie | Video only | Frame-level |

Genie의 핵심은 video-only 학습과 frame-level control을 동시에 달성하려는 것이다.

## 2.3 전체 구조

Figure 3의 Genie는 세 구성 요소로 이루어진다.

```text
입력 비디오
├─ Video Tokenizer -> 이산 video token z
└─ Latent Action Model -> frame 사이의 잠재 행동 a

video token z + 잠재 행동 a
-> Dynamics Model
-> 다음 frame의 token 예측
-> Tokenizer decoder로 영상 복원
```

| 구성 요소 | 역할 |
|---|---|
| Video Tokenizer | 픽셀 영상을 작은 이산 token으로 압축하고 다시 복원함 |
| Latent Action Model | 연속된 frame 사이에서 일어난 변화를 이산 행동으로 표현함 |
| Dynamics Model | 이전 video token과 행동을 보고 다음 frame token을 생성함 |

학습은 두 단계로 진행된다.

1. Video Tokenizer를 먼저 학습한다.
2. 고정된 tokenizer를 사용하여 Latent Action Model과 Dynamics Model을 함께 학습한다.

## 2.4 ST-Transformer

일반 Transformer가 영상의 모든 token끼리 attention을 계산하면 token 수에 대해 비용이 제곱으로 커진다. Genie는 공간 attention과 시간 attention을 나눈 Spatiotemporal Transformer를 세 구성 요소 모두에 사용한다.

```text
Spatial attention
-> 한 frame 안의 H × W token끼리 attention

Temporal attention
-> 같은 공간 위치의 token이 T개 frame을 따라 attention
```

Temporal attention에는 causal mask를 적용하여 미래 frame을 미리 보지 못하게 한다. 계산량의 큰 부분인 spatial attention이 frame 수에 대해 선형으로 증가하므로, 모든 시공간 token을 한꺼번에 처리하는 방식보다 긴 영상을 효율적으로 다룰 수 있다.

각 ST block은 spatial attention, temporal attention, feed-forward layer로 구성된다. 논문은 spatial layer 뒤의 별도 feed-forward layer를 생략하고 두 attention 뒤에 하나만 사용하여 계산량을 다른 부분에 배분한다.

## 2.5 Latent Action Model

Latent Action Model은 행동 라벨 대신 두 frame 사이의 중요한 변화를 나타내는 행동 코드를 학습한다.

```text
과거 frame x_(1:t) + 다음 frame x_(t+1)
-> LAM encoder
-> 연속 latent action
-> VQ codebook에서 가장 가까운 이산 action 선택

과거 frame + 이산 latent action
-> LAM decoder
-> 다음 frame 복원
```

LAM decoder가 다음 frame을 복원하려면 latent action이 과거와 미래 사이의 중요한 차이를 담아야 한다. VQ-VAE 방식으로 행동을 작은 codebook에 제한하며, 논문의 주요 실험에서는 가능한 행동을 `8개`로 설정한다.

행동 수를 작게 제한한 이유는 다음과 같다.

- 사용자가 controller 버튼처럼 선택할 수 있어야 함.
- 모델이 불필요하게 세분화된 행동 코드를 만드는 것을 막음.
- 각 행동이 생성 결과에 실제 영향을 주도록 유도함.

LAM의 encoder와 decoder는 학습 신호를 만들기 위해 사용된다. 추론할 때는 VQ codebook만 남기고 나머지 LAM은 버리며, 사용자가 직접 0부터 7 사이의 action code를 선택한다.

처음에는 각 번호가 무엇을 하는지 알 수 없지만, 논문은 서로 다른 시작 이미지에서도 같은 code의 의미가 비교적 일관되었다고 보고한다. 새로운 게임의 controller 버튼을 직접 눌러 의미를 파악하는 것과 비슷하다.

## 2.6 Video Tokenizer

픽셀 영상을 직접 예측하면 차원이 너무 크기 때문에 VQ-VAE 기반 tokenizer로 이산 token으로 압축한다.

```text
T개의 RGB frame
-> ST-ViViT encoder
-> 이산 video token
-> ST-ViViT decoder
-> frame 복원
```

공간 정보만 압축하는 tokenizer와 달리, Genie의 `ST-ViViT`는 시간 attention도 사용한다. 따라서 시점 `t`의 token은 현재 frame만이 아니라 이전 frame의 정보도 포함한다.

논문의 tokenizer 설정은 다음과 같다.

| 항목 | 설정 |
|---|---|
| 파라미터 | 약 2억 |
| Patch size | 4 |
| Codebook | 1,024개 code |
| Code embedding | 32차원 |

## 2.7 Dynamics Model

Dynamics Model은 decoder-only MaskGIT Transformer이다. 이전 frame의 token과 잠재 행동을 입력받아 다음 frame의 token을 예측한다.

```text
입력: z_(1:t-1), latent action a_(1:t-1)
출력: 다음 frame token z_hat_t

loss = Cross-Entropy(z_hat_(2:T), 실제 z_(2:T))
```

학습할 때 `z_(2:T-1)`의 일부를 무작위로 mask한다. Mask 비율은 0.5와 1 사이에서 sampling한다. Latent action은 Dynamics Model을 통해 gradient가 LAM으로 역전파되지 않도록 stop-gradient를 적용한다.

행동을 별도 token으로 이어 붙이는 대신 frame token에 additive embedding으로 더했을 때 생성 결과의 controllability가 높아졌다고 보고한다.

## 2.8 추론: 생성 환경을 플레이하는 과정

Figure 8의 추론 과정은 다음과 같다.

```text
1. 사용자가 시작 이미지 한 장을 제공함.
2. Tokenizer가 이미지를 video token으로 변환함.
3. 사용자가 0~7 사이의 latent action 하나를 선택함.
4. Dynamics Model이 다음 frame token을 생성함.
5. Tokenizer decoder가 token을 픽셀 frame으로 복원함.
6. 생성된 frame과 다음 action으로 과정을 반복함.
```

학습 영상의 action을 LAM으로 추론해 넣으면 원본과 비슷한 trajectory를 재생할 수 있고, action을 바꾸면 같은 시작점에서 새로운 trajectory를 생성할 수 있다.

## 2.9 데이터와 학습 설정

### Platformers 데이터

공개 인터넷 영상에서 2D platform game 관련 영상을 수집했다.

| 항목 | 내용 |
|---|---|
| 초기 수집 | 5,500만 개의 16초 clip, 20만 시간 이상 |
| 최종 filtering 후 | 680만 개의 16초 clip, 약 3만 시간 |
| Frame rate | 10 FPS |
| 해상도 | 160 × 90 |
| Sequence length | 16 frame |

논문은 Platformers 외에도 RT-1 데이터와 기존 robot dataset을 행동 라벨 없이 영상으로만 합친 `Robotics` 데이터로 별도의 모델을 학습한다.

### 최종 모델 크기

| 구성 요소 | 파라미터 |
|---|---:|
| Video Tokenizer | 약 0.2B |
| Latent Action Model | 약 0.3B |
| Dynamics Model | 10.1B |
| 합계 | 약 10.7B, 논문에서는 약 11B로 표현 |

최종 Dynamics Model은 batch size 512로 125K step 학습했으며, 총 942B token과 256개의 TPU v5p를 사용했다.

## 2.10 평가 지표

논문은 영상 품질과 행동 제어 가능성을 따로 평가한다.

### FVD

Frechet Video Distance는 실제 비디오와 생성 비디오의 feature distribution 차이를 측정한다. 낮을수록 실제 영상과 비슷한 품질을 의미한다.

### Delta-t PSNR

논문은 행동이 결과에 실제 영향을 주는지 측정하기 위해 다음 지표를 제안한다.

```text
Delta-t PSNR
= PSNR(실제 frame, 실제 영상에서 추론한 action으로 생성한 frame)
- PSNR(실제 frame, 무작위 action으로 생성한 frame)
```

값이 클수록 올바른 action을 넣었을 때 무작위 action보다 실제 미래를 더 잘 재현한다. 즉, 생성 품질이 아니라 **행동에 따른 변화가 구분되는 정도**를 측정한다. 논문은 `t = 4`에서 이 값을 보고한다.

## 2.11 실험 결과

### Scaling

Figure 9에서는 Dynamics Model을 약 40M에서 2.7B 파라미터까지 늘릴수록 training loss가 일관되게 감소했다. 2.3B 모델에서도 batch size를 128, 256, 448로 늘릴수록 loss가 감소했다.

이를 근거로 최종 10.1B Dynamics Model과 batch size 512를 선택했다. 다만 Figure 9는 주로 training loss의 scaling을 보여주므로, 모델 크기 증가가 모든 실제 상호작용 지표를 같은 비율로 높였다고 바로 결론 내릴 수는 없다.

### 다양한 이미지 Prompt

Figure 10에서는 다음과 같은 학습 분포 밖의 한 장짜리 이미지를 prompt로 사용한다.

- Text-to-image model이 만든 이미지
- 손으로 그린 sketch
- 실제 사진

같은 latent action을 반복했을 때 캐릭터나 대상이 움직이는 game-like trajectory가 생성되었다. Figure 12에서는 foreground와 background가 서로 다른 속도로 움직이는 parallax도 나타났다.

이 결과는 시작 이미지 형식의 다양성에 대한 qualitative evidence이다. 논문은 prompt별 정량 성공률이나 사람 평가를 보고하지 않으므로, 모든 임의 이미지가 안정적인 게임으로 변환된다는 의미는 아니다.

### Robotics 모델

Robotics 데이터로 학습한 2.5B 모델은 test split에서 FVD `82.7`을 기록했다. Figure 13에서 같은 action code가 서로 다른 시작 frame에서도 `위`, `아래`, `왼쪽`과 같이 비교적 일관된 로봇 팔 움직임을 만들었다. Figure 11에서는 로봇이 물체를 누를 때 포장지가 변형되는 모습도 생성했다.

그러나 이 실험은 생성된 영상의 controllability를 보여주는 것이며, 생성 결과로 실제 로봇을 제어한 실험은 아니다.

## 2.12 잠재 행동으로 Agent 학습하기

논문은 Genie의 LAM이 발견한 행동이 새로운 환경의 imitation learning에도 사용 가능한지 CoinRun에서 평가한다.

```text
1. 목표 환경의 expert video를 frozen LAM에 넣음.
2. 각 frame 사이에 latent action label을 자동으로 붙임.
3. 관찰을 보고 latent action을 예측하는 policy를 학습함.
4. 소량의 실제 action data로 latent action을 환경의 실제 action에 연결함.
```

Figure 15에서 LAM 기반 policy는 latent action과 실제 action의 mapping에 expert sample을 약 200개 사용했을 때 oracle behavioral cloning과 비슷한 level 해결률에 도달했다. Oracle은 처음부터 expert의 실제 action label로 학습한 모델이다.

이 실험은 latent action이 단순한 영상 압축 code를 넘어 행동의 의미를 어느 정도 일관되게 담았다는 근거이다. 다만 완전한 zero-shot agent 학습은 아니며, latent action을 실제 환경 action으로 변환하기 위한 소량의 labeled sample이 필요하다.

## 2.13 Ablation Study

### Latent Action Model의 입력

Table 2는 LAM이 원본 pixel을 볼 때와 tokenizer의 token을 볼 때를 비교한다.

| LAM 입력 | 데이터 | FVD ↓ | Delta-t PSNR ↑ |
|---|---|---:|---:|
| Token | Platformers | **38.8** | 1.33 |
| Pixel | Platformers | 40.1 | **1.91** |
| Token | Robotics | 257.8 | 1.65 |
| Pixel | Robotics | **136.4** | **2.07** |

Platformers에서는 token 입력이 FVD만 약간 좋았지만, 두 데이터 모두 pixel 입력의 controllability가 더 높았다. 저자들은 tokenization 과정에서 움직임 정보 일부가 사라질 수 있다고 해석하고 최종 LAM에는 pixel을 사용한다.

### Tokenizer 구조

Table 3은 tokenizer의 attention 구조를 비교한다.

| Tokenizer | 파라미터 | Memory | FVD ↓ | Delta-t PSNR ↑ |
|---|---:|---:|---:|---:|
| Spatial-only ViT | 230M | **0.3GB** | 114.5 | 1.39 |
| C-ViViT | 225M | 1.6GB | 272.7 | 1.37 |
| ST-ViViT | 205M | 0.9GB | **81.4** | **1.66** |

ST-ViViT는 spatial-only ViT보다 memory는 더 사용하지만 영상 품질과 controllability가 모두 좋았다. Full space-time attention을 사용하는 C-ViViT는 memory 사용량이 가장 많으면서 성능은 낮았다.

## 2.14 한계와 비판적으로 볼 부분

### 짧은 Memory

Genie가 한 번에 기억하는 길이는 16 frame이다. 10 FPS 기준으로 약 1.6초이므로, 상호작용이 길어지면 환경 구조나 물체 상태가 일관되게 유지되기 어렵다.

### 느린 생성

논문의 Genie는 약 1 FPS로 작동한다. 일반적인 게임처럼 즉각 반응하는 속도와는 차이가 있다.

### Autoregressive 오류와 Hallucination

생성한 frame을 다시 입력으로 사용하므로 오류가 누적될 수 있고, 논문도 비현실적인 미래를 hallucinate할 수 있다고 밝힌다.

### 잠재 행동의 의미

행동 code는 라벨 없이 학습되므로 `action 0 = 왼쪽 이동`처럼 의미가 미리 정해져 있지 않다. 여러 prompt에서 의미가 일관되는 경향을 보였지만, 모든 환경에서 동일한 물리적 의미가 보장되는 것은 아니다.

### 평가의 많은 부분이 정성적임

Sketch, 실제 사진, text-to-image prompt의 interactive generation 결과는 주로 선택된 사례를 시각적으로 보여준다. 범용적인 prompt 성공률이나 장기적인 환경 일관성을 정량적으로 측정하지 않았다.

### 재현성

저자들은 학습 데이터, 최종 model checkpoint, 데이터 예시를 공개하지 않았다고 명시한다. Appendix F에 단일 중급 TPU나 GPU에서 실행할 수 있는 소규모 예제를 설명하지만, 11B 모델의 주요 결과를 그대로 재현하기는 어렵다.

## 2.15 논문의 의미

Genie의 핵심은 단순히 게임 화면을 생성한 것이 아니라, **행동 라벨이 없는 비디오에서 작은 이산 행동 공간을 발견하고 그 행동으로 다음 frame 생성을 제어했다**는 점이다.

이 방식은 인터넷 비디오처럼 action annotation이 없는 대규모 데이터를 interactive model 학습에 사용할 가능성을 보여준다. 동시에 현재 결과는 2D platformer 중심이고, 짧은 memory와 느린 생성 속도를 가지므로 일반적인 물리 세계 simulator가 완성되었다고 보기는 어렵다.

---

# 3. V-JEPA 2와 Genie 비교

| 구분 | V-JEPA 2 | Genie |
|---|---|---|
| 핵심 목표 | 영상 이해·미래 예측·실제 로봇 계획 | 비디오만으로 조작 가능한 생성 환경 학습 |
| 주요 데이터 | 22M 비디오·이미지 + 62시간 미만 Droid | 약 3만 시간의 2D platformer video |
| 사전학습 행동 라벨 | 사용하지 않음 | 사용하지 않음 |
| 행동 학습 | Droid의 상태 차이로 실제 7D 행동 구성 | 연속 frame에서 8개의 latent action을 비지도 학습 |
| 영상 표현 | 연속적인 ViT embedding | VQ-VAE의 이산 video token |
| 예측 대상 | 미래 frame의 표현 | 다음 frame의 이산 token |
| 픽셀 생성 | 하지 않음 | Tokenizer decoder로 생성 |
| 제어 입력 | 연속 7D robot action | 사용자가 선택하는 이산 latent action |
| 행동 선택 | CEM과 MPC로 목표 energy 최소화 | 사용자가 controller처럼 action code 선택 |
| 대표 검증 | 실제 Franka robot의 zero-shot 조작 | Platformer 생성, Robotics video 생성, CoinRun imitation |
| 계산상 특징 | 픽셀을 만들지 않아 후보 행동 평가가 비교적 빠름 | 매 frame을 autoregressive하게 생성하므로 느림 |
| 대표 한계 | 카메라 민감도, image goal, 장기 계획 | 16-frame memory, 약 1 FPS, hallucination |

## 3.1 같은 점

- 두 모델 모두 비디오를 통해 시간에 따른 상태 변화를 학습한다.
- 대규모 관찰 데이터에는 행동 라벨이 없어도 학습을 시작할 수 있다.
- Transformer를 이용해 과거 문맥에서 미래 상태를 예측한다.
- 단순한 영상 분류를 넘어 행동 또는 상호작용과 연결하려 한다.

## 3.2 가장 중요한 차이

```text
V-JEPA 2
미래가 어떤 모습인지 픽셀로 그리지 않고,
미래 상태의 표현이 목표 표현에 얼마나 가까운지 계산함.

Genie
사용자가 선택한 잠재 행동에 따라
다음 영상 frame 자체를 생성함.
```

V-JEPA 2는 목표 달성을 위한 행동을 자동으로 탐색한다. Genie의 주요 interactive generation에서는 행동 후보의 의미를 사용자가 익힌 뒤 직접 선택한다.

또한 V-JEPA 2-AC는 실제 로봇의 상태·행동 신호를 사용하지만, Genie는 영상 변화로부터 latent action을 발견한다. 따라서 둘 다 `video learning`을 사용한다는 이유만으로 행동 supervision의 조건까지 같다고 보면 안 된다.

---

# 4. 논문을 읽을 때 주의할 점

## 4.1 `Unlabeled`의 의미가 서로 다름

- V-JEPA 2 사전학습: action, reward, task label을 사용하지 않음.
- V-JEPA 2-AC: task와 성공 여부는 사용하지 않지만 robot state와 그 차이로 만든 action은 사용함.
- Genie: Platformers 학습에서 실제 action label 없이 latent action을 발견함.

따라서 V-JEPA 2-AC까지 완전히 action-free로 학습했다고 설명하면 틀리다.

## 4.2 World Model의 출력이 다름

Genie는 사람이 볼 수 있는 미래 frame을 생성한다. V-JEPA 2는 미래의 latent representation만 예측하므로 예측 결과를 화면으로 직접 확인할 수 없다. 대신 행동 후보를 많이 평가하는 데 유리하다.

## 4.3 V-JEPA 2의 `Zero-shot`

Zero-shot은 배포한 두 연구실에서 추가 학습 데이터를 수집하지 않았다는 의미이다. 모델은 그 전에 같은 종류의 Franka robot이 포함된 Droid 데이터로 V-JEPA 2-AC 후속학습을 받았다.

## 4.4 Genie의 `Foundation World Model`

저자들은 11B 규모, 다양한 prompt, 비디오만을 이용한 학습을 근거로 Genie를 foundation world model이라고 표현한다. 그러나 주요 모델의 데이터는 2D platform game에 집중되어 있고, 범용 물리 환경에서 실제 행동 계획까지 검증한 것은 아니다.

## 4.5 실험 수치의 직접 비교

두 논문은 데이터, 출력, 평가 과제가 완전히 다르다. V-JEPA 2의 robot success rate와 Genie의 FVD를 비교해 어느 모델이 더 좋은 World Model인지 결정할 수 없다.

## 4.6 정성적 결과와 정량적 결과

V-JEPA 2는 분류, anticipation, Video QA, robot success rate를 정량 평가한다. Genie의 핵심 주장인 다양한 prompt의 playability는 시각적 사례가 큰 비중을 차지한다. 논문의 그림은 가능성을 보여주지만 전체 입력에서의 평균 성공률을 의미하지 않는다.

---

# 5. 최종 정리

V-JEPA 2는 가려진 비디오의 픽셀이 아니라 teacher encoder의 표현을 예측하도록 학습한다. 2,200만 개의 비디오·이미지, 최대 10억 파라미터, 긴 학습과 높은 시공간 해상도로 확장했을 때 영상의 움직임 이해와 미래 행동 예측 성능이 향상되었다. 이후 encoder를 고정하고 62시간 미만의 Droid 데이터로 행동 조건부 predictor를 학습한 V-JEPA 2-AC는, 목표 이미지와 예측 미래 표현의 거리를 줄이는 행동을 CEM과 MPC로 탐색하여 두 새로운 연구실의 Franka robot에서 reach, grasp, pick-and-place를 수행했다.

Genie는 행동 라벨이 없는 인터넷 비디오에서 8개의 이산 latent action을 학습하고, 이전 frame token과 latent action을 조건으로 다음 frame을 생성한다. Video Tokenizer, Latent Action Model, Dynamics Model 모두 효율적인 ST-Transformer를 사용하며, 약 11B 규모로 확장했다. Text-to-image 결과, sketch, 실제 사진을 시작점으로 game-like trajectory를 만들고, Robotics 영상과 CoinRun imitation에도 latent action을 적용했다. 다만 16-frame memory, 약 1 FPS의 생성 속도, 정성 평가의 비중, 비공개 checkpoint와 데이터라는 한계가 있다.

> **V-JEPA 2는 표현 공간에서 미래를 예측해 목표 달성 행동을 찾고, Genie는 영상에서 잠재 행동을 발견해 그 행동에 따른 다음 화면을 생성한다.**

---

# 6. 논문 속 근거 위치

## V-JEPA 2

- **Figure 1:** 전체 학습과 활용 구조
- **Figure 2:** V-JEPA 2 사전학습과 V-JEPA 2-AC 후속학습
- **Equation 1:** Masked video의 표현 예측 목적함수
- **Figure 3:** Data, model, 학습 길이, resolution scaling의 누적 효과
- **Table 1:** VideoMix22M의 구성과 sampling weight
- **Figure 4:** 데이터 규모와 YT1B curation 효과
- **Figure 5:** 모델 크기와 progressive resolution training
- **Equations 2~4:** Teacher-forcing loss, rollout loss, 전체 V-JEPA 2-AC loss
- **Figure 6:** V-JEPA 2-AC 학습 구조
- **Equation 5 / Figure 7:** Goal-conditioned energy와 MPC planning
- **Figure 8:** Single-goal reaching 결과
- **Figure 9:** 행동에 따른 energy landscape
- **Tables 2~3:** 실제 로봇 조작 성공률과 Cosmos 비교
- **Table 4:** Motion·appearance classification
- **Table 5:** EK100 action anticipation
- **Tables 6~8:** Video QA encoder 비교, scaling, 최종 8B 결과
- **Section 4.3:** 로봇 계획의 한계
- **Section 9:** 결론과 향후 연구

## Genie

- **Figure 1:** 다양한 prompt에서 interactive environment를 생성하는 개념
- **Table 1:** World model, video model, Genie의 학습 데이터와 제어 수준 비교
- **Figure 3:** 세 구성 요소와 전체 학습 구조
- **Figure 4:** ST-Transformer 구조
- **Figure 5:** Latent Action Model
- **Figure 6:** Video Tokenizer
- **Figure 7:** Dynamics Model
- **Figure 8:** 한 장의 prompt와 latent action을 이용한 autoregressive inference
- **Figure 9:** Model size와 batch size scaling
- **Figures 10~13:** OOD prompt, parallax, Robotics 생성 결과
- **Figures 14~15:** CoinRun trajectory와 behavioral cloning 결과
- **Table 2:** LAM의 pixel 입력과 token 입력 비교
- **Table 3:** Video tokenizer 구조 비교
- **Section 5:** 결론과 모델의 한계

