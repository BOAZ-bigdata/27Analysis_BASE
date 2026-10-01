[🔗 Paper - V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)

# Introduction

V-JEPA 2는 **영상의 다음 픽셀 자체를 생성하는 것이 아니라, 영상 안에서 중요한 정보를 latent representation 공간에서 예측하도록 학습하는 self-supervised world model**임.

기존 video generation model처럼 미래 frame을 RGB pixel 단위로 완벽하게 복원하는 것이 목적이 아님.

대신

**현재까지 관찰한 영상 → 가려진/미래 영역의 representation 예측**

방식으로 물체, 움직임, 시간적 변화와 같은 high-level 정보를 학습함.

Meta는 V-JEPA 2를 100만 시간 이상의 video/image data로 사전학습하고, 이후 소량의 robot interaction data를 추가하여 실제 robot planning까지 연결함.

논문의 큰 흐름은 아래 두 단계로 볼 수 있음.

```text
1. V-JEPA 2
Internet Video
→ Self-Supervised Representation Learning
→ 영상 이해 / 행동 예측

2. V-JEPA 2-AC
V-JEPA 2 + Robot Action Data
→ Action-Conditioned World Model
→ Robot Planning
```

# V-JEPA 2

## JEPA

JEPA는 **Joint Embedding Predictive Architecture**의 약자.

일반적인 generative model은 가려진 부분의 pixel을 직접 복원하려고 하지만, JEPA는 **embedding 공간에서 target representation을 예측함.**

```text
Pixel Prediction
현재 영상 → 미래 RGB Pixel 예측

JEPA
현재 영상 → 미래/가려진 부분의 Embedding 예측
```

픽셀에는 조명, 배경 texture 등 task와 크게 관련 없는 세부 정보도 매우 많음.

V-JEPA는 이러한 모든 pixel 정보를 복원하는 대신, **물체와 움직임 등 의미있는 representation을 예측하는 것**에 집중함.

# Architecture

V-JEPA 2는 크게

- **Context Encoder**
- **Target Encoder**
- **Predictor**

로 구성됨.

## 1. Video Tokenization

입력 video를 작은 spatio-temporal patch인 **tubelet** 단위로 나눔.

Image ViT가 이미지를 patch로 나누는 것과 비슷하지만, video이므로 공간뿐 아니라 시간축까지 포함함.

```text
Video
→ Spatio-Temporal Tubelets
→ Video Tokens
```

## 2. Masking

video token 중 일부 영역을 mask함.

Context Encoder는 **보이는 token들만 이용해서 representation을 생성함.**

반면 Target Encoder는 원본 video를 이용해, 모델이 맞혀야 하는 target representation을 생성함.

## 3. Context Encoder

Context Encoder $E_\theta$는 관찰 가능한 video token을 입력받아 현재 장면의 latent representation을 생성함.

즉,

```text
Visible Video Tokens
→ Context Encoder
→ Context Representation
```

## 4. Target Encoder

Target Encoder $E_{\bar{\theta}}$는 Predictor가 맞혀야 할 target representation을 만들어줌.

Target Encoder는 Context Encoder와 같은 계열의 encoder지만, gradient로 직접 학습하지 않고 **Context Encoder weight의 EMA(Exponential Moving Average)** 로 업데이트됨.

즉 target을 안정적으로 제공하는 teacher 역할에 가까움.

## 5. Predictor

Predictor $P_\phi$는 Context Encoder의 representation과 mask 정보를 이용하여 **가려진 영역의 representation을 예측함.**

$$
\hat{z}
=
P_\phi(E_\theta(x), M)
$$

- $x$: 관찰 가능한 video token
- $E_\theta$: Context Encoder
- $M$: mask 위치 정보
- $P_\phi$: Predictor
- $\hat{z}$: 예측한 target representation

Target Encoder가 만든 실제 target embedding과 비교하여 학습함.

$$
\mathcal{L}
=
\left\|
\hat{z}
-
\operatorname{sg}(E_{\bar{\theta}}(y))
\right\|_1
$$

- $y$: target 영역
- $E_{\bar{\theta}}$: EMA Target Encoder
- $\operatorname{sg}$: stop-gradient
- $\hat{z}$: Predictor가 예측한 representation

즉 핵심은

**가려진 부분의 pixel을 복원하는 것이 아니라, 가려진 부분이 어떤 의미의 representation을 가져야 하는지를 맞히도록 학습한다는 것.**

# V-JEPA 2-AC

V-JEPA 2 자체는 video만 보면서 세상의 움직임을 학습하기 때문에 **어떤 action을 수행하면 어떤 결과가 발생하는지 직접적으로 알지는 못함.**

이를 robot control에 사용하기 위해 만든 것이 **V-JEPA 2-AC(Action-Conditioned)** 임.

V-JEPA 2에서 학습된 visual encoder를 기반으로,

```text
현재 Visual State
+
Robot Action
→
다음 Visual State의 Representation
```

을 예측하도록 학습함.

논문에서는 DROID dataset의 **62시간 미만 robot video**를 이용해 action-conditioned predictor를 post-training함.

## Planning

Robot에게 목표 image가 주어졌다고 하자.

V-JEPA 2-AC는 여러 action sequence를 가정한 뒤

```text
현재 상태
+
후보 Action Sequence
→
예측된 미래 Representation
```

을 계산함.

그리고 예측된 미래 representation이 **goal image의 representation과 가장 가까워지는 action**을 선택함.

즉 직접 reward function을 학습하는 것이 아니라,

**"이 action을 하면 목표 장면에 가까워질 것인가?"**

를 latent space에서 예측하여 planning함.

# Results

V-JEPA 2는 video representation 자체만으로도 다양한 video understanding task에서 높은 성능을 보임.

- **Something-Something v2**: 77.3 Top-1 Accuracy
- **EPIC-KITCHENS-100 Action Anticipation**: 39.7 Recall@5
- LLM과 연결한 Video QA
    - **PerceptionTest**: 84.0
    - **TempCompass**: 76.9

V-JEPA 2-AC는 별도의 target environment 전용 학습 없이 두 연구실의 Franka robot에서 pick-and-place task를 수행함.

즉 논문의 핵심은 단순히 video encoder 성능을 높였다는 것보다,

**Internet Video로 학습한 representation을 실제 physical world의 prediction과 planning까지 연결했다는 것**에 있음.

# 핵심 정리

V-JEPA 2의 핵심 아이디어는 아래와 같음.

```text
Video
→ 일부 영역 Mask
→ Context Encoder
→ Predictor
→ Masked Target의 Latent Representation 예측
```

그리고 이를 action-conditioned model로 확장하면

```text
현재 상태 + Action
→ 미래 상태의 Representation 예측
→ Goal Representation과 비교
→ Action Planning
```

이 가능해짐.

결국 V-JEPA 2는 **pixel generation보다 representation prediction에 집중하여 video에서 physical dynamics를 학습하고, 이를 robot planning으로 연결하려는 world model**이라고 볼 수 있음.

# Limitations

- 미래를 실제 RGB frame으로 생성하는 model은 아니기 때문에, 사람이 직접 predicted world를 영상으로 확인하기는 어려움.
- Robot 실험은 주로 image-goal 기반 manipulation task에 집중되어 있음.
- V-JEPA 2-AC의 robot interaction data는 비교적 작지만, 아직 다양한 robot embodiment와 장기 planning까지 일반화되었다고 보기는 어려움.

논문이 보여준 것은 완전한 범용 physical world model이라기보다, **대규모 passive video learning → action-conditioned prediction → real-world planning**으로 이어지는 가능성에 가까움.
