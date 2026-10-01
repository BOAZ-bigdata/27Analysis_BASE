[🔗 Paper - Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391)

# Introduction

Genie는 Google DeepMind에서 발표한 **Generative Interactive Environment**, 즉 사용자의 action에 따라 다음 장면이 변화하는 interactive world를 생성하는 world model임.

기존 world model들은 보통

```text
현재 상태 + 실제 Action Label
→ 다음 상태
```

형태로 학습함.

문제는 일반적인 Internet Video에는

```text
왼쪽으로 이동
점프
오른쪽으로 이동
```

같은 action label이 존재하지 않는다는 것.

Genie의 핵심 아이디어는 **action label이 없는 video만 보고도 frame 사이의 변화에서 latent action을 스스로 찾아내도록 하는 것**임.

논문의 모델은 약 **11B parameters** 규모이며, unlabeled Internet Video를 기반으로 학습됨.

# Main Goal

Genie가 해결하려는 핵심 문제는 아래와 같음.

> **Action label이 없는 video만 가지고 controllable world model을 만들 수 있는가?**

즉 단순히 다음 영상을 생성하는 video generation model이 아니라,

**사용자가 action을 입력했을 때 그 action에 맞게 world가 변화하는 interactive environment**를 생성하는 것이 목적임.

# Architecture

Genie는 크게 세 가지 component로 구성됨.

1. **Spatiotemporal Video Tokenizer**
2. **Latent Action Model**
3. **Dynamics Model**

전체 구조는 대략 아래와 같음.

```text
Video
↓
Video Tokenizer
↓
Discrete Video Tokens
          +
Latent Action Model
↓
Latent Action
          ↓
Dynamics Model
↓
Next Video Tokens
↓
Next Frame
```

# 1. Spatiotemporal Video Tokenizer

Raw video frame을 그대로 Transformer에 넣으면 연산량이 매우 큼.

따라서 Genie는 먼저 video를 **discrete token representation**으로 압축함.

```text
Raw Video
→ Spatiotemporal Video Tokenizer
→ Discrete Video Tokens
```

NLP에서 문장을 token sequence로 바꾸는 것과 비슷하게, video도 모델이 처리하기 쉬운 token sequence로 변환한다고 보면 됨.

Dynamics Model은 실제 RGB pixel이 아니라 이 token들을 대상으로 다음 상태를 예측함.

# 2. Latent Action Model

Genie에서 가장 중요한 부분.

일반적인 robot/game dataset에는

```text
State_t
Action_t
State_t+1
```

처럼 어떤 action을 수행했는지 기록되어 있음.

하지만 Internet Video에는 보통 Action 정보가 없음.

Genie는 연속된 frame 사이의 변화를 보고

$$
(x_t, x_{t+1})
\rightarrow
a_t
$$

와 같이 **두 상태 사이에서 어떤 action이 발생했는지를 latent variable로 추론함.**

여기서 $a_t$는 사람이 직접 지정한

```text
LEFT
RIGHT
JUMP
```

같은 label이 아님.

모델이 video dynamics를 설명하기 위해 스스로 발견한 **latent action code**임.

즉,

```text
Frame t
→ 어떤 변화가 발생했는가?
→ Latent Action 추론
→ Frame t+1
```

방식으로 action space를 학습함.

# 3. Dynamics Model

Dynamics Model은 과거 video token과 latent action을 입력받아 **다음 video token을 autoregressive하게 예측함.**

$$
p(z_{t+1}\mid z_{\leq t}, a_t)
$$

- $z_{\leq t}$: 현재까지의 video token
- $a_t$: Latent Action
- $z_{t+1}$: 다음 상태의 video token

즉

```text
현재까지의 World State
+
Action
→
다음 World State
```

를 학습하는 부분.

일반적인 video generation과 다른 점은 **action을 condition으로 넣기 때문에 사용자가 생성되는 world를 control할 수 있다는 것**임.

# Training

Genie의 학습 흐름을 단순화하면 아래와 같음.

## Stage 1. Video Tokenizer

먼저 video를 compact한 discrete token으로 표현할 수 있도록 Video Tokenizer를 학습함.

## Stage 2. Latent Action + Dynamics

그 다음 video sequence에서

- 어떤 latent action이 일어났는지 추론하고
- 해당 action을 조건으로 다음 상태를 예측하도록

Latent Action Model과 Dynamics Model을 학습함.

핵심은 **실제 action label을 supervision으로 사용하지 않는다는 것.**

따라서 대규모 Internet Video를 그대로 world model 학습 데이터로 사용할 수 있음.

# Inference

학습이 끝난 뒤에는 시작 image가 주어지고, action을 선택하면서 다음 frame을 계속 생성할 수 있음.

```text
Initial Image
↓
Action
↓
Generated Next Frame
↓
Action
↓
Generated Next Frame
↓
...
```

논문에서는 text, synthetic image, 실제 photograph, sketch 등 다양한 형태에서 시작한 interactive environment를 보여줌.

즉 하나의 image를 단순히 video로 움직이게 만드는 것이 아니라, **action에 따라 변화하는 playable world로 확장함.**

# Why Latent Action?

Internet Video를 world model 학습에 바로 사용하기 어려운 가장 큰 이유 중 하나가 **Action Label이 없다는 것**임.

예를 들어 game video를 보면 사람은

```text
캐릭터가 오른쪽으로 이동했다.
→ 아마 오른쪽 이동 action이 있었을 것
```

이라고 추론할 수 있음.

Genie의 Latent Action Model도 비슷하게 **관찰된 상태 변화로부터 원인이 된 control signal을 역으로 추론함.**

이를 통해

```text
Unlabeled Video
→ Latent Action Discovery
→ Action-Controllable World Model
```

이 가능해짐.

# Results

Genie는 약 11B parameter 규모로 확장되었으며, 논문의 주요 실험은 다양한 **2D platformer video**를 중심으로 진행됨.

학습 과정에서 실제 game action label을 제공하지 않았음에도, 모델 내부에서 비교적 일관된 latent action space가 형성됨.

또한 학습된 latent action을 이용하면

- 생성된 environment를 frame-by-frame으로 control하고
- 처음 보는 video의 behavior를 imitation하는 데 활용할 수 있음

을 보여줌.

# 핵심 정리

Genie의 가장 중요한 아이디어는 **video generation 자체가 아니라 action discovery와 controllable world generation을 함께 학습했다는 것**임.

```text
Unlabeled Internet Video
↓
Video Tokenizer
↓
Latent Action Model
↓
Action Discovery
↓
Dynamics Model
↓
Action-Controllable Generated World
```

즉 기존에는

```text
Action Label이 있는 Interaction Data
→ World Model
```

이 필요했다면,

Genie는

```text
Unlabeled Video
→ Action을 스스로 추론
→ World Model
```

방향을 제시함.

# Significance

Genie의 의의는 **대규모 Internet Video 자체를 embodied agent를 위한 경험 데이터로 사용할 가능성을 보여줬다는 것**에 있음.

인터넷에는 실제 robot action dataset보다 훨씬 많은 video가 존재함.

따라서 영상 속 상태 변화에서 action과 dynamics를 학습할 수 있다면, 앞으로 agent가 실제 환경에서 직접 모든 경험을 수집하지 않고도 대규모 데이터를 이용할 수 있음.

# Limitations

- 논문의 주요 대규모 실험은 2D platformer 환경에 집중되어 있음.
- Latent Action은 사람이 정의한 실제 action과 정확히 같은 의미를 가지는 것이 아님.
- 긴 시간 생성할수록 visual consistency가 유지되기 어려움.
- 2024년의 Genie는 최근 Genie 계열처럼 고해상도 real-time 3D world를 생성하는 모델은 아님.

즉 Genie는 완성된 범용 simulator라기보다, **unlabeled video에서 controllable world model을 학습할 수 있다는 가능성을 보여준 초기 foundation world model**로 보는 것이 적절함.
