# 1. Genie: Generative Interactive Environments

## 1) 개요 — Abstract

**한 줄 요약:**
인터넷 영상만 학습해서 **사용자가 직접 조작할 수 있는 가상 환경**을 만드는 World Model이다.

Genie는 별도의 행동(action) 라벨 없이 인터넷 영상에서 프레임 사이의 움직임을 학습한다. 이 과정에서 모델이 스스로 `latent action`을 만들고, 사용자는 이 action을 이용해 생성된 환경을 프레임 단위로 조작할 수 있다.

모델은 크게 세 부분으로 구성된다.

* Video Tokenizer
* Latent Action Model
* Dynamics Model

전체 규모는 약 11B parameters이며, 텍스트로 만든 이미지, 실제 사진, 손으로 그린 그림 등을 시작 화면으로 넣어도 조작 가능한 환경을 생성할 수 있다. 

---

## 2) 연구 동기와 문제점 — Introduction

기존 영상 생성 모델은 자연스러운 영상을 만들 수 있지만 **사용자가 그 영상 속 세계를 직접 조작하기 어렵다.**

반대로 기존 World Model은 사용자의 행동에 따라 미래 상태를 예측할 수 있지만 보통 다음과 같은 데이터가 필요하다.

**현재 상태 + 실제 Action → 다음 상태**

문제는 인터넷 영상에는 “이 장면에서 오른쪽 버튼을 눌렀다”, “로봇 팔을 왼쪽으로 움직였다” 같은 **action label이 거의 없다는 점**이다.

Genie는 여기서 질문을 바꿨다.

**“Action 정보를 따로 주지 않아도 영상 속 움직임만 보고 행동을 알아낼 수 없을까?”**

그래서 대규모 인터넷 영상에서 프레임 사이 변화를 분석해 latent action을 학습하고, 이를 이용해 조작 가능한 환경을 만든다. 

---

## 3) 관련 연구 동향 — Related Works

기존 연구는 크게 두 방향으로 나뉜다.

| 구분               | 특징                  | 한계                 |
| ---------------- | ------------------- | ------------------ |
| World Model      | Action에 따른 미래 상태 예측 | Action 데이터가 필요함    |
| Video Generation | 이미지·텍스트에서 영상 생성     | 프레임 단위 제어가 어려움     |
| Genie            | 영상만으로 학습하면서 제어 가능   | 장시간 일관성과 생성 속도에 한계 |

GAIA-1이나 UniSim 같은 기존 World Model은 text나 action 정보를 함께 사용했다. Genie는 **공개 인터넷 영상만으로도 action-controllable world model을 만들 수 있다는 점**에 초점을 둔다. 

---

## 4) 연구 접근법과 모델 구조 — Method

### ① Video Tokenizer

원본 영상의 픽셀을 그대로 처리하면 계산량이 너무 크기 때문에 영상을 **작은 discrete token으로 압축**한다.

즉,

**Video → Video Token z**

형태로 바꾼다.

Genie는 VQ-VAE와 ST-Transformer를 이용해 공간 정보뿐 아니라 **시간에 따른 변화까지 포함한 영상 표현**을 만든다. 

### ② Latent Action Model, LAM

Genie의 핵심 부분이다.

LAM은 연속된 두 장면의 차이를 보고

**“이 장면에서 다음 장면으로 넘어가기 위해 어떤 행동이 있었는가?”**

를 스스로 추정한다.

실험에서는 latent action의 종류를 **8개**로 제한했다.

예를 들면 실제 키 입력 정보가 없어도 모델 내부적으로

* Action 1 → 오른쪽 이동과 비슷한 변화
* Action 2 → 점프와 비슷한 변화
* Action 3 → 왼쪽 이동과 비슷한 변화

처럼 일관된 움직임을 학습할 수 있다.

중요한 점은 이 action의 의미를 사람이 미리 정한 것이 아니라 **영상에서 모델이 스스로 발견했다는 것**이다. 

### ③ Dynamics Model

Dynamics Model은

**과거 영상 정보 + latent action → 다음 장면**

을 예측한다.

영상은 token 형태로 들어가며, MaskGIT 기반 Transformer를 사용해 다음 frame의 token을 예측한다. 예측한 token과 실제 token 사이의 Cross-Entropy Loss로 학습한다. 

### 전체 구조

**영상 → Video Tokenizer → 영상 표현**

동시에

**영상의 프레임 변화 → Latent Action Model → 행동 표현**

을 구한 뒤,

**영상 표현 + 행동 표현 → Dynamics Model → 다음 장면**

으로 연결된다.

---

## 5) 실험 구성과 결과 — Experiments

### 데이터

주요 학습 데이터는 인터넷에서 수집한 **2D 플랫폼 게임 영상**이다.

* 약 30,000시간
* 약 6.8M개의 16초 clip
* 10 FPS
* 해상도 160 × 90

추가로 RT-1 계열의 Robotics 데이터에도 적용했다. 이때도 원래 존재하는 robot action 정보는 사용하지 않고 **영상만 학습 데이터로 사용했다.** 

### 평가 지표

두 가지를 주로 평가한다.

**FVD ↓**

생성한 영상의 품질을 평가한다.
낮을수록 실제 영상과 비슷하다.

**Delta-t PSNR ↑**

latent action이 실제 생성 결과에 얼마나 영향을 주는지를 평가한다.

수식은 다음처럼 보면 된다.

`Delta-t PSNR = PSNR(실제 프레임, 올바른 action으로 생성한 프레임) - PSNR(실제 프레임, 랜덤 action으로 생성한 프레임)`

값이 크다는 것은 **올바른 action과 랜덤 action을 넣었을 때 결과 차이가 크다**는 뜻이다.

즉, latent action이 영상 생성에 실제로 강하게 작용하고 있다는 의미다. 

### 주요 결과

모델 크기와 batch size를 증가시킬수록 training loss가 지속적으로 감소했다.

최종 Genie 모델은 약 **10.7B parameters** 규모이며, 약 942B token으로 학습됐다.

또 학습 데이터에서 보지 못한 형태의 입력인

* Text-to-Image 모델로 만든 이미지
* 손으로 그린 그림
* 실제 사진

에서도 게임처럼 움직이는 환경을 생성했다.

로봇 영상에서도 action label 없이 일관된 latent action을 학습했다.

CoinRun 실험에서는 latent action을 실제 action으로 연결할 때 **약 200개의 expert sample만으로 oracle behavioral cloning과 같은 수준의 결과**를 보였다고 보고했다. 

---

## 6) 무엇을 발견했고 한계점은 무엇인가? — Discussion

### 주요 발견

가장 중요한 결과는 **action label이 없어도 영상에서 의미 있는 action representation을 학습할 수 있다는 것**이다.

또 Latent Action Model에 tokenized video를 넣는 것보다 **원본 pixel을 직접 넣었을 때 controllability가 더 높았다.**

이는 영상 tokenization 과정에서 움직임과 관련된 일부 정보가 손실될 수 있다는 점을 보여준다. 

### 한계점

첫째, autoregressive 방식이라 여러 프레임을 연속으로 생성하면서 **비현실적인 장면을 만들어낼 수 있다.**

둘째, 모델이 사용하는 context가 **16 frame 정도로 제한**되어 있어 긴 시간 동안 같은 세계를 일관되게 유지하기 어렵다.

셋째, 당시 모델은 약 **1 FPS 수준**으로 작동해 실제 게임처럼 실시간으로 사용하기에는 느리다. 

---

## 7) 결론 및 주요 요약 — Conclusion

Genie는 단순히 게임 영상을 생성하는 모델이 아니다.

핵심은

**“Action 정보가 없는 일반 영상도 interactive world model을 학습하는 데이터로 사용할 수 있다.”**

는 가능성을 보여줬다는 것이다.

즉, 사람이 직접 action label을 수집하지 않아도 대규모 인터넷 영상을 이용해 조작 가능한 환경을 만들 수 있다.

이런 환경이 충분히 발전하면 강화학습 Agent가 학습할 수 있는 가상 환경을 자동으로 만들어내는 방향으로도 이어질 수 있다. 

---

## 8) 기존 연구 대비 차별점

기존 World Model의 기본 구조는

**Video + 실제 Action → World Model**

이다.

Genie는

**Video → Latent Action 자동 발견 → Controllable World Model**

이라는 구조다.

따라서 가장 큰 차이는

**“행동 정보를 주고 세계를 학습한 것이 아니라, 영상에서 행동 자체를 찾아냈다.”**

는 점이다.

---

## 9) 핵심 아이디어

Genie의 핵심 생각은 간단하다.

**프레임 사이의 변화에는 행동에 대한 정보가 들어 있다.**

예를 들어 캐릭터가 왼쪽에서 오른쪽으로 이동했다면 실제 키보드 기록을 몰라도 영상의 변화를 통해 “무언가 오른쪽 이동을 발생시킨 행동이 있었다”는 사실을 학습할 수 있다.

Genie는 이런 변화들을 몇 개의 latent action으로 묶어 **사용자가 조작할 수 있는 버튼처럼 만든다.**

---

## 10) 주요 수식

가장 이해해야 할 수식은 controllability 평가에 사용한 Delta-t PSNR이다.

`Delta-t PSNR = PSNR(x_t, x_pred_t) - PSNR(x_t, x_random_t)`

여기서

* `x_t` : 실제 정답 frame
* `x_pred_t` : 정답 영상에서 추론한 latent action을 사용해 생성한 frame
* `x_random_t` : 랜덤 latent action을 사용해 생성한 frame

이다.

값이 크면 랜덤 action을 넣었을 때 정답 영상에서 많이 벗어난다는 뜻이다.

따라서 **latent action이 실제 영상의 움직임을 잘 제어하고 있다고 해석한다.**

---

# 2. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

## 1) 개요 — Abstract

**한 줄 요약:**
대규모 인터넷 영상으로 먼저 세상의 움직임을 학습하고, 소량의 로봇 데이터를 추가해 **실제 로봇의 행동 계획까지 수행하는 World Model**이다.

V-JEPA 2는 **100만 시간 이상의 인터넷 영상**으로 self-supervised pretraining을 한다.

이후 62시간 미만의 robot interaction 데이터를 이용해 **V-JEPA 2-AC**를 추가 학습한다.

주요 결과는 다음과 같다.

* Something-Something v2: **77.3 Top-1 Accuracy**
* Epic-Kitchens-100: **39.7 Recall@5**
* PerceptionTest: **84.0**
* TempCompass: **76.9**
* 실제 Franka robot에서 zero-shot manipulation 수행

새 환경의 로봇 데이터를 다시 수집하거나 task별 reward를 설계하지 않고도 실제 manipulation이 가능했다. 

---

## 2) 연구 동기와 문제점 — Introduction

기존 로봇 World Model은 많은 **robot interaction data**가 필요하다.

즉,

**상태 → Action → 다음 상태**

가 포함된 데이터를 실제 로봇을 움직이며 수집해야 한다.

하지만 이런 데이터는 비용이 많이 들고 규모를 키우기 어렵다.

반면 인터넷에는 엄청난 양의 영상이 존재한다.

그래서 V-JEPA 2는

**“대부분의 세상에 대한 지식은 인터넷 영상에서 먼저 배우고, 실제 행동 데이터는 조금만 사용하면 되지 않을까?”**

라는 접근을 사용한다.

또 하나의 문제는 기존 generative video model이 미래의 **모든 픽셀을 예측하려 한다는 점**이다.

예를 들어 공이 움직이는 장면에서 중요한 것은 공의 이동 방향이지 배경에 있는 나뭇잎 하나하나의 정확한 위치가 아니다.

V-JEPA 2는 pixel 자체가 아니라 **representation space에서 예측 가능한 핵심 정보만 학습한다.** 

---

## 3) 관련 연구 동향 — Related Works

기존 World Model은 미래를

* Pixel space에서 예측하거나
* Learned representation space에서 예측하거나
* Keypoint 같은 구조화된 공간에서 예측

하는 방식으로 발전해왔다.

하지만 실제 로봇에 적용된 기존 연구는 대부분 **로봇이 배치될 환경에서 직접 interaction data를 수집한 뒤 해당 환경에 맞춰 학습**하는 방식이었다.

V-JEPA 2는 여기에서 한 단계 더 나아가

**대규모 일반 영상으로 범용 representation 학습 → 적은 robot interaction data 추가 → 새로운 환경에서 zero-shot planning**

을 시도했다. 

---

## 4) 연구 접근법과 모델 구조 — Method

전체 과정은 크게 **두 단계**다.

### Stage 1. V-JEPA 2 Pretraining

영상의 일부 patch를 가린다.

모델은 남아 있는 영상 정보를 보고 **가려진 부분의 representation을 예측한다.**

여기서 중요한 것은 가려진 **pixel 자체를 복원하는 것이 아니라 feature representation을 맞힌다는 점**이다.

구성은

* Encoder
* Predictor
* EMA Target Encoder

로 이루어진다.

Encoder와 Predictor는 ViT 기반이며, V-JEPA 2에서는 시간·높이·너비 정보를 다루기 위해 **3D-RoPE**를 사용한다. 

또 V-JEPA에서 V-JEPA 2로 확장하면서 네 가지를 키웠다.

* 데이터 규모 확대
* 모델 규모 확대
* 학습 시간 증가
* 영상 해상도와 길이 증가

이 조합으로 downstream task 평균 accuracy가 기존 84.2에서 **88.2까지 증가**했다. 

### Stage 2. V-JEPA 2-AC

이미 학습한 V-JEPA 2 Encoder는 고정한다.

그 위에 새로운 **Action-Conditioned Predictor**를 학습한다.

입력은

* 이전 video representation
* Robot action
* Robot end-effector state

이며, 이를 이용해 **다음 영상의 representation**을 예측한다.

Predictor는 약 **300M parameter Transformer**이며 block-causal attention을 사용한다.

즉,

**V-JEPA 2 = 세상이 어떻게 움직이는지 이해**

**V-JEPA 2-AC = 특정 행동을 하면 세상이 어떻게 변할지 예측**

으로 역할을 나눌 수 있다. 

---

## 5) 실험 구성과 결과 — Experiments

실험은 크게 세 종류로 보면 된다.

### ① Understanding

영상에서 행동과 객체를 얼마나 잘 표현하는지 확인한다.

대표 결과는 Something-Something v2에서

**77.3% Top-1 Accuracy**

이다.

특히 이 benchmark는 단순한 객체 인식보다 **움직임과 시간 관계를 이해해야 하는 문제**라서 V-JEPA 2의 video representation 능력을 보여준다. 

### ② Prediction

Epic-Kitchens-100에서 사람이 다음에 어떤 행동을 할지 예측한다.

결과는

**39.7 Recall@5**

이며, 논문에서는 기존 최고 결과 대비 약 **44% 상대 향상**이라고 보고한다. 

### ③ Planning

V-JEPA 2-AC를 실제 Franka robot에 적용했다.

주요 task는

* Reach
* Grasp
* Reach with Object
* Pick-and-Place

이다.

두 개의 서로 다른 연구실에서 실험했으며, 새로운 물체와 새로운 환경에서도 수행했다.

Pick-and-Place의 평균 성공률은

* Cup: **80%**
* Box: **65%**

였다.

같은 표에서 Octo는 각각 15%, 10%로 보고되지만, 모델의 사전학습 데이터와 학습 방식이 서로 다르므로 단순한 절대 성능 순위라기보다는 **해당 실험 조건에서의 비교 결과**로 보는 것이 적절하다. 

---

## 6) 무엇을 발견했고 한계점은 무엇인가? — Discussion

### 주요 발견

가장 중요한 결과는

**좋은 영상 representation을 대규모 데이터로 먼저 학습하면, 비교적 적은 robot action 데이터만 추가해도 실제 planning이 가능하다**

는 점이다.

또 pixel 영상을 직접 생성하는 모델보다 latent representation을 예측하면 planning에 필요한 계산량을 크게 줄일 수 있었다.

논문에서 한 번의 action을 찾는 데 걸린 시간은

* Cosmos: 약 **4분**
* V-JEPA 2-AC: 약 **16초**

였다. 

### 한계점

**첫째, 장기 예측이 어렵다.**

미래 representation을 반복적으로 예측하면 오차가 누적된다.

그래서 긴 행동 sequence를 한 번에 계획하기 어렵다.

**둘째, 카메라 위치에 민감하다.**

V-JEPA 2-AC는 별도의 camera calibration 없이 영상만 보고 robot coordinate를 이해해야 한다.

카메라 위치가 달라지면 좌표축을 잘못 추론해 행동 오류가 발생할 수 있다. 

**셋째, Image Goal이 필요하다.**

현재는 최종 목표 상태를 이미지 형태로 제공해야 한다.

예를 들어

**“이 컵이 저 위치에 놓인 이미지”**

를 goal로 주는 방식이다.

“컵을 상자 안에 넣어”처럼 자연어만으로 목표를 주는 것은 후속 연구 과제로 남아 있다. 

---

## 7) 결론 및 주요 요약 — Conclusion

V-JEPA 2가 보여준 핵심은 하나의 self-supervised video representation을

**Understanding → Prediction → Planning**

까지 연결할 수 있다는 것이다.

대규모 인터넷 영상에서 먼저 일반적인 세상의 움직임을 배우고, 소량의 robot interaction data를 추가하면 실제 로봇의 행동 계획으로 확장할 수 있었다.

따라서 World Model을 학습하기 위해 모든 지식을 실제 로봇 경험에서 얻어야 하는 것은 아니라는 가능성을 보여준다. 

---

## 8) 기존 연구 대비 차별점

기존 로봇 World Model은 보통

**많은 Robot Interaction Data → World Model → Planning**

형태다.

V-JEPA 2는

**대규모 인터넷 영상 → 범용 Representation 학습**

후,

**소량의 Robot Interaction Data → Action-Conditioned World Model**

을 추가한다.

또 하나 중요한 차이는 **예측 대상**이다.

기존 Video Generation Model은

**미래 Pixel을 생성**

하려는 경우가 많다.

V-JEPA 2는

**미래 Representation을 예측**

한다.

즉, 화면을 예쁘게 다시 그리는 것보다 **행동에 필요한 물체·움직임·상태 정보가 어떻게 변할지를 학습하는 데 집중한다.**

---

## 9) 핵심 아이디어

V-JEPA 2를 가장 짧게 설명하면

**“세상의 모든 픽셀을 예측할 필요는 없다. 행동과 판단에 필요한 중요한 정보만 예측하면 된다.”**

이다.

그리고 이 representation을 인터넷 영상으로 충분히 잘 학습해두면 실제 robot data의 양을 크게 줄이면서도 planning으로 확장할 수 있다는 것이 논문의 핵심이다.

---

## 10) 주요 수식

### ① V-JEPA 2 Pretraining

논문의 핵심 목적은 다음처럼 이해하면 된다.

`Loss = L1(예측한 masked 영역의 representation, 실제 masked 영역의 representation)`

즉,

**가려진 영역의 feature를 예측하고 실제 feature와의 차이를 최소화한다.**

여기서 실제 target representation은 EMA Encoder가 만들고, target 쪽에는 gradient가 전달되지 않도록 Stop-Gradient를 적용한다. 

수식을 복잡하게 외우기보다는

**“Pixel 복원 Loss가 아니라 Representation 예측 Loss다.”**

라고 이해하는 것이 가장 중요하다.

### ② Robot Planning

행동 계획에서는 후보 action을 실행했을 때 예측되는 미래 representation과 목표 이미지의 representation을 비교한다.

안 깨지는 형태로 쓰면:

`Energy = L1(예측된 미래 representation, Goal representation)`

그리고

`Best Action = Energy가 가장 작은 Action Sequence`

를 선택한다. 

즉,

**“이 행동을 하면 미래 모습이 목표 모습과 얼마나 가까워지는가?”**

를 계산하고, 가장 가까워지는 행동을 고르는 방식이다.

---

# Genie와 V-JEPA 2 비교

| 구분        | Genie                            | V-JEPA 2                            |
| --------- | -------------------------------- | ----------------------------------- |
| 발표        | 2024                             | 2025                                |
| 핵심 목표     | 조작 가능한 가상 세계 생성                  | 이해·예측·행동 계획                         |
| 주요 데이터    | 인터넷 게임 영상                        | 100만 시간 이상 인터넷 영상                   |
| Action 활용 | 영상에서 latent action을 스스로 발견       | 후처리 단계에서 실제 robot action 사용         |
| 미래 예측     | Video Token                      | Representation                      |
| 주요 구조     | Tokenizer + LAM + Dynamics Model | Encoder + Predictor + V-JEPA 2-AC   |
| 대표 응용     | 플레이 가능한 생성 환경                    | 실제 Robot Planning                   |
| 핵심 문제     | Action label 없이 제어를 배울 수 있는가?    | 관찰에서 배운 지식을 실제 행동으로 연결할 수 있는가?      |
| 핵심 아이디어   | **영상 변화에서 action을 발견**           | **Pixel 대신 중요한 representation을 예측** |

둘의 차이를 가장 간단히 말하면 이렇다.

**Genie:**
“Action이 없는 영상에서 행동 자체를 찾아내고, 그 행동으로 세계를 조작하자.”

**V-JEPA 2:**
“영상을 전부 다시 그리지 말고, 세계를 이해하고 행동하는 데 필요한 핵심 상태만 예측하자.”
