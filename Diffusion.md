# Denoising Diffusion Probabilistic Models

**Jonathan Ho, Ajay Jain, Pieter Abbeel · NeurIPS 2020**

## 1. 논문 개요

* **Diffusion Probabilistic Model을 이용한 고품질 이미지 생성**
* 데이터에 점진적으로 노이즈를 추가한 뒤, **노이즈 제거 과정을 학습하여 이미지를 생성하는 방식**
* Diffusion Model과 **Denoising Score Matching**, **Langevin Dynamics** 사이의 연결 제시
* 기존 Variational Bound를 단순화한 **새로운 학습 목적함수 \(L_{\text{simple}}\)** 제안
* CIFAR-10에서 당시 최고 수준의 생성 품질 달성 

---

# 2. 핵심 아이디어

### Forward Process

**원본 이미지 → 점진적으로 Gaussian Noise 추가 → 순수 Noise에 가까운 상태**

$$
x_0 \rightarrow x_1 \rightarrow x_2 \rightarrow \cdots \rightarrow x_T
$$

* \(x_0\): 실제 데이터
* \(x_T\): 거의 표준 정규분포 \(N(0,I)\)
* 각 단계에서 작은 Gaussian noise 추가
* **고정된 Markov Chain**
* 학습 대상 아님
* Noise 크기 \(\beta_t\)를 미리 설정

논문에서는 \(\beta_t\)를 학습하지 않고 상수로 고정. 따라서 forward process 자체에는 학습 파라미터가 없음. 

---

### Reverse Process

**Noise → 점진적으로 Noise 제거 → 원본 데이터 분포 복원**

$$
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_1 \rightarrow x_0
$$

* Forward Process의 반대 과정
* Neural Network가 학습하는 부분
* 각 단계에서 현재 noisy image \(x_t\)를 입력
* 이전 단계 \(x_{t-1}\)의 분포 예측
* Gaussian transition으로 모델링

$$
p_\theta(x_{t-1}|x_t)
$$

즉,

> **Forward:** Noise 추가
> **Reverse:** Noise 제거 방법 학습

---

# 3. Forward Process

각 단계에서

$$
q(x_t|x_{t-1})
=
\mathcal{N}
(
x_t;
\sqrt{1-\beta_t}x_{t-1},
\beta_t I
)
$$

### 의미

* 기존 이미지 신호를 조금 감소
* Gaussian Noise를 조금 추가
* 해당 과정 반복
* 충분히 큰 \(T\)에서는 데이터 정보가 거의 사라짐

논문의 실험에서는

$$
T=1000
$$

사용.

Noise variance는

$$
\beta_1=10^{-4}
\rightarrow
\beta_T=0.02
$$

까지 **선형 증가**하도록 설정. 

---

# 4. 중요한 특징: 임의의 timestep으로 바로 이동 가능

Forward Process를 한 단계씩 계산할 필요 없음.

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

$$
\epsilon \sim N(0,I)
$$

### 의미

원본 이미지 \(x_0\)와 random noise \(\epsilon\)만 있으면

> **원하는 timestep \(t\)의 noisy image \(x_t\)를 한 번에 생성 가능**

### 학습 효율 향상

매 학습마다

1. 원본 이미지 \(x_0\) 선택
2. 임의의 timestep \(t\) 선택
3. Noise \(\epsilon\) 생성
4. 바로 \(x_t\) 생성
5. Noise 예측 학습

→ 1000단계를 실제로 순차 실행하면서 학습할 필요 없음.

---

# 5. Reverse Process

Reverse Process:

$$
p_\theta(x_{t-1}|x_t)
=
\mathcal{N}
(
x_{t-1};
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
)
$$

### 모델의 역할

현재 noisy image \(x_t\)를 보고

> **어떤 noise가 포함되어 있는지 예측**

논문의 핵심 선택:

$$
\epsilon_\theta(x_t,t)
$$

를 학습.

즉 모델이 직접 깨끗한 이미지 \(x_0\)를 출력하는 것이 아니라

> **추가된 Gaussian Noise \(\epsilon\) 예측**

---

# 6. 왜 Noise를 예측하는가?

원래는 reverse distribution의 평균

$$
\mu_\theta(x_t,t)
$$

을 직접 예측할 수도 있음.

논문에서는 이를 재parameterization하여

$$
\epsilon_\theta(x_t,t)
$$

를 예측하도록 변경.

### 장점

* 학습 목적함수 단순화
* Denoising Score Matching과 연결
* Langevin Dynamics와 연결
* 실험적으로 더 높은 이미지 생성 품질

논문에서는 \(x_0\) 직접 예측도 실험했으나 초기 실험에서 생성 품질이 좋지 않았다고 설명. 

---

# 7. 학습 과정

논문의 Algorithm 1을 간단하게 표현하면

```text
실제 이미지 x₀
      ↓
Random timestep t 선택
      ↓
Gaussian Noise ε 생성
      ↓
xₜ 생성
      ↓
U-Net에 (xₜ, t) 입력
      ↓
Noise ε̂ 예측
      ↓
실제 Noise ε와 비교
      ↓
MSE Loss
```

Loss:

$$
L_{\text{simple}}
=
E_{t,x_0,\epsilon}
[
||\epsilon-\epsilon_\theta(x_t,t)||^2
]
$$

즉,

> **실제로 넣은 Noise와 모델이 예측한 Noise의 차이 최소화**

---

# 8. Simplified Training Objective

기존 Diffusion Model은 **Variational Lower Bound** 기반 학습.

논문에서는 이를 단순화한

$$
L_{\text{simple}}
$$

제안.

$$
L_{\text{simple}}
=
E
\left[
\left\|
\epsilon-
\epsilon_\theta
(
\sqrt{\bar{\alpha}_t}x_0+
\sqrt{1-\bar{\alpha}_t}\epsilon,t
)
\right\|^2
\right]
$$

### 특징

* 단순한 MSE 형태
* timestep별 복잡한 weighting 제거
* 구현 간단
* 작은 \(t\)의 쉬운 denoising task 비중 감소
* 큰 \(t\)의 어려운 denoising task에 상대적으로 집중
* **실험에서 더 좋은 sample quality** 

---

# 9. Sampling 과정

학습 이후에는 실제 이미지 불필요.

먼저

$$
x_T \sim N(0,I)
$$

즉 **완전한 Random Noise에서 시작**.

이후

```text
Random Noise xₜ
      ↓
Noise 예측 εθ(xₜ,t)
      ↓
Noise 일부 제거
      ↓
xₜ₋₁
      ↓
Noise 예측
      ↓
...
      ↓
x₀
      ↓
생성 이미지
```

### Algorithm 2

$$
x_T \rightarrow x_{T-1}\rightarrow \cdots \rightarrow x_0
$$

을 순차적으로 수행.

논문 기준 \(T=1000\).

→ 이미지 하나를 생성하기 위해 **약 1000번의 neural network evaluation 필요**

---

# 10. Network Architecture

Reverse Process 모델로 **U-Net 사용**

### 구성

* U-Net backbone
* Group Normalization
* Transformer의 **Sinusoidal Position Embedding**으로 timestep \(t\) 전달
* 16×16 feature map에서 **Self-Attention 적용**
* 모든 timestep에서 **동일한 network parameter 공유**



즉 별도의 모델 1000개 사용 X.

> **하나의 U-Net이 timestep 정보를 입력받아 모든 denoising step 담당**

---

# 11. 전체 구조

```text
[Training]

Real Image x₀
      ↓
Gaussian Noise 추가
      ↓
Noisy Image xₜ
      ↓
U-Net
      ↓
Noise ε 예측
      ↓
실제 Noise와 MSE 비교
      ↓
학습
```

```text
[Generation]

Random Noise xₜ
      ↓
U-Net
      ↓
Noise 제거
      ↓
xₜ₋₁
      ↓
U-Net
      ↓
Noise 제거
      ↓
...
      ↓
Generated Image x₀
```

---

# 12. Denoising Score Matching과의 연결

Noise prediction objective:

$$
||\epsilon-\epsilon_\theta(x_t,t)||^2
$$

가 **여러 noise level에서 수행하는 Denoising Score Matching과 유사한 형태**.

논문의 주요 이론적 기여 중 하나:

> **Diffusion Model의 Variational Inference ↔ Denoising Score Matching 연결**

또한 Reverse Sampling 과정이 **Langevin Dynamics와 유사한 형태**를 가짐. 

---

# 13. 실험 설정

### Dataset

* CIFAR-10
* CelebA-HQ 256×256
* LSUN Bedroom 256×256
* LSUN Church 256×256
* LSUN Cat 256×256

### 기본 설정

* \(T=1000\)
* Linear noise schedule
* \(\beta_1=10^{-4}\)
* \(\beta_T=0.02\)
* U-Net backbone
* Adam Optimizer
* EMA 사용

CIFAR-10 모델 약 **35.7M parameters**, LSUN/CelebA-HQ 모델 약 **114M parameters**. 

---

# 14. 주요 결과

### CIFAR-10

Simplified objective 사용:

* **Inception Score: 9.46**
* **FID: 3.17**
* 당시 unconditional generation에서 매우 높은 성능



특히 Ablation 결과:

| 방법                                             |       IS |      FID |
| ---------------------------------------------- | -------: | -------: |
| \(\tilde{\mu}\) prediction + Variational Bound |     8.06 |    13.22 |
| \(\epsilon\) prediction + Variational Bound    |     7.67 |    13.51 |
| **\(\epsilon\) prediction + \(L_{simple}\)**   | **9.46** | **3.17** |

### 핵심 결과

> **Noise prediction + Simplified Objective 조합에서 생성 품질 크게 향상**

또한 reverse variance를 직접 학습시키는 방법은 **학습이 불안정하고 성능도 낮음**. 

---

# 15. LSUN 결과

256×256 이미지에서도 높은 생성 품질.

* LSUN Bedroom: **FID 4.90**
* LSUN Church: **FID 7.89**
* LSUN Cat: **FID 19.75**

ProgressiveGAN과 비슷하거나 일부 경우 더 좋은 수준의 생성 품질 확인. 

---

# 16. Progressive Generation

Diffusion의 생성 과정에서

> **큰 구조 → 세부적인 구조**

순서로 이미지 형성.

초기 reverse step:

* 전체적인 형태
* 객체 배치
* 얼굴 구조 등

후반 reverse step:

* texture
* 작은 디테일
* 세부적인 픽셀 정보

논문의 Figure 6에서도 reverse process가 진행될수록 **coarse feature → fine detail** 순서로 이미지가 형성되는 모습 확인. 

---

# 17. Progressive Compression

Diffusion 과정을 **점진적 압축/복원 과정**으로도 해석.

* 초기 정보 → 이미지의 큰 구조
* 추가적인 정보 → 세부적인 디테일
* 많은 bit가 사람이 거의 인지하지 못하는 세부적인 이미지 정보에 사용

논문의 CIFAR-10 분석에서 lossless codelength의 절반 이상이 **사람이 거의 인지하지 못하는 distortion을 표현하는 데 사용**된다고 분석. 

단, 논문의 compression 방법 자체는 **실용적인 압축 알고리즘이 아닌 개념적 proof-of-concept**. 

---

# 18. Autoregressive Model과의 관계

Diffusion Model을

> **일반화된 Autoregressive Decoding**

으로 해석 가능.

Autoregressive Model:

```text
Pixel 1 → Pixel 2 → Pixel 3 → ...
```

Diffusion Model:

```text
Noise
 → 큰 구조
 → 중간 구조
 → 세부 구조
 → 최종 이미지
```

특정 diffusion process를 구성하면 autoregressive model과 동일한 형태가 될 수 있음을 논문에서 설명.

따라서 Gaussian Diffusion을

> **data coordinate 순서를 넘어선 generalized ordering을 사용하는 autoregressive model**

로 해석. 

---

# 19. 기존 Score-based Model과 차이

NCSN 등 기존 Score Matching 모델과 비교.

### DDPM

* U-Net + Self-Attention
* Forward Process에서 데이터 scale 조절
* 최종 \(x_T\)가 Gaussian prior와 거의 일치하도록 signal 제거
* 작은 \(\beta_t\) 사용
* Sampling coefficient를 forward process에서 수학적으로 도출
* **Sampler 자체를 Variational Inference로 직접 학습**

### NCSN

* RefineNet 사용
* Sampler coefficient를 학습 이후 별도로 설정
* Sampler 자체를 직접 최적화하는 구조가 아님



---

# 20. 논문의 핵심 기여

### ① 고품질 Diffusion Image Generation

Diffusion Model도 GAN과 경쟁 가능한 이미지 생성 품질 달성.

### ② Noise Prediction Parameterization

Reverse mean을 직접 예측하는 대신

$$
\epsilon_\theta(x_t,t)
$$

즉 **Noise 예측 방식 사용**.

### ③ Simplified Objective

복잡한 Variational Bound 대신

$$
||\epsilon-\epsilon_\theta(x_t,t)||^2
$$

형태의 간단한 학습 목적함수 사용.

### ④ Score Matching과 연결

Diffusion Model 학습이

> **Denoising Score Matching + Annealed Langevin Dynamics**

와 밀접하게 연결됨을 이론적으로 설명.

### ⑤ Progressive Generation / Compression 해석

Diffusion generation을

> **coarse-to-fine progressive decoding**

과정으로 해석.

---

# 21. 한계

### 느린 Sampling

* \(T=1000\)
* 이미지 하나 생성 시 반복적인 denoising 필요
* GAN처럼 한 번의 forward pass로 생성하는 방식보다 느림

실제 논문의 256×256 모델에서는 **128개 이미지 sampling에 약 300초** 소요. 

### Likelihood 성능

* 이미지 sample quality는 뛰어남
* 하지만 likelihood 기반 생성모델과 비교하면 log-likelihood 성능은 경쟁력이 낮음 

---

# 22. 결론

* Diffusion Model을 이용한 **고품질 이미지 생성 가능성 입증**
* Forward Diffusion + Learned Reverse Denoising 구조
* **Gaussian Noise를 직접 예측하는 방식** 제안
* 단순한 **MSE 기반 \(L_{simple}\)** 학습
* Diffusion Model과 **Variational Inference / Denoising Score Matching / Langevin Dynamics** 연결
* Autoregressive Model 및 Progressive Compression과의 관계 제시
* 이후 Diffusion 기반 생성모델 발전의 핵심 기반이 된 구조

논문에서 최종적으로 강조하는 부분도 **Diffusion Model을 단순한 생성 기법이 아니라 여러 기존 생성모델·학습 방법과 연결되는 프레임워크로 보는 것**. 

### 한 줄 요약

> **이미지에 Noise를 점진적으로 추가하고, U-Net이 각 단계의 Noise를 예측하도록 학습하여 Random Noise에서 고품질 이미지를 복원하는 생성 모델.**
