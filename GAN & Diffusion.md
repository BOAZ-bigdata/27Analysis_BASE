# GAN & Diffusion Paper Review

## 0. 전체 개요

GAN과 Diffusion은 모두 **새로운 데이터를 생성하기 위한 생성 모델(Generative Model)**이지만, 데이터를 생성하는 방식이 완전히 다름.

GAN은 **Generator와 Discriminator가 서로 경쟁하면서 데이터 분포를 학습**하고, Diffusion은 **데이터에 노이즈를 넣는 과정을 정의한 뒤 그 노이즈를 역으로 제거하는 방법을 학습**함.

| 구분     | GAN                                     | Diffusion                                  |
| ------ | --------------------------------------- | ------------------------------------------ |
| 논문     | Generative Adversarial Nets             | Denoising Diffusion Probabilistic Models   |
| 핵심 문제  | 기존 생성 모델의 복잡한 확률계산·추론 문제                | Diffusion Model도 고품질 이미지를 생성할 수 있는가        |
| 핵심 해결법 | Generator와 Discriminator의 경쟁            | Noise 추가 → Noise 예측 → 반복적 제거               |
| 핵심 구성  | Generator + Discriminator               | Forward Process + Reverse Process          |
| 학습 목표  | 생성 분포 \(p_g\)를 실제 분포 \(p_{data}\)에 접근시킴 | 입력 이미지에 추가된 Noise \(\epsilon\)을 예측         |
| 생성 시작점 | Random Noise \(z\)                      | Gaussian Noise \(x_T\)                     |
| 생성 방식  | Generator 한 번의 Forward Pass             | 여러 단계의 Denoising                           |
| 대표 키워드 | Adversarial Learning, Minimax Game      | Denoising, Gaussian Noise, Reverse Process |

GAN은 Generator \(G\)와 Discriminator \(D\)를 동시에 학습시키는 **two-player minimax game**을 제안했으며, 이상적인 경우 생성 분포가 실제 데이터 분포와 같아지고 판별기는 모든 입력에 대해 \(1/2\)을 출력하게 됨. 

DDPM은 Diffusion Model에 **denoising score matching 및 Langevin dynamics와의 연결**을 도입하고, 단순화한 Noise Prediction Loss를 사용하여 CIFAR-10에서 FID 3.17을 기록함. 

### 전체 구조

#### GAN

```text
Random Noise z
      ↓
  Generator G
      ↓
   Fake Image
      ↓
┌───────────────┐
│ Discriminator │
└───────────────┘
   ↑         ↑
Fake       Real

Generator:
"진짜처럼 만들어야 함"

Discriminator:
"진짜와 가짜를 구별해야 함"

        ↓

서로 경쟁하며 학습

        ↓

생성 데이터 분포
        ≈
실제 데이터 분포
```

#### Diffusion

```text
실제 이미지 x0
      ↓
Noise 추가
      ↓
x1
      ↓
Noise 추가
      ↓
x2
      ↓
...
      ↓
xT ≈ Gaussian Noise
```

학습 후 생성 과정은 반대 방향.

```text
Gaussian Noise xT
      ↓
Noise 제거
      ↓
xT-1
      ↓
Noise 제거
      ↓
...
      ↓
생성 이미지 x0
```

---

# 1. GAN

## Generative Adversarial Nets

Goodfellow et al., 2014

---

## 1.1 어떤 문제를 해결하려 했는가? — Abstract

기존 Deep Generative Model은 **데이터가 어떤 확률분포에서 생성되는지를 학습하는 것**이 목표.

하지만 당시 생성 모델들은 학습 과정에서 복잡한 확률 계산이나 근사 추론, Markov Chain 등이 필요한 경우가 많았음.

GAN 논문은 이를 완전히 다른 방식으로 해결.

> **확률분포를 직접 계산하지 말고, 가짜 데이터를 만드는 모델과 그것을 판별하는 모델을 경쟁시키자.**

두 개의 모델을 사용.

```text
Generator G
=
가짜 데이터를 생성

Discriminator D
=
진짜 데이터와 가짜 데이터를 구별
```

Generator의 목표.

```text
Random Noise
     ↓
Generator
     ↓
Fake Image
     ↓
Discriminator를 속임
```

Discriminator의 목표.

```text
Real Image ──→ Real이라고 판정

Fake Image ──→ Fake라고 판정
```

둘을 동시에 학습시키면서 Generator가 실제 데이터 분포를 모방하도록 만드는 구조. 논문은 이를 **minimax two-player game**으로 정의함. 

### 핵심

**좋은 생성 모델을 직접 설계하는 대신, 좋은 판별기를 상대하게 만들어 생성 모델을 학습시킨다.**

---

# 1.2 연구 동기와 문제점 — Introduction

당시 Deep Learning은 이미지 분류와 같은 **Discriminative Model**에서 큰 성공을 거두고 있었음.

```text
이미지
 ↓
Neural Network
 ↓
고양이 / 강아지
```

하지만 Generative Model은 상대적으로 어려웠음.

논문이 지적한 핵심 문제는 크게 두 가지.

* Maximum Likelihood 기반 학습에서 복잡한 확률 계산 발생
* Generative Model에서 효율적인 Backpropagation 구조를 활용하기 어려움

저자들은 이러한 계산을 우회하는 새로운 생성 모델 학습 방법을 제안함. 

논문에서는 Generator와 Discriminator의 관계를 **위조지폐범과 경찰**에 비유함.

```text
Generator
=
위조지폐범

목표
→ 경찰이 구별하지 못하는 가짜 돈 제작
```

```text
Discriminator
=
경찰

목표
→ 진짜 돈과 위조지폐 구별
```

경찰의 감별 능력이 좋아지면 위조범도 더 정교하게 만들어야 함.

```text
G 향상
 ↓
D가 구별법 학습
 ↓
D 향상
 ↓
G가 더 진짜 같은 데이터 생성
 ↓
...
```

경쟁이 반복되면서 생성 데이터가 실제 데이터에 가까워지는 아이디어. 

---

# 1.3 관련 연구 동향 — Related Works

GAN 이전에도 다양한 Generative Model 존재.

| 방법             | 핵심 특징                                   | 주요 어려움                    |
| -------------- | --------------------------------------- | ------------------------- |
| RBM            | Undirected Graphical Model              | Partition Function 계산 문제  |
| DBM            | 여러 Hidden Layer 사용                      | MCMC 기반 학습 복잡             |
| DBN            | Directed + Undirected 구조                | 두 구조의 계산 문제를 모두 가짐        |
| Score Matching | Score를 이용한 분포 학습                        | 확률밀도 구조 필요                |
| NCE            | 실제 데이터와 Noise 구별                        | 고정 Noise Distribution 사용  |
| GSN            | Markov Chain을 학습                        | Sampling에 Markov Chain 필요 |
| VAE 계열         | Latent Variable + Approximate Inference | 근사 추론 필요                  |

특히 RBM·DBM과 같은 모델은 Partition Function이나 그 Gradient 계산이 어렵고 MCMC가 필요했음. GAN은 이와 달리 **Sampling 단계에서 Markov Chain을 필요로 하지 않는 구조**를 제안함. 

### GAN의 접근

```text
기존 방식

복잡한 확률분포 정의
        ↓
Likelihood 계산
        ↓
근사 추론 / MCMC
        ↓
모델 학습
```

GAN.

```text
Random Noise
     ↓
Generator
     ↓
Fake Data
     ↓
Discriminator
     ↓
Gradient
     ↓
Generator 개선
```

즉,

**확률밀도를 직접 계산하는 문제를 분류 문제로 전환한 것**이 중요한 차이.

---

# 1.4 연구 접근법과 모델 구조 — Method

GAN의 핵심 구성은 두 개.

### 1.4.1 Generator

Generator \(G\)는 Random Noise \(z\)를 받아 데이터 \(x\)를 생성.

$$
z \sim p_z(z)
$$

```text
Random Noise z
      ↓
 Generator G
      ↓
 Generated x
```

즉,

$$
x = G(z;\theta_g)
$$

Generator가 만들어내는 데이터 분포를

$$
p_g
$$

라고 표현.

Generator의 목표는

```text
p_g
 ↓
p_data에 최대한 가까워지기
```

---

## 1.4.2 Discriminator

Discriminator \(D(x)\)는 입력 데이터가 **실제 데이터에서 왔을 확률**을 출력.

```text
입력 x
  ↓
Discriminator
  ↓
0 ~ 1
```

예를 들어

```text
D(x) = 0.95

→ 진짜일 확률이 높다고 판단
```

반대로

```text
D(G(z)) = 0.03

→ Generator가 만든 가짜라고 판단
```

논문에서는 \(G\)와 \(D\) 모두 Multilayer Perceptron으로 구현하고 Backpropagation으로 학습할 수 있다고 설명함. 

---

# 1.5 GAN의 Adversarial Training

GAN의 핵심은 두 모델이 **반대 목적을 가지고 있다는 것**.

### Discriminator의 목표

```text
Real Image
 ↓
D(x) → 1

Fake Image
 ↓
D(G(z)) → 0
```

### Generator의 목표

```text
Fake Image
 ↓
D(G(z)) → 1
```

즉 Generator는 Discriminator를 속이려고 함.

```text
              Generator
                  │
              Fake Image
                  │
                  ▼
            Discriminator
             ↙          ↘
          Fake          Real
           ↑             ↑
        D의 목표       G의 목표
```

이 경쟁이 **Adversarial**의 의미.

---

# 1.6 주요 수식

## 1.6.1 GAN Minimax Objective

GAN의 핵심 수식.

$$
\min_G \max_D V(D,G)
=
\mathbb{E}_{x\sim p_{data}}
[\log D(x)]
+
\mathbb{E}_{z\sim p_z}
[\log(1-D(G(z)))]
$$



두 항을 나누어 보면.

### 첫 번째 항

$$
\mathbb{E}_{x\sim p_{data}}[\log D(x)]
$$

Discriminator 입장.

```text
Real Image x
     ↓
D(x)
     ↓
1에 가깝게 만들기
```

즉 실제 데이터를 Real이라고 맞히는 능력.

### 두 번째 항

$$
\mathbb{E}_{z\sim p_z}
[\log(1-D(G(z)))]
$$

Discriminator는

```text
Fake Image
 ↓
D(G(z))
 ↓
0
```

으로 만들려고 함.

그러나 Generator는 반대.

```text
Fake Image
 ↓
D(G(z))
 ↓
1
```

로 만들려고 함.

그래서

```text
Discriminator → V를 크게

Generator → V를 작게
```

만드는 Minimax Game.

---

# 1.7 Generator 학습 시 실제 사용되는 목적함수

이론적으로 Generator는

$$
\min_G \log(1-D(G(z)))
$$

를 학습.

그런데 학습 초반에는 Generator가 매우 못 만들기 때문에

```text
G(z)
 ↓
매우 어색한 Fake Image
 ↓
D가 너무 쉽게 구별
 ↓
D(G(z)) ≈ 0
```

이 상황에서는 Generator에 전달되는 Gradient가 약해질 수 있음.

따라서 논문에서는 실무적으로

$$
\max_G \log D(G(z))
$$

를 사용할 수 있다고 제안.

최종적으로 도달하려는 지점은 같으면서 초기 학습에서 더 강한 Gradient를 제공함. 

---

# 1.8 이론적으로 GAN이 성공하면 어떻게 되는가?

고정된 Generator에 대한 최적의 Discriminator는

$$
D_G^*(x)
=
\frac{p_{data}(x)}
{p_{data}(x)+p_g(x)}
$$



만약 Generator가 완벽하게 실제 데이터 분포를 학습하면

$$
p_g = p_{data}
$$

따라서

$$
D^*(x)
=
\frac{p_{data}}
{p_{data}+p_{data}}
=
\frac{1}{2}
$$

즉,

```text
Real
  ↓
D = 0.5

Fake
  ↓
D = 0.5
```

Discriminator가 더 이상 구별할 수 없음.

논문의 Figure 1도 이 과정을 보여줌. 학습이 진행되면서 생성 분포 \(p_g\)가 실제 분포 \(p_{data}\)에 접근하고 최종적으로 \(D(x)=1/2\)가 됨. 

---

# 1.9 Jensen-Shannon Divergence와 GAN

Generator의 최종 Objective를 정리하면

$$
C(G)
=
-\log 4
+
2\cdot JSD
(p_{data}\parallel p_g)
$$

가 됨.



JSD는 두 확률분포의 차이를 나타내는 값.

```text
p_data      p_g

다름
 ↓
JSD 큼

비슷함
 ↓
JSD 작음

완전히 동일
 ↓
JSD = 0
```

따라서 GAN 학습의 이론적 목표는 결국

$$
p_g \rightarrow p_{data}
$$

이며, 최적점에서

$$
p_g=p_{data}
$$

가 됨.

---

# 1.10 실험 구성과 결과 — Experiments

논문에서는 세 데이터셋 사용.

* MNIST
* Toronto Face Database
* CIFAR-10

Generator에서는 Rectifier Linear Activation과 Sigmoid를 사용했고, Discriminator에서는 Maxout과 Dropout을 사용함. 

### Quantitative Evaluation

논문은 생성 샘플에 Gaussian Parzen Window를 적용하여 Test Set의 Log-Likelihood를 추정.

| Model            |       MNIST |           TFD |
| ---------------- | ----------: | ------------: |
| DBN              |     138 ± 2 |     1909 ± 66 |
| Stacked CAE      |   121 ± 1.6 | **2110 ± 50** |
| Deep GSN         |   214 ± 1.1 |     1890 ± 29 |
| Adversarial Nets | **225 ± 2** |     2057 ± 26 |

GAN은 MNIST에서 비교 모델 중 가장 높은 값을 기록했지만 TFD에서는 Stacked CAE보다 낮음. 저자들도 Parzen Window 기반 평가가 고차원 공간에서 한계가 있고 분산이 높다고 지적함. 

따라서 이 논문의 실험 결과를

> "GAN이 모든 기존 생성 모델보다 압도적으로 성능이 좋았다."

라고 해석하면 과함.

정확하게는

> **새로운 Adversarial Framework가 실제로 학습 가능하고 경쟁력 있는 생성 결과를 만들 수 있음을 보였다.**

정도가 적절함.

---

## 1.11 생성 이미지 실험

논문 Figure 2에서는 MNIST, TFD, CIFAR-10 생성 샘플을 제시함.

특히 생성 이미지 오른쪽에 **가장 가까운 Training Example**을 함께 표시하여 모델이 Training Data를 그대로 복사한 것이 아니라는 점을 확인하려 함. 

또 Figure 3에서는 Latent Variable \(z\) 사이를 선형 보간했을 때 숫자가 부드럽게 변화하는 모습을 보여줌. 

```text
z1 ─────────────── z2

 ↓      ↓      ↓

숫자 모양이 점진적으로 변화
```

즉 Latent Space가 단순히 Training Image를 저장하는 공간이 아니라 일정한 구조를 학습하고 있음을 보여주는 보조적인 결과.

---

# 1.12 무엇을 발견했고 한계점은 무엇인가? — Discussion

원 논문에는 별도의 `Discussion`이라는 제목 대신 **Advantages and disadvantages**에서 장단점을 직접 정리함.

### 주요 발견

#### 1. Markov Chain 없이 Sampling 가능

기존 일부 생성 모델과 달리 생성할 때 Markov Chain이 필요하지 않음.

```text
Noise z
 ↓
Generator
 ↓
Sample
```

Forward Propagation으로 생성 가능.

---

#### 2. Backpropagation만으로 학습 가능

Generator와 Discriminator 모두 Differentiable Function으로 구성할 수 있어 기존 Neural Network Training 방식을 활용 가능.

---

#### 3. Explicit Probability Density가 없어도 됨

Generator가 직접

$$
p_g(x)
$$

를 계산하지 않아도 Sample 생성 가능.

---

### 한계점

#### 1. \(p_g(x)\)를 명시적으로 계산할 수 없음

논문에서도 GAN의 주요 단점으로

> 생성 분포 \(p_g(x)\)에 대한 explicit representation이 없음

을 지적함. 

---

#### 2. Generator와 Discriminator의 균형 필요

```text
G 너무 강함
      ↓
D가 따라가지 못함

D 너무 강함
      ↓
G가 충분한 Gradient를 받기 어려움
```

따라서 두 네트워크의 학습을 잘 조율해야 함.

---

#### 3. 생성 다양성 감소 문제

논문은 Generator를 너무 많이 업데이트하면 여러 \(z\)가 동일하거나 비슷한 \(x\)로 매핑되는 **“Helvetica scenario”**가 나타날 수 있다고 설명함. 

```text
z1 ─┐
z2 ─┤
z3 ─┼─→ 비슷한 x
z4 ─┤
z5 ─┘
```

즉 다양한 Noise를 넣어도 충분히 다양한 데이터를 생성하지 못하는 문제가 생길 수 있음.

---

# 1.13 결론 및 주요 요약 — Conclusion

GAN 논문의 핵심 결론은 **Adversarial Training이라는 새로운 Generative Modeling Framework가 실제로 가능하다는 것**.

```text
Generator
     ↕ 경쟁
Discriminator
     ↓
데이터 분포 학습
```

논문은 이후 확장 방향으로

* Conditional Generative Model
* Approximate Inference
* Conditional Distribution Modeling
* Semi-supervised Learning
* Generator와 Discriminator 학습 효율 개선

등을 제안함.  

---

# 1.14 기존 연구 대비 차별점

| 기존 생성 모델                    | GAN                                     |
| --------------------------- | --------------------------------------- |
| Likelihood 중심               | Adversarial Objective                   |
| 확률분포 계산 필요                  | Explicit Density 계산 불필요                 |
| Approximate Inference 필요 가능 | 학습 시 별도 Inference 불필요                   |
| MCMC Sampling 필요 가능         | Generator Forward Pass로 Sampling        |
| 하나의 생성 모델                   | Generator + Discriminator               |
| 데이터를 직접 기준으로 생성 모델 학습       | Discriminator Gradient를 통해 Generator 학습 |

가장 중요한 차별점은

> **생성 문제를 Generator와 Discriminator 사이의 게임으로 재정의한 것**

---

# 1.15 핵심 아이디어

### 기존 방식

```text
Real Data
   ↓
확률분포 직접 모델링
   ↓
Likelihood
   ↓
Generative Model
```

### GAN

```text
         Real Data
             ↓
        ┌─────────┐
        │    D    │
        └─────────┘
             ↑
        Fake Data
             ↑
             G
             ↑
        Random Noise
```

### 한 문장 요약

**가짜 데이터를 만드는 Generator와 이를 구별하는 Discriminator를 경쟁시켜 실제 데이터와 유사한 생성 분포를 학습하는 방법.**

---

# 1.16 코드 관점의 이해

논문 Algorithm 1에서는 Discriminator를 \(k\)회 학습한 후 Generator를 한 번 학습하며, 실제 실험에서는 \(k=1\)을 사용함. 

논문 구조를 단순화한 개념 코드.

```python
for real_images in dataloader:

    # 1. random noise
    z = sample_noise()

    # 2. Generator
    fake_images = generator(z)

    # 3. Discriminator training
    real_score = discriminator(real_images)
    fake_score = discriminator(fake_images.detach())

    d_loss = -(
        log(real_score)
        + log(1 - fake_score)
    )

    update(discriminator, d_loss)

    # 4. Generator training
    fake_images = generator(z)
    fake_score = discriminator(fake_images)

    g_loss = -log(fake_score)

    update(generator, g_loss)
```

핵심 흐름.

```text
sample_noise()
      ↓
Generator
      ↓
Fake Image
      ↓
Discriminator

────────────────

Real + Fake
      ↓
D 학습

────────────────

Fake
 ↓
D를 속이도록
G 학습
```

위 코드는 논문 원본 구현을 복사한 것이 아니라 **Algorithm 1의 학습 원리를 이해하기 위한 단순화 코드**.

---

---

# 2. Diffusion

## Denoising Diffusion Probabilistic Models

Ho, Jain, Abbeel, 2020

---

# 2.1 어떤 문제를 해결하려 했는가? — Abstract

Diffusion Probabilistic Model이라는 아이디어 자체는 이미 존재했음.

핵심 구조.

```text
Real Image
   ↓
조금씩 Noise 추가
   ↓
Gaussian Noise
```

그리고 반대로

```text
Gaussian Noise
   ↓
조금씩 Noise 제거
   ↓
Realistic Image
```

하지만 당시 Diffusion Model은 **고품질 이미지 생성 모델로서의 성능이 충분히 입증되지 않은 상태**였음.

이 논문이 해결하려 한 핵심 문제.

> **Diffusion Probabilistic Model을 실제 고품질 이미지 생성 모델로 만들 수 있는가?**

저자들은 Diffusion Model과

* Denoising Score Matching
* Langevin Dynamics
* Variational Inference

사이의 연결을 이용하여 새로운 학습 방법을 구성함.

결과적으로 CIFAR-10에서

* Inception Score = **9.46**
* FID = **3.17**

을 기록함. 

---

# 2.2 연구 동기와 문제점 — Introduction

당시 대표적인 Generative Model.

```text
GAN
VAE
Flow
Autoregressive Model
Energy-Based Model
Score Matching
```

이미 여러 방법에서 고품질 이미지 생성 결과가 나오고 있었음. 

Diffusion Model의 아이디어는 단순함.

```text
데이터
 ↓
작은 Gaussian Noise
 ↓
조금 더 Noise
 ↓
조금 더 Noise
 ↓
...
 ↓
거의 완전한 Noise
```

그리고 이를 뒤집어

```text
Noise
 ↓
조금 복원
 ↓
조금 더 복원
 ↓
...
 ↓
Data
```

하면 새로운 이미지를 생성할 수 있음.

논문의 중요한 기여는 단순히 Diffusion을 사용했다는 것이 아니라,

> **Reverse Process를 Noise Prediction 문제로 바꾸고 간단한 Objective로 학습하면 높은 Sample Quality를 얻을 수 있다는 것을 보여준 것**

---

# 2.3 관련 연구 동향 — Related Works

DDPM은 여러 생성 모델 및 확률 모델링 연구와 연결됨.

### 1. VAE

```text
Data
 ↓
Latent Variable
 ↓
Data Reconstruction
```

Diffusion 역시 Latent Variable Model로 볼 수 있음.

다만 여러 단계의 Latent Variable

$$
x_1,x_2,\dots,x_T
$$

를 사용하는 구조.

---

### 2. Flow

Diffusion도 데이터를 다른 분포로 변환한다는 점에서 Flow와 유사한 면이 있음.

하지만 Diffusion에서는 Forward Process가 **학습 대상이 아니라 미리 정해진 과정**이라는 차이가 있음.

---

### 3. Score Matching

이 논문의 중요한 연결점.

Noise가 섞인 데이터에서 원래 데이터 방향을 찾아가는 과정이 **Score Matching**과 연결됨.

---

### 4. Langevin Dynamics

Reverse Sampling 과정이 Langevin Dynamics와 비슷한 형태를 가짐.

특히 논문은 \(\epsilon\)-prediction Parameterization이 여러 Noise Level에서의 Denoising Score Matching과 연결됨을 보임. 

### 핵심 차이

논문은 기존 NCSN 등과 달리 **Sampler 자체를 Variational Inference를 이용한 Latent Variable Model로 직접 학습**한다고 강조함. 

---

# 2.4 연구 접근법과 모델 구조 — Method

Diffusion에는 크게 두 과정 존재.

```text
Forward Process
+
Reverse Process
```

---

# 2.5 Forward Process

Forward Process는 **실제 데이터에 조금씩 Gaussian Noise를 추가하는 과정**.

```text
x0
 ↓
x1
 ↓
x2
 ↓
x3
 ↓
...
 ↓
xT
```

여기서

$$
x_0
$$

는 실제 이미지.

$$
x_T
$$

는 거의 Gaussian Noise.

Forward Process는

$$
q(x_t|x_{t-1})
$$

로 표현됨.

직관적으로

```text
이전 이미지
   +
조금의 Gaussian Noise
   ↓
다음 단계 이미지
```

---

## 2.5.1 한 번에 원하는 timestep 만들기

Diffusion의 중요한 특징은

```text
x0 → x1 → x2 → ... → xt
```

를 모두 순서대로 계산하지 않아도

$$
x_0
$$

에서 바로

$$
x_t
$$

를 만들 수 있다는 것.

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

$$
\epsilon \sim \mathcal{N}(0,I)
$$

이 관계는 이후 Noise Prediction Training의 핵심이 됨. 

### 직관적 의미

```text
xt

=

원본 이미지 일부
+
Gaussian Noise 일부
```

초반 timestep.

```text
원본 비율 ↑
Noise 비율 ↓
```

후반 timestep.

```text
원본 비율 ↓
Noise 비율 ↑
```

---

# 2.6 Reverse Process

생성할 때는 Forward Process의 반대 방향으로 진행.

```text
xT
 ↓
xT-1
 ↓
xT-2
 ↓
...
 ↓
x0
```

Reverse Process는

$$
p_\theta(x_{t-1}|x_t)
$$

를 학습.

즉 모델이 해결해야 하는 문제는

> **Noise가 섞인 \(x_t\)를 보고 한 단계 전의 더 깨끗한 \(x_{t-1}\)를 만들어내는 것**

---

# 2.7 핵심 아이디어: Noise 자체를 예측

Reverse Process의 평균을 직접 예측할 수도 있지만 저자들은 다른 Parameterization을 사용.

Network가

$$
\epsilon_\theta(x_t,t)
$$

를 출력하도록 함.

의미.

```text
현재 이미지 xt
      +
timestep t
      ↓
Neural Network
      ↓
이 이미지에 포함된 Noise 예측
```

즉 모델에게

> "원본 이미지를 바로 만들어."

라고 하는 대신

> **"지금 이미지에 들어간 Noise가 무엇인지 맞혀."**

라고 학습시키는 것.

논문은 이 \(\epsilon\)-prediction을 통해 Reverse Process를 표현하며, 이것이 Langevin Dynamics 및 Denoising Score Matching과 연결된다고 설명함. 

---

# 2.8 Diffusion Model 구조

논문의 Reverse Process Neural Network는 **U-Net**을 기반으로 함.

추가적으로

* Group Normalization
* Sinusoidal Timestep Embedding
* Self-Attention

을 사용.



전체 구조를 단순화하면.

```text
Noisy Image xt
      +
Timestep t
      ↓
    U-Net
      ↓
Predicted Noise εθ
```

그리고 예측된 Noise를 이용하여

```text
xt
 ↓
Noise 제거
 ↓
xt-1
```

을 반복.

---

# 2.9 핵심 학습 Loss

DDPM 논문에서 가장 중요한 수식 중 하나.

$$
L_{\text{simple}}
=
\mathbb{E}_{t,x_0,\epsilon}
\left[
\left\|
\epsilon-
\epsilon_\theta
(
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
t
)
\right\|^2
\right]
$$



복잡해 보이지만 실제 의미는 매우 단순함.

먼저 Noise 생성.

$$
\epsilon
\sim
\mathcal N(0,I)
$$

그 Noise를 실제 이미지에 넣음.

```text
x0
 +
ε
 ↓
xt
```

Network에 \(x_t\) 입력.

```text
xt
 ↓
U-Net
 ↓
ε_pred
```

그리고

```text
실제 넣은 Noise ε
        vs
모델이 예측한 Noise ε_pred
```

를 비교.

즉

$$
Loss
=
\|\epsilon-\epsilon_{pred}\|^2
$$

### 핵심

**Diffusion Training은 복잡해 보이지만 실제 구현 관점에서는 Noise Prediction MSE 문제로 단순화할 수 있음.**

---

# 2.10 Diffusion 학습 과정

논문 Algorithm 1을 쉽게 풀면 다음과 같음. 

### Step 1

실제 이미지 선택.

```text
x0
```

### Step 2

Random Timestep 선택.

```text
t ∈ {1, ..., T}
```

### Step 3

Gaussian Noise 생성.

```text
ε ~ N(0, I)
```

### Step 4

Noisy Image 생성.

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

### Step 5

Network가 Noise 예측.

```text
xt + t
 ↓
U-Net
 ↓
ε_pred
```

### Step 6

Loss 계산.

$$
Loss
=
\|\epsilon-\epsilon_{pred}\|^2
$$

전체.

```text
Real Image x0
      ↓
Random t 선택
      ↓
Random Noise ε
      ↓
xt 생성
      ↓
U-Net
      ↓
Noise ε 예측
      ↓
True Noise와 비교
      ↓
MSE Loss
```

---

# 2.11 Diffusion Sampling 과정

학습이 끝나면 실제 이미지가 필요 없음.

시작은 순수 Gaussian Noise.

$$
x_T\sim\mathcal N(0,I)
$$

그리고

```text
xT
 ↓
Model이 Noise 예측
 ↓
Noise 일부 제거
 ↓
xT-1
 ↓
Noise 예측
 ↓
Noise 제거
 ↓
...
 ↓
x0
```

논문 Algorithm 2도 이와 같은 반복적 Reverse Sampling 구조를 사용함. 

---

# 2.12 실험 구성과 결과 — Experiments

논문에서는 모든 실험에서

$$
T=1000
$$

을 사용.

Noise Variance는

$$
\beta_1=10^{-4}
$$

에서

$$
\beta_T=0.02
$$

까지 선형적으로 증가하도록 설정함. 

주요 데이터.

* CIFAR-10
* CelebA-HQ
* LSUN Bedroom
* LSUN Church
* LSUN Cat

---

## 2.12.1 CIFAR-10 결과

주요 결과.

| Model                     | Inception Score |      FID |
| ------------------------- | --------------: | -------: |
| NCSN                      |            8.87 |    25.32 |
| SNGAN                     |            8.22 |     21.7 |
| SNGAN-DDLS                |            9.09 |    15.42 |
| DDPM — \(L\)              |            7.67 |    13.51 |
| **DDPM — \(L_{simple}\)** |        **9.46** | **3.17** |

논문의 \(L_{simple}\) 모델은 Unconditional CIFAR-10에서 FID 3.17을 기록함. 

중요한 점은 단순히

> "Diffusion이 GAN을 이겼다."

는 결론이 아님.

이 논문이 직접 입증한 것은

> **이전까지 고품질 생성 모델로 크게 주목받지 못했던 Diffusion Model도 적절한 Parameterization과 Objective를 사용하면 매우 높은 Sample Quality를 얻을 수 있다.**

는 것.

---

# 2.13 Ablation Study

논문에서는 Reverse Process의 Parameterization도 비교.

대표적으로

```text
Mean μ 예측
vs
Noise ε 예측
```

을 비교.

결과적으로

$$
\epsilon
$$

을 예측하면서

$$
L_{simple}
$$

을 사용하는 조합이 가장 좋은 Sample Quality를 기록.

또 Reverse Variance를 학습하도록 했을 때는 Training이 불안정하고 Sample Quality가 악화되는 결과도 나타남. 

즉 단순히 Diffusion이라는 구조가 중요했던 것이 아니라,

> **무엇을 Network가 예측하게 할 것인가 + 어떤 Loss로 학습할 것인가**

가 성능을 크게 좌우함.

---

# 2.14 Progressive Generation

Diffusion의 흥미로운 특징.

Denoising 과정을 관찰하면 이미지가 한 번에 만들어지는 것이 아님.

```text
Noise
 ↓
큰 구조
 ↓
형태
 ↓
세부 특징
 ↓
Fine Detail
```

논문의 Progressive Generation 실험에서는 **Large-scale Image Feature가 먼저 나타나고 세부적인 Detail이 나중에 생성되는 현상**을 확인함. 

즉 Diffusion은 이미지 정보를

```text
Coarse
 ↓
Fine
```

방향으로 복원하는 성질을 보임.

---

# 2.15 무엇을 발견했고 한계점은 무엇인가? — Discussion

원 논문에 독립된 `Discussion` 장은 없으므로 이 부분은 **Experiments·Related Work·Conclusion 내용을 기준으로 재구성한 것**.

### 주요 발견

#### 1. Diffusion Model도 고품질 생성 가능

가장 중요한 결과.

기존 Diffusion Model은 구조가 단순하지만 고품질 Sample Generation 가능성이 충분히 입증되지 않았음.

DDPM은 CIFAR-10 FID 3.17을 기록하면서 가능성을 보여줌.

---

#### 2. Noise Prediction이 효과적

직접 Image나 Reverse Mean을 예측하기보다

$$
\epsilon
$$

을 예측하도록 하는 Parameterization이 효과적.

특히

$$
L_{simple}
$$

과 결합했을 때 가장 좋은 Sample Quality를 보임.

---

#### 3. Sample Quality와 Likelihood가 항상 같이 좋아지는 것은 아님

흥미로운 실험 결과.

논문은 원래 Variational Bound를 직접 최적화하면 더 좋은 Codelength를 얻지만, \(L_{simple}\)을 사용했을 때 Sample Quality가 더 좋았다고 보고함. 

즉

```text
Likelihood 성능 향상
≠
항상 이미지 품질 향상
```

---

#### 4. Progressive Representation 존재

Diffusion 과정에서

```text
초기 Reverse Step
→ 큰 구조 결정

후기 Reverse Step
→ 세부 정보 결정
```

과 같은 Coarse-to-Fine 특성이 나타남.

---

### 한계점

#### 1. Sampling Step이 많음

논문의 기본 설정은

$$
T=1000
$$

이므로 하나의 이미지를 생성하기 위해 Reverse Network를 매우 많이 실행해야 함.

실제 논문의 구현 정보에서도 CIFAR-10 256장 Sampling에 약 17초, 256×256 모델의 128장 Sampling에는 약 300초가 걸렸다고 보고함. 

따라서 당시 DDPM의 명확한 실용적 부담은 **반복적 Sampling 비용**.

---

#### 2. Likelihood 측면에서는 최고 성능이 아님

논문은 Diffusion Model의 Lossless Codelength가 다른 Likelihood 기반 생성 모델과 비교해 경쟁력이 부족하다고 직접 밝힘. 

따라서

```text
Sample Quality
=
매우 우수

Likelihood Modeling
=
항상 최고는 아님
```

으로 구분해야 함.

---

#### 3. Progressive Compression은 실제 압축 시스템이 아님

논문에서 제안한 Progressive Lossy Compression 해석은 흥미로운 분석이지만, 저자 스스로 이를 **Proof of Concept**으로 설명하며 실제 고차원 데이터에 바로 사용할 수 있는 Practical Compression System은 아니라고 밝힘. 

---

# 2.16 결론 및 주요 요약 — Conclusion

논문의 최종 결론.

Diffusion Model을 이용해 **고품질 이미지 생성이 가능함을 보였으며**, Diffusion Model이 다음 개념들과 연결됨을 확인.

```text
Diffusion Model
     │
     ├─ Variational Inference
     │
     ├─ Denoising Score Matching
     │
     ├─ Langevin Dynamics
     │
     ├─ Energy-Based Model
     │
     ├─ Autoregressive Model
     │
     └─ Progressive Lossy Compression
```



논문의 의의는 단순히 새로운 이미지 생성기를 하나 만든 것이 아니라,

> **Diffusion이라는 기존 생성 모델을 고품질 생성 모델로 발전시키고 여러 Generative Modeling 방법론 사이의 연결을 정리한 것**

---

# 2.17 기존 연구 대비 차별점

| 기존 접근                               | DDPM                                         |
| ----------------------------------- | -------------------------------------------- |
| Diffusion Model의 낮은 Sample Quality  | 고품질 Image Generation 입증                      |
| Reverse Mean 직접 Modeling 가능         | Noise \(\epsilon\) Prediction                |
| Standard Variational Bound          | \(L_{simple}\) Objective                     |
| Diffusion과 Score Model이 별개로 보일 수 있음 | Denoising Score Matching과 연결 제시              |
| Sampling 과정 중심 해석                   | Progressive Compression·Autoregressive 관점 추가 |
| 일반적인 Neural Network                 | U-Net + Timestep Embedding + Attention       |

가장 중요한 차별점은

> **Diffusion Model의 Reverse Process를 Noise Prediction 문제로 단순화하여 고품질 생성이 가능하다는 것을 실험적으로 보여준 것**

---

# 2.18 핵심 아이디어

### 학습

```text
Real Image x0
       ↓
Random timestep t
       ↓
Gaussian Noise ε
       ↓
Noisy Image xt
       ↓
     U-Net
       ↓
Predicted Noise εθ
       ↓
True Noise ε와 비교
```

### 생성

```text
Gaussian Noise
       ↓
Noise Prediction
       ↓
Denoising
       ↓
Noise Prediction
       ↓
Denoising
       ↓
...
       ↓
Generated Image
```

### 한 문장 요약

**실제 이미지에 Noise를 추가하는 과정의 역과정을 Neural Network가 학습하도록 하여 순수 Noise에서 이미지를 생성하는 방법.**

---

# 2.19 주요 수식 정리

## 수식 1. Noisy Image 생성

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

여기서

| 기호                 | 의미                           |
| ------------------ | ---------------------------- |
| \(x_0\)            | 원본 이미지                       |
| \(x_t\)            | timestep \(t\)의 noisy image  |
| \(\epsilon\)       | Gaussian Noise               |
| \(\bar{\alpha}_t\) | 현재 timestep에 남아 있는 원본 신호의 정도 |

직관적으로

```text
Noisy Image
=
Original Signal
+
Noise
```

---

## 수식 2. Network

$$
\epsilon_\theta(x_t,t)
$$

의미.

```text
Noisy Image xt
+
현재 timestep t
     ↓
Neural Network
     ↓
추정 Noise
```

---

## 수식 3. Training Objective

$$
L_{simple}
=
\mathbb E
\left[
\|
\epsilon-
\epsilon_\theta(x_t,t)
\|^2
\right]
$$

즉 사실상

```text
실제 Noise
    -
예측 Noise
    ↓
MSE
```

라고 이해하면 됨.

---

# 2.20 코드 관점의 이해

## Training

논문 Algorithm 1을 단순화한 코드.

```python
for x0 in dataloader:

    # random timestep
    t = random_timestep()

    # Gaussian noise
    noise = torch.randn_like(x0)

    # create noisy image
    xt = (
        sqrt(alpha_bar[t]) * x0
        + sqrt(1 - alpha_bar[t]) * noise
    )

    # predict noise
    predicted_noise = model(xt, t)

    # noise prediction loss
    loss = mse(
        predicted_noise,
        noise
    )

    update(model, loss)
```

핵심.

```text
x0
 ↓
add_noise()
 ↓
xt
 ↓
model()
 ↓
noise_pred
 ↓
MSE
```

---

## Sampling

```python
x = torch.randn(image_shape)

for t in reversed(range(T)):

    predicted_noise = model(x, t)

    x = denoise(
        x,
        predicted_noise,
        t
    )

generated_image = x
```

흐름.

```text
randn()
 ↓
xT
 ↓
model()
 ↓
denoise()
 ↓
xT-1
 ↓
model()
 ↓
denoise()
 ↓
...
 ↓
x0
```

이 역시 논문의 실제 Source Code를 그대로 옮긴 것이 아니라 **Algorithm 1·2를 이해하기 위해 단순화한 개념 코드**임. 논문 자체도 Training에서는 무작위 \(t\)와 Noise를 뽑아 Noise Prediction Error를 최소화하고, Sampling에서는 \(x_T\sim N(0,I)\)에서 시작해 \(t=T,\dots,1\) 순으로 Reverse Process를 수행함. 

---

# 3. GAN vs Diffusion 최종 비교

| 구분               | GAN                                         | Diffusion                                   |
| ---------------- | ------------------------------------------- | ------------------------------------------- |
| 기본 아이디어          | 두 Network의 경쟁                               | Noise를 넣고 다시 제거                             |
| 입력               | Random Noise \(z\)                          | Gaussian Noise \(x_T\)                      |
| 핵심 Network       | Generator + Discriminator                   | Denoising Network                           |
| 학습 문제            | Minimax Game                                | Noise Prediction                            |
| 핵심 Loss          | Adversarial Loss                            | Noise MSE                                   |
| 생성 과정            | Generator Forward Pass                      | Iterative Denoising                         |
| 장점               | 직접적이고 빠른 Sampling                           | 높은 Sample Quality, 안정적인 Denoising Objective |
| 논문이 직접 지적한 주요 한계 | G-D Synchronization, Explicit \(p_g(x)\) 부재 | Likelihood 경쟁력 한계, 많은 Reverse Step          |
| 이론적 목표           | \(p_g=p_{data}\)                            | Reverse Process가 Forward Diffusion을 역으로 복원  |
| 핵심 직관            | **속이면서 학습**                                 | **노이즈를 지우면서 학습**                            |

가장 쉽게 기억하면 다음 두 문장으로 정리할 수 있음.

```text
GAN
=
"진짜처럼 만들어서 판별기를 속여라."
```

```text
Diffusion
=
"이미지에 들어간 Noise를 맞히고,
그 Noise를 반복해서 제거하라."
```
