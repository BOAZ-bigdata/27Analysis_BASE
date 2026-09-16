# GAN & Diffusion 논문 정리

## 정리한 논문

### GAN

- **논문명:** *Generative Adversarial Nets*
- **저자:** Ian J. Goodfellow 외 7명
- **발표:** NeurIPS 2014
- **링크:** [arXiv 1406.2661](https://arxiv.org/abs/1406.2661)

### Diffusion

- **논문명:** *Denoising Diffusion Probabilistic Models*
- **저자:** Jonathan Ho, Ajay Jain, Pieter Abbeel
- **발표:** NeurIPS 2020
- **링크:** [arXiv 2006.11239](https://arxiv.org/abs/2006.11239)

---

# 1. Generative Adversarial Nets

## 1.1 연구 배경

기존의 깊은 생성 모델은 확률분포를 계산하거나 잠재변수를 추론하는 과정이 복잡했다. 특히 Boltzmann machine 계열은 Markov Chain을 이용한 근사가 필요했고, 다른 확률 모델들도 학습 과정에서 다루기 어려운 확률 계산이 발생했다.

GAN 논문은 이러한 계산을 피하고, **두 신경망을 경쟁시키는 방식으로 생성 모델을 학습하는 새로운 프레임워크**를 제안한다.

논문이 제시한 핵심 비유는 다음과 같다.

- Generator는 가짜 지폐를 만드는 위조범임.
- Discriminator는 진짜와 가짜를 구분하는 경찰임.
- 두 모델이 경쟁할수록 Generator는 더 진짜 같은 데이터를 만들고, Discriminator는 더 정교하게 가짜를 판별하게 됨.

## 1.2 모델 구성

### Generator `G`

Generator는 사전에 정한 noise distribution에서 무작위 벡터 `z`를 뽑아 데이터 공간의 샘플 `G(z)`로 변환한다.

```text
z ~ p_z(z)
z -> G(z) -> generated sample
```

Generator가 만든 샘플들의 분포를 `p_g`라고 한다. 학습 목표는 `p_g`가 실제 데이터 분포 `p_data`와 같아지는 것이다.

### Discriminator `D`

Discriminator는 입력 `x`가 실제 학습 데이터에서 나온 확률을 하나의 값으로 출력한다.

```text
D(x) ≈ 1: 실제 데이터라고 판단
D(x) ≈ 0: Generator가 만든 데이터라고 판단
```

Generator와 Discriminator는 모두 multilayer perceptron으로 구성되며 역전파로 학습된다.

## 1.3 Adversarial Objective

논문의 Equation 1은 두 모델의 목표를 하나의 minimax game으로 표현한다.

```text
min_G max_D V(D, G)

V(D, G)
= E_x[log D(x)]
  + E_z[log(1 - D(G(z)))]
```

각 항의 의미는 다음과 같다.

| 항 | 의미 |
|---|---|
| `log D(x)` | 실제 데이터를 실제라고 판단하도록 D를 학습 |
| `log(1 - D(G(z)))` | 생성 데이터를 가짜라고 판단하도록 D를 학습 |
| `max_D` | D는 두 항을 모두 크게 만들어 분류 성능을 높임 |
| `min_G` | G는 두 번째 항을 줄여 D가 생성 데이터를 구분하지 못하게 만듦 |

즉, Discriminator는 목적함수를 최대화하고 Generator는 최소화한다. 한 모델의 목표가 다른 모델의 목표와 반대이기 때문에 adversarial training이라고 부른다.

## 1.4 실제 학습 방법

논문의 Algorithm 1에서는 D와 G를 번갈아 학습한다.

```text
1. 실제 데이터 x를 minibatch로 추출함.
2. noise z를 추출하고 G(z)를 생성함.
3. 실제 x는 1, 생성된 G(z)는 0으로 구분하도록 D를 업데이트함.
4. 새로운 z를 추출함.
5. D가 G(z)를 실제라고 판단하도록 G를 업데이트함.
6. 이 과정을 반복함.
```

원칙적으로 D를 `k`번 업데이트한 뒤 G를 한 번 업데이트한다. 논문의 실험에서는 계산 비용이 가장 적은 `k = 1`을 사용했다.

### Generator 목적함수의 수정

학습 초기에는 G가 만든 결과가 실제 데이터와 크게 다르기 때문에 D가 가짜를 쉽게 구분한다. 이때 `log(1 - D(G(z)))`는 포화되어 G에 전달되는 gradient가 매우 작아질 수 있다.

논문은 이를 해결하기 위해 실제 구현에서 다음 목적을 사용할 수 있다고 설명한다.

```text
log(1 - D(G(z)))를 최소화하는 대신
log D(G(z))를 최대화함.
```

두 방식은 같은 최적점을 가지지만, 수정된 방식이 학습 초기에 더 강한 gradient를 제공한다.

## 1.5 이론적 결과

### 고정된 G에서 최적의 D

논문의 Proposition 1에 따르면 G가 고정되었을 때 최적의 Discriminator는 다음과 같다.

```text
D*(x) = p_data(x) / (p_data(x) + p_g(x))
```

어떤 위치에서 실제 데이터의 밀도가 높으면 D는 1에 가까운 값을 출력하고, 생성 데이터의 밀도가 높으면 0에 가까운 값을 출력한다.

### 전체 게임의 최적점

최적의 D를 목적함수에 대입하면 Generator의 기준은 다음과 같이 정리된다.

```text
C(G) = -log(4) + 2 × JSD(p_data || p_g)
```

Jensen-Shannon Divergence는 두 분포가 같을 때 0이 된다. 따라서 Theorem 1에서 제시한 전역 최적점은 다음과 같다.

```text
p_g = p_data
D(x) = 1/2
```

이 지점에서는 실제 데이터와 생성 데이터를 구분할 수 없으므로 D가 모든 입력에 1/2을 출력한다.

단, 이 결과는 다음 조건을 가정한 이론적 결과이다.

- G와 D의 표현력이 충분함.
- 각 단계에서 D가 현재 G에 대한 최적점까지 학습됨.
- 신경망 파라미터가 아니라 확률분포 공간에서 수렴을 분석함.

실제 신경망에서는 D를 매번 완전히 학습할 수 없고 목적함수도 비볼록이므로, 논문의 증명이 실제 학습의 안정적인 수렴까지 보장하지는 않는다.

## 1.6 실험

### 실험 설정

| 항목 | 내용 |
|---|---|
| 데이터셋 | MNIST, Toronto Face Database, CIFAR-10 |
| Generator | ReLU와 Sigmoid를 사용한 네트워크 |
| Discriminator | Maxout activation과 Dropout을 사용한 네트워크 |
| 학습 | D와 G를 번갈아 역전파로 업데이트 |
| 정량 평가 | Gaussian Parzen Window로 test log-likelihood 추정 |

### 정량 결과

논문의 Table 1은 다음 결과를 보고한다.

| 모델 | MNIST | TFD |
|---|---:|---:|
| DBN | 138 ± 2 | 1909 ± 66 |
| Stacked CAE | 121 ± 1.6 | **2110 ± 50** |
| Deep GSN | 214 ± 1.1 | 1890 ± 29 |
| Adversarial Nets | **225 ± 2** | 2057 ± 26 |

GAN은 MNIST에서는 가장 높은 추정 log-likelihood를 기록했지만, TFD에서는 Stacked CAE보다 낮았다.

저자들은 Parzen Window 추정치의 분산이 크고 고차원 데이터에서는 잘 작동하지 않는다는 한계를 함께 언급한다. 따라서 이 수치만으로 모델의 우수성을 단정하기는 어렵다.

### 생성 결과

- **Figure 2:** MNIST, TFD, CIFAR-10에서 생성한 샘플을 보여준다.
- 생성 이미지의 오른쪽에는 가장 가까운 학습 이미지를 배치하여 단순 암기가 아님을 확인한다.
- 저자들은 기존 모델보다 명확히 우수하다고 주장하지 않고, 당시 생성 모델들과 경쟁 가능한 수준이며 adversarial framework의 가능성을 보여준다고 해석한다.
- **Figure 3:** 두 latent vector 사이를 선형 보간했을 때 숫자 모양이 연속적으로 변화한다. 학습된 latent space가 단순히 학습 이미지를 불연속적으로 저장한 공간은 아니라는 점을 보여준다.

## 1.7 논문이 정리한 장점과 단점

### 장점

- 학습과 생성에 Markov Chain이 필요하지 않다.
- gradient 계산에 역전파만 사용한다.
- 별도의 approximate inference를 학습할 필요가 없다.
- 생성 시 G를 한 번 통과하면 샘플을 얻을 수 있다.
- 이론상 G와 D에 다양한 미분 가능한 함수를 사용할 수 있다.

### 단점

- `p_g(x)`가 명시적으로 표현되지 않아 특정 데이터의 likelihood를 직접 계산하기 어렵다.
- 학습 중 D와 G의 수준을 맞춰야 한다.
- G만 지나치게 학습하면 여러 `z`가 같은 `x`로 매핑되어 생성 결과의 다양성이 줄어들 수 있다.

마지막 문제를 논문은 `Helvetica scenario`라고 표현한다. 이는 이후 일반적으로 말하는 mode collapse와 연결되는 현상이다.

## 1.8 논문의 의미

이 논문의 핵심은 당시 최고 이미지 품질을 기록한 것이 아니라, **생성 문제를 두 신경망의 경쟁으로 바꾸고 확률밀도나 Markov Chain을 직접 다루지 않아도 생성 모델을 학습할 수 있음을 보인 것**이다.

---

# 2. Denoising Diffusion Probabilistic Models

## 2.1 연구 배경

Diffusion probabilistic model은 데이터에 노이즈를 점차 추가하는 과정과 그 반대 과정을 이용하는 생성 모델이다. 기존에도 관련 방법이 있었지만, GAN과 비교할 만한 고품질 이미지 생성 결과는 충분히 제시되지 않았다.

DDPM 논문의 주요 목표는 다음 두 가지이다.

1. Diffusion model도 고품질 이미지를 생성할 수 있음을 실험으로 보임.
2. Reverse Process의 noise-prediction parameterization이 denoising score matching과 연결됨을 설명함.

## 2.2 전체 구조

Diffusion model은 Forward Process와 Reverse Process로 구성된다.

```text
Forward Process q
x_0 -> x_1 -> ... -> x_T
실제 이미지에 Gaussian noise를 조금씩 추가함.

Reverse Process p_theta
x_T -> x_(T-1) -> ... -> x_0
noise에서 시작해 단계적으로 이미지를 복원함.
```

- `x_0`: 실제 이미지
- `x_t`: t단계까지 noise가 추가된 이미지
- `x_T`: 거의 표준 Gaussian noise가 된 상태

Forward Process는 고정되어 있으며 학습되는 파라미터가 없다. 모델이 학습하는 것은 Reverse Process이다.

## 2.3 Forward Process

Equation 2에서 한 단계의 Forward Process는 다음 Gaussian distribution으로 정의된다.

```text
q(x_t | x_(t-1))
= Normal(sqrt(1 - beta_t) × x_(t-1), beta_t × I)
```

`beta_t`는 t단계에서 추가하는 noise의 양이다. `beta_t`가 작기 때문에 한 단계에서는 이미지가 조금만 변하지만, 이 과정을 여러 번 반복하면 원본 정보가 점차 사라진다.

### 원하는 timestep으로 바로 이동

`alpha_t = 1 - beta_t`, `alpha_bar_t = alpha_1 × ... × alpha_t`로 두면 Equation 4에 따라 `x_0`에서 `x_t`를 바로 만들 수 있다.

```text
x_t
= sqrt(alpha_bar_t) × x_0
  + sqrt(1 - alpha_bar_t) × epsilon

epsilon ~ Normal(0, I)
```

따라서 학습할 때 Forward Process를 1단계부터 t단계까지 순서대로 계산할 필요가 없다. 무작위 timestep `t`를 뽑은 뒤 해당 시점의 noisy image를 한 번에 만들 수 있다.

## 2.4 Reverse Process

Reverse Process는 `x_t`로부터 조금 더 깨끗한 `x_(t-1)`을 생성한다.

```text
p_theta(x_(t-1) | x_t)
= Normal(mu_theta(x_t, t), Sigma_theta(x_t, t))
```

논문에서는 Reverse Process의 variance를 학습하지 않고 timestep에 따라 정해진 값으로 고정한다. 저자들은 학습 가능한 diagonal variance를 사용했을 때 학습이 불안정하고 sample quality가 낮아졌다고 보고한다.

남은 문제는 Gaussian distribution의 평균 `mu_theta`를 어떻게 나타낼 것인가이다.

## 2.5 Noise Prediction Parameterization

Reverse Process의 평균을 직접 예측할 수도 있지만, 논문은 `x_t`에 들어간 noise `epsilon`을 예측하도록 모델을 구성한다.

```text
입력: x_t, timestep t
출력: x_t에 포함된 noise의 예측값 epsilon_theta(x_t, t)
정답: Forward Process에서 실제로 사용한 epsilon
```

Equation 11은 예측한 noise를 이용해 Reverse Process의 평균을 계산할 수 있음을 보여준다. 이 방식은 여러 noise level에서의 denoising score matching과 연결되며, sampling 과정은 annealed Langevin dynamics와 유사한 형태가 된다.

논문에서 중요한 점은 score matching을 별도의 방식으로 추가한 것이 아니라, **Reverse Process를 noise prediction 형태로 다시 나타냈을 때 variational inference와 denoising score matching의 연결이 드러난다**는 것이다.

## 2.6 Simplified Training Objective

원래 Diffusion model은 negative log-likelihood의 variational bound를 최소화하도록 학습할 수 있다. 하지만 저자들은 Equation 14의 단순화된 목적함수가 구현하기 쉽고 sample quality도 더 좋았다고 보고한다.

```text
L_simple
= E[||epsilon - epsilon_theta(x_t, t)||²]
```

즉, 실제로 추가한 noise와 모델이 예측한 noise 사이의 평균제곱오차를 줄인다.

이 목적함수는 원래 variational bound의 가중치를 제거한 형태이다. 그 결과 noise가 매우 적은 쉬운 단계보다, noise가 많은 어려운 denoising 단계에 상대적으로 더 집중하게 된다.

### 학습 과정: Algorithm 1

```text
1. 실제 이미지 x_0를 추출함.
2. 1부터 T 사이에서 timestep t를 무작위로 선택함.
3. Gaussian noise epsilon을 추출함.
4. x_0와 epsilon을 이용해 x_t를 바로 만듦.
5. 모델이 x_t와 t를 보고 epsilon을 예측함.
6. 실제 epsilon과 예측값의 MSE를 줄임.
```

각 training example에서는 timestep 하나만 사용하므로 학습 중에 1,000단계를 모두 순차 실행하는 것은 아니다.

## 2.7 Sampling

Algorithm 2에서는 완전한 Gaussian noise `x_T`에서 시작한다.

```text
1. x_T ~ Normal(0, I)를 생성함.
2. 모델이 현재 x_t의 noise를 예측함.
3. 예측값을 이용해 x_(t-1)을 계산함.
4. t = T부터 1까지 반복함.
5. 최종 x_0를 얻음.
```

학습과 달리 sampling은 `x_t`가 있어야 `x_(t-1)`을 만들 수 있으므로 순차적으로 실행된다. 이 논문에서는 `T = 1000`을 사용하기 때문에 이미지 한 장을 생성할 때 신경망을 1,000번 평가한다.

## 2.8 모델과 실험 설정

| 항목 | 논문의 설정 |
|---|---|
| Reverse Process 모델 | U-Net 계열 네트워크 |
| Normalization | Group Normalization |
| timestep 입력 | Transformer의 sinusoidal position embedding 사용 |
| Self-Attention | 16 × 16 feature map에서 사용 |
| Diffusion step | `T = 1000` |
| Noise schedule | `beta_1 = 0.0001`에서 `beta_T = 0.02`까지 선형 증가 |
| 이미지 범위 | 픽셀 값을 `[-1, 1]`로 scaling |

모든 timestep마다 다른 모델을 만드는 것이 아니라, 하나의 U-Net이 timestep 정보를 입력받으며 전체 과정에서 파라미터를 공유한다.

## 2.9 주요 실험 결과

### CIFAR-10

Table 1에서 `L_simple`로 학습한 unconditional DDPM의 결과는 다음과 같다.

| 지표 | 결과 |
|---|---:|
| Inception Score | 9.46 ± 0.11 |
| FID | 3.17 |
| Test NLL | 3.75 bits/dim 이하 |

- Inception Score는 높을수록 좋고 FID와 NLL은 낮을수록 좋다.
- FID 3.17은 당시 unconditional CIFAR-10 모델 중 최고 수준이었다.
- 이 값은 관례에 따라 training set을 기준으로 계산한 FID이다.
- Test set을 기준으로 계산하면 FID는 5.24이다.

### LSUN 256 × 256

| 데이터셋 | FID |
|---|---:|
| LSUN Church | 7.89 |
| LSUN Bedroom | 4.90 |

저자들은 LSUN에서 ProgressiveGAN과 비슷한 수준의 sample quality를 얻었다고 평가한다.

## 2.10 Ablation Study

Table 2는 Reverse Process의 예측 대상과 학습 목적함수를 비교한다.

| Parameterization과 목적함수 | IS | FID |
|---|---:|---:|
| Mean 예측 + variational bound + learned variance | 7.28 | 23.69 |
| Mean 예측 + variational bound + fixed variance | 8.06 | 13.22 |
| Noise 예측 + variational bound + fixed variance | 7.67 | 13.51 |
| Noise 예측 + `L_simple` | **9.46** | **3.17** |

이 표에서 확인할 수 있는 내용은 다음과 같다.

1. Variance를 학습하는 것보다 고정했을 때 결과가 좋았다.
2. Noise를 예측하기만 한다고 FID가 바로 개선되는 것은 아니다.
3. Noise prediction과 `L_simple`을 함께 사용했을 때 sample quality가 크게 좋아졌다.
4. 정식 variational bound로 학습한 모델은 NLL이 더 좋았지만, `L_simple` 모델은 FID가 더 좋았다.

따라서 논문의 최고 결과를 단순히 `noise prediction의 효과`라고만 해석하면 안 된다. Noise parameterization과 손실의 가중 방식이 함께 영향을 준 결과이다.

## 2.11 Progressive Coding 분석

논문은 Diffusion의 생성 과정을 progressive decompression으로 해석한다.

- Reverse Process 초반에는 이미지의 큰 구조가 먼저 나타난다.
- 후반으로 갈수록 사람이 잘 인식하지 못하는 세부 정보가 추가된다.
- CIFAR-10에서 고품질 모델의 rate는 1.78 bits/dim, distortion은 1.97 bits/dim이었다.
- 저자들은 lossless codelength의 절반 이상이 거의 보이지 않는 세부 차이를 표현하는 데 사용된다고 분석한다.

이 결과를 바탕으로 저자들은 Diffusion model이 likelihood 값은 다른 likelihood-based model보다 낮더라도, 사람이 중요하게 보는 구조를 먼저 복원하는 lossy compressor로서 좋은 inductive bias를 가질 수 있다고 해석한다.

## 2.12 Interpolation

Figure 8에서는 두 실제 이미지에 동일한 정도의 noise를 추가한 뒤, 두 noisy latent 사이를 선형 보간하고 Reverse Process로 복원한다.

그 결과 자세, 피부색, 머리 모양, 표정, 배경 등이 자연스럽게 변하는 중간 이미지가 생성되었다. 더 큰 timestep에서 보간할수록 원본의 세부 정보가 적게 남아 더 다양하고 새로운 결과가 나타났다.

## 2.13 논문의 결론과 한계

논문은 Diffusion model이 높은 품질의 이미지를 생성할 수 있음을 보였고, 다음 방법들 사이의 연결을 정리했다.

- Variational inference
- Denoising score matching
- Annealed Langevin dynamics
- Autoregressive decoding
- Progressive lossy compression

다만 실험 결과에서 다음 한계도 확인된다.

- 좋은 sample quality에 비해 NLL은 다른 likelihood-based model보다 경쟁력이 낮았다.
- 한 번의 생성에 1,000번의 순차적인 network evaluation이 필요했다.
- 논문의 실험은 주로 unconditional image generation을 대상으로 한다.

---

# 3. GAN과 DDPM 비교

| 구분 | GAN | DDPM |
|---|---|---|
| 핵심 아이디어 | G와 D의 경쟁 | Gaussian noise를 추가한 과정의 역과정 학습 |
| 시작점 | 낮은 차원의 noise vector `z` | 이미지와 같은 차원의 Gaussian noise `x_T` |
| 학습 모델 | Generator와 Discriminator | timestep을 입력받는 U-Net 하나 |
| 학습 신호 | D가 실제와 가짜를 구분한 결과 | 실제로 추가한 noise와 예측 noise의 MSE |
| 목적함수 | Minimax adversarial objective | Variational bound 또는 `L_simple` |
| 생성 과정 | G의 forward pass 한 번 | Reverse Process를 1,000단계 반복 |
| 확률밀도 | `p_g(x)`가 명시적으로 주어지지 않음 | Variational bound로 likelihood 평가 가능 |
| 논문이 언급한 학습 문제 | G와 D의 동기화, 다양성 붕괴 | Learned variance의 불안정성, 긴 sampling 과정 |
| 대표 결과 | MNIST·TFD·CIFAR-10 생성 가능성 제시 | CIFAR-10에서 IS 9.46, FID 3.17 |

두 논문의 가장 큰 차이는 학습 신호이다.

```text
GAN
생성 결과가 실제처럼 보이는지를 D의 판단으로 간접 학습함.

DDPM
Forward Process에서 직접 넣은 noise를 정답으로 사용함.
```

GAN은 sampling에 Markov Chain이 필요하지 않아 생성이 빠르지만, D와 G의 균형을 맞춰야 한다. DDPM은 단일 noise-prediction loss로 각 timestep을 학습할 수 있지만, 생성할 때는 여러 timestep을 순서대로 거쳐야 한다.

---

# 4. 논문을 읽을 때 주의할 점

## 4.1 GAN의 수렴 증명

`p_g = p_data`라는 결과는 이상적인 분포 공간과 최적의 D를 가정한 증명이다. 실제 신경망의 파라미터 학습이 항상 이 지점에 도달한다는 의미는 아니다.

## 4.2 GAN의 실험 수치

GAN 논문의 likelihood는 Parzen Window로 추정한 값이다. 논문 자체에서도 이 방법이 고차원에서 부정확하다고 설명하므로, Table 1의 수치를 현대 생성 모델의 likelihood와 직접 비교하면 안 된다.

## 4.3 DDPM의 FID

FID 3.17은 unconditional CIFAR-10의 training set 기준 결과이다. 다른 논문과 비교할 때는 조건부 여부와 비교에 사용한 split이 같은지 확인해야 한다.

## 4.4 DDPM의 성능 향상 원인

Table 2를 보면 noise prediction을 variational bound와 결합했을 때 FID는 13.51이다. 최고 결과인 3.17은 noise prediction과 `L_simple`을 함께 적용했을 때 나온 값이다.

## 4.5 직접적인 성능 비교의 한계

GAN 논문은 2014년에 새로운 학습 프레임워크를 처음 제안한 연구이고, DDPM은 2020년에 Diffusion model의 이미지 품질을 크게 개선한 연구이다. 발표 시기, 모델 구조, 평가 방식이 다르므로 두 원 논문의 숫자로 GAN과 Diffusion 전체의 우열을 결정할 수 없다.

---

# 5. 최종 정리

GAN 논문은 Generator와 Discriminator가 minimax game을 수행하도록 하여, Markov Chain이나 복잡한 확률 계산 없이 생성 모델을 학습하는 방법을 제안했다. 이론적으로 생성 분포가 실제 분포와 같아지면 Discriminator는 모든 입력에 1/2을 출력한다. 실제 학습에서는 두 모델을 번갈아 업데이트해야 하며, G와 D의 균형과 생성 다양성 문제가 남는다.

DDPM 논문은 Gaussian noise를 단계적으로 추가하는 고정된 Forward Process와 이를 되돌리는 학습 가능한 Reverse Process를 사용한다. Reverse Process가 실제로 추가된 noise를 예측하도록 단순한 MSE로 학습했을 때 CIFAR-10에서 FID 3.17을 기록했다. 학습 시에는 임의의 timestep을 바로 선택할 수 있지만, 생성 시에는 1,000단계를 순차적으로 거쳐야 한다.

> GAN은 판별기의 피드백으로 생성 분포를 학습하고, DDPM은 Forward Process에서 넣은 noise를 정답으로 사용해 역과정을 학습한다.

---

# 6. 논문 속 근거 위치

## GAN

- **Equation 1:** G와 D의 minimax objective
- **Algorithm 1:** Discriminator와 Generator를 번갈아 업데이트하는 과정
- **Figure 1:** `p_g`가 `p_data`에 가까워지는 과정
- **Proposition 1:** 고정된 G에서 최적의 D
- **Theorem 1:** `p_g = p_data`, `D = 1/2`인 전역 최적점
- **Table 1:** MNIST와 TFD의 Parzen Window log-likelihood 추정
- **Figure 2:** MNIST, TFD, CIFAR-10 생성 결과
- **Figure 3:** Latent vector의 선형 보간
- **Section 6:** 저자들이 정리한 장점과 단점

## DDPM

- **Figure 2:** Forward Process와 Reverse Process의 방향
- **Equation 2:** 한 단계의 Forward Process
- **Equation 4:** `x_0`에서 임의의 `x_t`를 직접 만드는 분포
- **Equation 11:** Noise prediction으로 Reverse Process의 평균을 나타내는 방법
- **Algorithm 1:** Noise-prediction training
- **Algorithm 2:** Reverse Process sampling
- **Equation 14:** `L_simple`
- **Table 1:** CIFAR-10 sample quality와 NLL
- **Table 2:** Parameterization과 objective ablation
- **Figures 5~7:** Progressive coding과 generation
- **Figure 8:** Latent interpolation

