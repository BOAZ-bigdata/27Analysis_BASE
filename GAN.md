# Generative Adversarial Nets

**Ian Goodfellow et al., 2014**

## 1. 연구 목적

* 새로운 **생성 모델 학습 프레임워크** 제안
* 생성 모델과 판별 모델을 **적대적으로 동시에 학습**
* 복잡한 확률 계산이나 Markov Chain 없이 생성 모델 학습
* Backpropagation만으로 전체 시스템 학습 가능 

---

## 2. 핵심 아이디어

### Generator \(G\)

* 실제 데이터와 유사한 샘플을 생성하는 모델
* 입력: random noise \(z\)
* 출력: 생성 데이터 \(G(z)\)
* 목표: **Discriminator가 생성 데이터를 실제 데이터로 판단하도록 만듦**

### Discriminator \(D\)

* 입력 데이터가 실제 데이터인지 생성 데이터인지 판별하는 모델
* 출력 \(D(x)\):

  * 실제 데이터일 확률
* 목표:

  * 실제 데이터 → 1
  * 생성 데이터 → 0

### 전체 구조

```text
Random Noise z
      ↓
 Generator G
      ↓
 Fake Sample G(z)
      ↓
 Discriminator D
      ↓
 Real / Fake
```

* \(G\)와 \(D\)의 **경쟁 구조**
* Generator = 위조지폐 제작자
* Discriminator = 위조지폐를 잡는 경찰
* 경쟁을 반복하면서 두 모델의 성능 향상 

---

## 3. GAN의 학습 목표

**Two-player Minimax Game**

$$
\min_G \max_D V(D,G)
$$

$$
V(D,G)
=
E_{x\sim p_{data}}[\log D(x)]
+
E_{z\sim p_z}[\log(1-D(G(z)))]
$$

### Discriminator

$$
\max_D
$$

* 실제 데이터의 \(D(x)\) 증가
* 생성 데이터의 \(D(G(z))\) 감소
* Real / Fake 구분 성능 최대화

### Generator

$$
\min_G
$$

* \(D(G(z))\)가 증가하도록 학습
* 생성 데이터를 실제 데이터처럼 보이게 만드는 방향

즉,

```text
D : 진짜와 가짜를 더 잘 구별
G : D가 구별하지 못하도록 더 진짜 같은 데이터 생성
```



---

## 4. 학습 과정

### Step 1. Discriminator 학습

* 실제 데이터 \(x\) sampling
* noise \(z\) sampling
* \(G(z)\) 생성
* 실제 데이터와 생성 데이터를 이용해 D 업데이트

목표

```text
D(x) → 1
D(G(z)) → 0
```

### Step 2. Generator 학습

* 새로운 noise \(z\) sampling
* \(G(z)\) 생성
* D를 통과시켜 결과 확인
* D가 생성 데이터를 실제 데이터로 판단하도록 G 업데이트

목표

```text
D(G(z)) → 1
```

### Step 3. 반복

```text
D 업데이트
    ↓
G 업데이트
    ↓
D 업데이트
    ↓
G 업데이트
    ↓
...
```

실제 논문에서는

* D를 \(k\)번 업데이트
* G를 1번 업데이트
* 실험에서는 **k = 1 사용** 

---

## 5. Generator 학습 시 사용한 개선 방법

기본 목적함수

$$
\min_G \log(1-D(G(z)))
$$

문제점

* 초기에는 Generator 품질이 매우 낮음
* Discriminator가 너무 쉽게 fake 판별
* \(D(G(z)) \approx 0\)
* gradient가 약해지는 **saturation 문제**

대안

$$
\max_G \log D(G(z))
$$

* 동일한 최종 해를 가짐
* 초기 학습에서 더 강한 gradient 제공
* Generator 학습 안정성 개선 

---

# 6. 이론적 결과

## 최적 Discriminator

Generator \(G\)가 고정된 경우

$$
D_G^*(x)
=
\frac{p_{data}(x)}
{p_{data}(x)+p_g(x)}
$$

의미

* 어떤 \(x\)가 실제 데이터에서 나왔을 가능성과
* Generator에서 나왔을 가능성을 비교하는 형태 

---

## GAN의 최적 상태

최종 목표

$$
p_g = p_{data}
$$

즉,

```text
Generator가 만드는 데이터 분포
=
실제 데이터 분포
```

이때

$$
D(x)=\frac12
$$

* 실제와 생성 데이터를 구분할 수 없는 상태
* Discriminator의 정답 확률 50%

논문의 Figure 1에서도

```text
초기
pg ≠ pdata
↓
D가 두 분포 구별
↓
G가 D의 gradient를 이용하여 pdata 방향으로 이동
↓
pg = pdata
↓
D(x) = 1/2
```

의 과정 제시 

---

## Jensen-Shannon Divergence와의 관계

Generator의 목적함수

$$
C(G)
=
-\log 4
+
2\cdot JSD(p_{data}\parallel p_g)
$$

JSD는

$$
JSD \ge 0
$$

이며

$$
JSD=0
$$

일 때만

$$
p_g=p_{data}
$$

따라서 GAN의 global optimum

$$
p_g=p_{data}
$$

최솟값

$$
C(G)=-\log4
$$



---

# 7. 실험

### Dataset

* MNIST
* Toronto Face Database \(TFD\)
* CIFAR-10

### Generator

* Rectifier Linear Activation
* Sigmoid Activation

### Discriminator

* Maxout Activation
* Dropout 사용

### Noise

* Generator의 가장 아래 입력 layer에만 적용 

---

## 8. 평가 방법

GAN은 명시적인 확률 밀도

$$
p_g(x)
$$

를 직접 계산하기 어려움.

따라서 생성된 sample에

**Gaussian Parzen Window**

적용.

이를 통해 test data의 log-likelihood 추정.

### 결과

| Model                |       MNIST |           TFD |
| -------------------- | ----------: | ------------: |
| DBN                  |     138 ± 2 |     1909 ± 66 |
| Stacked CAE          |   121 ± 1.6 | **2110 ± 50** |
| Deep GSN             |   214 ± 1.1 |     1890 ± 29 |
| **Adversarial Nets** | **225 ± 2** |     2057 ± 26 |

* MNIST에서 비교 모델 중 가장 높은 결과
* TFD에서는 Stacked CAE보다 낮은 결과
* 저자도 기존 방식보다 생성 품질이 우월하다고 단정하지 않음
* 기존 생성 모델과 경쟁 가능한 결과로 평가 

---

# 9. 생성 결과

논문 Figure 2

* MNIST 숫자
* TFD 얼굴
* CIFAR-10 이미지 생성

오른쪽 열

* 생성 이미지와 가장 가까운 training example
* 단순한 training data 암기가 아니라는 점 확인 목적

또한

* Cherry-picking하지 않은 random sample
* Markov Chain을 사용하지 않아 sample 간 mixing 문제 없음 

---

# 10. 장점

* **Markov Chain 불필요**
* **Approximate inference 불필요**
* Backpropagation만으로 학습 가능
* Sampling 시 단순 forward propagation 사용
* 다양한 differentiable function 적용 가능
* 복잡한 확률분포를 명시적으로 계산할 필요 없음
* Sharp distribution 표현 가능 

---

# 11. 단점

### 명시적 probability density 부재

$$
p_g(x)
$$

직접 계산 불가능.

→ likelihood 평가 어려움

### G와 D의 학습 균형 필요

* Generator와 Discriminator의 학습 속도 조절 필요
* D 업데이트 없이 G만 과도하게 학습할 경우 문제 발생

### Generator Collapse 가능성

* 여러 \(z\)가 동일한 \(x\)로 mapping되는 현상
* 생성 데이터 다양성 감소

논문에서는 이를 **Helvetica scenario**로 표현 

---

# 12. 기존 생성 모델과의 차이

| 항목           | 기존 생성 모델                  | GAN                       |
| ------------ | ------------------------- | ------------------------- |
| 학습 방식        | Likelihood / inference 중심 | Adversarial learning      |
| 구조           | 단일 생성 모델 중심               | Generator + Discriminator |
| Markov Chain | 필요한 경우 많음                 | 불필요                       |
| Inference    | 필요한 경우 많음                 | 학습 중 불필요                  |
| Sampling     | 복잡할 수 있음                  | Forward propagation       |
| \(p(x)\) 계산  | 일부 모델 가능                  | 명시적으로 표현하지 않음             |
| 핵심 목표        | 데이터 분포 모델링                | 실제와 구분 불가능한 데이터 생성        |

논문의 Table 2에서도 GAN의 주요 특징을 **sampling의 단순함**, **Markov Chain 불필요**, **differentiable function 사용 가능** 등으로 정리. 

---

# 13. Future Work

논문에서 제시한 확장 방향

* **Conditional GAN**

  * 조건 \(c\)를 G와 D 모두에 입력
  * \(p(x|c)\) 학습

* Approximate inference network 추가

  * \(x\rightarrow z\) 예측

* 다양한 conditional distribution 학습

* Semi-supervised learning 활용

  * Discriminator의 feature 활용

* G와 D의 coordination 개선

  * 학습 효율성 향상 

---

# 14. 논문 핵심 정리

### Problem

기존 deep generative model

* Intractable probabilistic computation
* Approximate inference 필요
* MCMC 및 Markov Chain 의존
* 학습 및 sampling 복잡

### Solution

**Generative Adversarial Nets**

```text
Generator G
vs
Discriminator D
```

두 모델을 **동시에 적대적으로 학습하는 프레임워크**.

### Training

```text
Noise z
   ↓
Generator
   ↓
Fake Data
   ↓
Discriminator
   ↙       ↘
Real      Fake
```

* D: Real / Fake 구별
* G: D를 속이도록 학습
* D와 G를 번갈아 업데이트

### Theoretical Goal

$$
p_g=p_{data}
$$

$$
D(x)=\frac12
$$

### 핵심 기여

* **Adversarial learning 개념 제안**
* Generator와 Discriminator의 **two-player minimax framework**
* Likelihood를 직접 계산하지 않는 생성 모델 학습
* Markov Chain 및 approximate inference 불필요
* Backpropagation 기반 end-to-end 학습

