# Diffusion

DDPM은 **데이터에 Gaussian noise를 점진적으로 추가하는 forward process를 정의하고, 이 과정을 거꾸로 되돌리는 reverse process를 neural network로 학습하는 생성 모델**임.

논문의 핵심은 diffusion이라는 개념 자체를 처음 제안한 것이 아니라, 기존 diffusion probabilistic model을 **noise prediction 문제로 parameterization하고 단순한 MSE objective로 학습해 고품질 image generation이 가능함을 보인 것**임.

---

# 1. Introduction

Diffusion probabilistic model은 이전에도 존재했지만, 당시 image generation에서는 GAN이나 autoregressive model보다 널리 사용되는 방식이 아니었음.

DDPM 논문은 diffusion model과 다음 개념의 연결을 활용함.

- Variational Inference
- Denoising Score Matching
- Langevin Dynamics

특히 reverse process를 **noise prediction network**로 parameterization한 것이 실질적인 핵심임.

논문은 CIFAR-10에서

```
Inception Score: 9.46
FID: 3.17
```

을 기록하며 diffusion model이 실제로 매우 높은 품질의 이미지를 생성할 수 있음을 보임.

---

# 2. Overall Diffusion Process

Diffusion model은 두 process로 구성됨.

- **Forward Process q**: 실제 이미지에 Gaussian noise를 점진적으로 추가함
- **Reverse Process p_θ**: noisy sample에서 noise를 제거하여 data distribution으로 돌아감

대표적인 그림은 Hugging Face의 DDPM 설명 자료가 가장 직관적임.

**Forward / Reverse Process Figure:**

[https://huggingface.co/blog/annotated-diffusion](https://huggingface.co/blog/annotated-diffusion)

원 논문의 전체 diffusion formulation과 sampling 결과는 아래에서 확인할 수 있음.

**Original DDPM Paper:**

[https://arxiv.org/pdf/2006.11239](https://arxiv.org/pdf/2006.11239)

**Algorithm 1·2 / Figure 2 주변:**

[https://arxiv.org/pdf/2006.11239#page=4](https://arxiv.org/pdf/2006.11239#page=4)

---

# 3. Forward Process

원본 데이터를 x₀라고 할 때, forward process는 x₀에 작은 Gaussian noise를 반복해서 추가함.

각 transition은 다음과 같은 Gaussian distribution으로 정의됨.

$$
q(x_t\mid x_{t-1}) = \mathcal{N}\left(x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_t I\right)
$$

β_t는 timestep t에서 얼마나 많은 noise를 추가할지를 결정하는 variance schedule임.

β_t가 충분히 작고 timestep T가 충분히 크면 마지막 x_T는 거의 standard Gaussian distribution에 가까워짐.

논문에서는

$$
T=1000
$$

을 사용함.

즉 실제 이미지를 약 1000개의 작은 noise step을 통해 Gaussian noise에 가까운 상태로 변환함.

---

# 4. α_t와 ᾱ_t

계산을 단순하게 하기 위해 다음을 정의함.

$$
\alpha_t=1-\beta_t
$$

그리고

$$
\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s
$$

로 정의함.

이 notation을 사용하면 x₀에서 특정 timestep의 x_t를 다음과 같이 바로 sampling할 수 있음.

$$
x_t=\sqrt{\bar{\alpha}_t}\,x_0+\sqrt{1-\bar{\alpha}_t}\,\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)
$$

이 식은 DDPM을 이해할 때 매우 중요함.

x_t가 결국 **원본 signal과 Gaussian noise의 weighted combination**으로 표현된다는 의미임.

---

# 5. Closed-form Sampling이 중요한 이유

Forward Process의 정의만 보면 x_t를 얻기 위해 x₁부터 x_t까지 모든 step을 순서대로 계산해야 할 것처럼 보임.

하지만 위 식을 사용하면 x₀에서 원하는 timestep의 x_t를 한 번에 만들 수 있음.

따라서 training에서는 매 iteration마다

1. training image x₀를 sampling
2. timestep t를 random하게 sampling
3. Gaussian noise ε를 sampling
4. 식을 이용해 x_t를 직접 계산

하면 됨.

즉 training할 때 1000개의 forward diffusion step을 실제로 모두 수행할 필요가 없음.

---

# 6. Reverse Process

생성 시에는 Gaussian noise에서 시작해 forward process의 반대 방향으로 이동해야 함.

Reverse transition은 다음과 같이 parameterized Gaussian distribution으로 정의됨.

$$
p_\theta(x_{t-1}\mid x_t) = \mathcal{N}\left(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t)\right)
$$

여기서 neural network가 현재 noisy sample x_t와 timestep t를 이용해 reverse distribution의 parameter를 예측함.

문제는 실제 reverse distribution을 직접 알 수 없기 때문에 이를 neural network로 approximate해야 한다는 것임.

DDPM의 핵심은 이 reverse process의 mean을 직접 예측하기보다 **noise ε를 예측하는 방식으로 다시 parameterization**했다는 데 있음.

---

# 7. Noise Prediction

Forward Process에서 x_t는 다음 식으로 만들어졌음.

$$
x_t=\sqrt{\bar{\alpha}_t}\,x_0+\sqrt{1-\bar{\alpha}_t}\,\epsilon
$$

따라서

$$
x_t
$$

안에는 실제 image signal과 우리가 직접 추가한 noise

$$
\epsilon
$$

이 함께 들어있음.DDPM은 neural network

$$
\epsilon_\theta(x_t,t)
$$

가

**실제로 추가된 noise ε를 예측하도록 학습**

함.

입력은

- noisy image
    
    $$
    x_t
    $$
    
- timestep t

이고, 출력은 x_t와 동일한 shape의 predicted noise임.

이렇게 예측한 noise를 이용하면 reverse mean μ_θ를 계산하고 이전 timestep의 sample x_{t-1}를 만들 수 있음.

---

# 8. Simplified Training Objective

DDPM에서 가장 중요한 식 중 하나가 다음 objective임.

$$
L_{\mathrm{simple}} = \mathbb{E}_{t,x_0,\epsilon} \left[ \left\| \epsilon-\epsilon_\theta(x_t,t) \right\|_2^2 \right]
$$

즉 모델이 예측한 noise와 실제로 x_t를 만들 때 사용한 noise 사이의 MSE를 최소화함.

원래 diffusion probabilistic model은 variational bound에서 출발하지만, 실제 training은 결과적으로 **noise prediction regression 문제**로 단순화됨.

이것이 DDPM이 구현 관점에서 생각보다 단순한 이유임.

---

# 9. Training Algorithm

원 논문의 Algorithm 1은 다음 과정으로 구성됨.

먼저 실제 이미지 x₀를 sampling하고, 1부터 T 사이에서 timestep t를 random하게 선택함.

이후

$$
\epsilon\sim\mathcal{N}(0,I)
$$

를 sampling하고

$$
x_0
$$

와

$$
\epsilon
$$

을 이용해

$$
x_t
$$

를 직접 계산함.

U-Net은 x_t와 timestep t를 입력받아 ε를 예측함.

마지막으로 실제

$$
\epsilon
$$

과 predicted

$$
\epsilon_\theta(x_t,t)
$$

사이의 squared error를 계산해 network parameter를 update함.즉 training에서 모델이 직접 맞히는 target은 clean image

$$
x_0
$$

가 아니라

**noise \epsilon**

임.

**Original Algorithm 1:**

[https://arxiv.org/pdf/2006.11239#page=4](https://arxiv.org/pdf/2006.11239#page=4)

---

# 10. Sampling Algorithm

학습이 끝난 뒤에는 실제 이미지가 필요하지 않음.

x_T를 standard Gaussian distribution에서 sampling하고, T부터 1까지 reverse step을 반복함.

각 timestep에서 U-Net이 현재 x_t에 포함된 noise를 예측하고, 이 prediction을 이용해 조금 더 denoised된 x_{t-1}를 sampling함.

원 논문에서는 T=1000을 사용했기 때문에 sample 하나를 생성하려면 neural network evaluation이 매우 많이 필요함.

이 점이 GAN과 비교했을 때 DDPM의 대표적인 단점임.

**Original Algorithm 2:**

[https://arxiv.org/pdf/2006.11239#page=4](https://arxiv.org/pdf/2006.11239#page=4)

---

# 11. Noise Schedule

논문에서는 forward process의 variance β_t를 고정된 linear schedule로 사용함.

$$
\beta_1=10^{-4},\qquad \beta_T=0.02,\qquad T=1000
$$

β를 첫 timestep에서 마지막 timestep까지 linear하게 증가시킴.

초기에는 원본 image의 정보를 거의 유지하면서 작은 noise만 추가하고, timestep이 커질수록 noise가 누적되어 최종적으로 Gaussian noise에 가까워짐.

이후 Improved DDPM 등의 연구에서는 cosine schedule과 같은 다른 방식이 제안됨.

---

# 12. Model Architecture

DDPM의 noise prediction network는 **U-Net 기반 architecture**임.

논문 implementation의 주요 요소는 다음과 같음.

- U-Net backbone
- residual block
- Group Normalization
- timestep embedding
- self-attention
- 모든 timestep에서 동일한 network parameter 공유

timestep마다 별도의 network를 학습하는 것이 아니라, 하나의 U-Net이 현재 timestep 정보를 함께 입력받아 다양한 noise level을 처리함.

DDPM 논문은 U-Net 구조 자체를 새롭게 제안한 논문은 아니므로, 세부 U-Net architecture는 별도로 보는 것이 이해하기 좋음.

**U-Net / DDPM architecture 설명:**

[https://huggingface.co/blog/annotated-diffusion#the-neural-network](https://huggingface.co/blog/annotated-diffusion#the-neural-network)

---

# 13. Timestep Embedding

같은 image라도 timestep이 다르면 noise의 정도가 완전히 다름.

t가 작으면 x_t는 원본 image에 가까우며, t가 크면 대부분의 정보가 noise로 손상된 상태임.

따라서 noise prediction network는 x_t뿐 아니라 현재 timestep t도 알아야 함.

DDPM은 timestep을 sinusoidal positional embedding과 유사한 방식으로 embedding하여 U-Net 내부에 전달함.

이 덕분에 하나의 network가 여러 noise scale을 동시에 학습할 수 있음.

---

# 14. Variational Bound와 DDPM

DDPM은 단순한 image denoising network가 아니라 latent variable probabilistic model임.

전체 objective는 negative log-likelihood의 variational upper bound에서 유도됨.

Forward posterior q는 고정되어 있고, 학습 가능한 reverse model p_θ가 forward posterior의 reverse transition을 근사하도록 학습함.

각 transition이 Gaussian이므로 KL divergence를 analytic하게 계산할 수 있음.

하지만 실제 image sample quality를 높이는 데는 variational bound를 그대로 최적화하는 것보다 **simplified noise prediction objective L_simple**이 더 효과적이었음.

---

# 15. 왜 ε Prediction인가?

논문에서는 reverse process의 mean μ를 직접 예측하는 parameterization과 noise ε를 예측하는 parameterization을 비교함.

주요 실험 결과는 다음과 같음.

| Parameterization / Objective | CIFAR-10 FID |
| --- | --- |
| μ prediction + Variational Bound | 13.22 |
| \epsilon prediction + Variational Bound | 13.51 |
| \epsilon prediction + L_simple | **3.17** |

즉 단순히 noise prediction을 사용했다는 것뿐 아니라 **\epsilon prediction과 simplified objective의 조합**이 sample quality를 크게 향상시킴.

이 실험이 DDPM 논문의 핵심 empirical result 중 하나임.

---

# 16. Experiments

## CIFAR-10

Unconditional CIFAR-10에서 다음 결과를 기록함.

$$
\text{Inception Score}=9.46\pm0.11,\qquad \text{FID}=3.17
$$

당시 unconditional generation에서 매우 높은 성능이었음.

## LSUN 256×256

논문은 LSUN Bedroom과 Church 데이터셋에서도 고해상도 image generation을 수행함.

대표 FID:

| Dataset | FID |
| --- | --- |
| LSUN Bedroom | 4.90 |
| LSUN Church | 7.89 |

논문에서는 ProgressiveGAN과 비교 가능한 수준의 sample quality를 보고함.

---

# 17. Progressive Generation

DDPM의 reverse process를 관찰하면 image의 모든 detail이 한 번에 생성되지 않음.

초기 reverse step에서는 전체적인 shape이나 큰 구조가 결정되고, 뒤쪽 timestep으로 갈수록 texture와 세부적인 feature가 복원됨.

논문에서는 이를 **progressive lossy decompression** 관점에서도 설명함.

이 성질은 diffusion이 왜 coarse-to-fine 방식의 generation처럼 보이는지를 이해하는 데 중요함.

**Original Paper Figure 4 관련 내용:**

[https://arxiv.org/pdf/2006.11239](https://arxiv.org/pdf/2006.11239)

---

# 18. Score Matching과의 관계

DDPM의 noise prediction은 단순한 engineering trick이 아님.

논문은 diffusion objective가 denoising score matching과 연결됨을 설명함.

score는 probability density의 log-gradient로 정의됨.

$$
\nabla_x\log p(x)
$$

직관적으로는 현재 sample이 data density가 더 높은 방향으로 이동하려면 어느 방향으로 가야 하는지를 나타냄.

여러 noise level에서 ε를 예측하는 network는 이러한 score와 관련된 정보를 학습하게 됨.

또한 reverse sampling process는 Langevin Dynamics와 연결해서 해석할 수 있음.

이 부분이 DDPM을 score-based generative model과 이어주는 이론적 배경임.

---

# 19. GAN과 비교

| GAN | DDPM |
| --- | --- |
| Generator + Discriminator | Noise prediction network |
| Adversarial objective | Denoising / variational objective |
| 한 번의 Generator forward로 sampling 가능 | 반복적인 reverse process 필요 |
| 학습이 불안정할 수 있음 | 상대적으로 안정적 |
| Mode collapse 가능 | distribution coverage가 상대적으로 좋음 |
| Sampling 빠름 | Sampling 느림 |

GAN은 Generator와 Discriminator 사이의 균형이 중요한 반면, DDPM은 실제로 추가한 noise를 target으로 사용할 수 있기 때문에 **supervised regression과 비슷한 형태의 안정적인 objective**를 갖는다는 차이가 있음.

---

# 20. Limitations

## Slow Sampling

원 논문의 DDPM은 T=1000을 사용함.

따라서 sample 하나를 생성하기 위해 U-Net을 반복적으로 실행해야 함.

이후 DDIM, Improved DDPM 등의 연구에서는 sampling step을 크게 줄이는 방법이 제안됨.

## Computation Cost

pixel space에서 고해상도 image를 반복적으로 처리하기 때문에 계산량과 memory 사용량이 큼.

이 문제는 이후 Latent Diffusion에서 image를 compressed latent space로 옮겨 diffusion을 수행하는 방식으로 개선됨.

## Likelihood와 Sample Quality의 차이

논문에서는 true variational bound를 더 직접적으로 최적화하는 것이 likelihood에는 유리하지만, 실제 generated sample의 FID는 L_simple이 더 좋았음.

즉 likelihood와 perceptual sample quality가 항상 같은 방향으로 움직이는 것은 아님.

---

# 21. Conclusion

DDPM의 핵심 기여는 diffusion probabilistic model을 **실제로 강력한 image generator로 만든 training parameterization**에 있음.

Forward process에서 어떤 noise ε를 추가했는지 알고 있기 때문에 이를 직접 supervision으로 사용할 수 있고, U-Net은 noisy image와 timestep을 입력받아 그 noise를 예측함.

결국 복잡한 probabilistic formulation이 실제 training에서는 다음 형태의 단순한 문제로 정리됨.

$$
L_{\mathrm{simple}} = \mathbb{E} \left[ \left\| \epsilon-\epsilon_\theta(x_t,t) \right\|_2^2 \right]
$$

이 구조가 이후 diffusion model의 기본 training framework가 됨.

## 핵심 정리

- Forward Process는 Gaussian noise를 점진적으로 추가
- x_t는 x₀에서 closed-form으로 직접 sampling 가능
- Reverse Process는 U-Net으로 학습
- 모델은 clean image가 아니라 noise ε를 예측
- objective는 noise prediction MSE인 L_simple
- timestep embedding으로 하나의 network가 모든 noise level을 처리
- 원 논문은 T=1000, linear β schedule 사용
- \epsilon prediction + L_simple으로 CIFAR-10 FID 3.17 기록
- GAN보다 training은 안정적이지만 sampling은 느림
- 이후 DDIM, Improved DDPM, Latent Diffusion의 기반이 됨