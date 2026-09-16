# GAN

GAN은 **Generator G와 Discriminator D를 동시에 학습시키는 adversarial game을 통해 실제 데이터의 분포를 학습하는 생성 모델**을 제안한 논문임.

논문의 핵심은 특정한 CNN 구조를 제안한 것이 아니라, generative modeling 자체를 **두 모델의 경쟁 문제로 바꾼 새로운 학습 framework**를 제안했다는 데 있음.

> Generator는 실제 데이터와 유사한 sample을 만들고, Discriminator는 실제 sample과 generated sample을 구분함.
> 
> 
> 두 모델을 함께 학습하면 이상적으로 Generator의 분포가 실제 데이터 분포에 가까워짐.
> 

---

# 1. Introduction

당시 Deep Learning의 성공은 classification과 같은 discriminative model에 집중되어 있었음.

Generative model은 단순히 label을 예측하는 것이 아니라 실제 데이터가 어떤 분포에서 생성되는지를 학습해야 함.

기존 deep generative model들은 학습 과정에서 approximate inference, Markov Chain, 복잡한 probability computation 등이 필요한 경우가 많았음.

GAN은 이 문제를 **Generator와 Discriminator라는 두 neural network의 minimax game**으로 정의함.

논문에서 강조한 장점은 다음과 같음.

- Markov Chain이 필요하지 않음
- approximate inference network가 필수적이지 않음
- 학습에 일반적인 backpropagation을 사용할 수 있음
- 생성 시 Generator의 forward computation만 필요함

---

# 2. Overall Architecture

GAN은 두 모델로 구성됨.

- **Generator G**: latent variable z를 입력받아 generated sample 를 만듦
    
    $$
    G(z)
    $$
    
- **Discriminator D**: 입력 sample이 실제 데이터에서 왔을 확률을 추정함

전체 구조는 아래 그림이 가장 직관적임.

GAN Overall Architecture

![GAN Overall Architecture](https://developers.google.com/static/machine-learning/gan/images/gan_diagram.svg)

GAN Overall Architecture

**Architecture reference:**

[https://developers.google.com/machine-learning/gan/gan_structure](https://developers.google.com/machine-learning/gan/gan_structure)

원 논문에서는 network block diagram보다 **p_data와 p_g가 학습 과정에서 어떻게 가까워지는지**를 Figure 1로 설명함.

**Original Paper Figure 1:**

[https://arxiv.org/pdf/1406.2661#page=2](https://arxiv.org/pdf/1406.2661#page=2)

---

# 3. Generator

Generator는 random variable z를 입력으로 받아 data space의 sample을 생성함.

z는 일반적으로 simple prior distribution에서 sampling함.

예를 들어,

- Uniform distribution
- Gaussian distribution

등을 사용할 수 있음.

Generator는 deterministic mapping G(z; θ_g)를 학습함.

여기서 중요한 점은 GAN이

$$
p_g(x)
$$

를 직접 계산하는 explicit density model이 아니라는 것임.

Generator가 z를 data space로 mapping하면서 결과적으로 **implicit distribution p_g**가 만들어짐.

즉 GAN은 probability density를 직접 계산하기보다 **sample을 만들어내는 방법 자체를 학습**함.

---

# 4. Discriminator

Discriminator D(x)는 입력 x가 실제 data distribution에서 왔을 확률을 출력함.

$$
D(x)
$$

가 1에 가까울수록 실제 데이터라고 판단하고, 0에 가까울수록 Generator가 만든 sample이라고 판단함.

Discriminator의 학습 목표는 다음 두 가지임.

- 실제 데이터 x에 대해서 D(x)를 크게 만들기
- generated sample 에 대해서 를 작게 만들기
    
    $$
    G(z)
    $$
    
    $$
    D(G(z))
    $$
    

Generator는 반대로

$$
D(G(z))
$$

가 커지도록 학습함.

이렇게 두 모델의 목적이 반대이기 때문에 **adversarial training**이라고 부름.

---

# 5. Minimax Objective

GAN의 핵심 objective는 다음과 같음.

$$
\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{\mathrm{data}}}\left[\log D(x)\right] + \mathbb{E}_{z \sim p_z}\left[\log\left(1-D(G(z))\right)\right]
$$

첫 번째 항은 실제 데이터를 실제라고 맞히는 정도를 나타냄.

$$
\mathbb{E}_{x \sim p_{\mathrm{data}}}\left[\log D(x)\right]
$$

두 번째 항은 Generator가 만든 sample을 가짜라고 맞히는 정도를 나타냄.

$$
\mathbb{E}_{z \sim p_z}\left[\log\left(1-D(G(z))\right)\right]
$$

Discriminator는 이 objective를 maximize하고, Generator는 minimize함.

따라서 GAN은 **two-player minimax game**으로 해석할 수 있음.

---

# 6. Generator Objective와 Saturation 문제

원래 minimax formulation에서 Generator는 다음 값을 최소화함.

$$
\log\left(1-D(G(z))\right)
$$

학습 초기에는 Generator의 sample 품질이 낮기 때문에 Discriminator가 쉽게 fake를 판별함.

따라서

$$
D(G(z)) \approx 0
$$

이 되기 쉬움.

이 경우 Generator가 받는 gradient가 매우 작아질 수 있음.

논문에서는 실제 학습에서 Generator가 다음 objective를 사용하면 더 강한 gradient를 받을 수 있다고 설명함.

$$
\max_G \mathbb{E}_{z \sim p_z}\left[\log D(G(z))\right]
$$

즉 generated sample이 **real로 판단될 확률 자체를 직접 높이는 방식**임.

이 방식은 이후 **non-saturating GAN loss**라고 불리게 됨.

---

# 7. Training Algorithm

GAN은 Generator와 Discriminator를 번갈아 업데이트함.

먼저 real data minibatch와 noise minibatch를 sampling한 뒤, Generator가 만든 sample과 실제 데이터를 이용해 Discriminator를 학습함.

Discriminator update에서는 실제 sample에 높은 probability를 주고 generated sample에는 낮은 probability를 주도록 parameter를 업데이트함.

이후 새로운 noise minibatch를 sampling하고, Generator의 output을 Discriminator에 통과시킨 뒤 Generator parameter를 업데이트함.

이때 Discriminator의 parameter는 Generator update에서 고정하고, **Discriminator를 통해 전달된 gradient만 Generator까지 backpropagation**함.

원 논문의 기본 실험에서는 discriminator update 횟수 k를 1로 사용함.

---

# 8. Discriminator가 Generator를 학습시키는 방식

Generator에는 정답 이미지가 직접 주어지지 않음.

즉 generated image와 특정 target image 사이의 pixel loss를 계산하는 구조가 아님.

Generator가 받는 supervision은 **Discriminator의 판단**에서 나옴.

Discriminator가 generated sample의 어떤 특징 때문에 fake라고 판단했는지에 대한 gradient가 Generator까지 전달됨.

따라서 Discriminator는 단순한 classifier인 동시에 Generator 입장에서는 **학습되는 loss function의 역할**도 함.

이 점이 GAN의 중요한 아이디어 중 하나임.

---

# 9. Optimal Discriminator

Generator G를 고정했다고 가정하면 최적의 Discriminator는 다음과 같음.

$$
D^*(x)=\frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x)+p_g(x)}
$$

어떤 x가 실제 데이터에서 나타날 가능성이 높다면 D*(x)는 1에 가까워짐.

반대로 generated distribution에서 나타날 가능성이 높다면 0에 가까워짐.

이 식은 Discriminator가 결국 **p_data와 p_g의 상대적인 density를 비교하는 역할**을 한다는 것을 보여줌.

---

# 10. Global Optimum과 Jensen-Shannon Divergence

최적의 Discriminator D*를 objective에 대입하면 Generator가 최종적으로 최적화하는 문제는 Jensen-Shannon Divergence와 연결됨.

$$
C(G)=-\log 4+2\,\mathrm{JSD}\left(p_{\mathrm{data}}\,\|\,p_g\right)
$$

JSD는 두 probability distribution이 얼마나 다른지를 나타냄.

JSD가 최소가 되는 조건은

$$
p_g=p_{\mathrm{data}}
$$

임.

따라서 GAN의 이론적 global optimum은 Generator가 실제 데이터 분포를 정확히 재현하는 상태임.

이때 Discriminator는 두 분포를 구분할 수 없으므로

$$
D(x)=\frac{1}{2}
$$

가 됨.

즉 학습이 완벽하게 이루어진 상태에서 Discriminator는 모든 sample에 대해 50% 정도의 확률만 줄 수 있음.

---

# 11. Figure 1의 의미

원 논문의 Figure 1은 GAN을 neural network architecture 관점보다 **distribution 관점**에서 설명함.

초기에는

$$
p_g
$$

와

$$
p_{\mathrm{data}}
$$

가 서로 다른 위치에 존재함.

Discriminator를 학습하면 두 distribution을 가장 잘 구분할 수 있는 D가 만들어짐.

이후 Generator를 업데이트하면 generated distribution p_g가 실제 데이터가 많이 존재하는 방향으로 이동함.

이 과정을 반복하면서

$$
p_g \rightarrow p_{\mathrm{data}}
$$

가 되는 것이 GAN 학습의 이상적인 목표임.

**Original Figure 1:**

[https://arxiv.org/pdf/1406.2661#page=2](https://arxiv.org/pdf/1406.2661#page=2)

---

# 12. Architecture Used in the Original Paper

현재 GAN이라고 하면 CNN 기반 image generator를 떠올리기 쉽지만, **2014년 원 논문의 실험 모델은 MLP 기반**임.

논문에서 사용한 주요 구성은 다음과 같음.

| Component | Original GAN |
| --- | --- |
| Generator | Multilayer Perceptron |
| Discriminator | Multilayer Perceptron |
| Generator activation | ReLU, output에서 sigmoid |
| Discriminator activation | Maxout |
| Regularization | Dropout in Discriminator |
| Training | Backpropagation + momentum |

Convolution 기반 GAN 구조가 일반화된 것은 이후 DCGAN 등의 연구에서 발전한 것임.

---

# 13. Experiments

논문은 다음 데이터셋에서 GAN을 평가함.

- MNIST
- Toronto Face Database
- CIFAR-10

당시에는 GAN의 explicit likelihood를 직접 계산할 수 없었기 때문에 generated sample에 Gaussian Parzen Window를 fitting한 뒤 test log-likelihood를 추정함.

MNIST에서 adversarial nets는 다음 값을 기록함.

$$
\text{Parzen Window log-likelihood}=225\pm2
$$

다만 논문에서도 이 평가법의 variance가 높고 high-dimensional space에서 신뢰하기 어렵다는 한계를 언급함.

따라서 GAN 논문의 의미는 특정 benchmark의 수치보다 **adversarial framework 자체가 실제 generative model로 작동한다는 것을 보인 것**에 더 가까움.

---

# 14. Latent Space

논문에서는 latent vector 사이를 interpolation한 뒤 Generator output의 변화를 확인함.

두 latent vector 사이를 연속적으로 이동하면 generated image도 갑자기 끊기지 않고 점진적으로 변화함.

이는 Generator가 단순히 training sample을 외워서 출력하는 것이 아니라 latent space에서 어느 정도 **연속적인 representation을 학습**했음을 보여주는 관찰임.

또한 generated sample과 training data의 nearest neighbor를 비교하여 단순 memorization 여부도 확인함.

---

# 15. Advantages

원 논문에서 강조하는 GAN의 장점은 다음과 같음.

- Markov Chain이 필요하지 않음
- Generator sampling이 한 번의 forward computation으로 가능
- backpropagation만으로 전체 system을 학습할 수 있음
- explicit probability density를 설계하지 않아도 됨
- 다양한 differentiable neural network를 G와 D에 사용할 수 있음

특히 **sampling이 빠르다**는 점은 이후 GAN의 중요한 장점으로 이어짐.

---

# 16. Limitations

## Explicit density가 없음

GAN은

$$
p_g(x)
$$

를 직접 계산하는 구조가 아니기 때문에 exact likelihood 계산이 어려움.

이 때문에 모델의 probabilistic evaluation이 쉽지 않음.

## Generator와 Discriminator의 균형

G와 D를 동시에 학습하기 때문에 두 모델의 학습 속도가 크게 차이나면 optimization이 불안정해질 수 있음.

원 논문에서도 Generator가 지나치게 적은 종류의 output만 만들게 되는 형태의 failure 가능성을 언급함.

이 문제는 이후 GAN 연구에서 **mode collapse**라는 대표적인 문제로 발전해서 다뤄짐.

---

# 17. Conclusion

GAN 논문의 핵심은 특정 architecture가 아니라 **새로운 generative learning framework**임.

Generator가 직접 likelihood를 계산하는 대신 sample을 생성하고, Discriminator가 real data와 generated data를 구분하면서 Generator에 학습 signal을 제공함.

이론적으로는 최적의 상태에서

```
$$p_g=p_{\mathrm{data}}$$
$$D(x)=\frac{1}{2}$$
```

가 됨.

이후 DCGAN, Pix2Pix, CycleGAN, StyleGAN 등 대부분의 GAN 계열 연구는 이 기본 adversarial framework 위에서 architecture와 loss를 발전시킨 것이라고 볼 수 있음.

## 핵심 정리

- GAN = Generator + Discriminator
- 두 모델은 minimax game으로 학습
- Generator는 explicit density 대신 sample generation을 학습
- Discriminator gradient가 Generator의 학습 signal이 됨
- global optimum에서는
    
    $$
    p_g=p_{\mathrm{data}}
    $$
    
- 이때
    
    $$
    D(x)=\frac{1}{2}
    $$
    
- 실제 Generator 학습에서는 non-saturating objective가 더 유용
- 원 논문의 실험 architecture는 CNN이 아니라 MLP
- 학습 불안정성과 mode collapse가 대표적인 후속 문제