# Learning Transferable Visual Models From Natural Language Supervision

**Alec Radford et al., 2021**

**CLIP: Contrastive Language-Image Pre-training**

## 1. 연구 목적

* 기존 이미지 분류 모델은 미리 정해진 클래스에 의존
* 새로운 시각 개념을 학습하려면 추가적인 라벨 데이터와 재학습이 필요
* 인터넷의 대규모 **이미지-텍스트 쌍**을 학습 데이터로 활용
* 자연어를 통해 새로운 분류 작업을 정의하는 **Zero-shot 이미지 분류 모델** 개발
* NLP의 대규모 사전학습과 Zero-shot 전이 능력을 컴퓨터 비전에 적용

---

## 2. 기존 이미지 분류 방식의 문제

일반적인 이미지 분류 모델은 다음과 같이 학습된다.

```text
Image
  ↓
Image Encoder
  ↓
Fixed Classifier
  ↓
Predetermined Class
```

예를 들어 ImageNet으로 학습한 모델은 사전에 정의된 1,000개 클래스만 예측할 수 있다.

새로운 클래스를 추가하려면

* 라벨이 있는 데이터 수집
* 출력 분류기 수정
* 모델 Fine-tuning 또는 재학습

과정이 필요하다.

CLIP은 고정된 클래스 분류기 대신 **자연어로 분류할 클래스를 정의**한다.

---

## 3. 핵심 아이디어

CLIP은 이미지와 텍스트를 동일한 임베딩 공간에 배치한다.

### Image Encoder

* 이미지 \(x\)를 특징 벡터로 변환

$$
f_I(x)
$$

### Text Encoder

* 텍스트 \(t\)를 특징 벡터로 변환

$$
f_T(t)
$$

### 학습 목표

* 실제로 함께 등장한 이미지와 텍스트의 유사도 증가
* 서로 관련 없는 이미지와 텍스트의 유사도 감소

```text
Image ──→ Image Encoder ──→ Image Embedding
                                  ↕ similarity
Text  ──→ Text Encoder  ──→ Text Embedding
```

즉, CLIP은 문장을 직접 생성하는 대신

> 이 이미지와 어떤 텍스트가 가장 잘 대응하는가?

를 학습한다.

---

## 4. Contrastive Learning

하나의 미니배치에 \(N\)개의 이미지-텍스트 쌍이 있다고 가정한다.

```text
(Image 1, Text 1)
(Image 2, Text 2)
...
(Image N, Text N)
```

가능한 조합은 총

$$
N \times N
$$

개이다.

이 가운데 실제로 대응하는 조합은 대각선의 \(N\)개이다.

```text
             Text 1   Text 2   Text 3
Image 1        ✓         ×        ×
Image 2        ×         ✓        ×
Image 3        ×         ×        ✓
```

CLIP의 목표는

* 올바른 \(N\)개 쌍의 유사도 증가
* 잘못된 \(N^2-N\)개 쌍의 유사도 감소

이다.

---

## 5. CLIP의 학습 과정

이미지와 텍스트 특징을 각각 추출한다.

$$
I_f = \operatorname{ImageEncoder}(I)
$$

$$
T_f = \operatorname{TextEncoder}(T)
$$

각 특징을 선형 변환하고 L2 정규화한다.

$$
I_e
=
\operatorname{Normalize}(I_fW_i)
$$

$$
T_e
=
\operatorname{Normalize}(T_fW_t)
$$

이미지와 텍스트 사이의 코사인 유사도를 계산한다.

$$
S_{ij}
=
\exp(t)\,I_{e,i}^{\mathsf T}T_{e,j}
$$

여기서 \(t\)는 학습되는 temperature 관련 파라미터이다.

### 대칭적 Cross-Entropy Loss

이미지를 기준으로 올바른 텍스트를 찾는 손실:

$$
L_{\text{image}}
=
\operatorname{CE}(S,\operatorname{label})
$$

텍스트를 기준으로 올바른 이미지를 찾는 손실:

$$
L_{\text{text}}
=
\operatorname{CE}(S^{\mathsf T},\operatorname{label})
$$

최종 손실:

$$
L
=
\frac{
L_{\text{image}}+L_{\text{text}}
}{2}
$$

즉,

```text
Image → Correct Text
Text  → Correct Image
```

두 방향의 분류 문제를 동시에 학습한다.

---

## 6. 생성 모델 대신 Contrastive Learning을 사용한 이유

초기에는 이미지가 주어졌을 때 caption을 생성하는 모델을 실험했다.

```text
Image → Caption Generation
```

하지만 정확한 문장을 생성하는 작업은 계산량이 크고 학습 효율이 낮았다.

실험 결과:

* Transformer 언어 모델 방식은 Bag-of-Words 예측보다 약 3배 느리게 학습
* 예측 목적함수를 Contrastive 목적함수로 교체하자 Zero-shot 전이 효율이 추가로 약 4배 향상

따라서 CLIP은 정확한 문장을 생성하지 않고

```text
어떤 텍스트가 이미지와 대응하는가?
```

만을 예측한다.

---

## 7. 학습 데이터

논문은 인터넷에서 수집한 새로운 데이터셋을 사용한다.

### WIT: WebImageText

* 약 **4억 개의 이미지-텍스트 쌍**
* 다양한 공개 인터넷 출처에서 수집
* 약 50만 개의 검색어 사용
* 검색어마다 최대 20,000개의 이미지-텍스트 쌍 수집
* 전체 단어 수는 GPT-2 학습에 사용된 WebText와 유사한 규모

검색어에는 다음이 포함된다.

* Wikipedia에서 자주 등장하는 단어
* 상호정보량이 높은 Bigram
* 검색량이 높은 Wikipedia 문서 제목
* 기존 목록에 없는 WordNet Synset

기존 이미지 데이터셋의 제한된 클래스 대신 광범위한 자연어 표현을 학습 데이터로 사용한다.

---

## 8. 모델 구조

### Image Encoder

두 종류의 아키텍처를 실험한다.

#### ResNet

* ResNet-50
* ResNet-101
* RN50x4
* RN50x16
* RN50x64
* ResNet-D 개선 적용
* Anti-aliased Blur Pooling 적용
* Global Average Pooling 대신 Attention Pooling 사용

#### Vision Transformer

* ViT-B/32
* ViT-B/16
* ViT-L/14
* ViT-L/14@336px

가장 좋은 결과는 **ViT-L/14@336px**에서 나타났다.

### Text Encoder

Transformer 구조 사용:

* 12개 Layer
* Width 512
* 8개 Attention Head
* 약 63M Parameters
* Vocabulary 크기 49,152
* Lower-cased BPE Tokenization
* 최대 문장 길이 76
* `[EOS]` 토큰의 최종 활성값을 텍스트 표현으로 사용

---

## 9. 학습 설정

* 모든 모델을 처음부터 학습
* ImageNet 사전학습 가중치 사용하지 않음
* 사전학습된 언어 모델 사용하지 않음
* 학습 기간: **32 Epochs**
* Optimizer: Adam
* Decoupled Weight Decay 사용
* Cosine Learning Rate Schedule
* Batch Size: **32,768**
* Data Augmentation: Random Square Crop
* Mixed Precision 사용
* Gradient Checkpointing 사용
* Temperature Parameter도 학습

가장 큰 모델의 학습 비용:

| 모델 | 학습 자원 | 학습 시간 |
|---|---:|---:|
| RN50x64 | 592 V100 GPUs | 18일 |
| ViT-L/14 | 256 V100 GPUs | 12일 |

ViT-L/14는 마지막에 336픽셀 해상도로 1 Epoch 추가 학습된다.

---

# 10. Zero-shot 분류

CLIP은 별도의 분류기 학습 없이 자연어만으로 새로운 분류 작업을 수행한다.

예를 들어 분류 클래스가 다음과 같다고 가정한다.

```text
plane
car
dog
bird
```

각 클래스 이름을 문장으로 변환한다.

```text
A photo of a plane.
A photo of a car.
A photo of a dog.
A photo of a bird.
```

텍스트 인코더로 각 문장의 임베딩을 계산한다.

입력 이미지와 각 텍스트 임베딩 사이의 코사인 유사도를 비교하고 가장 높은 클래스를 선택한다.

$$
P(y=k\mid x)
=
\operatorname{softmax}
\left(
\frac{
\cos(f_I(x),f_T(p_k))
}{\tau}
\right)
$$

여기서 \(p_k\)는 클래스 \(k\)를 표현하는 자연어 Prompt이다.

### 핵심 해석

Text Encoder는 자연어 설명을 이용해 분류기의 가중치를 생성하는 **Hypernetwork**처럼 작동한다.

```text
Class Description
       ↓
 Text Encoder
       ↓
Classifier Weights
```

따라서 새로운 작업을 수행할 때 학습 데이터를 추가하지 않고 클래스 설명만 바꾸면 된다.

---

# 11. Prompt Engineering

클래스 이름만 사용하는 것보다 문장 형태의 Prompt를 사용하는 것이 효과적이다.

### 기본 Prompt

```text
A photo of a {label}.
```

ImageNet에서 이 Prompt만 사용해도 정확도가 약 **1.3%p 향상**된다.

### Task-specific Prompt

동일한 단어가 여러 의미를 가지는 문제를 문맥으로 해결한다.

예:

```text
boxer
```

만 입력하면 운동선수인지 개 품종인지 불분명하다.

Oxford-IIIT Pets에서는 다음과 같이 사용한다.

```text
A photo of a boxer, a type of pet.
```

다른 예:

```text
A photo of {label}, a type of food.
A photo of {label}, a type of aircraft.
A satellite photo of {label}.
```

OCR 작업에서는 인식 대상 문자를 따옴표로 감싸는 것이 도움이 되었다.

---

## Prompt Ensembling

하나의 클래스에 여러 Prompt를 사용한다.

```text
A photo of a dog.
A photo of a small dog.
A photo of a big dog.
A blurry photo of a dog.
```

각 Prompt의 텍스트 임베딩을 평균하여 하나의 분류기 가중치로 사용한다.

ImageNet에서는 80개의 Prompt를 결합했다.

### 효과

* 단일 기본 Prompt 대비 추가로 약 **3.5%p 향상**
* Prompt Engineering과 Ensembling을 합하면 약 **5%p 향상**
* 대략 모델 계산량을 4배 늘린 것과 유사한 성능 개선

---

# 12. 주요 실험 결과

## ImageNet Zero-shot 분류

가장 큰 CLIP 모델의 결과:

| 모델 | ImageNet Top-1 |
|---|---:|
| Visual N-Grams | 11.5% |
| **CLIP** | **76.2%** |

* ImageNet 라벨 학습 데이터 128만 장을 사용하지 않고 76.2% 달성
* 지도학습으로 훈련된 기존 ResNet-50과 유사한 수준
* Top-5 정확도는 약 **95%**
* Inception-V4의 Top-5 성능과 유사

---

## 27개 데이터셋 평가

Zero-shot CLIP과 ImageNet으로 학습된 ResNet-50 특징 위의 지도학습 Linear Classifier를 비교했다.

결과:

* 27개 데이터셋 중 **16개에서 Zero-shot CLIP이 우수**
* ImageNet, CIFAR-10, CIFAR-100, STL-10 등에서 경쟁력 있는 결과
* STL-10에서 **99.3%** 정확도
* Kinetics-700에서 ResNet-50보다 **14.5%p 향상**
* UCF-101에서 **7.7%p 향상**

자연어 데이터가 명사 중심의 ImageNet보다 동작과 관련된 동사를 폭넓게 포함하기 때문에 Action Recognition에서도 좋은 결과를 보인 것으로 해석된다.

---

## Few-shot 학습과 비교

Zero-shot CLIP은 동일한 CLIP 특징 위에서 학습한 **4-shot Linear Classifier**의 평균 성능과 비슷하다.

또한 다른 공개 모델들의 특징을 사용하는 경우에는 가장 좋은 **16-shot Classifier**와 유사한 성능을 보인다.

ImageNet에서도 Zero-shot CLIP은 같은 특징 공간에서 클래스당 16개 샘플을 사용한 분류기와 비슷하다.

다만 데이터셋별 차이가 매우 크다.

* 일부 데이터셋에서는 1-shot보다 낮음
* 일부 데이터셋에서는 클래스당 184개 샘플과 비슷한 효과
* 중앙값: 클래스당 5.4개
* 평균: 클래스당 20.8개

---

# 13. 잘 수행하는 작업

CLIP은 다음과 같은 작업에서 강한 Zero-shot 성능을 보인다.

* 일반적인 객체 분류
* 음식 분류
* 자동차 종류 분류
* 사람의 행동 인식
* 동영상 행동 분류
* OCR
* 장면 분류
* 지리적 위치 추정
* 이미지-텍스트 검색
* 감정이 표현된 이미지 분류

특히 자연어 데이터에 자주 등장하는 시각적 개념에서 좋은 성능을 보인다.

---

# 14. 어려워하는 작업

CLIP은 다음 작업에서 상대적으로 낮은 성능을 보인다.

### 세밀한 전문 분류

* 항공기 세부 기종
* 꽃의 세부 종
* 일부 자동차 모델

### 추상적 또는 체계적인 추론

* 이미지 속 물체 개수 세기
* 가장 가까운 자동차까지의 거리 추정
* 합성 이미지의 구조적 속성 판단

### 전문 영역

* 위성 이미지 분류
* 림프절 종양 탐지
* 독일 교통 표지판 인식

### 분포 밖 데이터

손글씨 숫자인 MNIST에서는 Zero-shot 정확도가 약 **88%**였다.

이는 Raw Pixel 위의 단순 Logistic Regression보다 낮다.

즉, 대규모 데이터 학습이 모든 Out-of-Distribution 문제를 해결하는 것은 아니다.

---

# 15. Representation Learning

CLIP의 특징 표현 위에 Linear Classifier를 학습하여 표현 품질을 평가한다.

### 결과

* 대규모 지도학습 및 자기지도학습 모델과 경쟁력 있는 성능
* 최고의 CLIP 모델은 여러 데이터셋에서 기존 공개 모델보다 강한 Linear Probe 성능
* 동일한 계산량에서는 Vision Transformer 기반 CLIP이 ResNet 기반 CLIP보다 효율적
* Zero-shot 성능과 Linear Probe 성능 사이에 높은 상관관계 존재

상관계수:

$$
r=0.82
$$

하지만 대부분의 데이터셋에서 Zero-shot 성능은 완전 지도학습 Linear Probe보다 약 10~25%p 낮았다.

이는 CLIP의 표현 자체는 강하지만, 자연어만으로 그 표현을 완전히 활용하는 능력에는 개선 여지가 있음을 의미한다.

---

# 16. Scaling Law

CLIP의 모델 크기와 계산량을 증가시키면 평균 Zero-shot 오류가 일관되게 감소한다.

* 5개 ResNet 모델 비교
* 약 44배의 계산량 범위
* 평균 오류율이 Log-Log Linear Trend를 따름

```text
More Compute
     ↓
Larger CLIP Model
     ↓
Lower Zero-shot Error
```

다만 개별 데이터셋에서는 성능 변화가 불규칙할 수 있다.

논문은 현재 방식만으로 전체 작업에서 SOTA에 도달하려면 약 **1,000배의 추가 계산량**이 필요할 것으로 추정한다.

---

# 17. Distribution Shift에 대한 강건성

기존 ImageNet 모델은 ImageNet과 다른 분포의 이미지에서 성능이 크게 떨어진다.

평가 데이터:

* ImageNetV2
* ImageNet-A
* ImageNet-R
* ObjectNet
* ImageNet Sketch
* ImageNet-Vid
* YouTube-BB

Zero-shot CLIP은 유사한 ImageNet 정확도를 가진 기존 모델보다 분포 변화에 강했다.

### 예시

| Dataset | ResNet-101 | Zero-shot CLIP |
|---|---:|---:|
| ImageNet | 76.2% | 76.2% |
| ImageNetV2 | 64.3% | 70.1% |
| ImageNet-A | 2.7% | 77.1% |
| ImageNet-R | 37.7% | 88.9% |
| ObjectNet | 32.6% | 72.3% |
| ImageNet Sketch | 25.2% | 60.2% |

Zero-shot CLIP은 기존 모델과 비교해 Robustness Gap을 최대 약 **75% 감소**시켰다.

---

## ImageNet에 지도학습으로 적응시킨 경우

CLIP 특징에 ImageNet Linear Classifier를 학습하면 ImageNet 정확도는

$$
76.2\% \rightarrow 85.4\%
$$

로 **9.2%p 증가**한다.

그러나 다른 분포에서의 평균 성능은 오히려 조금 감소했다.

즉,

```text
특정 데이터셋에 대한 적응
        ↓
In-distribution 성능 증가
        ↓
Distribution Shift 강건성은 반드시 증가하지 않음
```

이는 데이터셋 비종속적 Zero-shot 평가의 중요성을 보여준다.

---

# 18. 데이터 중복 분석

인터넷에서 수집한 WIT 데이터에 평가 데이터가 포함되어 있을 가능성이 있다.

논문은 중복 이미지 탐지기를 사용해 사전학습 데이터와 평가 데이터의 중복을 분석했다.

결과:

* 35개 데이터셋 가운데 유의미한 정확도 차이가 나타난 데이터셋은 소수
* 중복으로 추정되는 샘플의 비율은 대부분 한 자릿수
* 전체 정확도 상승의 최대 추정치는 Birdsnap에서 약 **0.6%p**

따라서 데이터 중복이 일부 존재할 수 있지만 주요 실험 결과 전체를 설명할 정도는 아니라고 판단한다.

---

# 19. 장점

* 자연어를 이용한 유연한 Zero-shot 분류
* 새로운 클래스에 대한 추가 학습 불필요
* 고정된 출력 클래스에 제한되지 않음
* 4억 개의 이미지-텍스트 쌍으로 확장 가능한 학습
* 이미지와 텍스트를 연결하는 범용 표현 학습
* 일반 객체뿐 아니라 OCR, 행동, 장면 등 다양한 작업으로 전이
* 기존 ImageNet 모델보다 자연스러운 분포 변화에 강함
* Prompt 수정만으로 작업 정의 가능
* 이미지 검색과 텍스트 검색에도 활용 가능

---

# 20. 한계

### 많은 데이터와 계산량

* 4억 개의 이미지-텍스트 쌍 사용
* 32 Epoch 동안 총 128억 개의 이미지 샘플 처리
* 계산 및 데이터 효율성이 낮음

### 완전한 Out-of-Distribution 일반화 실패

* 학습 데이터와 크게 다른 입력에서는 성능 저하
* MNIST와 같은 손글씨 이미지에서 한계 확인

### 복잡한 추론 능력 부족

* 물체 개수 세기
* 공간적 관계 이해
* 거리 추정
* 작은 객체 탐지

등에서 성능이 낮다.

### 후보 클래스에 제한

CLIP은 주어진 텍스트 후보 중 하나를 선택한다.

따라서 자유롭게 새로운 설명을 생성하는 Image Captioning 모델보다 출력이 제한적이다.

### Zero-shot 평가의 방법론적 한계

* 모델 개발 과정에서 여러 Validation Set을 반복적으로 확인
* 엄밀한 의미의 완전한 Zero-shot 환경과 차이가 있음
* 평가 데이터셋이 CLIP의 능력에 맞춰 선택되었을 가능성 존재

### Few-shot 성능

CLIP은 Few-shot 학습을 직접 최적화하지 않았다.

일부 경우에는 Zero-shot에서 Few-shot으로 전환할 때 오히려 성능이 감소하는 비직관적 현상이 나타난다.

---

# 21. 편향과 사회적 영향

CLIP은 인터넷의 필터링되지 않은 이미지-텍스트 데이터를 사용하므로 사회적 편향을 학습한다.

FairFace를 이용한 실험에서는

* 비인간 범주로 잘못 분류된 비율이 인종 그룹별로 다르게 나타남
* 범죄 관련 범주로 분류되는 비율이 성별과 인종에 따라 다름
* 직업 및 외모 관련 단어에서 성별 편향이 관찰됨

특히 분류 후보에 어떤 클래스를 포함하는지에 따라 결과가 크게 달라졌다.

예를 들어 `child` 클래스를 추가하자 어린 사람의 이미지가 범죄 또는 비인간 범주로 잘못 분류되는 비율이 크게 감소했다.

즉,

```text
Prompt Design
Class Design
Decision Threshold
       ↓
Model Performance와 Bias에 직접적인 영향
```

CLIP의 유연성은 장점이지만 잘못된 클래스 설계가 새로운 피해를 만들 수 있다.

---

## 감시 기술 관련 위험

CLIP은 별도의 학습 데이터 없이 새로운 분류 작업을 만들 수 있기 때문에 특수한 감시 시스템의 개발 장벽을 낮출 수 있다.

실험 결과:

* 일반적인 CCTV 장면 분류: Top-1 정확도 91.8%
* 비슷한 오답 후보를 포함한 Stress Test: 51.1%
* 작은 객체의 존재 여부 판단: 무작위 수준에 가까움
* CelebA 100명 Zero-shot 신원 인식: 59.2%
* 후보를 1,000명으로 늘리면: 43.3%

현재 성능은 전문 감시 모델보다 낮지만, 과제별 학습 없이 사람의 이름이나 행동을 분류할 수 있다는 점에서 신중한 평가가 필요하다.

---

# 22. 기존 이미지 분류 모델과의 차이

| 항목 | 기존 지도학습 모델 | CLIP |
|---|---|---|
| 학습 데이터 | 이미지와 고정 라벨 | 이미지와 자연어 |
| 클래스 | 학습 시 고정 | 추론 시 자연어로 정의 |
| 새로운 작업 | Fine-tuning 필요 | Prompt만으로 수행 가능 |
| 출력 분류기 | 학습된 고정 가중치 | Text Encoder가 생성 |
| 전이 방식 | Supervised Transfer | Zero-shot Transfer |
| 표현 공간 | 주로 이미지 특징 공간 | 이미지-텍스트 공동 공간 |
| 분포 변화 | 상대적으로 취약 | 비교적 강건 |
| 주요 제한 | 새로운 라벨 필요 | Prompt와 후보 클래스에 의존 |

---

# 23. Future Work

논문에서 제안하거나 시사하는 발전 방향:

* Contrastive Learning과 Image Captioning 목적함수 결합
* Self-supervised Learning을 통한 데이터 효율성 개선
* Self-training 적용
* Zero-shot과 Few-shot 학습의 효과적인 결합
* 자연어로 표현하기 어려운 복잡한 작업을 지정하는 방법 개발
* 계산 효율성과 학습 효율성 개선
* 새로운 Zero-shot 전용 Benchmark 구축
* 모델의 편향과 사회적 영향을 조기에 평가하는 테스트 개발
* 민감한 응용 분야에 대한 정책 및 배포 기준 마련

---

# 24. 논문 핵심 정리

### Problem

기존 이미지 분류 모델은

* 미리 정해진 클래스만 예측
* 새로운 작업마다 라벨 데이터 필요
* 데이터셋별 Fine-tuning 필요
* 자연어로 표현되는 다양한 시각 개념을 충분히 활용하지 못함

### Solution

**CLIP**

```text
Image Encoder
      ↘
    Shared Embedding Space
      ↗
Text Encoder
```

이미지와 텍스트를 동일한 공간에 배치하고 올바른 이미지-텍스트 쌍을 찾도록 학습한다.

### Training

```text
N Images + N Texts
        ↓
N × N Similarity Matrix
        ↓
Correct Pair Similarity ↑
Incorrect Pair Similarity ↓
        ↓
Symmetric Cross-Entropy Loss
```

### Zero-shot Prediction

```text
Class Names
     ↓
Natural Language Prompts
     ↓
Text Encoder
     ↓
Zero-shot Classifier
     ↓
Compare with Image Embedding
```

### 핵심 기여

* 4억 개의 이미지-텍스트 쌍을 이용한 대규모 자연어 감독 학습
* 이미지와 텍스트의 Contrastive Pre-training
* 자연어 Prompt를 이용한 Zero-shot 이미지 분류
* ImageNet에서 Zero-shot Top-1 76.2% 달성
* 30개 이상의 다양한 비전 데이터셋에서 전이 능력 평가
* 기존 ImageNet 모델보다 높은 Distribution Shift 강건성 확인
* Prompt Engineering과 Prompt Ensembling의 중요성 제시
* 범용 비전 모델의 편향, 감시 활용 가능성 및 사회적 위험 분석

### 논문의 핵심 의미

CLIP은 이미지 분류를

```text
정해진 라벨을 예측하는 문제
```

에서

```text
자연어로 설명된 시각적 개념과 이미지를 연결하는 문제
```

로 전환했다.

이를 통해 컴퓨터 비전에서도 대규모 사전학습 모델이 자연어를 인터페이스로 사용하여 다양한 작업을 별도의 학습 없이 수행할 수 있음을 보여주었다.