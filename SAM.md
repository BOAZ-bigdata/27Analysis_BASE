# Segment Anything

**Alexander Kirillov et al., 2023**

## 1. Introduction

기존 이미지 분할 모델의 한계.

* 특정 데이터셋과 작업에 맞춰 학습.
* 새로운 객체나 이미지 분포에 대한 일반화 부족.
* Semantic, Instance, Interactive Segmentation별로 별도의 모델 필요.
* 대규모 범용 분할 데이터 부족.

논문의 목적.

* 이미지 분할을 위한 Foundation Model 구축.
* Prompt를 통해 다양한 객체를 분할하는 범용 모델 개발.
* 새로운 데이터셋과 작업에 대한 Zero-shot 전이.
* 대규모 분할 데이터셋 구축.

논문의 세 가지 핵심 구성.

```text
Task  → Promptable Segmentation
Model → Segment Anything Model
Data  → SA-1B Dataset
```

모델이 데이터 생성을 지원하고, 생성된 데이터로 모델을 개선하는 구조.

---

## 2. Segment Anything Task

제안한 작업.

**Promptable Segmentation**

이미지와 Segmentation Prompt가 주어졌을 때 유효한 Mask를 출력.

사용 가능한 Prompt.

* Foreground Point.
* Background Point.
* Bounding Box.
* Mask.
* Text.

전체 구조.

```text
Image + Prompt
       ↓
      SAM
       ↓
Valid Segmentation Mask
```

Prompt는 이미지에서 어떤 대상을 분할할 것인지 지정.

### Valid Mask

하나의 Prompt가 여러 객체를 의미할 수 있음.

예를 들어 사람의 옷 위에 Point를 입력한 경우.

```text
Point on shirt
      ↓
Shirt / Person / Shirt Detail
```

어떤 객체를 의미하는지 모호함.

SAM은 이 경우 평균적인 Mask를 생성하는 것이 아니라 가능한 객체 중 하나에 해당하는 유효한 Mask를 출력하도록 학습.

모호한 Prompt에 대해서는 여러 개의 Mask를 생성.

### Zero-shot Transfer

새로운 분할 작업을 적절한 Prompt로 변환.

예시.

```text
Object Detector
      ↓
Bounding Box
      ↓
SAM
      ↓
Instance Mask
```

기존 Object Detector의 Bounding Box를 SAM에 입력하여 Instance Segmentation 수행.

학습할 때 정의되지 않은 작업도 Prompt Engineering과 다른 모델의 결합을 통해 수행 가능.

---

## 3. Segment Anything Model

SAM의 세 가지 구성 요소.

```text
Image
  ↓
Image Encoder
  ↓
Image Embedding
  ↓
Mask Decoder ← Prompt Embedding ← Prompt Encoder
  ↓
Segmentation Masks
```

### Image Encoder

* MAE로 사전학습된 Vision Transformer 사용.
* 고해상도 이미지 처리.
* 이미지당 한 번만 실행.
* 생성된 Image Embedding을 여러 Prompt에서 재사용.

무거운 Image Encoder의 계산을 한 번만 수행하여 반복적인 Prompt 입력 비용 감소.

### Prompt Encoder

Sparse Prompt와 Dense Prompt를 구분하여 처리.

Sparse Prompt.

* Point.
* Bounding Box.
* Text.

Dense Prompt.

* Mask.

Point와 Box는 위치 정보와 Prompt 종류를 나타내는 Embedding으로 변환.

Mask Prompt는 Convolution을 이용해 Image Embedding과 같은 공간으로 변환.

Text Prompt는 CLIP Text Encoder 활용.

### Mask Decoder

Image Embedding과 Prompt Embedding을 결합하여 Mask 생성.

* Transformer 기반 Decoder 사용.
* Prompt와 이미지 사이의 양방향 Cross-Attention 수행.
* Mask와 예상 IoU Score 출력.
* Image Encoder보다 가벼운 구조.

Image Embedding이 미리 계산된 경우 Prompt 입력 후 약 50ms 안에 Mask 생성.

웹 브라우저에서도 대화형 분할 가능.

### Ambiguity 처리

하나의 Point가 전체 객체, 부분, 세부 부분을 동시에 의미할 수 있음.

SAM은 하나의 Prompt에 대해 3개의 Mask를 출력.

```text
Prompt
  ↓
Mask 1: Whole Object
Mask 2: Object Part
Mask 3: Subpart
```

각 Mask에 예상 IoU Score를 함께 출력.

학습 시 정답과 가장 잘 맞는 Mask의 Loss만 역전파.

단일 Mask만 출력하는 방식보다 모호한 Prompt를 효과적으로 처리함.

### Training Loss

Mask 학습에 두 Loss를 결합.

* Focal Loss.
* Dice Loss.

Interactive Segmentation 상황을 모방하여 하나의 Mask에 여러 차례 Prompt를 추가하는 방식으로 학습.

---

## 4. Segment Anything Data Engine

인터넷에는 자연 이미지가 많지만 Segmentation Mask는 충분하지 않음.

이를 해결하기 위해 모델과 데이터를 함께 개선하는 Data Engine 구축.

세 단계로 진행.

```text
Assisted-Manual
       ↓
Semi-Automatic
       ↓
Fully Automatic
```

### 1단계: Assisted-Manual

전문 Annotation 작업자가 SAM의 도움을 받아 Mask 생성.

* Foreground와 Background Point 입력.
* 필요할 경우 Brush와 Eraser로 수정.
* SAM이 실시간으로 Mask 제안.
* 특정 Semantic Class를 제한하지 않고 보이는 객체를 자유롭게 분할.

모델이 개선되면서 Mask당 평균 작업 시간 감소.

$$
34\text{초} \rightarrow 14\text{초}
$$

수집 결과.

* 이미지 약 12만 장.
* Mask 약 430만 개.

### 2단계: Semi-Automatic

SAM이 확실한 객체의 Mask를 자동 생성.

작업자는 아직 분할되지 않은 객체를 추가로 Annotation.

주요 목적.

* 눈에 잘 띄지 않는 객체 수집.
* 작은 객체와 다양한 객체의 Mask 확대.
* 이미지당 Mask 수 증가.

수집 결과.

* 이미지 약 18만 장 추가.
* Mask 약 590만 개 추가.
* 누적 Mask 약 1,020만 개.

### 3단계: Fully Automatic

사람의 Annotation 없이 SAM이 모든 Mask 생성.

이미지 전체에 \(32 \times 32\) Point Grid 입력.

```text
Regular Point Grid
        ↓
Multiple Masks per Point
        ↓
Confidence Filtering
        ↓
Stability Filtering
        ↓
Duplicate Removal
```

낮은 품질과 불안정한 Mask 제거.

Non-Maximum Suppression을 이용해 중복 Mask 제거.

작은 객체를 분할하기 위해 확대된 Image Crop도 처리.

최종 결과.

* 이미지 1,100만 장.
* Mask 11억 개.
* 이미지당 평균 약 100개 Mask.

---

## 5. Segment Anything Dataset

제안 데이터셋.

**SA-1B**

구성.

* 1,100만 개의 이미지.
* 11억 개의 Segmentation Mask.
* 고해상도 이미지.
* 라이선스를 확보한 이미지 사용.
* 공개 이미지의 얼굴과 차량 번호판 Blur 처리.

기존 최대 분할 데이터셋보다 약 400배 많은 Mask 포함.

### Mask Quality

SA-1B Mask의 99.1%가 완전 자동으로 생성됨.

자동 Mask와 전문가가 수정한 Mask 비교.

* 약 94%가 IoU 90% 이상.
* 약 97%가 IoU 75% 이상.
* 기존 연구의 작업자 간 일치도와 유사하거나 높은 수준.

자동 생성 Mask도 학습에 사용할 수 있을 정도로 품질이 높다고 판단.

### Dataset Properties

기존 데이터셋보다 이미지당 Mask 수가 많음.

* 작은 객체와 중간 크기 객체의 Mask 비율 증가.
* 이미지 중앙뿐 아니라 가장자리의 객체도 비교적 많이 포함.
* 다양한 크기와 복잡한 형태의 Mask 포함.

모든 데이터를 사용한 경우와 자동 생성 데이터만 사용한 경우의 성능 차이가 작게 나타남.

대규모 자동 Annotation이 효율적인 데이터 구축 방법이라고 판단.

---

## 6. Segment Anything RAI Analysis

SA-1B와 SAM의 지역적 편향과 사람 분할 성능 분석.

### Geographic Representation

SA-1B는 기존 COCO와 Open Images보다 유럽, 아시아, 오세아니아 및 중간 소득 국가의 비율이 높음.

하지만 다음 지역은 여전히 적게 포함됨.

* 아프리카.
* 라틴아메리카와 카리브 지역.
* 저소득 국가.

기존 데이터셋보다 다양성이 개선되었지만 전 세계를 균형 있게 대표하지는 못한다고 판단.

### Fairness in Segmenting People

사람 분할 성능을 다음 기준으로 비교.

* 인지된 성별 표현.
* 인지된 연령대.
* 인지된 피부색.

대부분의 그룹에서 성능 차이가 통계적으로 크게 나타나지 않음.

Point를 3개 제공했을 때 대부분의 그룹에서 약 90% 이상의 mIoU 기록.

그러나 SAM이 다른 시스템의 구성 요소로 사용될 경우 새로운 편향이 발생할 가능성 존재.

의류 분할에서는 인지된 성별 표현에 따른 편향 가능성도 확인됨.

제한된 실험만으로 전체 공정성을 보장할 수 없다고 판단.

---

## 7. Zero-Shot Transfer Experiments

SAM을 학습에서 보지 못한 데이터셋과 작업에 적용.

주요 평가 작업.

* Single Point Segmentation.
* Edge Detection.
* Object Proposal Generation.
* Instance Segmentation.
* Text-to-Mask.

---

### 7.1. Zero-Shot Single Point Valid Mask Evaluation

하나의 Foreground Point만으로 객체를 분할.

다양한 환경을 포함한 23개 Segmentation Dataset 사용.

비교 모델.

* RITM.
* SimpleClick.
* FocalClick.

결과.

* 23개 중 16개 데이터셋에서 RITM보다 높은 mIoU.
* 최대 약 47 IoU 차이.
* SAM의 3개 출력 중 정답과 가장 가까운 Mask를 선택하면 모든 데이터셋에서 RITM보다 높은 성능.
* 사람 평가에서도 SAM의 Mask가 RITM보다 높은 품질로 평가됨.

자동 평가지표에서는 낮은 성능을 보였지만 사람 평가에서는 더 좋은 Mask로 판단되는 사례 존재.

데이터셋의 Ground Truth가 가능한 모든 유효 Mask를 포함하지 않기 때문이라고 판단.

Point 수가 증가하면 기존 Interactive Segmentation 모델과의 성능 차이가 감소.

SAM은 많은 Point를 사용한 정밀 분할보다 하나의 모호한 Prompt에서 유효한 Mask를 생성하는 데 초점을 둔 모델.

---

### 7.2. Zero-Shot Edge Detection

SAM은 Edge Detection을 직접 학습하지 않음.

여러 Point Prompt에서 생성된 Mask의 경계를 결합하여 Edge Map 생성.

BSDS500 결과.

| Model | ODS | OIS | AP |
|---|---:|---:|---:|
| HED | 0.788 | 0.808 | 0.840 |
| SAM | 0.768 | 0.786 | 0.794 |

전용 학습 모델보다는 낮은 성능.

기존 Zero-shot Edge Detection 방법보다는 높은 성능.

Ground Truth에 없는 의미 있는 경계까지 예측하여 Recall은 높지만 Precision이 감소.

Edge Detection을 학습하지 않은 모델이라는 점을 고려하면 높은 전이 성능이라고 판단.

---

### 7.3. Zero-Shot Object Proposals

이미지의 객체 후보 Mask를 자동 생성.

LVIS에서 평가.

| Model | AR@1000 |
|---|---:|
| ViTDet-H | 63.0 |
| SAM | 59.3 |

전체 성능은 LVIS로 지도학습한 ViTDet-H보다 낮음.

하지만 다음 항목에서는 SAM이 더 높은 성능.

* Medium Objects.
* Large Objects.
* Common Objects.
* Rare Objects.

Small Object와 Frequent Object에서는 ViTDet-H보다 낮은 성능.

ViTDet-H는 LVIS의 Annotation 특성을 직접 학습하지만 SAM은 Zero-shot으로 적용된다는 차이 존재.

---

### 7.4. Zero-Shot Instance Segmentation

Object Detector가 생성한 Bounding Box를 SAM의 Prompt로 사용.

```text
Object Detector
      ↓
Bounding Box
      ↓
SAM
      ↓
Instance Mask
```

COCO와 LVIS에서 평가.

지도학습 ViTDet보다 Mask AP는 낮음.

하지만 사람 평가에서는 SAM Mask의 품질이 더 높게 평가됨.

* SAM 평균 품질 점수: 8.1.
* ViTDet-H 평균 품질 점수: 7.9.

SAM은 객체 경계를 더 자연스럽고 정밀하게 생성하는 경향.

ViTDet은 COCO와 LVIS의 Annotation 방식에 맞춰 학습되어 정량 평가에서 유리하다고 판단.

---

### 7.5. Zero-Shot Text-to-Mask

자유로운 텍스트로 객체를 지정하여 분할.

예시.

```text
“a wheel”
“beaver tooth grille”
“a wiper”
```

CLIP의 이미지와 텍스트 Embedding이 같은 공간에 정렬된다는 특성 활용.

학습에서는 CLIP Image Embedding을 Prompt로 사용.

추론에서는 CLIP Text Embedding을 Prompt로 사용.

단순한 객체 이름뿐 아니라 구체적인 표현도 일부 분할 가능.

텍스트만으로 대상을 찾지 못한 경우 Point Prompt를 추가하면 결과가 개선됨.

초기적인 가능성을 보여주는 실험이며 안정적인 Text-to-Mask 시스템은 아니라고 판단.

---

### 7.6. Ablations

Data Engine의 각 단계가 추가될수록 성능 향상.

자동 생성 Mask만 사용한 경우에도 전체 데이터를 사용한 경우보다 약 0.5 mIoU 낮은 수준.

자동 Mask가 모델 학습에 효과적이라고 판단.

데이터 크기 비교.

* 10만 이미지에서는 성능 크게 감소.
* 100만 이미지와 1,100만 이미지의 성능은 비교적 유사.
* 100만 이미지에도 약 1억 개의 Mask 포함.

Image Encoder 비교.

* ViT-B에서 ViT-L로 확장할 때 성능 크게 향상.
* ViT-L에서 ViT-H로 확장할 때 향상 폭 감소.

모델 규모 증가에 따른 성능 향상이 점차 포화되는 것으로 나타남.

---

## 8. Discussion

### Foundation Model

SAM은 대규모 데이터로 학습되고 다양한 후속 작업에 적용 가능한 모델.

다만 이미지 분할은 전체 컴퓨터 비전의 일부이므로 범위가 제한된 Foundation Model.

SAM의 주요 능력은 Self-supervised Learning보다 대규모 Mask를 이용한 Supervised Learning에서 발생.

Annotation을 자동화할 수 있다면 대규모 지도학습도 Foundation Model 구축에 효과적이라고 판단.

### Compositionality

SAM은 다른 모델과 결합할 수 있는 범용 분할 모듈.

예시.

```text
Object Detector + SAM
Gaze Detector + SAM
Text Encoder + SAM
3D Reconstruction System + SAM
```

Point, Box, Mask와 같은 단순한 Prompt를 공통 인터페이스로 사용.

모델 개발 당시 고려하지 않은 새로운 작업에도 활용 가능.

### Limitations

* 가느다란 구조를 놓치는 경우 존재.
* 작은 분리 영역을 잘못 생성하는 경우 존재.
* 확대 처리를 사용하는 전용 모델보다 경계가 덜 정밀할 수 있음.
* 많은 Point를 제공하는 정밀 Interactive Segmentation에서는 전용 모델보다 낮을 가능성.
* Prompt 처리 속도는 빠르지만 무거운 Image Encoder까지 포함하면 전체 처리가 실시간은 아님.
* Text-to-Mask 성능이 안정적이지 않음.
* Semantic Segmentation과 Panoptic Segmentation을 단순한 Prompt만으로 구현하기 어려움.
* 특정 전문 분야에서는 전용 모델보다 낮은 성능이 예상됨.

SAM은 모든 분할 문제에서 최고의 성능을 얻기 위한 모델이 아니라 다양한 작업과 데이터에 범용적으로 적용하기 위한 모델이라고 판단.

### Conclusion

논문의 핵심 구조.

```text
Promptable Segmentation Task
             +
Segment Anything Model
             +
     SA-1B Dataset
             ↓
General-Purpose Segmentation
```

핵심 기여.

* Promptable Segmentation 작업 제안.
* Point, Box, Mask, Text Prompt를 처리하는 SAM 개발.
* 모호한 Prompt에 대해 여러 유효 Mask를 생성하는 구조.
* 모델을 이용해 데이터를 수집하는 3단계 Data Engine 구축.
* 1,100만 이미지와 11억 Mask로 구성된 SA-1B 공개.
* 다양한 데이터셋과 분할 작업에서 Zero-shot 전이 성능 확인.
* 이미지 분할을 Foundation Model 방식으로 확장할 가능성 제시.

논문의 핵심 의미.

기존 이미지 분할.

```text
특정 데이터셋과 작업에 맞춰 학습한 모델
```

SAM.

```text
Prompt로 대상을 지정하면 다양한 환경에서 Mask를 생성하는 범용 모델
```

자연어 모델의 Prompt 기반 Foundation Model 개념을 이미지 분할에 적용한 연구.