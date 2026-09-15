# RAG & LoRA Paper Review

## 0. 전체 개요

RAG와 LoRA는 모두 대규모 언어모델을 활용하기 위한 기술이지만 해결하려는 문제의 종류가 다름.
RAG는 모델이 보는 정보를 바꾸고, LoRA는 모델이 정보를 처리하는 방식을 바꿈.

| 구분 | RAG | LoRA |
|---|---|---|
| 논문 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | LoRA: Low-Rank Adaptation of Large Language Models |
| 핵심 문제 | 모델 내부 지식만 사용할 때 발생하는 지식 한계 | 대규모 모델 전체를 Fine-tuning할 때 발생하는 높은 비용 |
| 핵심 해결법 | 외부 문서 검색 후 생성 모델에 전달 | 기존 모델을 고정하고 작은 저랭크 행렬만 학습 |
| 핵심 구성 | Retriever + Generator | Frozen Model + Low-Rank Update |
| 핵심 관점 | 필요한 지식을 어디서 가져올 것인지 | 모델을 어떻게 효율적으로 적응시킬 것인지 |
| 대표 키워드 | Retrieval, External Knowledge, Generation | Fine-tuning, Low-Rank, Parameter Efficiency |

### 전체 구조

#### RAG

```text
질문
  ↓
관련 문서 검색
  ↓
검색 문서 + 질문
  ↓
언어모델
  ↓
답변 생성
```

#### LoRA

```text
기존 언어모델
  ↓
모델 가중치 고정
  +
작은 저랭크 행렬만 학습
  ↓
특정 작업에 맞게 모델 적응
```

---

# 1. RAG

## Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

## 1.1 어떤 문제를 해결하려 했는가? — Abstract

기존 사전학습 언어모델은 학습 과정에서 얻은 지식을 모델 파라미터 내부에 저장하는 구조.

이러한 방식의 주요 문제점.

- 새로운 지식 추가의 어려움
- 기존 지식 수정의 어려움
- 최신 정보 반영의 어려움
- 생성 결과의 근거 확인 어려움
- 사실과 다른 정보 생성 가능성

RAG의 해결 방향은 **언어모델의 내부 지식과 외부 문서 지식을 함께 사용하는 구조**.

논문에서는 BART 기반 생성 모델을 **Parametric Memory**, Wikipedia 문서 인덱스를 **Non-Parametric Memory**로 구성.

### 기존 방식

```text
질문
  ↓
언어모델 내부 지식
  ↓
답변
```

### RAG 방식

```text
질문
  ↓
관련 문서 검색
  ↓
질문 + 검색 문서
  ↓
언어모델
  ↓
답변
```

### 핵심

**외부 지식을 검색하여 언어모델의 생성 과정에 추가하는 구조**

---

## 1.2 연구 동기와 문제점 — Introduction

대규모 언어모델은 많은 지식을 파라미터 내부에 저장할 수 있지만 해당 지식을 정확하게 불러오거나 수정하는 데 한계 존재.

### 기존 언어모델의 구조

```text
대규모 학습 데이터
       ↓
   언어모델 학습
       ↓
수많은 모델 파라미터
       ↓
    내부 지식
```

문제 발생 시 모델 파라미터 자체를 다시 학습해야 하는 구조.

```text
과거 정보 학습
      ↓
   언어모델
      ↓
새로운 정보 필요
      ↓
기존 모델 내부에는 없음
```

RAG에서는 외부 문서 저장소를 별도로 구성함으로써 이러한 문제 완화.

```text
언어모델
   +
외부 문서 저장소
```

외부 문서 변경만으로 새로운 지식 반영 가능.

---

## 1.3 관련 연구 동향 — Related Works

RAG 이전에도 검색 기술과 언어모델을 결합하려는 연구 존재.

주요 연구 방향.

| 연구 방향 | 특징 |
|---|---|
| Open-domain QA | 검색된 문서에서 정답 부분 추출 |
| Fact Checking | 주장과 관련된 증거 검색 |
| Memory Network | 외부 메모리를 신경망에 추가 |
| Dense Retrieval | 질문과 문서를 벡터로 표현하여 검색 |
| BART / T5 | 검색 없이 모델 내부 지식으로 다양한 생성 작업 수행 |

기존 검색 기반 연구는 특정 작업에 특화된 구조가 많았다는 특징.

RAG의 주요 확장점은 하나의 **Retrieval + Generation 구조를 여러 NLP 작업에 적용한 점**.

---

## 1.4 연구 접근법과 모델 구조 — Method

RAG의 전체 구조는 크게 **Retriever**와 **Generator**의 두 부분으로 구성.

```text
                 RAG

질문 x
  │
  ▼
Query Encoder
  │
  ▼
Retriever
  │
  ▼
Top-K 관련 문서
  │
  ▼
질문 + 관련 문서
  │
  ▼
Generator
  │
  ▼
답변 y
```

질문을 Query Encoder로 변환한 뒤 관련 문서를 검색하고, 검색된 문서를 Generator에 전달하여 최종 결과 생성.

### 1.4.1 Retriever

Retriever의 역할은 **질문과 관련성이 높은 문서를 찾는 작업**.

논문에서는 DPR(Dense Passage Retriever) 사용.

```text
질문
  ↓
Query Encoder
  ↓
질문 벡터 q(x)


문서
  ↓
Document Encoder
  ↓
문서 벡터 d(z)
```

질문 벡터와 문서 벡터의 유사도를 계산한 뒤 관련성이 높은 문서를 선택하는 구조.

```text
              질문
                ↓
          질문 벡터 생성
                ↓
   ┌────────────┼────────────┐
   ↓            ↓            ↓
문서 A 벡터   문서 B 벡터   문서 C 벡터
   ↓            ↓            ↓
유사도 높음   유사도 중간   유사도 낮음
   ↓
관련 문서 선택
```

### 1.4.2 Generator

Generator의 역할은 **검색된 문서를 이용한 최종 답변 생성**.

논문에서는 BART-large 사용.

```text
질문
 +
검색 문서
    ↓
  BART
    ↓
최종 답변
```

Retriever와 Generator의 역할 구분.

| 구성 | 역할 |
|---|---|
| Retriever | 관련 자료 탐색 |
| Generator | 자료를 기반으로 답변 생성 |

---

## 1.5 RAG-Sequence와 RAG-Token

RAG에는 검색된 문서를 사용하는 방식에 따라 두 가지 모델 존재.

### RAG-Sequence

하나의 문서를 중심으로 전체 문장을 생성하는 방식.

```text
문서 A
  ↓
단어 1
  ↓
단어 2
  ↓
단어 3
  ↓
전체 답변
```

전체 Sequence에 동일한 문서가 사용되는 구조.

### RAG-Token

각 단어를 생성할 때 서로 다른 문서를 참고할 수 있는 방식.

```text
문서 A ──→ 단어 1

문서 B ──→ 단어 2

문서 B ──→ 단어 3

문서 C ──→ 단어 4
```

하나의 답변 안에서도 여러 문서의 정보를 조합할 수 있는 구조.

### 비교

| 구분 | RAG-Sequence | RAG-Token |
|---|---|---|
| 문서 사용 기준 | 전체 문장 | 각 Token |
| 문서 변경 | 전체 답변 동안 동일 | Token마다 변경 가능 |
| 특징 | 단순한 구조 | 여러 문서 정보 결합 가능 |

---

## 1.6 실험 구성과 결과 — Experiments

### 실험용 외부 지식

2018년 12월 Wikipedia dump 사용.

Wikipedia 문서를 약 100단어 단위로 분할하여 약 2,100만 개 문서 조각 생성.

```text
Wikipedia
   ↓
문서 분할
   ↓
100단어 단위 Passage
   ↓
약 21M Passages
   ↓
Embedding 생성
   ↓
FAISS Index 구축
```

### 실험 Task

| Task | Dataset | 목적 |
|---|---|---|
| Open-domain QA | Natural Questions | 질문 답변 |
| Open-domain QA | TriviaQA | 질문 답변 |
| Open-domain QA | WebQuestions | 질문 답변 |
| Open-domain QA | CuratedTrec | 질문 답변 |
| Abstractive QA | MS-MARCO | 자유로운 문장형 답변 |
| Question Generation | Jeopardy | 질문 생성 |
| Fact Verification | FEVER | 사실 여부 판단 |

### 주요 결과

Natural Questions 결과.

| Model | Exact Match |
|---|---:|
| T5-11B | 34.5 |
| DPR | 41.5 |
| RAG-Token | 44.1 |
| RAG-Sequence | 44.5 |

RAG가 기존 Parametric-only 모델과 검색 기반 모델보다 높은 성능을 기록한 결과.

### 생성 결과의 사실성 평가

Jeopardy Question Generation의 사람 평가 결과.

| 평가 | 비율 |
|---|---:|
| BART가 더 사실적 | 7.1% |
| RAG가 더 사실적 | 42.7% |
| 둘 다 좋음 | 11.7% |

RAG가 BART보다 더 구체적이고 사실적인 결과를 생성하는 경향.

---

## 1.7 무엇을 발견했고 한계점은 무엇인가? — Discussion

### 주요 발견

#### 1. 외부 검색 지식의 효과

언어모델 내부 지식만 사용하는 것보다 외부 문서를 함께 사용하는 방식에서 높은 성능 확인.

#### 2. Retriever 학습의 중요성

Retriever를 학습하지 않고 고정한 경우 여러 Task에서 성능 감소.

단순한 검색 시스템 추가만으로 충분하지 않으며 **Task에 적합한 검색 학습의 중요성** 확인.

#### 3. 외부 지식 교체 가능성

```text
기존 모델
   +
2016 Wikipedia Index
```

에서

```text
기존 모델
   +
2018 Wikipedia Index
```

로 문서 Index만 변경하여 모델의 활용 지식 변경 가능.

모델 전체 재학습 없이 지식을 변경할 수 있다는 장점.

### 한계점

#### 1. Retrieval Collapse

일부 작업에서 질문과 관계없이 비슷한 문서를 반복해서 검색하는 현상 발생.

```text
질문 A ──┐
질문 B ──┼──→ 동일한 문서 검색
질문 C ──┘
```

이 경우 Generator가 검색 문서를 무시하게 되며 RAG의 장점 감소.

#### 2. 검색 문서 품질 의존

```text
질문
  ↓
잘못된 문서 검색
  ↓
잘못된 정보 전달
  ↓
잘못된 답변 가능
```

외부 문서를 사용하더라도 문서 자체의 오류와 편향 가능성 존재.

---

## 1.8 결론 및 주요 요약 — Conclusion

RAG의 핵심 결론은 **Parametric Memory와 Non-Parametric Memory의 결합 가능성**.

```text
언어모델 내부 지식
       +
검색 가능한 외부 지식
       ↓
Knowledge-Intensive Task
성능 향상
```

주요 효과.

- Open-domain QA 성능 향상
- 생성 결과의 사실성 향상
- 외부 문서를 통한 지식 갱신
- 하나의 구조를 다양한 NLP Task에 적용 가능

---

## 1.9 기존 연구 대비 차별점

| 기존 연구 | RAG |
|---|---|
| Task별 별도 검색 구조 | 하나의 Retrieval-Generation 구조 |
| 검색 문장에서 정답 추출 | 검색 결과 기반 자유로운 답변 생성 |
| 모델 내부 지식 중심 | 내부 지식 + 외부 지식 |
| 지식 수정 시 재학습 필요 | 외부 Index 변경 가능 |

핵심 차별점은 **검색과 생성의 통합**.

---

## 1.10 핵심 아이디어

### 기존 방식

```text
질문
 ↓
언어모델
 ↓
답변
```

### RAG

```text
질문
 ↓
Retriever
 ↓
관련 문서
 ↓
Generator
 ↓
답변
```

**한 문장 요약**

외부 문서를 검색한 뒤 해당 문서를 근거로 답변을 생성하는 구조.

---

## 1.11 주요 수식

RAG의 핵심 개념을 단순화한 수식.

\[
p(y|x)
=
\sum_z
p(z|x)p(y|x,z)
\]

각 항의 의미.

| 기호 | 의미 |
|---|---|
| \(x\) | 질문 |
| \(z\) | 검색된 문서 |
| \(y\) | 생성된 답변 |
| \(p(z|x)\) | 질문과 문서의 관련성 |
| \(p(y|x,z)\) | 질문과 문서를 기반으로 답변을 만들 확률 |

### 직관적 해석

```text
좋은 답변 확률

=

좋은 문서를 찾을 확률
        ×
그 문서를 보고 좋은 답을 만들 확률
```

---

## 1.12 코드 관점의 이해

논문 구조를 단순화한 개념 코드.

```python
query = encode(question)

documents = retrieve(query, top_k=5)

answer = generator(
    question=question,
    context=documents
)
```

전체 흐름.

```text
encode()
   ↓
retrieve()
   ↓
generator()
```

논문 원본 실험 코드를 그대로 재현한 코드가 아닌 **RAG 동작 원리를 설명하기 위한 단순화 코드**.

---

# 2. LoRA

## Low-Rank Adaptation of Large Language Models

## 2.1 어떤 문제를 해결하려 했는가? — Abstract

기존 Fine-tuning은 사전학습 모델의 대부분 또는 전체 파라미터를 다시 학습하는 방식.

모델 크기가 증가하면서 다음 문제 발생.

- GPU 메모리 사용 증가
- 학습 시간 증가
- Task별 모델 저장 공간 증가
- 모델 교체 비용 증가

GPT-3 175B와 같은 대규모 모델에서는 전체 Fine-tuning 비용이 매우 큰 문제.

LoRA의 해결 방향은 **기존 모델 가중치를 고정하고 작은 Low-Rank 행렬만 학습하는 방식**.

### 기존 Fine-tuning

```text
기존 모델

W1 수정
W2 수정
W3 수정
W4 수정
...
전체 파라미터 수정
```

### LoRA

```text
기존 모델 W
    ↓
   고정

    +

작은 A, B
    ↓
   학습
```

---

## 2.2 연구 동기와 문제점 — Introduction

대규모 언어모델을 특정 Task에 맞게 Fine-tuning하면 Task마다 별도의 모델 파라미터가 필요.

```text
기본 모델
   ├─ 금융 Fine-tuning 모델
   ├─ 의료 Fine-tuning 모델
   ├─ 법률 Fine-tuning 모델
   └─ 요약 Fine-tuning 모델
```

모델 하나의 크기가 매우 크기 때문에 Task가 증가할수록 저장 및 학습 비용도 크게 증가.

LoRA의 핵심 가설.

**Fine-tuning 과정에서 발생하는 모델 변화 전체가 실제로 필요한 것은 아니며, 중요한 변화는 낮은 차원의 공간으로 표현 가능하다는 가설**

---

## 2.3 관련 연구 동향 — Related Works

대표적인 Parameter-Efficient Fine-Tuning 방법.

| 방법 | 특징 | 한계 |
|---|---|---|
| Full Fine-Tuning | 전체 파라미터 학습 | 높은 메모리·저장 비용 |
| Adapter | 작은 Layer 추가 | 추가 추론 연산 발생 |
| Prefix Tuning | 학습 가능한 Prefix 추가 | 입력 길이 사용 및 최적화 문제 |
| BitFit | Bias만 학습 | 표현력 제한 |
| LoRA | Weight Update를 Low-Rank로 표현 | 적용 대상과 Rank 설정 필요 |

Adapter 방식은 Transformer 내부에 추가 Layer가 직렬로 삽입되는 구조로 추가 Inference Latency 발생 가능.

---

## 2.4 연구 접근법과 모델 구조 — Method

LoRA의 가장 중요한 구조.

```text
                    ┌─────────────┐
             ┌─────→│ 기존 W      │
             │      │ 고정        │
입력 x ──────┤      └──────┬──────┘
             │             │
             │             ↓
             │            합산 ──→ 출력
             │             ↑
             │      ┌──────┴──────┐
             └─────→│ A → B       │
                    │ 학습        │
                    └─────────────┘
```

기존 모델 \(W_0\)는 그대로 유지.

추가되는 변화량만 \(A\), \(B\)라는 작은 행렬로 표현.

---

## 2.5 Low-Rank의 의미

기존 Fine-tuning.

```text
큰 Weight 행렬 전체 수정

████████████
████████████
████████████
████████████
```

LoRA.

```text
큰 변화량

████████████
████████████
████████████
████████████

       ≈

작은 B     ×     작은 A

██               ████████████
██               ████████████
██
██
```

큰 변화량 \(\Delta W\)를 두 개의 작은 행렬 곱으로 표현.

\[
\Delta W = BA
\]

---

## 2.6 Rank의 의미

Rank \(r\)은 LoRA가 모델 변화를 표현할 때 사용하는 작은 내부 차원.

직관적인 의미.

```text
r = 1
중요한 변화 방향 1개

r = 4
중요한 변화 방향 4개

r = 8
중요한 변화 방향 8개
```

LoRA의 핵심 가정.

```text
전체 모델의 가능한 변화
        ↓
실제로 Task 수행에 중요한 변화
        ↓
상대적으로 적은 수의 방향
```

따라서 매우 작은 \(r\)만으로도 충분한 성능을 얻을 가능성 존재.

---

## 2.7 Transformer에 LoRA 적용

Transformer Attention의 대표 Weight.

```text
             Attention

입력
 │
 ├──── Wq : Query
 │
 ├──── Wk : Key
 │
 ├──── Wv : Value
 │
 └──── Wo : Output
```

논문에서는 이러한 Attention Weight 중 어떤 부분에 LoRA를 적용하는 것이 효과적인지 실험.

같은 Parameter Budget에서는 \(W_q\)와 \(W_v\)를 함께 학습하는 방식에서 전반적으로 좋은 성능 확인.

---

## 2.8 실험 구성과 결과 — Experiments

### 실험 모델

- RoBERTa
- DeBERTa
- GPT-2
- GPT-3 175B

### 주요 Task

| Dataset | Task |
|---|---|
| GLUE | 자연어 이해 |
| WikiSQL | 자연어 → SQL |
| MultiNLI | 자연어 추론 |
| SAMSum | 대화 요약 |
| E2E NLG | 자연어 생성 |

### GPT-3 175B 실험

| Method | Trainable Parameters | WikiSQL | MNLI |
|---|---:|---:|---:|
| Full Fine-Tuning | 175,255.8M | 73.8 | 89.5 |
| LoRA | 4.7M | 73.4 | 91.7 |
| LoRA | 37.7M | 74.0 | 91.6 |

매우 적은 Trainable Parameter로 Full Fine-Tuning과 비슷하거나 높은 성능 확인.

---

## 2.9 Rank 실험

LoRA가 실제로 작은 Rank에서도 작동하는지 확인하기 위한 실험.

\(W_q, W_v\) 적용 시 WikiSQL 결과.

| Rank | Accuracy |
|---:|---:|
| 1 | 73.4 |
| 2 | 73.3 |
| 4 | 73.7 |
| 8 | 73.8 |
| 64 | 73.5 |

Rank가 크게 증가하더라도 성능 차이가 크지 않은 결과.

Task Adaptation에 필요한 변화가 실제로 매우 낮은 Rank로 표현될 수 있다는 근거.

---

## 2.10 무엇을 발견했고 한계점은 무엇인가? — Discussion

### 주요 발견

#### 1. 전체 파라미터 학습의 불필요성

Task에 적응하기 위해 모든 모델 Weight를 수정할 필요가 없다는 결과.

#### 2. 매우 작은 Rank의 효과

일부 Task에서 \(r=1\) 또는 \(r=4\) 수준에서도 높은 성능 확인.

#### 3. 중요한 Weight 선택의 중요성

한 Weight에 높은 Rank를 사용하는 것보다 여러 중요한 Weight에 낮은 Rank를 사용하는 방식의 효과 확인.

#### 4. 추가 Inference Latency 제거 가능

학습 종료 후

\[
W=W_0+BA
\]

형태로 병합 가능.

추가 Layer 없이 기존 모델과 동일한 방식으로 추론 가능.

### 한계점

#### 1. 모든 Task에서 작은 Rank가 충분한 것은 아님

사전학습과 매우 다른 Task나 언어에서는 더 큰 Rank가 필요할 가능성.

#### 2. LoRA 적용 Weight 선택 문제

어떤 Weight에 LoRA를 적용할 것인지에 대한 완전한 이론적 기준 부재.

실험과 경험적 설정에 대한 의존성 존재.

---

## 2.11 결론 및 주요 요약 — Conclusion

LoRA의 핵심 결론.

```text
대규모 Pretrained Model
          ↓
     Weight 고정
          +
   작은 Low-Rank Update
          ↓
Task-specific Adaptation
```

주요 효과.

- Trainable Parameter 감소
- GPU Memory 감소
- Task별 저장 공간 감소
- 빠른 Task Switching
- Full Fine-Tuning 수준의 성능
- 추가 Inference Latency 제거 가능

---

## 2.12 기존 연구 대비 차별점

| 방식 | 학습 방법 |
|---|---|
| Full Fine-Tuning | 기존 Weight 전체 수정 |
| Adapter | 새로운 Layer 추가 |
| Prefix Tuning | 학습 가능한 Prefix 추가 |
| LoRA | 기존 Weight는 고정하고 변화량만 Low-Rank 형태로 학습 |

LoRA의 핵심 차별점은 **새로운 모델 구조를 크게 추가하는 것이 아니라 기존 Weight의 변화량을 효율적으로 표현하는 방식**.

---

## 2.13 핵심 아이디어

### 기존 Fine-Tuning

```text
W
↓
W 전체 수정
```

### LoRA

```text
W
↓
고정

+

A × B
↓
작은 변화만 학습
```

**한 문장 요약**

대규모 언어모델 전체를 수정하지 않고 작은 Low-Rank 행렬만 학습하여 모델을 특정 Task에 적응시키는 방식.

---

## 2.14 주요 수식

기존 Layer.

\[
h=W_0x
\]

Fine-Tuning 이후.

\[
h=(W_0+\Delta W)x
\]

LoRA의 핵심 가정.

\[
\Delta W=BA
\]

따라서

\[
h=W_0x+BAx
\]

실제 LoRA에서는 Scaling까지 포함.

\[
h=W_0x+\frac{\alpha}{r}BAx
\]

### 직관적 의미

```text
최종 모델 출력

=

기존 모델이 만든 출력
        +
Task에 맞게 조금 수정한 출력
```

---

## 2.15 코드 관점의 이해

기존 Fine-Tuning.

```python
W.requires_grad = True
```

LoRA 방식의 개념.

```python
W.requires_grad = False

A = trainable_matrix()
B = trainable_matrix()

base_output = x @ W.T
lora_output = x @ A.T @ B.T

output = base_output + lora_output
```

Scaling 포함.

```python
output = base_output + (alpha / r) * lora_output
```

학습 완료 후.

```python
W_new = W + (alpha / r) * B @ A
```

논문 원본 구현을 그대로 옮긴 코드가 아닌 **LoRA 구조를 이해하기 위한 단순화 코드**.

---

# 3. RAG와 LoRA 비교

## 3.1 해결하는 문제의 차이

### RAG

```text
모델이 필요한 지식을
가지고 있지 않음
        ↓
외부 자료 검색
        ↓
검색 결과 기반 생성
```

### LoRA

```text
모델을 특정 Task에
맞게 학습하고 싶음
        ↓
전체 Fine-Tuning 비용이 큼
        ↓
작은 Low-Rank Parameter만 학습
```

---

## 3.2 최종 비교

| 구분 | RAG | LoRA |
|---|---|---|
| 핵심 문제 | 지식 부족 | Fine-Tuning 비용 |
| 핵심 방법 | 검색 후 생성 | 작은 Parameter만 학습 |
| 외부 문서 | 필요 | 필수 아님 |
| 모델 학습 | Retriever와 Generator 학습 가능 | LoRA Parameter 학습 |
| 기본 모델 변경 | 필수 아님 | 기본 Weight 고정 |
| 대표 구성 | Retriever + Generator | Frozen Weight + A/B |
| 대표 수식 | \(p(y|x)=\sum_zp(z|x)p(y|x,z)\) | \(\Delta W=BA\) |
| 주요 장점 | 외부 지식 활용 및 지식 갱신 | 저비용 Fine-Tuning |

---

# 4. 핵심 정리

## 4.1 RAG

```text
질문
 ↓
검색
 ↓
관련 문서
 ↓
생성 모델
 ↓
답변
```

**핵심 개념**

외부 지식을 검색하여 답변 생성에 활용하는 구조.

---

## 4.2 LoRA

```text
기존 모델
 ↓
Weight 고정

+

Low-Rank 행렬
 ↓
Task-specific 학습
```

**핵심 개념**

전체 모델을 다시 학습하지 않고 필요한 변화만 작은 행렬로 학습하는 구조.

---

## 4.3 두 논문의 핵심 차이

**RAG**

모델이 **무엇을 참고할 것인지**에 대한 방법.

**LoRA**

모델을 **어떻게 효율적으로 학습할 것인지**에 대한 방법.

---

# 5. 참고 논문

1. Patrick Lewis et al., **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**
2. Edward Hu et al., **LoRA: Low-Rank Adaptation of Large Language Models**
