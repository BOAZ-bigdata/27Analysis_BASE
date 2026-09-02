# Improving Language Understanding by Generative Pre-Training

**Alec Radford et al., OpenAI**

## 핵심 질문

- 자연어 처리에서 많은 라벨 데이터가 필요한 문제의 해결
- 대규모 비라벨 텍스트를 먼저 학습한 뒤 필요한 과제에 적용하는 방법의 제안
- **Pre-training → Fine-tuning**을 이용한 범용 언어모델 구축

> **핵심 아이디어: 많은 글을 먼저 학습하여 기본적인 언어 능력을 만든 뒤, 원하는 문제에 맞게 추가 학습하는 방식**

---

## 연구 배경

- 자연어 처리 모델 학습에 많은 양의 라벨 데이터가 필요한 문제
- 사람이 직접 라벨을 만드는 데 필요한 높은 시간과 비용
- 기존 Word Embedding이 주로 단어 수준의 정보만 학습한다는 한계
- 문장 전체의 의미와 긴 문맥까지 학습할 수 있는 범용 모델의 필요성

---

## 관련 연구

- 라벨이 없는 데이터와 라벨이 있는 데이터를 함께 사용하는 Semi-supervised Learning의 활용
- Word Embedding이나 LSTM을 이용한 기존 사전학습 연구의 존재
- 기존 LSTM이 긴 문맥을 처리하는 데 가지는 한계
- 긴 문장의 관계를 처리하기 위한 Transformer 활용

---

## 연구 방법과 모델 구조

### Pre-training

- 약 7,000권 이상의 책으로 구성된 BooksCorpus 활용
- 앞의 Token을 바탕으로 다음 Token을 예측하는 방식
- 문법, 문맥, 단어 관계 등의 기본적인 언어 정보 학습
- 별도의 사람이 만든 정답 라벨 없이 학습 가능한 구조

대규모 텍스트 → Transformer → 다음 Token 예측 → 기본 언어 능력 학습

### Fine-tuning

- Pre-training된 모델에 특정 과제의 라벨 데이터를 추가 학습하는 방식
- 자연어 추론, 질문응답, 문장 유사도, 텍스트 분류 등에 적용
- 과제마다 새로운 모델을 만들지 않고 하나의 사전학습 모델을 재사용하는 구조

Pre-trained Transformer → 특정 과제 데이터 → Fine-tuning → 과제별 예측

### Architecture

- Decoder 기반 Transformer 구조
- 12개의 Transformer Layer
- 12개의 Attention Head
- 768차원의 Hidden State
- 최대 512 Token 입력
- Masked Multi-Head Self-Attention을 이용한 Token 관계 학습

Token 입력 → Embedding → Self-Attention → Feed Forward → 다음 Token 예측

---

## 주요 수식

### Pre-training

L₁(U) = Σ log P(uᵢ | uᵢ₋ₖ, ..., uᵢ₋₁)

- 앞에 나온 Token을 이용해 다음 Token의 예측 확률을 높이는 목적함수

### Fine-tuning

L₃(C) = L₂(C) + λL₁(C)

- 특정 과제의 정답 학습과 기존 언어모델 학습을 함께 사용하는 방식
- 기존 언어 능력을 유지하면서 새로운 과제에 적응하기 위한 목적

---

## 실험 및 결과

- 자연어 추론, 질문응답, 문장 유사도, 텍스트 분류를 이용한 평가
- 총 12개 데이터셋 중 9개에서 당시 최고 수준의 성능 달성
- Story Cloze에서 기존 최고 성능 대비 8.9%p 향상
- RACE에서 5.7%p 향상
- MultiNLI에서 1.5%p 향상
- Pre-training 제거 시 전체 성능이 크게 감소하는 결과

---

## 기존 연구 대비 차별점

- 문제마다 별도의 모델을 만드는 기존 방식에서 벗어난 범용 모델 활용
- 하나의 Pre-trained Transformer를 다양한 NLP 과제에 재사용하는 방식
- LSTM 대신 Transformer를 활용한 긴 문맥 처리
- 이후 LLM의 기본 학습 방식이 된 **Pre-training → Fine-tuning 구조의 제시**

---

## 한계점

- 특정 과제에 적용하기 위해 여전히 라벨 데이터를 이용한 Fine-tuning이 필요한 구조
- 현재 LLM과 비교하면 제한적인 모델 규모와 학습 데이터
- Zero-shot 능력이 일부 나타났지만 아직 제한적인 수준

---

## 핵심 정리

> **대규모 비라벨 텍스트로 Transformer를 먼저 학습하고, 필요한 과제에 맞게 Fine-tuning하여 하나의 모델을 여러 자연어 처리 문제에 활용하는 연구**

---

# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

**DeepSeek-AI**

## 핵심 질문

- 기존 Reasoning Model이 사람이 만든 고품질 풀이 데이터에 의존하는 문제
- 사람이 풀이 과정을 직접 알려주지 않아도 강화학습만으로 추론 능력을 발전시킬 수 있는지에 대한 검증
- 정답에 대한 Reward를 이용해 모델이 스스로 문제 해결 전략을 찾도록 하는 접근

> **핵심 아이디어: 풀이 방법을 직접 가르치는 대신 정답에 보상을 주어 모델이 스스로 더 좋은 추론 방법을 찾도록 만드는 방식**

---

## 연구 배경

- 고품질 Reasoning 데이터 제작에 필요한 높은 시간과 비용
- 사람이 직접 사고 과정을 작성해야 하는 기존 방식의 한계
- 인간이 만든 풀이 방식에 모델의 탐색이 제한될 가능성
- 대규모 Reasoning 데이터 제작의 어려움

기존 방식: 사람이 좋은 풀이 작성 → 모델이 해당 풀이 학습 → 추론 능력 향상

연구 질문: **사람이 어떻게 생각해야 하는지 직접 알려주지 않아도 정답 보상만으로 추론 능력을 발전시킬 수 있는가에 대한 질문**

---

## 관련 연구

- 중간 사고 과정을 생성한 뒤 답을 도출하는 Chain-of-Thought
- 여러 풀이를 생성하고 일관된 답을 선택하는 Self-Consistency
- 여러 사고 경로를 탐색하는 Tree-of-Thoughts
- 사람이 만든 Reasoning 구조에 의존하는 기존 방식과 달리 강화학습을 통해 스스로 전략을 찾도록 하는 접근

---

## 모델 구조

- DeepSeek-R1-Zero와 DeepSeek-R1 모두 DeepSeek-V3-Base 기반
- Mixture-of-Experts(MoE) 구조
- 약 671B개의 전체 Parameter
- Token 하나를 처리할 때 약 37B Parameter 활성화
- Multi-head Latent Attention 활용
- 전체 모델을 항상 사용하는 대신 필요한 Expert만 선택적으로 사용하는 구조

입력 Token → 필요한 Expert 선택 → 일부 Parameter 활성화 → 출력 생성

---

## DeepSeek-R1-Zero

- 기존 방식의 SFT 단계를 생략하고 DeepSeek-V3-Base에 바로 Reinforcement Learning 적용
- 사람이 만든 Reasoning 예시 없이 모델이 자유롭게 문제 해결 방법을 탐색하는 방식
- 최종 정답과 출력 형식에 대한 Reward를 이용한 학습

기존 방식: Pre-trained Model → SFT → RL

R1-Zero: DeepSeek-V3-Base → RL → DeepSeek-R1-Zero

---

## GRPO와 Reward

### GRPO

**Group Relative Policy Optimization**

- 하나의 문제에 여러 답변을 생성하는 방식
- 각 답변이 받은 Reward를 같은 그룹 안에서 비교하는 방식
- 평균보다 상대적으로 좋은 답변의 생성 확률을 높이는 학습

문제 → 여러 답 생성 → Reward 계산 → 상대적 성능 비교 → 좋은 답 강화

### 주요 수식

Aᵢ = (rᵢ - mean(r)) / std(r)

- rᵢ: 특정 답변의 Reward
- mean(r): 같은 문제에서 생성된 답들의 평균 Reward
- std(r): Reward의 표준편차
- Aᵢ: 같은 그룹의 다른 답과 비교한 상대적인 성능

### Reward

Reward = Accuracy Reward + Format Reward

- Accuracy Reward: 수학 문제의 정답 여부나 코드의 테스트 통과 여부
- Format Reward: Reasoning과 최종 답을 요구한 형식으로 출력했는지에 대한 평가
- Reasoning 과정을 직접 가르치기보다 결과가 올바른지를 중심으로 평가하는 방식

---

## 실험 및 주요 결과

- AIME 2024에서 R1-Zero의 Pass@1이 15.6%에서 77.9%로 향상
- 강화학습이 진행될수록 더 긴 Reasoning과 자기검증 행동의 증가
- 자신의 풀이를 다시 확인하거나 다른 해결 방법을 탐색하는 행동의 자연스러운 발생

### Aha Moment

- 문제 풀이 도중 "Wait"와 같은 표현을 사용하며 자신의 풀이를 다시 검토하는 행동
- 사람이 직접 자기검증 방법을 가르치지 않았음에도 나타난 Reflection 행동
- 강화학습을 통해 새로운 Reasoning 전략이 자연스럽게 나타날 수 있음을 보여주는 결과

---

## R1-Zero의 문제점

- 지나치게 긴 Reasoning
- 낮은 가독성
- 영어와 중국어가 섞이는 Language Mixing
- 일반적인 글쓰기 능력 부족
- Open-domain Question Answering 능력 부족

> **문제를 잘 푸는 능력과 사람이 읽기 좋은 답변을 만드는 능력이 서로 다른 문제라는 점**

---

## DeepSeek-R1의 개선

- R1-Zero의 높은 추론 능력을 유지하면서 가독성과 일반적인 응답 능력을 보완하는 목적
- Cold-start SFT를 이용한 읽기 좋은 Reasoning 방식 학습
- Reasoning RL을 이용한 추론 능력 향상
- Rejection Sampling을 이용한 좋은 답변 선택
- Reasoning과 일반 데이터를 함께 이용한 추가 SFT
- Helpfulness와 Safety까지 고려한 두 번째 RL

DeepSeek-V3-Base → Cold-start SFT → Reasoning RL → Rejection Sampling → SFT → Second RL → DeepSeek-R1

---

## Distillation

- 큰 DeepSeek-R1의 Reasoning 능력을 작은 모델에 전달하는 방법
- DeepSeek-R1이 생성한 약 80만 개의 데이터 활용
- Qwen과 Llama 등의 Base Model을 Fine-tuning하는 방식
- 작은 모델에서도 높은 Reasoning 성능이 나타나는 결과

DeepSeek-R1 → Reasoning 데이터 생성 → 작은 모델 학습 → DeepSeek-R1-Distill

---

## 기존 연구 대비 차별점

- 사람이 좋은 풀이 과정을 직접 만들어주는 방식에서 벗어난 접근
- 정답을 검증할 수 있는 문제와 Reward를 이용한 Reasoning 학습
- 강화학습 과정에서 자기검증과 Reflection 행동의 자연스러운 출현
- 큰 모델의 Reasoning 능력을 작은 모델에 전달하는 Distillation 가능성 제시

---

## 한계점

- 간단한 문제에서도 필요 이상으로 오래 생각하는 Overthinking 문제
- 중국어와 영어 외 언어에서 발생할 수 있는 Language Mixing 문제
- 검색엔진이나 계산기 등의 Tool Use 능력 부족
- Structured Output 능력의 추가 개선 필요

---

## 핵심 정리

> **정답을 검증할 수 있는 문제와 적절한 Reward를 제공하면 사람이 풀이 과정을 직접 가르치지 않아도 LLM이 스스로 Reasoning 전략을 발전시킬 수 있다는 연구**

---

# 두 논문 비교

| 구분 | GPT | DeepSeek-R1 |
| --- | --- | --- |
| 핵심 문제 | 라벨 데이터 부족 | Reasoning 데이터 의존 |
| 핵심 방법 | Pre-training + Fine-tuning | Reinforcement Learning |
| 학습 목표 | 다음 Token 예측 | 문제의 정답 도출 |
| 주요 능력 | 언어 이해와 표현 | 추론과 문제 해결 |
| 기본 구조 | Transformer | DeepSeek-V3-Base |
| 핵심 개념 | Pre-training, Fine-tuning | GRPO, Reward, Reasoning |
| 주요 발견 | 사전학습의 효과 | 자기검증과 Reflection의 자연스러운 발생 |
| 주요 의미 | 범용 LLM 학습 방식의 기반 | Reasoning Model 학습 방식의 발전 |

## 두 논문의 연결

GPT: 대규모 텍스트 → Pre-training → 기본 언어 능력 → Fine-tuning → 다양한 NLP 과제 해결

DeepSeek-R1: Pre-trained LLM → Reasoning 문제 → 여러 답 생성 → Reward → Reinforcement Learning → 추론 능력 향상

> **GPT가 LLM의 기본적인 언어 능력을 어떻게 학습할 것인지에 대한 연구라면, DeepSeek-R1은 이미 학습된 LLM의 추론 능력을 어떻게 더 발전시킬 것인지에 대한 연구**
