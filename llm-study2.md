---
layout: default
title: Basics of Deep Learning
---

## Machine Learning이란 무엇인가
- 많은 AI분야 중의 한 분야이다.
- Data를 통해 배우고, manual instructions을 최소화하고 예측값을 추론
- 통계적 알고리즘에 의존
- Supervised learning과 Unsupervised learning으로 나뉨
  - Supervised learning: 정답이 있음
  - Unsupervised learning: 정답이 없음

## Supervised Learning Framework
- STEP 1: 모델의 형태를 디자인하기
- STEP 2: 모델의 목적을 정의하기
- STEP 3: 학습된 데이터로 최적의 parameter를 찾기
- STEP 4: 학습하지 않았던 입력값을 주고 결과값 측정하기

## Regression Function
- x를 input으로 받아 y를 output으로 출력하는 함수
- f(x)를 찾는 것이 목적이 아닌, x를 가지고 y예측을 잘하는 f(x)를 찾는 것이 목표
- 좋은 Regression Function은 어떤 x가 y에 관련 있는지, 없는지를 잘 파악함
- 데이터에서 여러 값이 하나의 x에 있는 경우 좋은 값은 평균값이다.

## Parametric model
- 유한한 갯수의 parameter를 통해 그 관계를 학습하는 모델
- Ex) Linear regression 등
- 입력값들과 파라미터의 선형 결합으로 예측을 수행함
- 현실의 데이터들이 모두 linear model에 맞진 않지만 해석이 쉬움

## Non - parametric methods
- 함수의 형태에 대해 명시적인 가정을 하지 않음
- 데이터를 가능한 잘 따르는 함수 f(x)를 찾는 것이 목적
- 모델 형식이 틀릴 위험이 없음
- 매우 많은 수의 데이터가 필요
- Ex) K-NN, Decison Tree 등

## Regression function의 목적
- 입력 x가 주어졌을 때 y가 예측함수와 최대한 가까워야 한다는 것이 목적
- 보통 조건부 기대값을 f(x)로 사용
- 예측 오차를 제곱해서 평균을 낸 값(MSE)을 최소화 하려면 조건부 기대값을 사용 해야함

## Optimization
- Loss function이나 error를 최소화하는 파라미터를 찾는 것
- Linear regression에서는 MSE를 최소화함
- Good Optimization은 Good predictive performance와 연관됨

## Good fit vs Over-fit or Under-fit
- Good fit: 학습 데이터에 잘 맞는 모델
- Over-fit: 너무 학습데이터에 맞춰서 새로운 데이터에 성능이 떨어짐
- Under-fit: 학습데이터조차 잘 설명하지 못함

## Prediction accuracy vs Interpretability
- Prediction accuracy: Non-linear model이나 Deep learning모델은 정확도가 높음
- But 해석이 어려움
- Interpretability: Linear model은 해석하기가 쉬움
- But 정확도가 비교적 낮음

## Parsimony vs Complexity
- Parsimony: 변수가 적고 구조를 단순히 함
- Complexity: 변수가 많고 구조를 어렵게 함
- 간단하고 해석이 쉬운 모델일 수록 오버피팅을 방지하고 비용이 싼 경향이 있음
