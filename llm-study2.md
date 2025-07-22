---
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

## Linear Regression에서 인자 명칭
- 가정한 모델: $y = \beta_0 + \beta_1 x + \varepsilon$
- $\beta_0$: intercept, coefficients라고 부름
- $\beta_1$: slope, parameter라고 부름
- $\varepsilon$: error
  
## Linear Regression Model의 정확도 평가
- RSS(Residual sum of squares)
  - (실제값 - 예측값)의 제곱의 합
  - 0에 가까울수록 정확함
- RSE(Residual standard error)
  - RSS를 n-2로 나눈 후 루트를 씌움
  - 0에 가까울수록 정확함
- R-squared
  - (TSS(분산) - RSS)/TSS = 1 - RSS/TSS
  - RSS가 0이면 값이 1로 나오며 정확한 상태
  - RSS가 커져서 분산에 가까워지면 가장 부정확한 상태
- 각 인자는 상관관계가 있는 것이지 인과관계가 있는 것이 아님

## 왜 Least Squares를 쓰는가?
- Independent and Identically Distributed(I.I.D) 가정
  - 각 데이터는 서로 독립임
  - 모든 데이터는 같은 확률 분포에서 나옴
- 먼저 Maximum likelihood estimator(MLE)를 알아야함
  - 우리가 원하는 θ값을 찾을 때 사용
  - 데이터가 주어졌을 때 특정 데이터가 나타날 확률의 곱(IID가정이라 독립임) = Likelihood
  - 여기서 Likelihood Function을 최대화 하는 θ를 찾는것이 MLE임
  - 계산의 편의를 위해 log스케일로 사용
- 결론적으로 수학적으로 계산이 간단하고 통계적으로 정당하며, 실제로도 잘 작동하기 때문

## Linear Regression에서 Qualitative predictors는 어떻게 계산하는가
- 회귀모형에서 사용하려면 숫자로 변환해야함
- 이 상황에서 쓰는 것이 더미변수(dummy variables)임
- n개의 데이터에서 n-1개의 더미변수를 만들어야함
- 나머지 하나는 baseline으로 설정되기 때문에 n-1

## Interaction
- 여러가지 변수들은 서로 상호작용을함
- 해당 상호작용도 회귀식에 반영할 수 있음
- 2가지의 인자면 두 변수의 곱을 식에 반영하면 됨
- Hierachy
  - 2가지의 변수를 하는 경우 혼자 쓰이는 경우를 먼저 계산해야함
  - 그 이후 서로의 상호작용을 반영해야함
  - 당연하게도 독립적으로 영향을 미치는 부분이 훨씬 크기 때문임

## Nonlinear relationship
- Interaction에서 두 변수의 곱을 식에 반영하는 방식으로 2차 이상식도 반영이 가능
- 같은 변수가 두번 이상 곱해지면 됨
- 해당 변수의 파라미터가 양수면 시너지관계

## Least square은 항상 존재하는가?
- 그렇지 않음, 해당 값을 계산할때 역행렬이 들어가는데, 역행렬이 존재하지 않을 수 있음

## Best subset selection
- 가능한 모든 조합을 비교하여 가장 최적의 식을 찾는 것
- 매우 큰 데이터의 경우 지수스케일로 복잡도가 커지는 단점이 있음
- 차원이 커질수록 필요로하는 변수가 지수스케일로 커지기에 curse of dimensionality가 발생
- 오버피팅이 발생할 수 있음

## Stepwise selection(forward)
- 변수가 없는 상태에서 성능향상이 가장 큰 변수부터 추가
- 모든 조합을 계산하지 않는 장점이 있음
- best solution이 보장되지 않음

## Stepwise selection(backward)
- 모든 변수가 들어간 식에서 기여도가 가장 작은 변수를 제거하며 가장 최적의 식을 찾는 방법
- 데이터의 갯수 n이 변수의 갯수 p보다 항상 커야만 가능

## Stagewise selection
- 초기 식에 변수를 추가할 때 앞선 식을 상수로 취급하여 그대로 가져감
- 추후 파라미터가 업데이트 될 수 있음

## 어떻게 최적의 모델을 찾는가?
- 변수를 모두 포함하는 것이 무조건적으로 최적의 식을 도출할 것임
- 하지만 성능계산은 training data가 아니라, 반드시 test data로 해야함 
- 랜덤하게 10~20%를 test data로 빼둬야함
- Validation: training data중에서 test로 쓸 부분
- K-fold cross validation
  - 전체 데이터를 k로 나누어 각 부분을 test data로 쓰는 것
  - k번의 교차검증을 하는 것임
  - 편향성을 없앨 수 있음

## Regression vs Classification
- Regression: 수치값을 찾는 것
  - ex) 시험 점수: 87점
- Classification: 범주를 찾는 것
  - ex) 시험 등급: B등

## Sigmoid Function
- $\sigma(z) = \frac{1}{1 + e^{-z}}$
- 확률로 해석이 가능하기에 classification에서 씀
- 경계를 설정해 줄 수 있음
- ![sigmoid](./images/sigmoid.png)
- 출처: https://rgbitcode.com/blog/senspond/55

## Logistic Regression
- 주어진 변수를 바탕으로 결과값을 예측하는 모델
- Linear regression처럼 계산하고 sigmoid에 넣어 확률로 변환
- 특정 경계값을 정하여 그 이상이면 1 아니면 0
- 식이 non-linear하기에 파라미터가 closed form이 아님 -> 수식으로 못품
- 따라서, Optimization을 사용하여 근사적으로 구함

## Optimization
- Gradient Descent
  - 손실함수의 최소값을 구하기 위해 해당 함수의 기울기를 따라 조금씩 이동하는 방법
  - $\theta \leftarrow \theta - \alpha \cdot \nabla J(\theta)$
  - \theta : 모델의 파라미터(진행에 따라 업데이트 됨)
  - \alpha : 학습률 (learning rate)
  - \nabla J(\theta): 손실 함수 \( J(\theta) \)에 대한 기울기
