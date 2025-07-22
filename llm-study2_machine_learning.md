---
title: Basics of Deep Learning
---
## Machine Learning
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
- 가정한 모델: \\(y = \beta_0 + \beta_1 x + \varepsilon\\)
- \\(\beta_0\\): intercept, coefficients라고 부름
- \\(\beta_1\\): slope, parameter라고 부름
- \\(\varepsilon\\): error
  
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
- \\(\sigma(z) = \frac{1}{1 + e^{-z}}\\)
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

## Gradient Descent
- 손실함수의 최소값을 구하기 위해 해당 함수의 기울기를 따라 조금씩 이동하는 방법
- \\(\theta \leftarrow \theta - \alpha \cdot \nabla J(\theta)\\)
- \\(\theta\\): 모델의 파라미터(진행에 따라 업데이트 됨)
- \\(\alpha\\): 학습률 (learning rate)
- \\(\nabla J(\theta)\\): 손실 함수 \( J(\theta) \)에 대한 기울기
- 손실함수가 convex할때 global minimum을 보장함
- 하지만 대부분 non-convex하기에 local minumum 혹은 saddle point가 존재함
- 손실함수가 미분 가능할때만 작동 가능
- 데이터가 매우 많으면 학습 시간이 매우 길어짐

## Overfitting
- 모델이 traing data에 너무 과하게 접학하여 test data에 대해 적합하지 못하는 현상
- 모델구조가 너무 복잡해지면 오버피팅이 발생함
- training data가 너무 적으면 발생함
- noise까지 학습하면 발생함
- 그 결과로 training data에 대한 성능은 좋음
- 하지만 test data에 대한 성능이 나쁨

## Regularization
- 앞선 오버피팅을 방지하기위해 모델을 규제하는 방법
- 기존 손실함수에 regularization term을 추가함
  - regularization term: \\(\lambda\\) * Penalty
- 여기서 penalty의 종류에 따라 ridge와 lasso로 나뉨

## Ridge Regression(L2 정규화)
- 모델의 모든 파라미터를 작게 만듬
- 파라미터를 0으로 만들지는 않음
- 오버피팅을 줄이지만 모든 변수는 남음
- 각 파라미터를 작게 만들어 복잡도를 낮춤

## Lasso Regression(L1 정규화)
- 파라미터 일부를 아예 0으로 만듬
- 불필요한 파라미터를 제거 가능함
- 모델 해석이 쉬움
- 고차원 데이터에서 유용함
- 중요하지 않은 파라미터를 제거해서 모델을 단순하게 만듬

## Bias-Variance Trade-off
- High Bias and Low Variance
  - 단순 모델
  - 오버피팅 확률 낮음
  - Ex) linear regression
- Low Bias and Hugh Variance
  - 복잡 모델
  - 오버피팅 확률 높음
  - Ex) 고차 다항 회귀

## Decision Tree
- 데이터를 특정 조건에 따라 분할하며 예측하는 모델
- 쉽게 해석이 가능하고 트리구조로 설명이 쉬움

## Regression Tree
- 연속적인 값을 어떠한 방식으로 구분해야 하는지 찾는 모델
- greedy: 당장 선택할 조건만 보는 식
- 나누어진 구간의 평균값을 대표값으로 사용
- 분할기준: MSE가 최소가 되도록 분할함
  
## Classification Tree
- 범주형 값을 어떠한 방식으로 구분해야 하는지 찾는 모델
- 나누어진 구간에서 갯수가 가장 많은 값을 대표값으로 사용
- 분할기준: gini impurity, entropy를 사용

## Pruning(가지치기)
- 오버피팅된 트리구조에서 특정 가지를 잘라내는 과정
- Pre-pruning
  - 트리를 만드는 도중에 더 이상 나누지 않도록 멈추는 방식
  - 빠르고 간단함
  - 너무 일찍 멈추면 underfitting 가능성이 있음
- Post-pruning
  - 트리를 다 만들고 가지를 잘라내는 방식
  - 더욱 세밀하게 가지치기 가능
  - 구현이 복잡하고 시간이 오래걸림

## Bootstrapping
- 한정된 데이터를 복원추출로 여러번 샘플링해서 모델에 학습시키는 기법
- 데이터 부족을 극복할 수 있음
- 분산이 감소함
- overfitting 방지 효과

## Bagging(Bootstrap Aggregating)
- 여러개의 bootstrapped 데이터셋을 만듬
- 각 데이터셋으로 독립적인 모델을 학습시킴
- 예측 시에는 각 모델의 결과를 모아서 종합함
  - 회귀방식 -> 평균을 사용
  - 분류방식 -> 다수결을 사용
- overfitting을 방지함
- 분산이 감소함 -> 일반화 향상
- 예측 안정성 향상

## Random Forest
- 여러개의 Decision Trees를 만들어 각 트리의 예측 결과를 결합하는 모델
- 각 트리에서 분할 기준을 선택할 때 무작위하게 선택
- 정확도가 높음
- overfitting방지
- 해석이 어려움

## Boosting
- 여러개의 weak learner를 순차적으로 학습시키며 오류를 보완하는 앙상블 기법
- AdaBoosting
  - 이전에 틀린 문제의 가중치를 높이고
  - 이전에 맞은 문제의 가중치를 낮추는 방식
 
## SVM(Support Vector Machine)
- 데이터를 두 클래스로 나누는 최적의 경계선(Hyperplane)을찾는 지도학습 알고리즘
- 용어정리
  - Margin: Hyperplane에서 가장 가까운 데이터까지의 거리
  - Support Vector: Margin을 결장하는 핵심 데이터
- Margin이 가장 넓은 선을 찾는 것이 목표 -> optimal hyperplane을 찾기
- 현실 데이터는 대부분 직선으로 완벽히 나누어지지 않음 -> 확장의 필요성
  - soft margin: 약간의 노이즈를 허용 -> C라는 하이퍼파라미터를 사용
    - C가 크면 오차 적게 허용 -> overfitting 위험
    - C가 작으면 오차 많이 허용 -> 일반성 높아짐
  - kernel trick: 데이터의 차원을 높여서 해결

## Margin-based Loss function
- Margin이란 예측의 확신도를 수치로 표현한 값 -> Margin = \\(y \hat{y}\\)
  - \\(y \in \ {+1, -1\}\\): 실제 정답
  - \\(\hat{y}\\): 예측값
  - Margin이 +1에 가까우면 예측도 강하고 정답률도 높음
  - Margin이 -1에 가까우면 예측은 강하지만 정답률이 낮음
- Hinge Loss : max(0,1-\\(y \hat{y}\\))
  - SVM에서 사용하는 함수
  - Margin이 1보다 잡으면 loss가 존재하고 1이상이면 loss가 0임
    - 즉 정답일지라도 margin이 충분히 크지 않으면 패널티를 줌
  - 확신있는 예측을 하도록 학습을 유도함
- Log Loss(Logistic Loss): \\( \log(1 + e^{-y \hat{y}}) \\)
  - Logistic Regression에서 사용하는 함수
  - Margin이 커질수록 로그스케일로 감소
  - 확률적 해석이 가능함
- Exponential Loss: \\(e^{-y \hat{y}}\\)
  - Boosting 계열에서 사용
  - margin이 작아질수록 지수적으로 큰 패널티
  - 오분류를 강하게 억제하지만 overfitting의 위험이 있음

## Clustering
- 비슷한 데이터끼리 묶는 비지도학습
- lable(정답)이 없는 상태에서 패턴이나 구조를 자동으로 발견
- inner similarity는 최대화, inter similarity는 최소화
- Ex) K-means, Hierarchical clustering 등

## K-means
- 데이터를 K개의 클러스터로 나누는 알고리즘
- K를 선택
- 초기 중심점을 랜덤하게 설정
- 각 데이터를 가까운 중심점에 할당
- 각 클러스터의 평균값으로 중심점을 갱신
- 해당 과정 반복(변화가 없을 때까지)
- 유한한 데이터는 유한한 조합이 있기에 반드시 종료됨
- 초기 중심점에 따라 결과가 달라짐
- 최적의 K를 선택하는 방법으로 elbow-point가 있음

## Dimension Reduction
- 고차원 데이터를 낮은 차원으로 변환하는 기법
- Data Manifold
  - 데이터는 고차원에 있지만 실제로는 더 낮은 차원(manifold)에 깔려 있음
  - 이러한 data manifold를 저차원 공간으로 펼치는 것이 Dimension Reduction의 목적임
- MDS(Multidimensional Scaling)
  - 고차원 데이터의 점들 간 거리를 저차원에서도 최대한 유지하는 방식
  - 모든 데이터간의 유클리드 거리행렬을 계산
  - 그 거리를 최대한 보존하며 저차원에 매핑
- PCA(Principal Component Analysis)
  - 데이터의 분산을 최대한 보존하는 방향으로 축을 찾아 회전시켜 매핑하는 방식
  - 전체에서 분산이 가장 큰 방향을 찾음
  - 그에 수직인 방향에 두 번째로 큰 축을 찾음
  - n차원 데이터를 k차원(축으로 정한 갯수)으로 투영

## Nearest Neighbors
- 새로운 데이터가 주어졌을 때 가장 가까운 기존데이터와 비교해서 예측하는 알고리즘
- 가장 가까운 k개의 이웃을 찾고 다수결로 분류
- 거리 측정 방식은 유클리드 거리, 맨하탄 거리, 코사인 유사도 등도 가능함
- 학습시간: 빠름
- 예측시간: 느림 -> 전부 비교해야함
- 간단하고 직관적임
- 차원의 저주를 극복 못함

## Linear Classifiers(image기준)
- 이미지 데이터는 원래 2D배열임 이를 1D벡터로 펼침(flatten)
- 각 클래스별 가중치 벡터를 곱하여 해당 값을 출력으로 가짐
- 출력값을 시그모이드에 넣어 0~1로 압축 -> 확률처럼 해석 가능
- 클래스가 여러개일 경우 시그모이드가 아닌 softmax를 사용함

## Cross Entropy
- 예측값과 레이블(정답) 사이의 차이를 계산하는 손실함수
- Softmax + Cross Entropy는 classifier 문제의 표준 조합임

## KL Divergence(Kullback-Leibler Divergence)
- Cross Entropy는 예측값과 정답을 비교하지만
- KL Divergence은 두 확률 분포간의 유사도를 측정함
