# Human Activity Recognition using imaging sensor data with Federated setting
# Federated 환경에서 imaging 센서 데이터를 사용한 인간의 행동 인식
## 1. 프로젝트 개요

### 1) 프로젝트 설명

 본 연구는 인간의 행동을 센서로 측정한 센서 기반의 데이터를 이미지 데이터로 변환한 후 CNN모델을 사용하여 인간의 행동을 예측하도록 훈련하고, 개인 데이터를 공유하지 않는 연합학습을 사용하여 개인정보를 보호할 수 있는 프레임워크를 제안합니다. 
<br/>
<br/>

## 2. 사용한 기술 및 방법 설명

### 1) 전체 프로세스
<img width="100%" src="https://github.com/GayoungLee09/FL01/assets/81952512/58ab588f-5f74-49d6-8469-b6676a879a8f"/>

<br/> 
 스마트폰에 내장된 센서로 사람의 행동들이 수집된 수치(원시) 데이터를 정규화 한 후 극 좌표계로 바꾸고 GAF(Gramian Angular Field)를 사용하여 이미지 데이터로 변환합니다. 변환된 데이터를 FedAvg 기법을 사용한 연합 환경에서 CNN 모델을 사용하여 학습시킵니다. 최종적으로 새로운 테스트 입력이 들어올 때 사람의 행동을 예측하고자 합니다.
<br/> 
<br/>

### 2) 데이터셋
실험을 위해 UCI HAR 데이터셋을 사용하였습니다. UCI HAR 데이터셋은 6가지 행동(걷기, 계단 오르기, 계단 내려가기, 앉기, 일어서기, 눕기)을 하는 30명의 피실험자를 기록한 스마트폰 데이터셋 입니다. 실험 중 피실험자의 허리에 스마트폰이 부착되어 실험이 진행되었고 스마트폰에 내장된 3축 가속도 센서, 자이로 센서, 지자기 센서로 측정되었습니다.
<br/>
<br/>

### 3) 데이터 유형 변환

 먼저 원시의 센서 데이터는 수식(1)을 이용하여 [-1, 1] 사이의 값으로 정규화 합니다.
 
 $$\tag{1}\tilde{x_i} = \frac {(x_i - max(X)) + (x_i - min(X))}{max(X) - min(X)}$$
 
 정규화 된 센서 값들은 수식(2)를 사용하여 극 좌표$(θ,r)$로 변환합니다.
   
$$
\tag{2} g(\tilde{x}_i, t_i) = \begin{Bmatrix}\theta_i, r_i\end{Bmatrix} \quad \text{where}
\begin{cases} 
\theta_i = \arccos(\tilde{x}_i), & \tilde{x}_i \in \tilde{X} \\ 
r_i = t_i 
\end{cases}
$$
 
 정규화 된 시계열 센서 데이터를 극좌표로 인코딩하면 점 사이의 삼각합을 고려하여 시간 간격 사이의 상관 계수를 쉽게 추출할 수 있습니다. 시간 상관 관계는 Pearson의 상관 계수 기하학적 해석을 기반으로 역 코사인 각도에서 추출할 수 있습니다. 상관 계수는 벡터 사이 각도의 코사인과 동일합니다. 마지막으로 타임스탬프 와 사이의 상관관계는 $cos(θ_i,θ_j)$를 사용하여 계산되고, GAF는 수식(3)을 사용하여 G로 정의됩니다.

 $$\tag{3} G = \begin{bmatrix}
cos(\theta_1+\theta_1) & \cdots & cos(\theta_1+\theta_n) \\
\vdots & \ddots & \vdots \\
cos(\theta_n+\theta_1) & \cdots & cos(\theta_n+\theta_n)
\end{bmatrix}$$

 따라서 GAF는 타임스탬프가 증가함에 따라 시간 상관의 형태로 로컬 시간 관계를 보존할 수 있는 표현을 제공합니다. 실험에서는 극좌표로 변환한 후 좌표의 1초(128개의 값)를 하나의 이미지로 변환하였습니다. Time window의 길이는 1초이며, 이 이미지 데이터는 모델 학습 과정에 사용하였습니다.
 
<img width="55%" src = "https://github.com/GayoungLee09/FL01/assets/81952512/b08b6355-7e24-4696-be2c-a99c23feed09" />

(데이터 유형 변환 과정)
<br/>
<br/>

### 4) Federated Learning
 연합학습은 클라이언트(로컬) 모델에서 개인의 데이터로 학습을 각각 진행하고 학습 파라미터를 중앙(글로벌) 모델로 전달합니다. 글로벌 모델에서는 FedAvg 연합학습 알고리즘을 사용하여 로컬 모델로부터 전달받은 학습 파라미터를 계산하고 글로벌 모델을 학습한다. 글로벌 모델의 학습 파라미터는 다시 로컬 모델로 전달되고 과정을 반복하여 로컬 모델을 최적화합니다.
 이러한 연합 학습의 구조는 개인 데이터를 서로 공유하지 않고 학습이 가능해 사용자 데이터 간의 프라이버시를 보장할 수 있다는 장점이 있습니다. 따라서 개인 프라이버시의 문제가 있는 개인의 활동, 앱 사용 데이터, 자동차의 주행 정보, 개인의 병원 진료 기록 등의 데이터가 사용될 때 연합학습의 사용이 용이합니다. 
 수식(4)를 사용하여 모든 클라이언트가 보유한 전체 훈련 데이터에 대한 클라이언트의 로컬 훈련 데이터 비율을 계산하여 스케일링 인수를 얻을 수 있습니다. 계산된 배율 인수의 값을 기반으로 각 로컬 모델의 가중치를 배율 조정하여 각 구성 요소에 대해 매개변수 평균화를 수행하였습니다.

$$\tag{4} f(w) = \sum_{k=1}^{K}\frac{n_k}{n}F_k(w)\ where\ F_k(w) = \frac{1}{n_k}\sum_{i \in P_k}f_i(w)$$

<img width="55%" src = "https://github.com/GayoungLee09/FL01/assets/81952512/5e133f1a-c3ef-46fb-ad8b-521d260bc15e" />

(Federated Learning의 구조)
<br/>
<br/>

### 5) CNN
 CNN(Convolutional Neural Network)은 수동으로 특징을 추출할 필요 없이, 데이터로부터 직접 학습하는 신경망 아키텍처입니다. CNN은 이미지를 분석하기 위해 패턴을 찾는데 유용한 알고리즘으로 데이터에서 이미지를 직접 학습하고 패턴을 사용해 이미지를 분류합니다. 
<br/>
<br/>

## 3. 실험 및 결과

### 1) 실험 환경
 본 연구에서는 센서 데이터로 연합 학습시킨 모델을 비교 모델로 사용하였습니다. 모델은 50 라운드로 학습되고 내부 epoch는 10입니다. Optimizer로 SGD(Stochastic Gradient Descent)를 사용하며 learning rate는 0.001로 설정하였습니다. 30명의 실험자 중 21명의 데이터셋을 모델 학습에 사용하고 나머지 9명의 데이터셋은 검증에 사용하였습니다.
<br/>
<br/>

### 2) 실험 결과
 Deep Learning방법에서 인식 모델로 사용한 CNN의 구조는 Input - Convolution - Pooling - Convolution - Pooling - Dense - Dense - Dense - Output으로, 2개의 Convolution Layer와 2개의 Pooling Layer, 3개의 Fully Connected Layer로 이루어집니다. 
 Comparison Model과 제안하는 방법에서 인식 모델로 사용한 CNN의 구조는 Input - Convolution - Pooling - Convolution - Pooling - Convolution - Pooling - Flatten – Fully Connected – Fully Connected - Output으로, 3개의 Convolution Layer와 3개의 Pooling Layer, 2개의 Fully Connected Layer, 1개의 Flatten Layer로 이루어집니다.
 
|Method|Accuracy|Setting|
|:---:|:---:|:---:|
|Deep Learning|81.57|Non-Federated Setting|
|**Proposed Method**|**76.47**|**Federated Setting**|
|Comparison Model|71.44|Federated Setting|

 성능 검증을 실시한 결과는 Federated 환경에서의 학습으로 인해 일반적인 딥러닝 방식(81.57%)보다 정확도가 낮았습니다. 하지만, 제안하는 방법은 연합 학습을 사용하여 분산 환경에서 프라이버시 문제를 처리할 수 있어 개인정보 보호가 가능합니다. 제안하는 방법은 센서 데이터를 이미지 데이터로 변환한 후 이미지 데이터의 특징 추출 및 학습에 강력한 CNN 모델을 사용하여, Comparison Model(71.44%)을 사용한 방법보다 나은 76.47%의 인식률을 달성해 유의미한 결과를 도출하였습니다.
<br/>
<br/>

## 4. 느낀점
 연구를 통해, 새로운 접근 방법을 도입하고 학습 정확도를 향상시키는 과정에서 많은 성취와 흥미로운 깨달음을 얻었습니다. 

 특히, 분산된 데이터에서 모델을 향상시키는 Federated Learning의 강력한 장점이 매력적으로 느껴졌습니다. 분산된 디바이스들에서 학습하고 중앙 서버에 데이터를 집중시키지 않아도 되는 연합학습의 접근 방식이 데이터 프라이버시와 보안에 뛰어난 기여를 하는 것을 확인하면서, Federated Learning에 대하여 깊이있는 공부를 하였고 관련된 다른 분야와의 연구에도 관심을 갖게 되었습니다.

 또한, 다양한 학습 모델의 결과를 해석하고 개선하기 위한 여러가지 방법들을 실행해보며, 모델들의 구조와 내부 과정을 더 잘 이해할 수 있었고 모델의 성능을 향상시킬 수 있었습니다.

 이러한 깨달은 점들은 연구를 통해 새로운 지식을 얻으면서, 관련된 분야에 대한 더 깊은 이해와 탐구에 동기부여가 되었습니다.

 
