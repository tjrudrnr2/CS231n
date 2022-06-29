# Lecture 6

## Summary
Optimize를 위한 다양한 activation function, preprocessing, weight initialization,
hyperparameter tuning을 배우고 결국 자주 사용하는건 ReLU, Xavier Initialization, Batch Normalization,
Random Search이다.

## Content
### Activation Function
#### Sigmoid
![image](https://user-images.githubusercontent.com/70709889/176481024-721c3c60-420d-48f0-833d-a02bb4f9a1dd.png)

$\sigma(x)=\frac{1}{(1+e^{-x})}$
0~1 사이의 값만 출력가능한 기본적인 활성화함수이다. 

하지만 양수나 음수로 큰 값이 들어오면 기울기가 0에 수렴하기 때문에
gradient가 saturate (vanishning) 되고 학습이 잘 되지 않는다.

또한, output이 항상 양수이기 때문에 update가 비효율적이라는 단점이 있다.

#### tanh
![image](https://user-images.githubusercontent.com/70709889/176482886-14a05dd0-8f4c-44a7-9b86-39484399f980.png)

$f(x) = tanh(x)$

sigmoid의 두번째 단점을 해결한 활성화함수이다. 출력값이 -1~1로 변경되고 zero-center 되어
update의 비효율성을 줄였지만 여전히 양 끝단에서 gradient가 saturate되는 문제가 있다.

#### ReLU
![image](https://user-images.githubusercontent.com/70709889/176483308-23a3ce60-6dec-46c6-be36-af58e5fb4c1e.png)

$f(x) = max(0,x)$
음수는 0, 양수는 입력값 그대로 출력하며 가장 많이 사용되는 활성화함수이다. max 연산이기 때문에 연산속도가 빠르고
기울기가 1이므로 수렴속도가 빠르다는 장점이 있다. 하지만 zero-center이 아니기 때문에 양수에서는
saturate 문제를 해결했지만 음수에서는 gradient가 0이라는 단점이 있다. 즉, 음수일 경우 gradient가 죽어버리는
dead ReLU가 발생한다. 이를 방지하기 위해 초기화나 learning rate를 잘 확인하자.

#### Leaky ReLU
![image](https://user-images.githubusercontent.com/70709889/176484513-34297805-d9f5-44d8-9c68-929214e0261e.png)

$f(x)=max(0.01x, x)$
ReLU가 데이터의 절반을 버려버리는 문제를 해결한 버전이다. 

이와 비슷하게 음수인 부분의 기울기를 파라미터로 조절하는 RReLU가 있다. $PReLU=max(\alpha, x)$

![image](https://user-images.githubusercontent.com/70709889/176487053-6565ffda-b425-49a0-bb31-5256626a2a60.png)

ELU 같은 경우 다른 ReLU들과 비교하여 zero mean에 근접했다. 하지만 음수값일 때 saturation의 여지가
남아있지만 이는 noise에 robust하다고 한다.

#### Maxout
$max(w_{1}^{T}x+b_{1},w_{2}^{T}x+b_{2})$
이 함수는 형식이 정해져있지 않고 위의 값들을 비교하여 큰 값을 사용한다. 두 개의 선형함수를 사용한다는
점에서 ReLU와 Leaky ReLU와 비슷하며 선형이기 때문에 saturation 되지 않는다.

단점은 그만큼 파라미터 수가 두배로 증가한다는 것... (잘 쓰이지는 않을 듯?)

## Data Preprocessing
![image](https://user-images.githubusercontent.com/70709889/176489102-39acb619-67e9-4ed2-8242-be1d5e1d257d.png)

sigmoid에서 봤듯이 입력값이 항상 양수이면 optimize에 비효율적이다. 따라서, 데이터도
zero-center으로 만들어준 후 normalization을 해 줄 필요가 있다.

보통 이미지에서는 이미 각 차원 간에 scale이 맞춰져있기 때문에 zero-mean 정도만 사용하고 normalization을 적용하지 않는다.

## Weight initialization
가중치가 0이라면 행렬곱 연산도 제대로 되지 않을 뿐더러 모든 filter가 같은 결과를
뱉기 때문에 layer를 deep하게 쌓는 의미가 없다. (symmetry breaking) 따라서, 가중치를 초기에 초기화 해주는 것이 중요!

1. 작은 random gaussian 값들로 초기화
```
W = 0.01 * np.random.randn(D,H)
```
![image](https://user-images.githubusercontent.com/70709889/176490623-f46a6430-0a0c-4393-846d-9878f063c204.png)

tanh을 사용하여 layer 별 activation 값의 변화를 출력한 결과이다. tanh을 사용했기 때문에 평균은
0 주변으로 분포하지만 문제는 표준편차 값이 0으로 수렴한다는 것. 이것은 **가중치 값이 너무 작기 때문에**
layer를 거치면서 값이 점점 작아지는 것! 이 문제는 forward 뿐만 아니라 backward에서도 존재한다.
2. W 초기화에 0.01 대신 1.0을 사용
```
W = 1.0 * np.random.randn(fan_in, fan_out)
```
![image](https://user-images.githubusercontent.com/70709889/176491100-411ea52a-beff-4ca6-93b0-a6cebc3b40fe.png)

값들이 saturation 하는 것을 확인할 수 있다. tanh이기 때문에 연산을 진행하면서 값이 점점 양 끝으로 몰려
출력이 -1 혹은 1에 몰린 것이다. => 초기화에 사용될 적절한 가중치 값을 찾기가 너무 어렵다!!
3. Xavier initialization
```
W = np.random.rand(fan_in, fan_out) / np.sqrt(fan_in)
```
![image](https://user-images.githubusercontent.com/70709889/176491523-51f16869-20e3-41d4-8244-0b967a870581.png)

가장 널리 사용되는 방법이며 기존과 동일한 gaussian에 입력값의 제곱근으로 나눠준 것이다.

=> 입력의 수가 작으면 가중치가 크고 입력의 수가 많으면 가중치가 작다. (더 큰 값으로 나누니까)

![image](https://user-images.githubusercontent.com/70709889/176491995-7dd5d8b6-5fd8-457b-842a-17a1457efa9d.png)

하지만, 첫번째 결과는 tanh일 때이고 ReLU를 사용하면 값의 절반이 0이 되고 표준편차가 반감되기 때문에
잘 작동하지 않는다.

## Batch Normalization
결국, layer의 입력을 gaussian하게 만들고 싶은 것이기 때문에 convolution 연산을 거친 후, 현재
배치에서 계산한 평균으로 빼주고 표준편차로 나눠주어 강제로 normalization 해주는 것이다.

$x = \frac{x-E[x]}{\sigma(x)}$
![image](https://user-images.githubusercontent.com/70709889/176492674-a75af261-9f09-4d4b-a6ae-30e74b5367af.png)

따라서 위와 같이 layer와 activation function 사이에서 수행된다.

근데 꼭 입력 값을 linear한 구간으로 강제해야할까?

=> 유연성을 주기 위해 어느 정도의 saturation을 허용해주자.

$y = \gamma * x + \beta$
처음 연산 후에 scaling과 shift를 추가한 것이다.

## 학습할 때 체크할 점들
#### loss 값 찍어보기
![image](https://user-images.githubusercontent.com/70709889/176494616-47082072-29cf-441e-b156-4ec4beade1a9.png)

class가 10개인 softmax classfier의 loss는 -log(1/10)이 될 것이다. 이는 약 2.3이고 위와 같이 초기에
점검해볼 수 있다.

#### sanity check 
데이터 일부만 학습시켜 cost와 loss를 확인해보는 것. 정상적이라면 loss가 줄어들 것이다.

#### learning rate 바꿔보기
보통 1e-3, 1e-6과 같이 10의 지수승으로 지정하며 바꿔볼 때 역시 log를 취하여 지수 범위 위주로
비교한다. 

learning rate가 지나치게 작을 경우 minima로 수렴하지 못하기 때문에 loss와 cost가 좀처럼 변하지 않고

너무 클 경우에는 cost가 발산하면서 NaNs라는 값을 뱉는다.

#### 하이퍼파라미터 조절
모든 데이터를 학습 시키면서 tuning 하는 것은 시간이 오래 걸리기 때문에
**Cross-Validation**을 사용한다.

train set으로 학습시키고 validation set으로 평가하는 것이다.
1. coarse stage
loss값을 보면서 Epoch 몇번만으로 넓은 범위에서 적절한 값을 고르기
2. fine stage
Epoch를 길게 학습하면서 좁은 범위 설정하기. 이 때, 최적값이 범위의 중앙에 오게 설정!

cost가 이전 epoch보다 커지거나 급격하게 증가한다면 다른 hyperparameter를 선택하면 된다.

3. grid search
hyperparameter를 고정된 값과 고정된 범위로 tuning 하는 것

하지만 실제로는 random search가 효율적이다. 모델이 특정 파라미터에 민감하다면 random search는 
중요한 파라미터에게 더 많은 샘플링이 가능하다.

=> 아래 그림에서 grid search는 초록색 함수에 3개의 값을 찾지만 random searhc는 많은 시도를 하는 모습

![image](https://user-images.githubusercontent.com/70709889/176497022-b8b480c3-23c7-4771-ab43-ffbe0cb9833d.png)

## loss를 항상 monitor 하자
![image](https://user-images.githubusercontent.com/70709889/176497508-d725adf5-8366-4588-be5f-1dc9c1a82297.png)

loss 값은 항상 주시하면서 값이 상승하거나 수렴하지 않는지 확인하자.

근데 overfitting도  확인해야하니 validation accuracy까지 확인해야 하지않나?
