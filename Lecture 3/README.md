# Lecture 3

![image](https://user-images.githubusercontent.com/70709889/175818475-4b272474-ebbd-45b7-80d2-15aef3435000.png)

위와 같이 이미지 별로 class score가 나왔다고 해보자. 이는 각 이미지에 대해 해당 class일 확률을 의미한다. 따라서, score 값들로 classifier를 qualify 하고자 한다.

qualify 하기위한 다양한 loss function이 있고 loss라는 뜻 자체가 부정적인 어감이기 때문에 보통 classifier가 얼마나 구린지 판단한다.
### Multiclass SVM loss
![image](https://user-images.githubusercontent.com/70709889/175818690-9774d7d7-dc1e-4009-9294-7a060ed728fc.png)

예측한 score에서 정답 score를 뺀 값을 0과 argmax 하여 정답 클래스를 제외하고 sum한다. 이 때, +1은 safety margin 이며 결과에 영향을 미치지는 않는다.

고양이를 예시로 들면 $max(0, 5.1-3.2+1) + max(0,-1.7-3.2+1) = max(0,2.9) + max(0, -3.9) = 2.9$

loss가 2.9인 것을 알 수 있다.

multiclass SVM loss는 값 자체보다는 정답과 정답이 아닌 label간의 격차를 보기 때문에 위의 수식에서 정답인 label까지 포함하거나 값이 조금 변해도 큰 차이는 없다. 마찬가지로 sum 대신 mean을 취해도 상관없지만 제곱을 취해주면 의미가 달라진다. ∵ 실제 격차보다 제곱배로 커지거나 작아지니까

#### Regularization
![image](https://user-images.githubusercontent.com/70709889/175819889-d8080297-4f87-454b-9518-40aed46d8ef5.png)

예를 들어 loss가 0이 되는 상황을 생각해보자. 이 때의 weight는 unique하지 않다. 이는 train data에만 fit하기 위해 모델이 복잡해진다는 의미이고 따라서 test set에서의 weight까지 fit 하기위해 weight에 penalty를 주는 방법이 **regularization**이다.

loss fucntion과 같이 모델의 복잡도를 평가하는 regularization 기법은 다양하고 각 기법마다 어떤 것을 복잡하다고 정의하는지 다르다. 예를 들어, L2 regularization은 weight의 l2 norm을 기준으로, L1 regularization l1 norm을 기준으로 평가한다. 

### Softmax
![image](https://user-images.githubusercontent.com/70709889/175819655-de8d75d3-128e-4e10-969b-7a2dfd10cd60.png)

Softmax는 class score를 0~1 사이의 확률로 normalize하는 과정이다. 당연히 모든 값을 더하면 1이 되고 multiclass SVM과는 다르게 정답인 label의 값 자체를 보며 이를 (정답인 label이 나올 확률을) 최대화하는 것이 목표이다.

![image](https://user-images.githubusercontent.com/70709889/175819790-7a6338e1-effa-41c7-ab8c-998eb7828ad4.png)

확률을 maximize하는 것보다 log 값을 maximize하는 것이 쉽기 때문에 log를 취해준 후, 그 값을 최대화하려고 노력한다. 음수가 취해진 이유는 위에서와 같이 loss의 어감에 따른 관례상의 이유이다.

### Optimize
지금까지 classifier를 통해 해당 이미지가 정답 label일 확률을 구하는 것까지 배웠다. 얘를 maximize (사실 음수가 붙어서 minimize) 하기 위해 weight를 update 시켜야 하는데 어떻게 할 것인지가 관건이다.
weight를 update 하는 이유는 당연히 loss가 weight에 input을 곱한 값들로 연산이 되기 때문. input을 바꿀 수는 없으니까..

이를 위해 loss function을 차원상의 그래프라고 생각하고 우리의 목표는 minima를 찾아가는 것.

=> 현재 위치에서 gradient(미분값)을 구하자. gradient는 현재 위치에서 증가하는 방향으로의 기울기니까 반대로 gradient를 빼주면 minima를 향해 update 할 수 있다.

마찬가지로 다양한 optimization 기법이 존재하고 learning late(보폭)등 고려해야 할 부분이 많다.

모든 데이터를 탐색하여 update하기엔 too expensive 하기 때문에 데이터를 minibatch로 나눠 minibatch 단위로 update를 실행하는 SGD(Stochastic Gradient Descent)를 주로 사용한다. (이 때 minibatch가 bias가 생기지 않게 batchNormalization이 필요)

### Image Feature
옛날에는(상대적으로) 이미지에서 feature를 추출하고 이를 linear classifier에 집어넣어 학습을 진행하는 2-stage 방식이었는데 deep learning으로 넘어면서 feature를 추출하는 방식이 사라졌다. 하지만 feature를 추출하는 과정도 흥미롭다.

1. Histogram of Oriented Gradients (HoG)

이미지를 8x8 grid로 나누어 grid마다의 edge를 histogram으로 만든 것

2. Bag of Words (BoW)

NLP에서 영감을 받았다. concept은 '이미지를 설명하는 단어 빈도를 feature로 사용하자' 이다. 근데 이미지에 적용하는 과정이 어렵기 때문에 visual word 라는 것을 정의해야했다.

- 이미지에서 random patches를 뽑아 clustering (ex. K-mean)
- input image에서 clustering 한 visual word의 빈도를 통해 인코딩
