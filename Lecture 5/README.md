# Lectre 5

## Summary
- Fully Conected Layer
- Convolution Layer
- Pooling Layer

## Content
### Fully Conected Layer
![image](https://user-images.githubusercontent.com/70709889/176332959-b7e5e0ef-529e-4df4-b2b5-e1d4eea64eae.png)

32x32x3 input이 들어올 때 class가 10개라고 가정해보자. output은 class 별 확률 즉, 10x1로 나와야 할 것이다. input은 3072x1로 flatten 시켜 연산하기 때문에 가중치 W는 3072x10의 dim을 가질 것.

### Convolution layer
![image](https://user-images.githubusercontent.com/70709889/176333308-7219d26d-8721-43b5-890c-a9e3daab61d6.png)

- input을 filter가 sliding하며 행렬곱을 수행한다.
- 각 filter는 input channel을 모두 cover
- **32x32x3**를 5x5x3가 stride 1로 (1칸씩 이동하면서) sliding 한다면 28x28x1의 activation map이 나올 것
- filter의 개수가 6개라면 위의 activation map이 6개 나오므로 **28x28x6**의 output 생성
- 이러한 convolution layer과 activation 함수를 여러개 쌓아 network를 만든다
- 앞단 layer에서는 low-level feature가 생성되고 뒷단으로 갈수록 high-level feature가 생성된다

#### stride 크기에 관한 고찰
7x7 size와 3x3 filter가 있다고 가정해보자.
- stride 1
  - 5x5 output
- stride 2
  - 3x3 output
- stride 3
  - 3칸씩 건너뛰는게 input size에 fit하지 않으므로 적용할 수 없다

근데 이렇게 하면 corner에 있는 값들은 filter를 몇번 거치지 않으니까 값을 제대로 반영할 수 없다
=> zero-padding

7x7 input과 3x3 filter에 1칸 zero padding한다면 총 input size는 9x9가 될 것이고 stride 1로 돌린다면 7x7 output이 나온다. => input size가 유지되네? activation map의 size를 유지할 때고 사용한다!

※ output size 계산하는 공식

$\frac{(input size - filter size + 2padding)}{stride} + 1$

#### parameter 개수
파라미터 개수는 filter로 연산되는 개수로 생각하자.
32x32x3 input에서 10개의 5x5x3 filter가 있다면 10x5x5x3+1(bias)=760개이다.

## pooling layer
activation map을 downsampling 해주는 layer. 쉽게 말하면 값들을 max, averge등으로 요약해주는 역할

그렇기 때문에 보통 filter size와 stride를 같게 만들어준다. (겹치지 않도록)
![image](https://user-images.githubusercontent.com/70709889/176335178-3793c4e4-f6fa-4f13-bf8a-3b2efa1e036b.png)

output size는 convolution layer와 동일하게 구할 수 있다
