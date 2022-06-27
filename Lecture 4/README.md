# Lecture 4

## Summary
weight를 업데이트하기 위한 backward 과정에서 어떤 일이 일어나는지?

## Content
![image](https://user-images.githubusercontent.com/70709889/176002079-8288872e-adb4-424e-8ffe-13199e8c935c.png)

위와 같은 예제에서 backpropagation이 어떻게 흘러가는지 보자.

우리에게 필요한건 변수 x, y, z가 f에 미치는 영향력 즉,$\frac{df}{dx}, \frac{df}{dy}, \frac{df}{dz}$ 이다. $\frac{df}{dz}=q=3$ 
이라는 것은 바로 구할 수 있지만 $\frac{df}{dx}$ 
와 $\frac{df}{dy}$ 는 바로 구할 수 없기 때문에 chain rule을 사용하여

$\frac{df}{dx} = \frac{df}{dq} \frac{dq}{dx}$

$\frac{df}{dy} = \frac{df}{dq} \frac{dq}{dy}$

와 같이 구할 수 있다. 이 때, 앞의 $\frac{df}{dq}$ 부분을 global gradient,
뒤의 $\frac{dq}{dx}$를 local gradient라고 부른다.

local gradient는 forward 과정에서 미리 계산해놓고 global gradient는 backward 과정에서 미리 계산해놓은 local gradient를 사용하여 계산한다고 한다.

$\frac{df}{dq}=z=-4$ 이고 
$\frac{dq}{dx}와 \frac{dq}{dy}$는 각각 1, 1 이므로 위의 chain rule에 대입하면 $\frac{df}{dx}$
와 
$\frac{df}{dy}$는 각각 -4, -4라는 것을 알 수 있다.

![image](https://user-images.githubusercontent.com/70709889/176005533-5cb16f3f-b291-49cc-b9af-f1e7e983f7a5.png)

만약 backward 과정에서 들어오는 미분 값이 두개 이상이라면 모두 더하면 된다.
### Neural Networks
![image](https://user-images.githubusercontent.com/70709889/176006598-ad36c084-d434-4eb4-a791-2ab66a17e112.png)

위와 같은 network는 input layer를 제외하고 3-layer Neural Net이라고 부르며 각 hidden layer는 모든 노드 간에 연결이 존재하기 때문에 **Fully Connected Layer**라고 부른다.
