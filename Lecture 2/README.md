# Lecture 2

컴퓨터가 이미지를 분류하기 위해서는 단순히 edges를 비교하는 알고리즘으로는 한계가 있다.
=> Data-Driven Approach. image와 label dataset을 사용하여 classifier를 학습시키고 new image를 분류하자
## Nearest Neighbor (NN)
train data를 memorize한 후 그 중에서 test data와 가장 비슷한 이미지를 사용하여 라벨링하는 단순한 방법이다.

비슷한 이미지를 찾는 방법에는 다양한 방법이 있지만 본 강의에서는 L1, L2 distance를 사용한다.
- L1 distance : pixel 값의 차를 절대값 취하여 모두 더하는 방식
![image](https://user-images.githubusercontent.com/70709889/175791342-4413b0cc-1961-44b0-96fa-789de2a2987b.png)

- L2 distance : pixel 값의 차를 제곱하여 더한 뒤 제곱근 취하는 방식
![image](https://user-images.githubusercontent.com/70709889/175791364-e1aaad57-1681-4362-b9c7-534a4dad0e40.png)

## K-Nearest Neighbors (K-NN)
![image](https://user-images.githubusercontent.com/70709889/175791393-35f03926-7a3c-441c-90a3-c1daeb980af0.png)

이 때, 가장 유사한 이미지를 찾기 때문에 위의 그림과 같이 섬이 생기거나 다른 구역을 침범하는 케이스가 존재할 수도 있다. 이러한 예외사항들을 방지하기 위해 K개의 유사 이미지를 찾은 후 과반수 투표하는 방법이 K-Nearest Neighbors 방법이다.

K를 몇으로 할 지, metric은 어떤 방식으로 선택할 지등 사용자에 의해 좌우되는 값이 **Hyperparameter**이고 이 값을 setting 하기 위해 필요한 것이 train, validation, test set으로 데이터를 나누는 것. 여기서 더 나아가면 K-fold 방식이 된다.
