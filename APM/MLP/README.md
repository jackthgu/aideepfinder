# MLP (Multilayer Perceptron, 다층 퍼셉트론)

## 모델 설명
#### 단층 퍼셉트론
n개의 입력을 받아 1개의 출력을 리턴하는데, 입력이 뉴런에 전송될 때 가중치가 곱해지며 결과의 총합이 정해진 한계(임계값, `\theta`)를 넘을 경우에만 1을 리턴한다. 모델 자체가 단층 구조이기 때문에 선형   연산만 가능하다는 한계가 있다.

<img src="https://storage.googleapis.com/apmdata/slp_model.png" width="360" height="330">

<br>
#### 다층 퍼셉트론
단층 퍼셉트론을 이어 붙여서 만든 것으로 입력층과 출력층 사이에 n개 이상의 은닉층이 존재한다. 선형 연산만 가능한 단층 퍼셉트론의 한계를 극복하기 위해서는 각 층의 활성 함수에 대하여 비선형 함수만을 사용해야 한다.  
* 활성 함수가 선형이면 단층 퍼셉트론으로도 원하는 값을 구할 수 있기 때문에 은닉층의 의미가 없어짐

![png](https://storage.googleapis.com/apmdata/mlp_model.png)  

   
[Wikipedia](https://en.wikipedia.org/wiki/Multilayer_perceptron)

[Keras Implementation Code](https://keras.io/getting-started/sequential-model-guide/)

## Model Result Example (30min predication)
![png](https://storage.googleapis.com/apmdata/mlp.png)
