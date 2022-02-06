# 5.7.3 오차역전파법으로 구한 기울기 검증하기
"""
기울기를 구하는데는 두 가지 방법이 있다.
1. 수치 미분 : 느리다. 구현이 쉽다.
2. 해석적으로 수식을 풀기(오차 역전파법) : 빠르지만 실수가 있을 수 있다.
두 기울기 결과를 비교해서 오차역전파법을 제대로 구현했는지 검증한다.
이 작업을 기울기 확인gradient check라고 한다.
"""

import sys, os
sys.path.append('/Users/maengseongjin/Library/Mobile Documents/com~apple~CloudDocs/7. python/220201_deeplearning_from_scratch_1') # 부모 디렉토리 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net_softmax_copy import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 0~3 사이인 3개만 select
x_batch = x_train[:2]
print(x_train)
print(x_batch)
t_batch = t_train[:2]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

print(grad_numerical)
print(grad_backprop)

# 각 가중치의 차이의 절댓값을 구한 후, 그 절댓값들의 평균을 낸다.
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))