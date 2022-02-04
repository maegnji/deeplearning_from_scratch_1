import sys
import os
import numpy as np
sys.path.append('/Users/maengseongjin/Library/Mobile Documents/com~apple~CloudDocs/7. python/220201_deeplearning_from_scratch_1') # 부모 디렉토리 파일을 가져올 수 있도록 설정
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 평균 0, 표준편차 1의 가우시안 표준정규분포 난수를 matrix array(m,n) 생성

    def predict(self, x):
        return np.dot(x, self.W) # x, self.W의 행렬 곱

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)

        return loss

net = simpleNet()
print(net.W)

# W는 더미
# 아래 표현으로 대체할 수 있음
# f = lamdba w: net.loss(x,t)
def f(W):
    return net.loss(x,t)



x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

print(np.argmax(p))

t = np.array([0,0,1])
print(net.loss(x, t))

# 손실함수의 기울기 (2*3)
dW = numerical_gradient(f, net.W)
print(dW)