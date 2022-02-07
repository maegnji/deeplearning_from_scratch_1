import numpy as np
"""
신경망 학습의 목적 : 손실 함수의 값을 최대한 낮추는 매개변수를 찾는 것 - 최적화optimization
최적의 매개변수 값을 찾는 단서로 매개변수의 기울기(미분)을 이용함 - 확률적 경사 하강법(SGD)
SGD의 단점과 다른 최적화 기법을 소개
"""

# 6.1.1 모험가 이야기
# 6.1.2 확률적 경사 하강법(SGD)
"""
W ← W - η * ∂L/∂W
W : 갱신할 가중치 매개변수
∂L/∂W : W에 대한 손실 함수의 기울기
η : 학습률(정해진 상수값. 0.01, 0.001 등)
"""

# 최적화를 담당하는 클래스를 분리해 구현하면 기능을 모듈화하기 좋다.
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


# 6.1.3 SGD의 단점
"""
SGD는 단순하고구현이 쉽지만, 문제에 따라 비효율적일 때가 있다.
다음 함수의 최솟값을 구해보자
f(x, y) = 1/20 * x² + y²
각 점에서 함수의 기울기는 (x/10, 2y)로 y축 방향은 가파른데 x축 방향은 완만하다.
또 최솟값은 (0, 0)이지만 기울기 대부분은 그 방향을 가리키지 않는다.
따라서 SGD를 적용하면 y축으로 지그재그로 수렴한다.
SGD는 비등방성anisotropy 함수(방향에 따라 성질, 여기서는 기울기가 달라지는 함수)에서는
탐색 경로가 비효율적이다.
이러한 단점을 개선해주는 모멘텀, AdaGrad, Adam이라는 방법을 소개한다.
"""

# 6.1.4 모멘텀
"""
모멘텀Momentum : 물리에서의 운동량
v ← αv - η * ∂L/∂W
W ← W + v
W : 갱신할 가중치 매개변수
∂L/∂W : W에 대한 손실 함수의 기울기
η : 학습률
v : 속도. 기울기 방향으로 힘을 받아 물체가 가속되는 것을 나타냄
α : 마찰/저항에 해당(0.9)
마치 공이 바닥을 구르는 듯한 움직임을 보여준다.
"""

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None # v는 물체의 속도 (초기화때는 아무것도 담지 않음)

    def update(self, params, grads):
        # update()가 처음 호출될 때 매개변수와 같은 구조의 데이터를 dict로 저장
        if self.v is None:
            self.v = {} 
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]