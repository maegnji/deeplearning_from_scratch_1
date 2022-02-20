# coding: utf-8
import os
import sys
sys.path.append('/Users/maengseongjin/Library/Mobile Documents/com~apple~CloudDocs/7. python/220201_deeplearning_from_scratch_1') # 부모 디렉토리 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD
from common.util import shuffle_dataset

# 6.4.1 오버피팅
"""
오버피팅은 주로 다음의 경우에 일어난다.
 * 매개변수가 많고 표현력이 높은 모델
 * 훈련 데이터가 적음
강제로 오버피팅을 만들기 위해 MNIST 데이터 셋 중 300개만 사용하고 7층 네트워크를 사용해
복잡성을 높인다. 각 층의 뉴런은 100개, 활성화 함수는 ReLU.
"""


"""
훈련 데이터의 정확도는 100%지만 시험 데이터는 76% 수준이다.
이는 훈련 데이터에만 적응했기 때문에 훈련 때 사용하지 않은 범용 데이터에는 대응하지 못하는 것이다.
"""

# 6.4.2 가중치 감소
"""
가중치 감소weight decay : 학습 과정에서 큰 가중치에 대해서는 그에 상응하는 큰 패널티를
부과하여 오버피팅을 억제하는 방법
신경망 학습의 목적은 손실 함수의 값을 줄이는 것. 이때 예를 들어 가중치의 제곱 노름(L2 norm)을
손실 함수에 더하면 가중치가 커지는 것을 억제할 수 있다.
W : 가중치
L2 노름에 따른 가중치 감소 = 1/2 * λ * W²
λ(람다) : 정규화의 세기를 조절하는 하이퍼파라미터. 크게 설정할수록 큰 가중치에 대한 패널티가 커짐
이 코드에서는 0.1로 적용함.
결과는 훈련 데이터와 시험 데이터의 정확도 차이가 줄어들고 훈련 데이터의 정확도도 100%에
도달하지 못했음.
NOTE : L2 노름은 각 원소의 제곱들을 더한 것에 해당한다.
가중치 W = (w1, w2, ..., wn)이 있다면, L2 노름은 √(w1² + ... + wn²)이다.
이외에 L1, L∞도 있다.
L1 노름 : 절댓값의 합. |w1| + ... + |wn|
L∞ 노름 : Max 노름. 각 원소의 절댓값 중 가장 큰 것
"""

# 6.4.3 드롭아웃
"""
가중치 감소는 간단하게 구현할 수 있고 어느정도 오버피팅을 방지할 수 있지만 신경망 모델이
복잡해지면 가중치 감소만으로는 대응하기 어려워진다. 이때 드롭아웃 기법을 사용한다.
드롭아웃 : 뉴런을 임의로 삭제하면서 학습하는 방법. 훈련 때 은닉층의 뉴런을 무작위로 골라 삭제한다.
훈련때는 데이터를 흘릴 때마다 삭제할 뉴런을 무작위로 선택하고 시험 때는 모든 뉴런에 신호를 전달.
단, 시험 때는 각 뉴런의 출력에 훈련 때 삭제한 비율을 곱하여 출력한다.(안해도 됨)
"""


class Dropout:
    """
    순전파 때마다 mask에 삭제할 뉴런을 False로 표시한다. mask는 x와 같은 형상의 무작위 배열을
    생성하고 그 값이 dropout_ratio보다 큰 원소만 True로 설정한다.
    역전파 때의 동작은 ReLU와 같다.
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            # slef.mask는 x와 형상이 같은 배열을 무작위로 생성
            # 그 값이 dropout_ratio보다 큰 애들만 true로 설정
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
