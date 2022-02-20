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

"""
overfit_dropout.py 참고
훈련과 시험 데이터에 대한 정확도 차이가 줄어듬
훈련 데이터에 대한 정확도가 100%에 도달하지 않음.
epoch:301, train acc:0.73, test acc:0.6315
NOTE : 앙상블 학습ensemble learning : 개별적으로 학습시킨 여러 모델의 출력을 평균내 추론.
앙상블 학습을 사용하면 신경망의 정확도가 몇% 정도 개선된다는 것이 실험적으로 알려져 있음.
앙상블 학습은 드롭아웃과 밀접하다. 학습 때 뉴런을 무작위로 학습하는 것이 매번 다른 모델을
학습시키는 것으로 해석할 수 있다. 추론 때 삭제한 비율을 곱하는 것은 앙상블에서 모델의 평균과 같다.
"""

# 6.5 적절한 하이퍼파라미터 값 찾기
# 6.5.1 검증 데이터
"""
데이터셋을 훈련 데이터와 시험 데이터로 분리해 이용해서 오버피팅과 범용 성능 등을 평가했다.
하이퍼파라미터를 설정하고 검증할 때는 시험 데이터를 사용해서는 안 된다.
시험 데이터를 사용하여 하이퍼파라미터를 조정하면 하이퍼파라미터 값이 시험 데이터에 오버피팅된다.
따라서 하이퍼파라미터를 조정할때는 전용 확인 데이터가 필요하다.
이를 검증 데이터validation data라고 부른다.
NOTE :
 * 훈련 데이터 : 매개변수(가중치와 편향)의 학습에 이용
 * 검증 데이터 : 하이퍼파라미터의 성능을 평가
 * 시험 데이터 : 범용 성능을 확인하기 위해 마지막에(이상적으로는 한 번만) 이용
MNIST는 검증 데이터가 따로 없다. 훈련 데이터에서 20% 정도를 분리해서 사용할 수 있다.
"""

(x_train, t_train), (x_test, t_test) = load_mnist()

# 훈련 데이터를 뒤섞는다.
# shuffle_dataset은 commom/util.py에 있는데, np.random.shuffle을 활용함
x_train, t_train = shuffle_dataset(x_train, t_train)

# 20%를 검증 데이터로 분할
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]

