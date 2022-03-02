import sys
import os
sys.path.append('/Users/maengseongjin/Library/Mobile Documents/com~apple~CloudDocs/7. python/220201_deeplearning_from_scratch_1') # 부모 디렉토리 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient
from dataset.mnist import load_mnist
from common.trainer import Trainer
"""
다음과 같은 CNN을 구성한다.
→ Conv → ReLU → Pooling → Affine → ReLU → Affine → Softmax →
전체 구현은 simple_convnet.py 참고

    Parameters
    ----------
    input_size : 입력 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
"""
class SimpleConvNet:
    # input_dim : input데이터의 차원 (채널수, 높이, 너비)
    # filter_num : 필터 수 / filter_size : 필터 크기 / stride : 스트라이드 / pad : 패딩
    # hidden_size : 은닉층 뉴런수 / output_size : 출력층 뉴런수 / weight_init_std : 초기화 때의 가중치 표준편차
    # 합성곱 계층의 parameter는 dictionary형태로 저장 (con_param)
    def __init__(self, input_dim=(1, 28, 28), conv_param={'filter_num':30, 'filter_size':5,
                'pad':0, 'stride':1}, hidden_size=100, output_size=10, weight_init_std=0.01):
        # 초기화 인수로 주어진 하이퍼파라미터를 딕셔너리에서 꺼내고 출력 크기를 계산한다.
        
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        # 합성곱계층 출력크기 계산
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        # 이거는 뭐지..
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        # 가중치 매개변수 초기화
        self.params = {}
        # 첫번째 매개변수 (필터수30, 채널1, 필터크기28, 필터크기28)
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        # 필터수 만큼 편향
        self.params['b1'] = np.zeros(filter_num)
        # 두번째 매개변수
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        # 세번째 매개변수
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # CNN을 구성하는 계층 생성 (딕셔너리 형태로 저장)
        self.layers = OrderedDict()
        # 합성곱 계산 (convolution 내 im2col을 이용하여 1차원으로 변환 후, reshape)
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        # 최대값으로 pooling진행
        self.layers['pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        # 행렬곱 계산
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        # 행렬곱 계산
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        # 마지막 softmaxwithloss는 별도변수에 저장
        self.last_layer = SoftmaxWithLoss()

    # 추론 진행
    def predict(self, x):
        # 여기서 values가 뭐지??
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # 손실 함수 (x : 입력데이터, t : 정답테이블)
        """손실 함수를 구한다.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
    def loss(self, x, t):
        y = self.predict(x)
        # 마지막층까지 계속 forward로 진행
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]   
        
    def numerical_gradient(self, x, t):
        """기울기를 구한다（수치미분）.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads 

    # 오차역전파로 기울기 구현
    def gradient(self, x, t):
        # 순전파
        self.loss(x,t)

        # 역전파
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        # 역전파 계산을 위하여 뒤집기
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads

       
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
