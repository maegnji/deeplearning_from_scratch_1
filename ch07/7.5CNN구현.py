import sys
import os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient
from dataset.mnist import load_mnist
from common.trainer import Trainer
"""
다음과 같은 CNN을 구성한다.
→ Conv → ReLU → Pooling → Affine → ReLU → Affine → Softmax →
전체 구현은 simple_convnet.py 참고
"""
class SimpleCovNet:
    # input_dim : input데이터의 차원 (채널수, 높이, 너)
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
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        # 가중치 매개변수 초기화
        self.params = {}
        # 첫번째 매개변수 (필터수30, 채널1, 필터크기28, 필터크기28)
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        # 필터수 만큼 편향
        self.params['b1'] = np.zeros(filter_num)
        
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)