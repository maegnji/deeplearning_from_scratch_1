import sys, os
sys.path.append('/Users/maengseongjin/Library/Mobile Documents/com~apple~CloudDocs/7. python/220201_deeplearning_from_scratch_1') # 부모 디렉토리 파일을 가져올 수 있도록 설정
from common.gradient import numerical_gradient
from common.layers import *
from collections import OrderedDict
from dataset.mnist import load_mnist


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        # params : 신경망의 매개변수를 보관하는 딕셔너리 변수(인스턴트 변수)
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) # input_size, hidden_size에 맞는 random 표준정규분포 생성
        self.params['b1'] = np.zeros(hidden_size) # hidden_size과 동일한 형상에 0을 채움
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) # input_size, hidden_size에 맞는 random 표준정규분포 생성
        self.params['b2'] = np.zeros(output_size) # hidden_size과 동일한 형상에 0을 채움

        #계층 생성
        self.layers = OrderedDict() # 입력된 items들의 순서를 기억하는 dictionary클래스 (기본 Dict과 거의 비슷)
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    # 예측 수행
    def predict(self, x):
        # W1, W2 = self.params['W1'], self.params['W2']
        # b1, b2 = self.params['b1'], self.params['b2']

        # a1 = np.dot(x, W1) + b1
        # z1 = sigmoid(a1)
        # a2 = np.dot(z1, W2) + b2
        # y = softmax(a2)

        # return y

        # 순전파때는 dict에 추가한 순서대로 각 계층의 forward()만 호출하기만 하면 됨! (역전파는 그 반대)
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x


    # 손실함수
    # x : 입력데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y,t)
        # return cross_entropy_error(y, t)

    # 정확도 측정
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력데이터, t : 정답레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W : self.loss(x, t)
        # grads : 기울기 보관하는 딕셔너리 변수(numerical_gradient)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        # 순전파
        self.loss(x,t)

        # 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

