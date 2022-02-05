import sys, os
sys.path.append('/Users/maengseongjin/Library/Mobile Documents/com~apple~CloudDocs/7. python/220201_deeplearning_from_scratch_1') # 부모 디렉토리 파일을 가져올 수 있도록 설정
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        # params : 신경망의 매개변수를 보관하는 딕셔너리 변수(인스턴트 변수)
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) # input_size, hidden_size에 맞는 random 표준정규분포 생성
        self.params['b1'] = np.zeros(hidden_size) # hidden_size과 동일한 형상에 0을 채움
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) # input_size, hidden_size에 맞는 random 표준정규분포 생성
        self.params['b2'] = np.zeros(output_size) # hidden_size과 동일한 형상에 0을 채움

    # 예측 수행
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # 손실함수
    # x : 입력데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    # 정확도 측정
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
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
