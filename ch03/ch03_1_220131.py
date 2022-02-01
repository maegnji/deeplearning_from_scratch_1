# 3.6.2 신경망의 추론 처리

# 입력층 784개
# 출력층 10개
# 은닉층 50개, 100개

import sys, os
sys.path.append('/Users/maengseongjin/Library/Mobile Documents/com~apple~CloudDocs/6. python/220125_밑딥1') # 부모 디렉토리 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
from PIL import Image
import pickle

def get_data():
  (x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=True, one_hot_label=False)
  return x_test, t_test
# normalize : 입력이미지의 픽셀값을 0.0~1.0 사이의 값으로 정규화할지? / false이면, 입력이미지 그대로 0~255유지
# flatten : 입력이미지를 평탄하게(1차원) 배열로 만들지? / false이면, 입력이미지를 1*28*28의 3차원 배열로 저장
# one_hot_label : 레이블을 one-hot-encording형태로 저장할 지? / false면, 7이나 2같이 숫자형태로 레이블 저장

# 가중치값이 이미 정해진 sample_weight.pkl을 불러옴
def init_network():
  # with 함수 : 파일을 열면 닫아야하는 명령을 해야 하는데, with함수를 사용함으로써 열고 닫는것을 자동 처리 함
  # open() as ~ : 파일을 불러오는 함수
  with open("/Users/maengseongjin/Library/Mobile Documents/com~apple~CloudDocs/6. python/220125_밑딥1/ch03/sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)

  return network

def predict(network, x):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = softmax(a3)

  return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
  y = predict(network, x[i])
  p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다
  if p == t[i]:
    accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


# 3.6.3 배치처리
# 형상 확인

x, t = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']

print(x.shape)
print(x[0].shape)
print(W1.shape)
print(W2.shape)
print(W3.shape)
print(len(x))

batch_size = 100
accuract_cnt = 0

for i in range(0, len(x), batch_size):
  x_batch = x[i:i+batch_size]
  y_batch = predict(network, x_batch)
  p = np.argmax(y_batch, axis=1)
  accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy : "+str(float(accuracy_cnt) / len(x)))
