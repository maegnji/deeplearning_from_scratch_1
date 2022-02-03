import sys, os
sys.path.append('/Users/maengseongjin/Library/Mobile Documents/com~apple~CloudDocs/7. python/220201_deeplearning_from_scratch_1') # 부모 디렉토리 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from common.functions import *
from PIL import Image
import pickle

t = [0,0,1,0,0,0,0,0,0,0]

y= np.array([0.1,0.0,0.4,0.5,0,0,0,0,0,0])

print(sum_squares_error(np.array(t), np.array(y)))
print(cross_entropy_error(np.array(t), np.array(y)))

print(y.ndim)
print(y)
print(y.shape)
print(y.shape[0])
y = y.reshape(1,y.size)
print(y)
print(y.shape)
print(y[0])

y = y.shape[0]
print(y)


z= np.array([[0.1,0.0,0.4,0.5,0,0,0,0,0,0],[0.1,0.0,0.4,0.5,0,0,0,0,0,0]])
print(z[0])
print(z.shape[0])


# 훈련데이터는 60000, x가 인풋(784(28*28)), t(0~9)가 정답
(x_train, t_train) , (x_test, t_test) = \
    load_mnist(normalize = True, one_hot_label = True)

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10)

train_size = x_train.shape[0] # 60000
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# np.random.choice(60000. 10) : 0이상 60000미만 중, 무작위로 10개 select

