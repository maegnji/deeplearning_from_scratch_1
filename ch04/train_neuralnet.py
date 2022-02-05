import sys, os
import numpy as np
from two_layer_net import TwoLayerNet
sys.path.append('/Users/maengseongjin/Library/Mobile Documents/com~apple~CloudDocs/7. python/220201_deeplearning_from_scratch_1') # 부모 디렉토리 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
 
# 하이퍼파라미터
iters_num = 10000 # 반복횟수
train_size = x_train.shape[0] # 60000
batch_size = 100 # 미니배치 크기
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

# iters_num 만큼 반복
for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size) # 0~train_size 중, batch_size만큼 random 추출
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch) # 성능 개선판!

    # 매개변수 갱신 (가중치 갱신)
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에폭당 정확도 계산 (굳이 매번 정확도를 계산할 필요는 없어서, 1에폭당 계산)
    # i를 iter_per_epoch로 나눈 나머지가 0인경우,,,
    # if i % iter_per_epoch == 0:
    #     train_acc = network.accuracy(x_train, t_train)
    #     test_acc = network.accuracy(x_test, t_test)
    #     train_acc_list.append(train_acc)
    #     test_acc_list.append(test_acc)
    #     print("train acc, test acc |" + str(train_acc) + "," + str(test_acc))

    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print("train acc, test acc |" + str(train_acc) + "," + str(test_acc) + "," + str(i))