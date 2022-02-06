import numpy as np
import matplotlib.pylab as plt

y = np.array([[1,2,3],[3,4,5]])

print(y.shape)
print(y.shape[0])
print(y.shape[1])

def function_1(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
#plt.show()

def function_2(x):
    return x[0]**2 + x[1]**2

def numerical_gradient(f,x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원
        
    return grad


# f : 최적화하려는 함수
# init_x : 초기값
# lr : learning rate (학습률)
# step_num : 경사법에 따른 반복 횟수
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        # numerical_gradient로 함수의 기울기 설정
        grad = numerical_gradient(f,x)
        x -= lr * grad
    return x


# 그래프 그리기 위해서 x_history 추가
def gradient_descent2(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


init_x = np.array([-3.0, 4.0])
print(init_x)
test = numerical_gradient(function_2, init_x)
test2 = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)

print(init_x)
print(test)
print(test2)


# 그래프
init_x = np.array([-3.0, 4.0])
x, x_history = gradient_descent2(function_2, init_x, lr=0.1, step_num=20)

plt.plot([-5, 5], [0, 0], '--b')
plt.plot([0, 0], [-5, 5], '--b')
plt.plot(x_history[:, 0], x_history[:, 1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()

