import matplotlib.pyplot as plt
import numpy as np

# 2차 함수 및 그 미분 함수 정의
def quadratic_function(x):
    return x**2 - 4*x + 4

def derivative_quadratic(x):
    return 2*x - 4

# 그래디언트 하강법 구현
def gradient_descent_quadratic(start_x, learning_rate, epochs):
    x = start_x
    x_path = [x]
    for _ in range(epochs):
        grad_x = derivative_quadratic(x)
        x -= learning_rate * grad_x
        x_path.append(x)
    return x_path

# 초기 설정
start_x = 2.5  # 시작점 (매우 큰 값넣어도 됨)
learning_rate = 0.1  # 학습률
epochs = 30  # 에포크 수

# 그래디언트 하강법 실행
x_path = gradient_descent_quadratic(start_x, learning_rate, epochs)

# 함수 및 최적화 경로 시각화
x_vals = np.linspace(-2, 12, 400)
y_vals = quadratic_function(x_vals)

for i in range(1, len(x_path)):
    plt.plot(x_vals, y_vals, label='Quadratic Function')
    plt.scatter(x_path[:i], quadratic_function(np.array(x_path[:i])), color='r', marker='o')
    plt.title('Gradient Descent on a Quadratic Function')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.draw()
    plt.pause(0.1)
    plt.clf()

plt.show()

