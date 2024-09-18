import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 仿真总时间步数
sum_step = 20

# 初始状态和协方差矩阵（为每个目标分别设置）
states = [np.array([0, 0, 1, 1]),  # 目标1
          np.array([10, 0, -1, 1]),  # 目标2
          np.array([5, 10, 0, -1])]  # 目标3
covariances = [np.eye(4) * 2 for _ in range(3)]  # 初始状态协方差

# 状态转移矩阵
dt = 1  # 时间间隔
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# 测量矩阵
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# 测量噪声协方差矩阵
R = np.array([[1, 0],
              [0, 1]])

# 过程噪声协方差矩阵
Q = np.eye(4) * 0.1

# 生成真实轨迹和测量值
np.random.seed(42)
true_positions = [[] for _ in range(3)]
measurements = np.zeros((sum_step, 3, 2))  # 每个时间步长下的所有目标的测量值
for t in range(sum_step):
    for i in range(3):
        # 真实轨迹更新
        states[i][0] += states[i][2] * dt
        states[i][1] += states[i][3] * dt
        true_positions[i].append(states[i][:2].copy())

        # 测量值（带噪声）
        measurement = states[i][:2] + np.random.multivariate_normal([0, 0], R)
        measurements[t, i] = measurement

# 将初始状态重置为 [0, 0, 1, 1] 等
states = [np.array([0, 0, 1, 1]),
          np.array([10, 0, -1, 1]),
          np.array([5, 10, 0, -1])]
covariances = [np.eye(4) * 2 for _ in range(3)]

# 卡尔曼滤波跟踪 + JPDA
estimated_positions = [[] for _ in range(3)]
for t in range(sum_step):
    # 预测阶段
    predicted_states = []
    predicted_covariances = []
    for i in range(3):
        state = F.dot(states[i])
        P = F.dot(covariances[i]).dot(F.T) + Q
        predicted_states.append(state)
        predicted_covariances.append(P)

    # 计算关联概率（JPDA）
    all_association_probs = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            S = H.dot(predicted_covariances[i]).dot(H.T) + R
            innovation = measurements[t, j] - H.dot(predicted_states[i])
            likelihood = multivariate_normal.pdf(innovation, mean=np.zeros(2), cov=S)
            all_association_probs[i, j] = likelihood

    # 正规化概率
    all_association_probs /= np.sum(all_association_probs, axis=1, keepdims=True)

    # 更新阶段
    for i in range(3):
        state_update = np.zeros(4)
        P_update = np.zeros((4, 4))
        for j in range(3):
            S = H.dot(predicted_covariances[i]).dot(H.T) + R
            K = predicted_covariances[i].dot(H.T).dot(np.linalg.inv(S))
            innovation = measurements[t, j] - H.dot(predicted_states[i])
            state_update += all_association_probs[i, j] * (predicted_states[i] + K.dot(innovation))
            P_update += all_association_probs[i, j] * (predicted_covariances[i] - K.dot(S).dot(K.T))

        states[i] = state_update
        covariances[i] = P_update
        estimated_positions[i].append(states[i][:2].copy())

# 绘制结果
colors = ['g', 'r', 'b']
labels = ['Target 1', 'Target 2', 'Target 3']

plt.figure(figsize=(10, 8))
for i in range(3):
    true_pos = np.array(true_positions[i])
    estimated_pos = np.array(estimated_positions[i])

    plt.plot(true_pos[:, 0], true_pos[:, 1], colors[i] + '-', label=f'{labels[i]} True Position')
    plt.plot(estimated_pos[:, 0], estimated_pos[:, 1], colors[i] + '--', label=f'{labels[i]} Estimated Position')
    plt.plot(measurements[:, i, 0], measurements[:, i, 1], colors[i] + 'o', label=f'{labels[i]} Measurements')

plt.legend()
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Multi-Target Tracking with Kalman Filter and JPDA')
plt.grid()
plt.show()
