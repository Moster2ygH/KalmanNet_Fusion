import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist

# 仿真总时间步数
sum_step = 20

# 目标数量
num_targets = 3

# 杂波（虚警）设置
num_clutter = 5  # 每个时间步中的虚警数量
clutter_range = [0, 20]  # 虚警的测量范围

# 初始状态和协方差矩阵（为每个目标分别设置）
states = [np.array([0, 0, 1, 1]),  # 目标1
          np.array([10, 0, -1, 1]),  # 目标2
          np.array([5, 10, 0, -1])]  # 目标3
covariances = [np.eye(4) * 2 for _ in range(num_targets)]  # 初始状态协方差

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

# 杂波噪声协方差矩阵
clutter_noise = 10  # 虚警的噪声水平

# 生成真实轨迹和测量值
np.random.seed(42)
true_positions = [[] for _ in range(num_targets)]
measurements = []

for t in range(sum_step):
    # 生成目标的真实轨迹和测量值
    target_measurements = np.zeros((num_targets, 2))
    for i in range(num_targets):
        # 真实轨迹更新
        states[i][0] += states[i][2] * dt
        states[i][1] += states[i][3] * dt
        true_positions[i].append(states[i][:2].copy())

        # 测量值（带噪声）
        measurement = states[i][:2] + np.random.multivariate_normal([0, 0], R)
        target_measurements[i] = measurement

    # 生成杂波（虚警量测）
    clutter_measurements = np.random.uniform(clutter_range[0], clutter_range[1], (num_clutter, 2))

    # 将目标测量和杂波混合
    all_measurements = np.vstack((target_measurements, clutter_measurements))
    np.random.shuffle(all_measurements)  # 随机打乱顺序
    measurements.append(all_measurements) #measurements是20（）个list（array(8X2X1)）8个量测

# 将初始状态重置为 [0, 0, 1, 1] 等
states = [np.array([0, 0, 1, 1]),
          np.array([10, 0, -1, 1]),
          np.array([5, 10, 0, -1])]
covariances = [np.eye(4) * 2 for _ in range(num_targets)]

# 设置椭圆形跟踪门的阈值
gamma = 9.21  # 对应于2自由度，95%的卡方分布阈值

# 卡尔曼滤波跟踪 + JPDA
estimated_positions = [[] for _ in range(num_targets)]

for t in range(sum_step):
    # 预测阶段
    predicted_states = []
    predicted_covariances = []
    for i in range(num_targets):
        state = F.dot(states[i])
        P = F.dot(covariances[i]).dot(F.T) + Q
        predicted_states.append(state)
        predicted_covariances.append(P)

    # 计算关联概率（JPDA）带有椭圆形跟踪门
    num_measurements = measurements[t].shape[0]
    all_association_probs = np.zeros((num_targets, num_measurements))
    for i in range(num_targets):
        S = H.dot(predicted_covariances[i]).dot(H.T) + R
        S_inv = np.linalg.inv(S)
        for j in range(num_measurements):
            innovation = measurements[t][j] - H.dot(predicted_states[i])
            mahalanobis_dist = innovation.T.dot(S_inv).dot(innovation)
            if mahalanobis_dist <= gamma:
                likelihood = multivariate_normal.pdf(innovation, mean=np.zeros(2), cov=S)
                all_association_probs[i, j] = likelihood
            else:
                all_association_probs[i, j] = 0

    # 正规化概率
    all_association_probs /= np.sum(all_association_probs, axis=1, keepdims=True, where=(all_association_probs != 0))

    # 更新阶段
    for i in range(num_targets):
        state_update = np.zeros(4)
        P_update = np.zeros((4, 4))
        for j in range(num_measurements):
            S = H.dot(predicted_covariances[i]).dot(H.T) + R
            K = predicted_covariances[i].dot(H.T).dot(np.linalg.inv(S))
            innovation = measurements[t][j] - H.dot(predicted_states[i])
            state_update += all_association_probs[i, j] * (predicted_states[i] + K.dot(innovation))
            P_update += all_association_probs[i, j] * (predicted_covariances[i] - K.dot(S).dot(K.T))

        states[i] = state_update
        covariances[i] = P_update
        estimated_positions[i].append(states[i][:2].copy())

# 绘制结果
plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r']
labels = [f'Target {i + 1}' for i in range(num_targets)]

for i in range(num_targets):
    plt.plot(*np.array(true_positions[i]).T, colors[i] + '-', label=f'{labels[i]} True')
    plt.plot(*np.array(estimated_positions[i]).T, colors[i] + 'o-', label=f'{labels[i]} Estimated')

# 绘制杂波
for t in range(sum_step):
    plt.scatter(measurements[t][:, 0], measurements[t][:, 1], c='k', marker='x', s=50, alpha=0.5,
                label='Clutter' if t == 0 else "")

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Tracking Results with Elliptical Gating')
plt.legend()
plt.grid()
plt.show()
