import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

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
    measurements.append(all_measurements)

# 将初始状态重置为 [0, 0, 1, 1] 等
states = [np.array([0, 0, 1, 1]),
          np.array([10, 0, -1, 1]),
          np.array([5, 10, 0, -1])]
covariances = [np.eye(4) * 2 for _ in range(num_targets)]

# 卡尔曼滤波跟踪 + JPDA
estimated_positions = [[] for _ in range(num_targets)]
true_positions_array = [np.array(tp) for tp in true_positions]
all_measurements_array = np.array(measurements)

for t in range(sum_step):
    # 预测阶段
    predicted_states = []
    predicted_covariances = []
    for i in range(num_targets):
        state = F.dot(states[i])
        P = F.dot(covariances[i]).dot(F.T) + Q
        predicted_states.append(state)
        predicted_covariances.append(P)

    # 计算关联概率（JPDA）
    num_measurements = measurements[t].shape[0]
    all_association_probs = np.zeros((num_targets, num_measurements))
    for i in range(num_targets):
        for j in range(num_measurements):
            S = H.dot(predicted_covariances[i]).dot(H.T) + R
            innovation = measurements[t][j] - H.dot(predicted_states[i])
            likelihood = multivariate_normal.pdf(innovation, mean=np.zeros(2), cov=S)
            all_association_probs[i, j] = likelihood

    # 正规化概率
    all_association_probs /= np.sum(all_association_probs, axis=1, keepdims=True)

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


# 计算评价指标
def compute_metrics(true_positions, estimated_positions, measurements):
    tp = np.zeros(num_targets)  # 真阳性
    fp = np.zeros(num_targets)  # 假阳性
    fn = np.zeros(num_targets)  # 假阴性

    for i in range(num_targets):
        true_pos = np.array(true_positions[i])
        estimated_pos = np.array(estimated_positions[i])
        num_true = len(true_pos)
        num_estimated = len(estimated_pos)

        # 计算每个目标的跟踪准确度
        distances = np.linalg.norm(true_pos[:, None, :] - estimated_pos[None, :, :], axis=2)
        matched = np.zeros(num_estimated, dtype=bool)
        for j in range(num_true):
            if len(estimated_pos) == 0:
                break
            min_dist_idx = np.argmin(distances[j])
            if distances[j, min_dist_idx] < 2:  # 距离阈值
                matched[min_dist_idx] = True

        tp[i] = np.sum(matched)
        fp[i] = num_estimated - np.sum(matched)
        fn[i] = num_true - np.sum(matched)

    return tp, fp, fn


tp, fp, fn = compute_metrics(true_positions, estimated_positions, measurements)

# 绘制结果
colors = ['g', 'r', 'b']
labels = ['Target 1', 'Target 2', 'Target 3']

plt.figure(figsize=(12, 10))
for i in range(num_targets):
    true_pos = np.array(true_positions[i])
    estimated_pos = np.array(estimated_positions[i])

    plt.plot(true_pos[:, 0], true_pos[:, 1], colors[i] + '-', label=f'{labels[i]} True Position')
    plt.plot(estimated_pos[:, 0], estimated_pos[:, 1], colors[i] + '--', label=f'{labels[i]} Estimated Position')

# 绘制目标测量值和杂波
for t in range(sum_step):
    target_measurements = measurements[t][:num_targets]
    clutter_measurements = measurements[t][num_targets:]

    plt.plot(target_measurements[:, 0], target_measurements[:, 1], 'o', color='black', label='Measurements')
    plt.plot(clutter_measurements[:, 0], clutter_measurements[:, 1], 'x', color='magenta', label='Clutter')

plt.legend()
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Multi-Target Tracking with Kalman Filter and JPDA with Clutter')
plt.grid()
plt.show()

# 显示评价指标
print("True Positives (TP):", tp)
print("False Positives (FP):", fp)
print("False Negatives (FN):", fn)
