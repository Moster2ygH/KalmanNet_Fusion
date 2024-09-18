import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from scipy.stats import chi2

# 洛伦兹吸引器参数
sigma = 10.0
rho = 28.0
beta = 8 / 3

# 时间步长
dt = 0.01

# 过程噪声协方差矩阵和观测噪声协方差矩阵
Q = np.diag([0.1, 0.1, 0.1])
R = np.diag([1.0, 1.0, 1.0])
clutter_rate = 0.05  # 杂波产生率


def lorenz_attractor(x, y, z, dt):
    """ 使用洛伦兹吸引器的离散化模型计算下一个状态 """
    x_next = x + dt * sigma * (y - x)
    y_next = y + dt * (x * (rho - z) - y)
    z_next = z + dt * (x * y - beta * z)
    return x_next, y_next, z_next


def jacobian_F(x, y, z, dt):
    """ 计算状态转移方程的雅可比矩阵 """
    F = np.array([
        [1 - dt * sigma, dt * sigma, 0],
        [dt * (rho - z), 1 - dt, -dt * x],
        [dt * y, dt * x, 1 - dt * beta]
    ])
    return F


def ekf_update(x_pred, P_pred, z, H, Q, R):
    """ EKF 更新步骤 """
    # 计算卡尔曼增益
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    # 更新状态估计
    x_upd = x_pred + K @ (z - H @ x_pred)

    # 更新协方差矩阵
    I = np.eye(len(x_pred))
    P_upd = (I - K @ H) @ P_pred

    return x_upd, P_upd


def gating(z, x_pred, P_pred, H, R, gate_size=1):
    """ 使用跟踪门 (gating) 进行数据关联 """
    gates = []
    for z_i in z:
        S = H @ P_pred @ H.T + R
        innovation = z_i - H @ x_pred
        mahalanobis_distance = np.sqrt(innovation.T @ np.linalg.inv(S) @ innovation)
        gates.append(mahalanobis_distance < gate_size)
    return np.array(gates)


def compute_gamma(R, confidence_level=0.5):
    """ 计算归一化的跟踪门 `gamma` """
    # 获取观测噪声协方差矩阵的维度
    dim = R.shape[0]

    # 卡方分布的临界值
    chi2_threshold = chi2.ppf(confidence_level, dim)

    # 使用卡方分布临界值计算 `gamma`
    gamma = np.sqrt(chi2_threshold) * np.sqrt(np.max(np.diag(R)))

    return gamma

def jpda(z, x_pred, P_pred, H, R, Q, gate_size=3):
    """ 使用 JPDA 进行数据关联 """
    num_targets = len(x_pred)
    num_meas = len(z)

    # 计算归一化的 gamma
    # gamma = compute_gamma(R)
    gamma = 9.21
    # print('gamma=', gamma)

    # 初始化关联概率矩阵
    P_association = np.zeros((num_targets, num_meas))

    for i in range(num_targets):
        S = H @ P_pred[i] @ H.T + R
        S_inv = np.linalg.inv(S)
        for j in range(num_meas):
            innovation = z[j] - H @ x_pred[i]
            mahalanobis_distance = np.sqrt(innovation.T @ S_inv @ innovation)
            if mahalanobis_distance <= gamma:
                likelihood = multivariate_normal.pdf(innovation, mean=np.zeros(3), cov=S)
                P_association[i, j] = likelihood
            else:
                P_association[i, j] = 0

    # # 归一化关联概率
    # row_sums = P_association.sum(axis=1, keepdims=True)
    # # 添加检查避免除以零
    # row_sums[row_sums == 0] = 1  # 如果行和为零，则设置为1以避免除以零
    # P_association /= row_sums
    # 正规化概率
    # 计算P_association的归一化因子
    sum_P_association = np.sum(P_association, axis=1, keepdims=True)

    # 避免出现分母为零的情况
    sum_P_association[sum_P_association == 0] = np.finfo(float).eps  # 将零替换为一个极小的正数

    # 对P_association进行归一化
    P_association /= sum_P_association

    # print("P_association=", P_association)
    # 更新目标状态
    x_upd = np.zeros_like(x_pred)
    P_upd = np.zeros_like(P_pred)
    for i in range(num_targets):
        weighted_sum = np.zeros(3)
        total_prob = 0
        for j in range(num_meas):
            if P_association[i, j] > 0:
                z_associated = z[j]
                x_tmp, P_tmp = ekf_update(x_pred[i], P_pred[i], z_associated, H, Q, R)
                weighted_sum += P_association[i, j] * x_tmp
                total_prob += P_association[i, j]

        if total_prob > 0:
            x_upd[i] = weighted_sum / total_prob
            P_upd[i] = P_pred[i]  # 这里可以考虑对 P_upd 的进一步更新

    return x_upd, P_upd


def generate_measurements(x_true, R, num_clutter):
    """ 生成带有杂波的测量值 """
    z = np.copy(x_true)+np.random.multivariate_normal([0, 0, 0], R)
    clutter = np.random.uniform(low=-20, high=40, size=(num_clutter, 3))
    z = np.vstack([z, clutter])
    np.random.shuffle(z)
    return z


def calculate_rmse(true_states, estimated_states):
    """ 计算均方根误差 (RMSE) """
    return np.sqrt(np.mean(np.square(true_states - estimated_states), axis=0))


def calculate_mae(true_states, estimated_states):
    """ 计算平均绝对误差 (MAE) """
    return np.mean(np.abs(true_states - estimated_states), axis=0)


def main():
    num_targets = 2
    num_clutters = 2
    # 初始状态和协方差矩阵 (每个目标一个)
    np.random.seed(42)  # For reproducibility
    # initial_states = np.array([
    #     [1.0, 1.0, 1.0],
    #     [10.0, 10.0, 10.0],
    #     [20.0, 20.0, 20.0]
    # ])
    initial_states = np.array([
        [1.0, 1.0, 1.0],
        [5.0, 5.0, 5.0]
    ])
    x_true = initial_states.copy()  # 初始真实状态
    x = initial_states.copy()  # 初始估计状态
    P = np.array([np.eye(3) for _ in range(num_targets)])  # 初始状态协方差矩阵，为每个目标分配一个独立的矩阵

    # 观测矩阵 (直接观测状态)
    H = np.eye(3)

    # 进行一定次数的预测和更新
    num_steps = 100
    x_estimates = np.zeros((num_steps, num_targets, 3))
    x_trues = np.zeros((num_steps, num_targets, 3))
    P_updates = np.zeros((num_steps, num_targets, 3, 3))
    z_measures = np.zeros((num_steps, num_targets + num_clutters, 3))  # 量测中包含杂波

    for i in range(num_steps):

        # 生成带有杂波的测量值
        z = generate_measurements(x_true, R, num_clutter=num_clutters)
        z_measures[i] = z

        # 预测步骤
        x_pred = np.zeros_like(x_true)
        P_pred = np.zeros_like(P)
        for j in range(num_targets):
            x_pred[j] = lorenz_attractor(x[j, 0], x[j, 1], x[j, 2], dt)
            F = jacobian_F(x[j, 0], x[j, 1], x[j, 2], dt)
            P_pred[j] = F @ P[j] @ F.T + Q

        # JPDA 更新步骤
        x, P = jpda(z, x_pred, P_pred, H, R, Q)
        print("P=",i, P)
        # 记录当前状态估计
        x_estimates[i] = x
        P_updates[i] = P
        # 更新真实状态
        for j in range(num_targets):
            x_true[j] = np.array(lorenz_attractor(x_true[j, 0], x_true[j, 1], x_true[j, 2], dt))
        # 记录真实状态
        x_trues[i] = x_true
    # 计算 RMSE 和 MAE
    rmse = calculate_rmse(x_trues.reshape(-1, 3), x_estimates.reshape(-1, 3))
    mae = calculate_mae(x_trues.reshape(-1, 3), x_estimates.reshape(-1, 3))

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    # print("P_updates=", P_updates)
    # 绘制三维轨迹
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for j in range(num_targets):
        # print("x_trues=", x_trues[:, j, :])
        # print("x_estimates=", x_estimates[:, j, :])
        ax.plot(x_trues[:, j, 0], x_trues[:, j, 1], x_trues[:, j, 2], label=f'True Trajectory Target {j + 1}')
        ax.plot(x_estimates[:, j, 0], x_estimates[:, j, 1], x_estimates[:, j, 2], label=f'EKF Estimates Target {j + 1}')
        ax.scatter(z_measures[:, :, 0], z_measures[:, :, 1], z_measures[:, :, 2], c='r', s=1, label=f'Measurements')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Multi-Target Tracking using JPDA')

    # 调整图例，只修改字体大小
    ax.legend(loc='best', fontsize=6)  # 调整字体大小为 'small'

    plt.tight_layout()  # 自动调整布局
    plt.savefig('tracking_plot.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
