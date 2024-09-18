import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 洛伦兹吸引器参数
sigma = 10.0
rho = 28.0
beta = 8 / 3

# 时间步长
dt = 0.01

# 过程噪声协方差矩阵和观测噪声协方差矩阵
Q = np.diag([1, 1, 1])
R = np.diag([1.0, 1.0, 1.0])


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

def calculate_rmse(true_states, estimated_states):
    """ 计算均方根误差 (RMSE) """
    return np.sqrt(np.mean(np.square(true_states - estimated_states), axis=0))

def calculate_mae(true_states, estimated_states):
    """ 计算平均绝对误差 (MAE) """
    return np.mean(np.abs(true_states - estimated_states), axis=0)

def main():
    # 初始状态和协方差矩阵
    x_true = np.array([1.0, 1.0, 1.0])  # 初始真实状态
    x = np.array([1.0, 1.0, 1.0])  # 初始估计状态
    P = np.eye(3)  # 初始状态协方差矩阵

    # 观测矩阵 (直接观测状态)
    H = np.eye(3)

    # 进行一定次数的预测和更新
    num_steps = 1000
    x_estimates = np.zeros((num_steps, 3))
    x_trues = np.zeros((num_steps, 3))
    z_measures = np.zeros((num_steps, 3))

    for i in range(num_steps):
        # 记录真实状态
        x_trues[i] = x_true

        # 生成带噪声的测量值
        z = x_true + np.random.normal(0, np.sqrt(np.diag(R)))
        z_measures[i] = z

        # 预测步骤
        x_pred = lorenz_attractor(x[0], x[1], x[2], dt)
        x_pred = np.array(x_pred)

        # 计算雅可比矩阵
        F = jacobian_F(x[0], x[1], x[2], dt)

        # 更新状态协方差矩阵
        P_pred = F @ P @ F.T + Q

        # EKF 更新步骤
        x, P = ekf_update(x_pred, P_pred, z, H, Q, R)

        # 记录当前状态估计
        x_estimates[i] = x

        # 更新真实状态
        x_true = np.array(lorenz_attractor(x_true[0], x_true[1], x_true[2], dt))
    # 计算 RMSE 和 MAE
    rmse = calculate_rmse(x_trues, x_estimates)
    mae = calculate_mae(x_trues, x_estimates)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    # 绘制三维轨迹
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_trues[:, 0], x_trues[:, 1], x_trues[:, 2], label='True Trajectory', color='g')
    ax.plot(x_estimates[:, 0], x_estimates[:, 1], x_estimates[:, 2], label='EKF Estimates', color='b')
    ax.scatter(z_measures[:, 0], z_measures[:, 1], z_measures[:, 2], c='r', s=1, label='Measurements')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Target Tracking using EKF')
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
