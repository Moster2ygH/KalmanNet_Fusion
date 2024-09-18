import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os
import random
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
import torch.nn as nn
from Filters.EKF_test import EKFTest

from Simulations.Extended_sysmdl_test import SystemModel
from Simulations.utils import DataGen, Short_Traj_Split
import Simulations.config as config

from Pipelines.Pipeline_EKF import Pipeline_EKF

from datetime import datetime

from KNet.KalmanNet_nn import KalmanNetNN

from Simulations.Lorenz_Atractor.parameters import m1x_0, m2x_0, m, n, \
    f, h, hRotate, H_Rotate, H_Rotate_inv, Q_structure, R_structure

from Plot import Plot_extended as Plot
from Plot import Plot_KF
#计算所有分布对之间的Mahalanobis距离
def mahalanobis_distance(mean1, cov1, mean2, cov2):
    mean_diff = mean1 - mean2
    cov_inv = torch.inverse((cov1 + cov2) / 2)
    distance = torch.sqrt(torch.matmul(torch.matmul(mean_diff.T, cov_inv), mean_diff))
    return distance.item()
def calculate_similarity(dist_set1, dist_set2):
    min_distances_1_to_2 = []
    min_distances_2_to_1 = []

    for dist1 in dist_set1:
        min_dist = float('inf')
        for dist2 in dist_set2:
            dist = mahalanobis_distance(
                dist1['mean'], dist1['std_dev'],
                dist2['mean'], dist2['std_dev']
            )
            if dist < min_dist:
                min_dist = dist
        min_distances_1_to_2.append(min_dist)

    for dist2 in dist_set2:
        min_dist = float('inf')
        for dist1 in dist_set1:
            dist = mahalanobis_distance(
                dist2['mean'], dist2['std_dev'],
                dist1['mean'], dist1['std_dev']
            )
            if dist < min_dist:
                min_dist = dist
        min_distances_2_to_1.append(min_dist)

    avg_min_distance_1_to_2 = sum(min_distances_1_to_2) / len(min_distances_1_to_2)
    avg_min_distance_2_to_1 = sum(min_distances_2_to_1) / len(min_distances_2_to_1)

    return (avg_min_distance_1_to_2 + avg_min_distance_2_to_1) / 2
def generate_random_distribution(train_distributions):
    random_distributions = []
    for dist in train_distributions:
        mean = torch.tensor([[random.uniform(-100, 100)], [random.uniform(-10, 20)], [random.uniform(-10, 10)]])
        std_dev_values = [random.uniform(1, 30), random.uniform(1, 10), random.uniform(1, 10)]
        std_dev = torch.diag(torch.tensor(std_dev_values))
        random_distributions.append({'mean': mean, 'std_dev': std_dev})
    return random_distributions


def run_simulation(x0_distributions_test):
    # 你的现有代码，计算相似度和其他结果
    import torch
    import sys
    import os
    import random
    torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
    import torch.nn as nn
    from Filters.EKF_test import EKFTest

    from Simulations.Extended_sysmdl_test import SystemModel
    from Simulations.utils import DataGen, Short_Traj_Split
    import Simulations.config as config

    from Pipelines.Pipeline_EKF import Pipeline_EKF

    from datetime import datetime

    from KNet.KalmanNet_nn import KalmanNetNN

    from Simulations.Lorenz_Atractor.parameters import m1x_0, m2x_0, m, n, \
        f, h, hRotate, H_Rotate, H_Rotate_inv, Q_structure, R_structure

    from Plot import Plot_extended as Plot
    from Plot import Plot_KF

    current_datetime = datetime.now()

    # 格式化日期和时间为 "YYYYMMDD_HHMM" 格式
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M")
    ################
    ### Get Time ###
    ################
    today = datetime.today()
    now = datetime.now()
    strToday = today.strftime("%m.%d.%y")
    strNow = now.strftime("%H:%M:%S")
    strTime = strToday + "_" + strNow
    # path_results = 'KNet/'
    load_model_path = 'Results/saved_lor_Discrete-Time/'
    path_results = load_model_path + '20240712_2144'  #修改了训练和测试，训练时候加入了sigma值，测试时候用m1x0 m2x0 new_folder_path = os.path.join(base_folder_path, formatted_datetime)
    # path_results = load_model_path + '20240710_1111'  # new_folder_path = os.path.join(base_folder_path, formatted_datetime)
    #20240711_1559时候为-16db       38》4.1
    #20240710_1111为          9.7》4.2
    #20240711_1445为-13      38》4.1
    #20240712_2144为-14.52  4.6》4.1   但是同样20240711_1544 在用m1x0测试，不行
    #20240711_1544            NAN 》4.1

    #对比 随机初始训练是否效果更好
    #20240713_0858 是单条轨迹训练 然后测试        与同一个x0_test的测试集      -16   6.5   5.8
    # -3                       -1.4（test_init加入0.01的噪声）   -0.07》-2.28（test_init加入0.1的噪声）    6.97》-5（test_init加入0.5的噪声）

    # -14(随机产生test数据集)    -14（test_init加入0.01的噪声）      -12（test_init加入0.1的噪声）          -3》-5.3（test_init加入0.5的噪声） （跟踪图上 能改善 但初始状态也还是有问题）
    #20240712_2144                           与同一个x0_test的测试集       -16  -13  -12(和训练不一样的初始测试)


    #当测试分布很接近初值 EKF效果比KFNet好
    ###################
    ###  Settings   ###
    ###################
    args = config.general_settings()
    ### dataset parameters
    args.N_E = 1000
    args.N_CV = 100
    args.N_T = 200 #testset-size
    args.T = 100  # input sequence length
    args.T_short = 100
    args.T_test = 100
    args.randomInit_train = True  # if True, random initial state for training set 决定了产生数据集多条轨迹的 初始状态值是带方差的，不是完全准确的
    args.randomInit_cv = True  # if True, random initial state for cross validation set
    args.randomInit_test = True  # if True, random initial state for test set
    args.distribution = 'normal'
    ### training parameters
    KnownRandInit_train = False  # if true: use known random init for training, else: model is agnostic to random init

    args.use_cuda = True  # use GPU or not
    args.n_steps = 5000
    args.n_batch = 32
    args.lr = 1e-4
    args.wd = 1e-4

    if args.use_cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # print("Using GPU")
        else:
            raise Exception("No GPU found, please set args.use_cuda = False")
    else:
        device = torch.device('cpu')
        # print("Using CPU")

    offset = 0  # offset for the data
    chop = False  # whether to chop data sequences into shorter sequences 是否将数据序列切分成更短的序列
    # path_results = 'KNet/'
    DatafolderName = 'Simulations/Lorenz_Atractor/data' + '/'
    switch = 'partial'  # 'full' or 'partial' or 'estH'

    # noise q and r
    r2 = torch.tensor([0.1])  # [100, 10, 1, 0.1, 0.01]
    vdB = -20  # ratio v=q2/r2
    v = 10 ** (vdB / 10)
    q2 = torch.mul(v, r2)

    Q = q2[0] * Q_structure
    R = r2[0] * R_structure


    traj_resultName = ['traj_lorDT_rq1030_T100.pt']
    dataFileName = ['data_lor_v20_RVdb0.1-20_3060_test_22:00.pt']

    if (args.randomInit_train or args.randomInit_cv or args.args.randomInit_test):  # 训练集验证集测试集 的初始
        std_gen = 3  # 产生数据用
    else:
        std_gen = 0

    if (KnownRandInit_train):
        std_feed = 0  # 训练用
    else:
        std_feed = 1.71

    m1x_0 = torch.ones(m, 1)  # Initial State
    m1x_0[0, 0] = -50
    m1x_0[1, 0] = -5
    m2x_0 = std_feed * std_feed * torch.eye(m)  # Initial Covariance for feeding to filters and KNet 初始协方差，std_feed=0表示训练时候知道初始位置

    m1x_0_gen = torch.ones(m, 1)  # Initial State
    m2x_0_gen = std_gen * std_gen * torch.eye(m)  # Initial Covariance for generating dataset 初始状态用xx分布进行采样
    #########################################
    ###  Generate and load data DT case   ###
    #########################################
    sys_model_gen = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, x0_distributions_test)  # parameters for GT
    sys_model_gen.InitSequence(m1x_0_gen, m2x_0_gen)  # x0 and P0

    # print("Start Data Gen")
    DataGen(args, sys_model_gen, DatafolderName + dataFileName[0])
    # print("Data Load")
    [train_input_long, train_target_long, cv_input, cv_target, test_input, test_target, train_init, cv_init, test_init] = torch.load(
        DatafolderName + dataFileName[0], map_location=device)
    # 噪声比例系数，例如 0.1 表示噪声大小是数据大小的 10%
    noise_scale = 0.5
    # 生成与 train_init 形状相同的噪声 tensor
    noise = noise_scale * test_init * torch.randn_like(test_init)

    # 将噪声添加到原始 tensor 上
    test_init_noisy = test_init + noise
    test_init = test_init_noisy
    if chop:
        # print("chop training data")
        [train_target, train_input, train_init] = Short_Traj_Split(train_target_long, train_input_long, args.T_short)
        # [cv_target, cv_input] = Short_Traj_Split(cv_target, cv_input, args.T)
    else:
        # print("no chopping")
        train_target = train_target_long[:, :, 0:args.T_short]
        train_input = train_input_long[:, :, 0:args.T_short]
        cv_target = cv_target[:,:,0:args.T]
        cv_input = cv_input[:,:,0:args.T]
    # Model with full info
    sys_model = SystemModel(f, Q, hRotate, R, args.T_short, args.T_test, m, n,
                            x0_distributions_test)  # parameters for GT
    sys_model.InitSequence(m1x_0, m2x_0)  # x0 and P0

    # Model with partial info
    sys_model_partial = SystemModel(f, Q, h, R, args.T_short, args.T_test, m, n, x0_distributions_test, prior_Sigma=m2x_0)
    sys_model_partial.InitSequence(m1x_0, m2x_0)
    # Model for 2nd pass
    sys_model_pass2 = SystemModel(f, Q, h, R, args.T_short, args.T_test, m, n,
                                  x0_distributions_test)  # parameters for GT
    sys_model_pass2.InitSequence(m1x_0, m2x_0)  # x0 and P0

    ########################################
    ### Evaluate Observation Noise Floor ###
    ########################################
    N_T = len(test_input)
    loss_obs = nn.MSELoss(reduction='mean')
    MSE_obs_linear_arr = torch.empty(N_T)  # MSE [Linear]

    for j in range(0, N_T):
        H_Rotate_inv = H_Rotate_inv.to(device)
        reversed_target = torch.matmul(H_Rotate_inv, test_input[j])
        MSE_obs_linear_arr[j] = loss_obs(reversed_target, test_target[j]).item()
    MSE_obs_linear_avg = torch.mean(MSE_obs_linear_arr)
    MSE_obs_dB_avg = 10 * torch.log10(MSE_obs_linear_avg)

    # Standard deviation
    MSE_obs_linear_std = torch.std(MSE_obs_linear_arr, unbiased=True)

    # Confidence interval
    obs_std_dB = 10 * torch.log10(MSE_obs_linear_std + MSE_obs_linear_avg) - MSE_obs_dB_avg


    ########################
    ### Evaluate Filters ###
    ########################
    ### Evaluate EKF true
    print("Evaluate EKF true")
    [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(args, sys_model,
                                                                                              test_input, test_target, randomInit=True, test_init=test_init)
    print("MSE_EKF_dB_avg", MSE_EKF_dB_avg)
    ### Evaluate EKF partial
    print("Evaluate EKF partial")
    [MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial,
     EKF_out_partial] = EKFTest(args, sys_model_partial, test_input, test_target, randomInit=True, test_init=test_init) #, randomInit=True, test_init=test_init
    print("MSE_EKF_dB_avg_partial", MSE_EKF_dB_avg_partial)

    #####################
    ### Evaluate KNet ###
    #####################
    if switch == 'full':
        ## KNet with full info ####################################################################################
        ################
        ## KNet full ###
        ################
        ## Build Neural Network
        # print("KNet with full model info")
        KNet_model = KalmanNetNN()
        KNet_model.NNBuild(sys_model, args)
        # ## Train Neural Network
        KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KNet")
        KNet_Pipeline.setssModel(sys_model)
        KNet_Pipeline.setModel(KNet_model)
        # print("Number of trainable parameters for KNet:",
        #       sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
        KNet_Pipeline.setTrainingParams(args)
        # if (chop):
        #     [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(
        #         sys_model, cv_input, cv_target, train_input, train_target, path_results, randomInit=True,
        #         train_init=train_init)
        # else:
        #     [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(
        #         sys_model, cv_input, cv_target, train_input, train_target, path_results)
        ## Test Neural Network
        [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KNet_out, RunTime] = KNet_Pipeline.NNTest(sys_model,
                                                                                                              test_input,
                                                                                                              test_target,
                                                                                                              path_results)

    ####################################################################################
    elif switch == 'partial':
        ## KNet with model mismatch ####################################################################################
        ###################
        ## KNet partial ###
        ####################
        ## Build Neural Network
        # print("KNet with observation model mismatch")
        KNet_model = KalmanNetNN()
        KNet_model.NNBuild(sys_model_partial, args)
        ## Train Neural Network
        KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KNet")
        KNet_Pipeline.setssModel(sys_model_partial)
        KNet_Pipeline.setModel(KNet_model)
        KNet_Pipeline.setTrainingParams(args)
        # if (chop):
        #     [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(
        #         sys_model_partial, cv_input, cv_target, train_input, train_target, path_results, randomInit=True,
        #         train_init=train_init)
        # else:
        #     [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(
        #         sys_model_partial, cv_input, cv_target, train_input, train_target, path_results, randomInit=True,
        #             train_init=train_init)
        ## Test Neural Network
        # [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KNet_out, RunTime] = KNet_Pipeline.NNTest(
        #     sys_model_partial, test_input, test_target, path_results, randomInit=True, test_init=test_init)
        [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KNet_out, RunTime] = KNet_Pipeline.NNTest(
            sys_model_partial, test_input, test_target, path_results, randomInit=True, test_init=test_init) #, randomInit=True, test_init=test_init
    ###################################################################################
    elif switch == 'estH':
        # print("True Observation matrix H:", H_Rotate)
        ### Least square estimation of H
        # 最小二乘法估计
        X = torch.squeeze(train_target[:, :, 0])
        Y = torch.squeeze(train_input[:, :, 0])
        for t in range(1, args.T):
            X_t = torch.squeeze(train_target[:, :, t])
            Y_t = torch.squeeze(train_input[:, :, t])
            X = torch.cat((X, X_t), 0)
            Y = torch.cat((Y, Y_t), 0)
        Y_1 = torch.unsqueeze(Y[:, 0], 1)
        Y_2 = torch.unsqueeze(Y[:, 1], 1)
        Y_3 = torch.unsqueeze(Y[:, 2], 1)
        H_row1 = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X.T, X)), X.T), Y_1)
        H_row2 = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X.T, X)), X.T), Y_2)
        H_row3 = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X.T, X)), X.T), Y_3)
        H_hat = torch.cat((H_row1.T, H_row2.T, H_row3.T), 0)
        # print("Estimated Observation matrix H:", H_hat)

        def h_hat(x, jacobian=False):
            H = H_hat.reshape((1, n, m)).repeat(x.shape[0], 1, 1)  # [batch_size, n, m]
            y = torch.bmm(H, x)
            if jacobian:
                return y, H
            else:
                return y

        # Estimated model
        sys_model_esth = SystemModel(f, Q, h_hat, R, args.T, args.T_test, m, n, x0_distributions_test)
        sys_model_esth.InitSequence(m1x_0, m2x_0)

        ################
        ## KNet estH ###
        ################
        # print("KNet with estimated H")
        KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KNetEstH_" + dataFileName[0])
        KNet_Pipeline.setssModel(sys_model_esth)
        KNet_model = KalmanNetNN()
        KNet_model.NNBuild(sys_model_esth, args)
        KNet_Pipeline.setModel(KNet_model)
        KNet_Pipeline.setTrainingParams(args)
        # [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(
        #     sys_model_esth, cv_input, cv_target, train_input, train_target, path_results)
        ## Test Neural Network
        [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, Knet_out, RunTime] = KNet_Pipeline.NNTest(
            sys_model_esth, test_input, test_target, path_results, randomInit=True, test_init=test_init)

    ###################################################################################
    else:
        print("Error in switch! Please try 'full' or 'partial' or 'estH'.")


    return MSE_EKF_dB_avg, MSE_EKF_dB_avg_partial, MSE_test_dB_avg, test_target, EKF_out_partial, KNet_out


def main():
    # 定义训练分布
    base_folder_path = 'Results/saved_lor_Discrete-Time/test0/'
    load_model_path = 'Results/saved_lor_Discrete-Time/'
    current_datetime = datetime.now()

    # 格式化日期和时间为 "YYYYMMDD_HHMM" 格式
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M")

    # 将日期时间字符串拼接到文件夹路径中
    new_folder_path = os.path.join(base_folder_path, formatted_datetime)
    # 检查并创建文件夹
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    # 打印控制台日志
    log_file = False  # True则打印到日志文件中
    if log_file:
        log_file_path = os.path.join(new_folder_path, 'a.log')
        f = open(log_file_path, 'a')  # a.log 或者a.txt都可以
        sys.stdout = f  # 保存print输出
        sys.stderr = f  # 保存异常或错误信息
        f.flush()  #
    ################
    ### Get Time ###
    ################
    today = datetime.today()
    now = datetime.now()
    strToday = today.strftime("%m.%d.%y")
    strNow = now.strftime("%H:%M:%S")
    strTime = strToday + "_" + strNow

    x0_distributions_train =  [
    {'mean': torch.tensor([[1.0], [1], [1.0]]), 'std_dev': torch.diag(torch.tensor([1.0, 1.0, 1.0]))},
    {'mean': torch.tensor([[5.0], [2], [1.0]]), 'std_dev': torch.diag(torch.tensor([2.0, 2.0, 2.0]))},
    {'mean': torch.tensor([[10.0], [5], [2.0]]), 'std_dev': torch.diag(torch.tensor([5.0, 2.0, 3.0]))},
    {'mean': torch.tensor([[30.0], [10], [3.0]]), 'std_dev': torch.diag(torch.tensor([20.0, 8.0, 9.0]))},
    {'mean': torch.tensor([[50.0], [15], [5.0]]), 'std_dev': torch.diag(torch.tensor([30.0, 10.0, 4.0]))},
    {'mean': torch.tensor([[-20.0], [1], [1.0]]), 'std_dev': torch.diag(torch.tensor([1.0, 1.0, 1.0]))},
    {'mean': torch.tensor([[200.0], [2], [1.0]]), 'std_dev': torch.diag(torch.tensor([2.0, 2.0, 2.0]))},
    {'mean': torch.tensor([[-180.0], [-5], [2.0]]), 'std_dev': torch.diag(torch.tensor([5.0, 2.0, 3.0]))},
    {'mean': torch.tensor([[-68.0], [-10], [3.0]]), 'std_dev': torch.diag(torch.tensor([20.0, 8.0, 9.0]))},
    {'mean': torch.tensor([[-100.0], [15], [5.0]]), 'std_dev': torch.diag(torch.tensor([30.0, 10.0, 4.0]))},
]

    # 运行次数
    n_runs = 1

    # 初始化结果列表
    dist_similarities = []
    mse_ekf_db_avg_list = []
    mse_ekf_db_avg_partial_list = []
    mse_test_db_avg_list = []

    for i in range(n_runs):
        # 生成随机测试分布
        x0_distributions_test = generate_random_distribution(x0_distributions_train)
        # x0_distributions_test = [
        #     {'mean': torch.tensor([[-70.0], [1], [1.0]]), 'std_dev': torch.diag(torch.tensor([1.0, 1.0, 1.0]))},
        #     {'mean': torch.tensor([[100.0], [2], [1.0]]), 'std_dev': torch.diag(torch.tensor([2.0, 2.0, 2.0]))},
        #     {'mean': torch.tensor([[-200.0], [-5], [2.0]]), 'std_dev': torch.diag(torch.tensor([5.0, 2.0, 3.0]))},
        #     {'mean': torch.tensor([[68.0], [-10], [3.0]]), 'std_dev': torch.diag(torch.tensor([2.0, 8.0, 9.0]))},
        #     {'mean': torch.tensor([[-120.0], [15], [5.0]]), 'std_dev': torch.diag(torch.tensor([3.0, 1.0, 4.0]))},
        # ]

        dist_similarity = calculate_similarity(x0_distributions_train, x0_distributions_test)
        # 运行仿真并获得结果
        MSE_EKF_dB_avg, MSE_EKF_dB_avg_partial, MSE_test_dB_avg, test_target, EKF_out_partial, KNet_out = run_simulation(x0_distributions_test)

        # 存储结果

        dist_similarities.append(dist_similarity)
        mse_ekf_db_avg_list.append(MSE_EKF_dB_avg.item())  # 转为标量
        mse_ekf_db_avg_partial_list.append(MSE_EKF_dB_avg_partial.item())  # 转为标量
        mse_test_db_avg_list.append(MSE_test_dB_avg.item())  # 转为标量
    average_mse_ekf_db_avg_list = sum(mse_ekf_db_avg_list) / len(mse_ekf_db_avg_list)
    average_mse_ekf_db_avg_partial_list = sum(mse_ekf_db_avg_partial_list) / len(mse_ekf_db_avg_partial_list)
    average_mse_test_db_avg_list = sum(mse_test_db_avg_list) / len(mse_test_db_avg_list)
    print("average_mse_ekf_db_avg_list=", average_mse_ekf_db_avg_list)
    print("average_mse_ekf_db_avg_partial_list=", average_mse_ekf_db_avg_partial_list)
    print("average_mse_test_db_avg_list=", average_mse_test_db_avg_list)
    print("********Plot********")
    # ### Plot results ###
    # 1 画状态变化图
    from Plot import Plot_extended as Plot
    PlotfolderName = "Figures/Linear_CA/"
    PlotfileName0 = "/TrainPVA_position.png"
    PlotfileName1 = "/TrainPVA_velocity.png"
    PlotfileName2 = "/TrainPVA_acceleration.png"

    Plot = Plot(new_folder_path, PlotfileName0)
    print("Plot")

    Plot.plotTraj_CA(test_target, EKF_out_partial, KNet_out, dim=0,
                     file_name=new_folder_path + PlotfileName0)  # Position
    Plot.plotTraj_CA(test_target, EKF_out_partial, KNet_out, dim=1,
                     file_name=new_folder_path + PlotfileName1)  # Velocity
    Plot.plotTraj_CA(test_target, EKF_out_partial, KNet_out, dim=2,
                     file_name=new_folder_path + PlotfileName2)  # Acceleration
    # 保存结果
    results = {
        'dist_similarities': dist_similarities,
        'mse_ekf_db_avg_list': mse_ekf_db_avg_list,
        'mse_ekf_db_avg_partial_list': mse_ekf_db_avg_partial_list,
        'mse_test_db_avg_list': mse_test_db_avg_list,
    }
    difference_list = []

    # 使用循环计算差值
    for i in range(len(mse_test_db_avg_list)):
        difference = mse_test_db_avg_list[i] - mse_ekf_db_avg_partial_list[i]
        difference_list.append(difference)
    np.save('results.npy', results)  # 保存到文件

    # 提取并绘制结果
    plt.figure(figsize=(10, 6))

    # plt.plot(dist_similarities, mse_ekf_db_avg_list, 'o-', label='MSE_EKF_dB_avg')
    # plt.plot(dist_similarities, mse_ekf_db_avg_partial_list, 'x-', label='MSE_EKF_dB_avg_partial')
    # plt.plot(dist_similarities, mse_test_db_avg_list, 's-', label='MSE_test_dB_avg')
    plt.vlines(dist_similarities, ymin=0, ymax=difference_list, colors='b', label='MSE_EKF_dB_avg',
               linewidth=1.8)
    plt.xlabel('Distance Similarity')
    plt.ylabel('Diff Values (dB)')
    plt.title('Diff vs Distance Similarity')
    plt.legend()
    plt.grid(True)
    # 保存图形
    save_path = os.path.join(new_folder_path, 'Diff Values.png')
    plt.savefig(save_path)

    # 绘制竖线图
    plt.figure(figsize=(10, 6))
    dist_similarities_offset1 = [dist + 0.1 for dist in dist_similarities]
    dist_similarities_offset0 = [dist - 0.1 for dist in dist_similarities]
    plt.vlines(dist_similarities_offset0, ymin=-20, ymax=mse_ekf_db_avg_list, colors='b', label='MSE_EKF_dB_avg', linewidth=1.8)
    plt.vlines(dist_similarities_offset1, ymin=-20, ymax=mse_ekf_db_avg_partial_list, colors='g', label='MSE_EKF_dB_avg_partial', linewidth=1.8)
    plt.vlines(dist_similarities, ymin=-20, ymax=mse_test_db_avg_list, colors='r', label='MSE_test_dB_avg', linewidth=3)

    plt.xlabel('Distance Similarity')
    plt.ylabel('Values (dB)')
    plt.title('Results vs Distance Similarity')
    plt.legend()
    plt.grid(True)
    # 保存图形
    save_path = os.path.join(new_folder_path, 'results_plot.png')
    plt.savefig(save_path)

    plt.show()

if __name__ == "__main__":
    main()
