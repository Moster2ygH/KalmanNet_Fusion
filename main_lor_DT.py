import torch
import sys
import os
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
from Filters.EKF_test import EKFTest

from Simulations.Extended_sysmdl_test import SystemModel
from Simulations.utils import DataGen,Short_Traj_Split
import Simulations.config as config

from Pipelines.Pipeline_EKF import Pipeline_EKF

from datetime import datetime

from KNet.KalmanNet_nn import KalmanNetNN

from Simulations.Lorenz_Atractor.parameters import m1x_0, m2x_0, m, n,\
f, h, hRotate, H_Rotate, H_Rotate_inv, Q_structure, R_structure

from Plot import Plot_extended as Plot
from Plot import Plot_KF

print("Pipeline Start")
#0 获取当前的日期和时间
base_folder_path = 'Results/saved_lor_Discrete-Time/'
current_datetime = datetime.now()

# 格式化日期和时间为 "YYYYMMDD_HHMM" 格式
formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M")

# 将日期时间字符串拼接到文件夹路径中
new_folder_path = os.path.join(base_folder_path, formatted_datetime)
# 检查并创建文件夹
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

# 打印控制台日志
log_file = False #True则打印到日志文件中
if log_file:
    log_file_path = os.path.join(new_folder_path, 'a.log')
    f = open(log_file_path, 'a')   # a.log 或者a.txt都可以
    sys.stdout = f    # 保存print输出
    sys.stderr = f    # 保存异常或错误信息
    f.flush()  #
################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)
#path_results = 'KNet/'
path_results = new_folder_path #new_folder_path = os.path.join(base_folder_path, formatted_datetime)
#path_results = load_model_path + '20240710_1111'  # new_folder_path = os.path.join(base_folder_path, formatted_datetime)


###################
###  Settings   ###
###################
args = config.general_settings()
### dataset parameters
args.N_E = 1500
args.N_CV = 100
args.N_T = 200
args.T = 100 #input sequence length
args.T_short = 100
args.T_test = 100
args.randomInit_train = True #if True, random initial state for training set 决定了产生数据集多条轨迹的 初始状态值是带方差的，不是完全准确的
args.randomInit_cv = True #if True, random initial state for cross validation set
args.randomInit_test = True #if True, random initial state for test set
args.distribution = 'normal'
### training parameters
KnownRandInit_train = False # if true: use known random init for training, else: model is agnostic to random init

args.use_cuda = True # use GPU or not
args.n_steps = 2000
args.n_batch = 32
args.lr = 1e-4
args.wd = 1e-4

if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

offset = 0 # offset for the data
chop = False # whether to chop data sequences into shorter sequences 是否将数据序列切分成更短的序列
# path_results = 'KNet/'
DatafolderName = 'Simulations/Lorenz_Atractor/data' + '/'
switch = 'partial' # 'full' or 'partial' or 'estH'
   
# noise q and r
r2 = torch.tensor([0.1]) # [100, 10, 1, 0.1, 0.01]
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = torch.mul(v,r2)

Q = q2[0] * Q_structure
R = r2[0] * R_structure

print("1/r2 [dB]: ", 10 * torch.log10(1/r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1/q2[0]))

traj_resultName = ['traj_lorDT_rq1030_T100.pt']
dataFileName = ['data_lor_v20_RVdb0.1-20_3060_main_0712_prior_Sigma.pt']

if(args.randomInit_train or args.randomInit_cv or args.args.randomInit_test):#训练集验证集测试集 的初始
   std_gen = 3 #产生数据用
else:
   std_gen = 0

if(KnownRandInit_train):
   std_feed = 0 #训练用
else:
   std_feed = 1.71

m1x_0 = torch.ones(m, 1)  # Initial State
m1x_0[0, 0] = 5
m1x_0[1, 0] = 2

m2x_0 = std_feed * std_feed * torch.eye(m)  # Initial Covariance for feeding to filters and KNet 初始协方差，std_feed=0表示训练时候知道初始位置
print("m2x_0=", m2x_0)
# x0_distributions_gen = [
#     {'mean': torch.tensor([[1.0], [1], [1.0]]), 'std_dev': torch.diag(torch.tensor([1.0, 1.0, 1.0]))},
#     {'mean': torch.tensor([[5.0], [2], [1.0]]), 'std_dev': torch.diag(torch.tensor([2.0, 2.0, 2.0]))},
#     {'mean': torch.tensor([[10.0], [5], [2.0]]), 'std_dev': torch.diag(torch.tensor([5.0, 2.0, 3.0]))},
#     {'mean': torch.tensor([[30.0], [10], [3.0]]), 'std_dev': torch.diag(torch.tensor([20.0, 8.0, 9.0]))},
#     {'mean': torch.tensor([[50.0], [15], [5.0]]), 'std_dev': torch.diag(torch.tensor([30.0, 10.0, 4.0]))},
#     {'mean': torch.tensor([[-20.0], [1], [1.0]]), 'std_dev': torch.diag(torch.tensor([1.0, 1.0, 1.0]))},
#     {'mean': torch.tensor([[200.0], [2], [1.0]]), 'std_dev': torch.diag(torch.tensor([2.0, 2.0, 2.0]))},
#     {'mean': torch.tensor([[-180.0], [-5], [2.0]]), 'std_dev': torch.diag(torch.tensor([5.0, 2.0, 3.0]))},
#     {'mean': torch.tensor([[-68.0], [-10], [3.0]]), 'std_dev': torch.diag(torch.tensor([20.0, 8.0, 9.0]))},
#     {'mean': torch.tensor([[-100.0], [15], [5.0]]), 'std_dev': torch.diag(torch.tensor([30.0, 10.0, 4.0]))},
# ]
x0_distributions_gen = [
    {'mean': torch.tensor([[5.0], [2], [1.0]]), 'std_dev': torch.diag(torch.tensor([3.0, 3.0, 3.0]))},
]
m1x_0_gen = torch.ones(m, 1)  # Initial State
m2x_0_gen = std_gen * std_gen * torch.eye(m) # Initial Covariance for generating dataset 初始状态用xx分布进行采样

# 定义参数 参数写进txt文件

args_txt = {
    "训练数据集大小N_E": args.N_E,  # 输入训练数据集大小（序列数）
    "验证数据集大小N_CV": args.N_CV,
    "测试数据集大小N_T": args.N_T,
    "数据集的初始条件offset": offset,  # Init condition of dataset
    "use_cuda": args.use_cuda,  # use GPU or not
    "T": args.T,
    "T_short": args.T_short,
    "T_test": args.T_test,
    "使用已知随机初始值进行训练KnownRandInit_train": KnownRandInit_train,  # if true: use known random init for training
    "训练集使用随机初始状态randomInit_train": args.randomInit_train,  # if True, random initial state for training set
    "input distribution for the random initial state": args.distribution,
    "训练次数n_steps": args.n_steps,
    "n_batch": args.n_batch,
    "lr": args.lr,
    "wd": args.wd,
    "chop是否切割数据集": chop,
    "switch": switch,
    "r2": r2,
    "vdB": vdB,
    "std_feed": std_feed,
    "std_gen": std_gen,
    "m1x_0": m1x_0,
    "m1x_0_gen": m1x_0_gen,
}

#########################################
###  Generate and load data DT case   ###
#########################################
sys_model_gen = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n, x0_distributions_gen)# parameters for GT
sys_model_gen.InitSequence(m1x_0_gen, m2x_0_gen)# x0 and P0


print("Start Data Gen")
DataGen(args, sys_model_gen, DatafolderName + dataFileName[0])
print("Data Load")
print(dataFileName[0])
[train_input_long,train_target_long, cv_input, cv_target, test_input, test_target,train_init,cv_init,test_init] =  torch.load(DatafolderName + dataFileName[0], map_location=device)

if chop: 
   print("chop training data")    
   [train_target, train_input, train_init] = Short_Traj_Split(train_target_long, train_input_long, args.T_short)
   # [cv_target, cv_input] = Short_Traj_Split(cv_target, cv_input, args.T)
else:
   print("no chopping") 
   train_target = train_target_long[:,:,0:args.T_short]
   train_input = train_input_long[:,:,0:args.T_short]
   cv_target = cv_target[:,:,0:args.T]
   cv_input = cv_input[:,:,0:args.T]

# # 噪声比例系数，例如 0.1 表示噪声大小是数据大小的 10%
# noise_scale = 0.04
# # 生成与 train_init 形状相同的噪声 tensor
# noise = noise_scale * train_init * torch.randn_like(train_init)
#
# # 将噪声添加到原始 tensor 上
# train_init_noisy = train_init + noise
# train_init = train_init_noisy

# 使用广播机制，将每个子张量赋值为 m1x_0
train_init = m1x_0.expand(args.N_E, -1, -1)
cv_init = m1x_0.expand(args.N_CV, -1, -1)
test_init = m1x_0.expand(args.N_T, -1, -1)

print("trainset size:",train_target.size())
print("cvset size:",cv_target.size())
print("testset size:",test_target.size())
# Model with full info
sys_model = SystemModel(f, Q, hRotate, R, args.T_short, args.T_test, m, n, x0_distributions_gen, prior_Sigma=m2x_0)# parameters for GT
sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0

# Model with partial info
sys_model_partial = SystemModel(f, Q, h, R, args.T_short, args.T_test, m, n, x0_distributions_gen, prior_Sigma=m2x_0)
sys_model_partial.InitSequence(m1x_0, m2x_0)
# Model for 2nd pass
sys_model_pass2 = SystemModel(f, Q, h, R, args.T_short, args.T_test, m, n, x0_distributions_gen, prior_Sigma=m2x_0)# parameters for GT
sys_model_pass2.InitSequence(m1x_0, m2x_0)# x0 and P0

########################################
### Evaluate Observation Noise Floor ###
########################################
N_T = len(test_input)
loss_obs = nn.MSELoss(reduction='mean')
MSE_obs_linear_arr = torch.empty(N_T)# MSE [Linear]

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

print("Observation Noise Floor(test dataset) - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
print("Observation Noise Floor(test dataset) - STD:", obs_std_dB, "[dB]")


########################
### Evaluate Filters ###
########################
### Evaluate EKF true
print("Evaluate EKF true")
[MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(args, sys_model, test_input, test_target)
print("MSE_EKF_dB_avg", MSE_EKF_dB_avg)
### Evaluate EKF partial
print("Evaluate EKF partial")
[MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKFTest(args, sys_model_partial, test_input, test_target)
print("MSE_EKF_dB_avg_partial", MSE_EKF_dB_avg_partial)

# ### Save trajectories
# trajfolderName = 'Filters' + '/'
# DataResultName = traj_resultName[0]
# EKF_sample = torch.reshape(EKF_out[0],[1,m,args.T_test])
# target_sample = torch.reshape(test_target[0,:,:],[1,m,args.T_test])
# input_sample = torch.reshape(test_input[0,:,:],[1,n,args.T_test])
# torch.save({
#             'EKF': EKF_sample,
#             'ground_truth': target_sample,
#             'observation': input_sample,
#             }, trajfolderName+DataResultName)

#####################
### Evaluate KNet ###
#####################
if switch == 'full':
   ## KNet with full info ####################################################################################
   ################
   ## KNet full ###
   ################  
   ## Build Neural Network
   print("KNet with full model info")
   KNet_model = KalmanNetNN()
   KNet_model.NNBuild(sys_model, args)
   # ## Train Neural Network
   KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KNet")
   KNet_Pipeline.setssModel(sys_model)
   KNet_Pipeline.setModel(KNet_model)
   print("Number of trainable parameters for KNet:",sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
   KNet_Pipeline.setTrainingParams(args) 
   if(chop):
      [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, randomInit=True, cv_init=cv_init, train_init=train_init)
   else:
      [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, randomInit=True, cv_init=cv_init, train_init=train_init)
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KNet_out, RunTime] = KNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)

####################################################################################
elif switch == 'partial':
   ## KNet with model mismatch ####################################################################################
   ###################
   ## KNet partial ###
   ####################
   ## Build Neural Network
   print("KNet with observation model mismatch")
   KNet_model = KalmanNetNN()
   KNet_model.NNBuild(sys_model_partial, args)
   ## Train Neural Network
   KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KNet")
   KNet_Pipeline.setssModel(sys_model_partial)
   KNet_Pipeline.setModel(KNet_model)
   KNet_Pipeline.setTrainingParams(args)
   if(chop):
      [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model_partial, cv_input, cv_target, train_input, train_target, path_results, randomInit=True, cv_init=cv_init, train_init=train_init)
   else:
      [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model_partial, cv_input, cv_target, train_input, train_target, path_results, randomInit=True, cv_init=cv_init, train_init=train_init)
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KNet_out, RunTime] = KNet_Pipeline.NNTest(sys_model_partial, test_input, test_target, path_results)

###################################################################################
elif switch == 'estH':
   print("True Observation matrix H:", H_Rotate)
   ### Least square estimation of H
   #最小二乘法估计
   X = torch.squeeze(train_target[:,:,0])
   Y = torch.squeeze(train_input[:,:,0])
   for t in range(1,args.T):
      X_t = torch.squeeze(train_target[:,:,t])
      Y_t = torch.squeeze(train_input[:,:,t])
      X = torch.cat((X,X_t),0)
      Y = torch.cat((Y,Y_t),0)
   Y_1 = torch.unsqueeze(Y[:,0],1)
   Y_2 = torch.unsqueeze(Y[:,1],1)
   Y_3 = torch.unsqueeze(Y[:,2],1)
   H_row1 = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X.T,X)),X.T),Y_1)
   H_row2 = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X.T,X)),X.T),Y_2)
   H_row3 = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X.T,X)),X.T),Y_3)
   H_hat = torch.cat((H_row1.T,H_row2.T,H_row3.T),0)
   print("Estimated Observation matrix H:", H_hat)

   def h_hat(x, jacobian=False):
    H = H_hat.reshape((1, n, m)).repeat(x.shape[0], 1, 1) # [batch_size, n, m] 
    y = torch.bmm(H,x)
    if jacobian:
        return y, H
    else:
        return y

   # Estimated model
   sys_model_esth = SystemModel(f, Q, h_hat, R, args.T, args.T_test, m, n, x0_distributions_gen)
   sys_model_esth.InitSequence(m1x_0, m2x_0)

   ################
   ## KNet estH ###
   ################
   print("KNet with estimated H")
   KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KNetEstH_"+ dataFileName[0])
   KNet_Pipeline.setssModel(sys_model_esth)
   KNet_model = KalmanNetNN()
   KNet_model.NNBuild(sys_model_esth, args)
   KNet_Pipeline.setModel(KNet_model)
   KNet_Pipeline.setTrainingParams(args)
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model_esth, cv_input, cv_target, train_input, train_target, path_results)
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,Knet_out,RunTime] = KNet_Pipeline.NNTest(sys_model_esth, test_input, test_target, path_results, randomInit=True, test_init=test_init)
   
###################################################################################
else:
   print("Error in switch! Please try 'full' or 'partial' or 'estH'.")

# 定义一个函数，用于绘制平滑损失曲线
def plot_smoothed_loss(tensor, save_path):
    # 将张量转换为 numpy 数组
    tensor_np = tensor.numpy()

    # 应用 Savitzky-Golay 滤波器进行平滑
    window_size = 9  # 窗口大小
    poly_order = 4    # 多项式阶数
    smoothed_loss = savgol_filter(tensor_np, window_size, poly_order)

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(tensor_np, label='Original Loss', color='blue', linestyle='--')
    plt.plot(smoothed_loss, label='Smoothed Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (dB)')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    # 保存图像到文件
    plt.savefig(save_path)

    # 显示图像
    plt.show()

    # 输出确认信息
    print(f"损失曲线图已保存到 {save_path}")

#####参数保存#####
import pickle
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
print("********result save********")
# 1.将文件训练等参数写入 txt 文件
args_txt_file_path = os.path.join(new_folder_path, 'training_params.txt')
with open(args_txt_file_path, 'w') as file:
    for key, value in args_txt.items():
        file.write(f"{key} = {value}\n")
# 输出文件路径，方便确认
print(f"训练参数已保存到 {args_txt_file_path}")
#2 将训练过程和结果保存为 'results.pt' 文件
# 定义文件保存路径
result_file_path = os.path.join(new_folder_path, "results.pt")
# 保存tensor到指定文件路径
# 保存张量
torch.save({
    'time': formatted_datetime,
    'chop': chop,
    'T': args.T,
    'T_short': args.T_short,
    'r2': r2,
    'args.n_steps': args.n_steps,
    'args.randomInit_train': args.randomInit_train,
    'args.randomInit_test': args.randomInit_test,
    'KnownRandInit_train': KnownRandInit_train,
    'm1x_0': m1x_0,
    'm2x_0': m2x_0,
    'm2x_0_gen': m2x_0_gen,
    'std_gen': std_gen,
    'std_feed': std_feed,
    'test_target': test_target,
    'EKF_out': EKF_out,
    'EKF_out_partial': EKF_out_partial,
    'KNet_out': KNet_out,
    'MSE_train_dB_epoch': MSE_train_dB_epoch,
    'MSE_cv_dB_epoch': MSE_cv_dB_epoch,
    'MSE_test_dB_avg': MSE_test_dB_avg,
}, result_file_path)
print(f"result已保存到: {result_file_path}")
#3 保存测试结果到test_results.csv
import csv
# 检查文件是否存在，如果不存在则创建并写入标题行
result_file_path_save = os.path.join(base_folder_path, 'train_results.csv')
file_exists = os.path.isfile(result_file_path_save)

with open(result_file_path_save, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        # 如果文件不存在，写入标题行
        writer.writerow(['formatted_datetime', 'chop', 'args.T', 'args.T_short', 'args.randomInit_train', 'args.randomInit_test', 'KnownRandInit_train ',
                         'r2', 'm1x_0', 'm2x_0', 'm2x_0_gen', 'std_gen', 'std_feed',
                         'MSE_EKF_dB_avg', 'MSE_EKF_dB_avg_partial', 'MSE_test_dB_avg'])

    # 写入当前训练的结果
    writer.writerow([formatted_datetime, chop, args.T, args.T_short, args.randomInit_train, args.randomInit_test, KnownRandInit_train,
                     r2, m1x_0, m2x_0, m2x_0_gen, std_gen, std_feed,
                     MSE_EKF_dB_avg, MSE_EKF_dB_avg_partial, MSE_test_dB_avg])

print(f"测试对比结果已保存到 {result_file_path_save}")
print("********result save********")

#4 保存KF和KNet MSE值 对比并打印
# 创建一个包含所有张量的字典
result_data_dict = {
    'time': formatted_datetime,
    'chop': chop,
    'T': args.T,
    'T_short': args.T_short,
    'r2': r2,
    'args.n_steps': args.n_steps,
    'args.randomInit_train': args.randomInit_train,
    'args.randomInit_test': args.randomInit_test,
    'KnownRandInit_train': KnownRandInit_train,
    'm1x_0': m1x_0,
    'm2x_0': m2x_0,
    'm2x_0_gen': m2x_0_gen,
    'std_gen': std_gen,
    'std_feed': std_feed,
    'test_target': test_target,
    'EKF_out': EKF_out,
    'EKF_out_partial': EKF_out_partial,
    'KNet_out': KNet_out,
    'MSE_EKF_dB_avg': MSE_EKF_dB_avg,
    'MSE_EKF_dB_avg_partial': MSE_EKF_dB_avg_partial,
    'MSE_test_dB_avg': MSE_test_dB_avg,
}
result_vs_dict = {
    'time': formatted_datetime,
    'chop': chop,
    'T': args.T,
    'T_short': args.T_short,
    'r2': r2,
    'args.n_steps': args.n_steps,
    'args.randomInit_train': args.randomInit_train,
    'args.randomInit_test': args.randomInit_test,
    'KnownRandInit_train': KnownRandInit_train,
    'm1x_0': m1x_0,
    'm2x_0': m2x_0,
    'm2x_0_gen': m2x_0_gen,
    'std_gen': std_gen,
    'std_feed': std_feed,
    'MSE_EKF_dB_avg': MSE_EKF_dB_avg,
    'MSE_EKF_dB_avg_partial': MSE_EKF_dB_avg_partial,
    'MSE_test_dB_avg': MSE_test_dB_avg,
}
# 定义保存文件的路径和名称
data_dict_file_path = os.path.join(new_folder_path, 'saved_tensors.pkl')

# 使用 pickle 序列化并保存字典到文件
with open(data_dict_file_path, 'wb') as f:
    pickle.dump(result_vs_dict, f)

print(f"张量已保存到文件: {data_dict_file_path}")

# 加载保存的文件并打印每个张量
with open(data_dict_file_path, 'rb') as f:
    loaded_data_dict = pickle.load(f)

# 打印加载的张量
print("********MSE对比********")
for key, tensor in loaded_data_dict.items():
    print(f"{key}:", end=" ")
    print(tensor)
print("********MSE对比********")

# 5.将文件训练测试结果result_data_dict写入 txt 文件
args_result_txt_file_path = os.path.join(new_folder_path, 'training_results.txt')
with open(args_result_txt_file_path, 'w') as file:
    for key, value in result_data_dict.items():
        file.write(f"{key} = {value}\n")
# 输出文件路径，方便确认
print(f"训练参数已保存到 {args_result_txt_file_path}")
####################
####################

print("********Plot********")
# ### Plot results ###
#1 画状态变化图
PlotfolderName = "Figures/Linear_CA/"
PlotfileName0 = "/TrainPVA_position.png"
PlotfileName1 = "/TrainPVA_velocity.png"
PlotfileName2 = "/TrainPVA_acceleration.png"

Plot = Plot(new_folder_path, PlotfileName0)
print("Plot")


Plot.plotTraj_CA(test_target, EKF_out_partial, KNet_out, dim=0, file_name=new_folder_path+PlotfileName0)#Position
Plot.plotTraj_CA(test_target, EKF_out_partial, KNet_out, dim=1, file_name=new_folder_path+PlotfileName1)#Velocity
Plot.plotTraj_CA(test_target, EKF_out_partial, KNet_out, dim=2, file_name=new_folder_path+PlotfileName2)#Acceleration

# print("Plot_KF")
# Plot_KF = Plot_KF.NNPlot_test(MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg)
#2 画出MSE_train_dB_epoch LOSS曲线

# 从文件中加载 MSE_train_dB_epoch 张量
loaded_tensor = torch.load(result_file_path)
loaded_MSE_train_dB_epoch = loaded_tensor['MSE_train_dB_epoch']
print(f"MSE_train_dB_epoch 已从 {result_file_path} 加载。")

# 调用函数绘制并保存图像
plot_filename_MSE_train = os.path.join(new_folder_path, "MSE_train_dB_epoch.png")
plot_smoothed_loss(loaded_MSE_train_dB_epoch, plot_filename_MSE_train)
plot_filename_MSE_cv = os.path.join(new_folder_path, "MSE_cv_dB_epoch.png")
plot_smoothed_loss(MSE_cv_dB_epoch, plot_filename_MSE_cv)
print("********Plot********")




   





