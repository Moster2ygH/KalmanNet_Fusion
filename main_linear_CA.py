import sys
import os
import torch
from datetime import datetime

from Simulations.Linear_sysmdl import SystemModel
import Simulations.config as config
import Simulations.utils as utils
from Simulations.Linear_CA.parameters import F_gen,F_CV,H_identity,H_onlyPos,\
   Q_gen,Q_CV,R_3,R_2,R_onlyPos,\
   m,m_cv

from Filters.KalmanFilter_test import KFTest

from KNet.KalmanNet_nn import KalmanNetNN

from Pipelines.Pipeline_EKF import Pipeline_EKF as Pipeline

from Plot import Plot_extended as Plot
#0 获取当前的日期和时间
base_folder_path = 'Results/saved_CA_tensors/'
current_datetime = datetime.now()

# 格式化日期和时间为 "YYYYMMDD_HHMMSS" 格式
formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

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
path_results = new_folder_path

print("Pipeline Start")
####################################
### Generative Parameters For CA ###
####################################
args = config.general_settings()
### Dataset parameters
args.N_E = 1000 #输入训练数据集大小（序列数）
args.N_CV = 100
args.N_T = 200
offset = 0 ### Init condition of dataset
args.randomInit_train = True #if True, random initial state for training set
args.randomInit_cv = True #if True, random initial state for cross validation set
args.randomInit_test = True #if True, random initial state for test set

args.T = 80
args.T_test = 100
### training parameters
KnownRandInit_train = True # if true: use known random init for training, else: model is agnostic to random init
KnownRandInit_cv = True
KnownRandInit_test = True
args.use_cuda = True # use GPU or not
args.n_steps = 40 #训练次数
args.n_batch = 10
args.lr = 1e-4
args.wd = 1e-4

# 定义参数 参数写进txt文件

args_txt = {
    "训练数据集大小N_E": 1000,  # 输入训练数据集大小（序列数）
    "验证数据集大小N_CV": 100,
    "测试数据集大小N_T": 200,
    "数据集的初始条件offset": 0,  # Init condition of dataset
    "use_cuda": True,  # use GPU or not
    "仿真步长T": 80,
    "T_test": 100,
    "使用已知随机初始值进行训练KnownRandInit_train": True,  # if true: use known random init for training
    "使用已知随机初始值进行验证KnownRandInit_cv": True,
    "使用已知随机初始值进行测试KnownRandInit_test": True,
    "训练集使用随机初始状态randomInit_train": True,  # if True, random initial state for training set
    "验证集使用随机初始状态randomInit_cv": True,  # if True, random initial state for cross validation set
    "测试集使用随机初始状态randomInit_test": True,  # if True, random initial state for test set
    "训练次数n_steps": 40,
    "batch_size n_batch": 10,
    "学习率lr": 1e-4,
    "wd": 1e-4
}

if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

if(args.randomInit_train or args.randomInit_cv or args.args.randomInit_test):#训练集验证集测试集 的初始
   std_gen = 1
else:
   std_gen = 0

if(KnownRandInit_train or KnownRandInit_cv or KnownRandInit_test):
   std_feed = 0
else:
   std_feed = 1

m1x_0 = torch.zeros(m) # Initial State
m1x_0_cv = torch.zeros(m_cv) # Initial State for CV
m2x_0 = std_feed * std_feed * torch.eye(m) # Initial Covariance for feeding to filters and KNet 初始协方差，std_feed=0表示知道初始位置
m2x_0_gen = std_gen * std_gen * torch.eye(m) # Initial Covariance for generating dataset 生成数据的初始协方差
m2x_0_cv = std_feed * std_feed * torch.eye(m_cv) # Initial Covariance for CV

#############################
###  Dataset Generation   ###
#############################
### PVA or P
Loss_On_AllState = False # if false: only calculate loss on position
Train_Loss_On_AllState = True # if false: only calculate training loss on position
CV_model = False # if true: use CV model, else: use CA model

DatafolderName = 'Simulations/Linear_CA/data/'
DatafileName = 'decimated_dt1e-2_T100_r0_randnInit.pt'

####################
### System Model ###
####################
# Generation model (CA)
sys_model_gen = SystemModel(F_gen, Q_gen, H_onlyPos, R_onlyPos, args.T, args.T_test)
sys_model_gen.InitSequence(m1x_0, m2x_0_gen)# x0 and P0

# Feed model (to KF, KalmanNet) 
if CV_model:
   H_onlyPos = torch.tensor([[1, 0]]).float()
   sys_model = SystemModel(F_CV, Q_CV, H_onlyPos, R_onlyPos, args.T, args.T_test)
   sys_model.InitSequence(m1x_0_cv, m2x_0_cv)# x0 and P0
else:
   sys_model = SystemModel(F_gen, Q_gen, H_onlyPos, R_onlyPos, args.T, args.T_test)
   sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0 m1x_0是初始状态，此处生成数据和喂给滤波器的都是0相当于已知的 m2x_0是喂给滤波器的初始协方差 此处设置为0

print("Start Data Gen")
utils.DataGen(args, sys_model_gen, DatafolderName+DatafileName)  #保存仿真数据集至DatafolderName+DatafileName Simulations\Linear_CA\data\decimated_dt1e-2_T100_r0_randnInit.pt
print("Load Original Data")
[train_input, train_target, cv_input, cv_target, test_input, test_target,train_init,cv_init,test_init] = torch.load(DatafolderName+DatafileName, map_location=device)
if CV_model:# set state as (p,v) instead of (p,v,a)
   train_target = train_target[:,0:m_cv,:]
   train_init = train_init[:,0:m_cv]
   cv_target = cv_target[:,0:m_cv,:]
   cv_init = cv_init[:,0:m_cv]
   test_target = test_target[:,0:m_cv,:]
   test_init = test_init[:,0:m_cv]

print("Data Shape")
print("testset state x size:",test_target.size())
print("testset observation y size:",test_input.size())
print("trainset state x size:",train_target.size())
print("trainset observation y size:",train_input.size())
print("cvset state x size:",cv_target.size())
print("cvset observation y size:",cv_input.size())

print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)

print("**********************Evaluation***********************")

##########################
### Evaluate KalmanNet ###
##########################
# Build Neural Network
KNet_model = KalmanNetNN()
KNet_model.NNBuild(sys_model, args)
print("Number of trainable parameters for KNet pass 1:",sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
## Train Neural Network
KNet_Pipeline = Pipeline(strTime, "KNet", "KNet")
KNet_Pipeline.setssModel(sys_model)
KNet_Pipeline.setModel(KNet_model)
KNet_Pipeline.setTrainingParams(args)
if (KnownRandInit_train):
   print("*************Train KNet with Known Random Initial State*************")
   print("Train Loss on All States (if false, loss on position only):", Train_Loss_On_AllState)
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, MaskOnState=not Train_Loss_On_AllState, randomInit = True, cv_init=cv_init,train_init=train_init)
else:
   print("*************Train KNet with Unknown Initial State*************")
   print("Train Loss on All States (if false, loss on position only):", Train_Loss_On_AllState) #这里cv是验证集
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, MaskOnState=not Train_Loss_On_AllState)
   
if (KnownRandInit_test): 
   print("*************Test KNet with Known Random Initial State*************")
   ## Test Neural Network
   print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,KNet_out,RunTime] = KNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results,MaskOnState=not Loss_On_AllState,randomInit=True,test_init=test_init)
else: 
   print("*************Test KNet with Unknown Initial State*************")
   ## Test Neural Network
   print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,KNet_out,RunTime] = KNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results,MaskOnState=not Loss_On_AllState)

####################

##############################
### Evaluate Kalman Filter ###
##############################

print("Evaluate Kalman Filter")
if args.randomInit_test and KnownRandInit_test:
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(args, sys_model, test_input, test_target, allStates=Loss_On_AllState, randomInit = True, test_init=test_init)
else:
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(args, sys_model, test_input, test_target, allStates=Loss_On_AllState)

### 保存训练参数和结果 ###
#先保存结果
now = datetime.now()
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)
# 创建文件夹路径

import pickle
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

#1 将测试状态保存为 'tensor.pt' 文件
# 定义文件保存路径
file_path1 = os.path.join(new_folder_path, 'test_target.pt')
file_path2 = os.path.join(new_folder_path, 'KF_out.pt')
file_path3 = os.path.join(new_folder_path, 'KNet_out.pt')
file_path4 = os.path.join(new_folder_path, 'training_params.txt')
# 保存张量到指定文件路径
torch.save(test_target, file_path1)
print(f"张量已保存到: {file_path1}")
torch.save(KF_out, file_path2)
print(f"张量已保存到: {file_path2}")
torch.save(KNet_out, file_path3)
print(f"张量已保存到: {file_path3}")
# 将参数写入 txt 文件
with open(file_path4, 'w') as file:
    for key, value in args_txt.items():
        file.write(f"{key} = {value}\n")
# 输出文件路径，方便确认
print(f"训练参数已保存到 {file_path4}")

# #2 保存当前训练完毕的模型
# model_file = os.path.join(path_results, 'best-model.pt')
# # 1）.定义源目标文件路径
# source_file = model_file
# # 2）.定义目标文件路径
# destination_file = os.path.join(new_folder_path, 'best-model.pt')
# # 3）. 复制文件
# shutil.copy2(source_file, destination_file)
# # 4). 输出确认信息
# print(f"文件已成功复制到 {destination_file}")

#3 保存KF和KNet MSE值 对比
# 创建一个包含所有张量的字典
data_dict = {
    'MSE_KF_linear_avg': MSE_KF_linear_avg,
    'MSE_KF_dB_avg': MSE_KF_dB_avg,
    'MSE_test_linear_avg': MSE_test_linear_avg,
    'MSE_test_dB_avg': MSE_test_dB_avg
}

# 定义保存文件的路径和名称
file_path5 = os.path.join(new_folder_path, 'saved_tensors.pkl')

# 使用 pickle 序列化并保存字典到文件
with open(file_path5, 'wb') as f:
    pickle.dump(data_dict, f)

print(f"张量已保存到文件: {file_path5}")

# 加载保存的文件并打印每个张量
with open(file_path5, 'rb') as f:
    loaded_data_dict = pickle.load(f)

# 打印加载的张量
for key, tensor in loaded_data_dict.items():
    print(f"{key}:")
    print(tensor)
    print()  # 为了输出间隔清晰

####################
####################
### Plot results ###
#1 画状态变化图
PlotfolderName = "Figures/Linear_CA/"
PlotfileName0 = "/TrainPVA_position.png"
PlotfileName1 = "/TrainPVA_velocity.png"
PlotfileName2 = "/TrainPVA_acceleration.png"

Plot = Plot(new_folder_path, PlotfileName0)
print("Plot")


Plot.plotTraj_CA(test_target, KF_out, KNet_out, dim=0, file_name=new_folder_path+PlotfileName0)#Position
Plot.plotTraj_CA(test_target, KF_out, KNet_out, dim=1, file_name=new_folder_path+PlotfileName1)#Velocity
Plot.plotTraj_CA(test_target, KF_out, KNet_out, dim=2, file_name=new_folder_path+PlotfileName2)#Acceleration

#2 画出MSE_train_dB_epoch LOSS曲线
# 定义一个函数，用于绘制平滑损失曲线
def plot_smoothed_loss(tensor, save_path):
    # 将张量转换为 numpy 数组
    tensor_np = tensor.numpy()

    # 应用 Savitzky-Golay 滤波器进行平滑
    window_size = 10  # 窗口大小
    poly_order = 2    # 多项式阶数
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
# 1. 将张量转换为 numpy 数组
# 保存 MSE_train_dB_epoch 张量到文件
file_path5 = os.path.join(new_folder_path, 'MSE_train_dB_epoch.pt')
torch.save(MSE_train_dB_epoch, file_path5)
print(f"MSE_train_dB_epoch 已保存到: {file_path5}")

# 从文件中加载 MSE_train_dB_epoch 张量
loaded_tensor = torch.load(file_path5)
print(f"MSE_train_dB_epoch 已从 {file_path5} 加载。")

# 调用函数绘制并保存图像
plot_filename = os.path.join(new_folder_path, "MSE_train_dB_epoch.png")
plot_smoothed_loss(loaded_tensor, plot_filename)
