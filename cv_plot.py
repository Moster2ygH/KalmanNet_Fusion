from Plot import Plot_extended as Plot
import torch
import os

folder_path = 'saved_cv_tensors'

# 定义文件保存路径
file_path1 = os.path.join(folder_path, 'test_target.pt')
file_path2 = os.path.join(folder_path, 'KF_out.pt')
file_path3 = os.path.join(folder_path, 'KNet_out.pt')

# 加载张量
test_target = torch.load(file_path1)
KF_out = torch.load(file_path2)
KNet_out = torch.load(file_path3)

####################
### Plot results ###
PlotfolderName = "Figures/Linear_CA/"
PlotfileName0 = "TrainPVA_position.png"
PlotfileName1 = "TrainPVA_velocity.png"
PlotfileName2 = "TrainPVA_acceleration.png"

Plot = Plot(PlotfolderName, PlotfileName0)
print("Plot")


Plot.plotTraj_CA(test_target, KF_out, KNet_out, dim=0, file_name=PlotfolderName+PlotfileName0)#Position
Plot.plotTraj_CA(test_target, KF_out, KNet_out, dim=1, file_name=PlotfolderName+PlotfileName1)#Velocity
Plot.plotTraj_CA(test_target, KF_out, KNet_out, dim=2, file_name=PlotfolderName+PlotfileName2)#Acceleration