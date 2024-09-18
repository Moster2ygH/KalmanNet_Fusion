main_lor_DT.py
0712修改
产生的训练和测试轨迹还是根据x0_distri产生
训练中：
sysmodel 加入了sigma0
train_init 加入了noise

测试中：
测试阶段不采用随机初始状态
采用m1x0 m2x0测试
但是测试轨迹还是根据x0_distri产生，所以初始状态可能偏差很大

