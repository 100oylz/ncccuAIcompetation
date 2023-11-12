# 文件夹路径
DIRPATh = r"D:\dataset\objectDetectionyolo\train"
# Format字符串
CSVFORMAT = "{}.csv"
IMAGEFORMAT = "{}.jpg"

# yolov8的网络参数
"""
Model | D (Deepen Factor) | W (Widen Factor) | R (Ratio)
---------------------------------------------------
n     | 0.33              | 0.25             | 2.0
s     | 0.33              | 0.50             | 2.0
m     | 0.67              | 0.75             | 1.5
l     | 1.00              | 1.00             | 1.0
X     | 1.00              | 1.25             | 1.0
"""
yolov8_d = 0.33
yolov8_w = 0.50
yolov8_r = 2.0

"""
最近邻插值 ('nearest')：
适用场景： 速度快，对于一些图像分割等任务可能效果不错。但在图像质量和平滑性上可能不如其他方法。

双线性插值 ('linear' 或 'bilinear')：
适用场景： 速度较快，对于一般的图像上采样任务通常效果良好。但在一些情况下可能会导致细节丢失。

双三次插值 ('bicubic')：
适用场景： 在需要更平滑的输出的图像上采样任务中，例如图像编辑等。相对于双线性插值，它更加平滑，但计算成本也更高。

三线性插值 ('trilinear')：
适用场景： 用于三维数据，例如立体图像或体积数据。对于涉及多个通道的三维数据，这是一个常见的选择。
"""
# Neck中UpSample的方法
neck_upsample_mode = 'nearest'
