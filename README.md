# 目标检测项目

## 2023年11月10日commit
1. config中需要更换文件路径
2. utils中写了load_image,load_csv,load_id_list,load_all等函数
3. 只需要调用load_all就可以获取所有的image数据\[channels,width,height\]的列表以及所有csv中的数据

## 网络模型选择

经过对于yolov8网络模型性能图的分析，拟采用yolov8-m的网络

![img](./assets/yolo-comparison-plots.png)

