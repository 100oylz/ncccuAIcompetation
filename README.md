# 目标检测项目

## 2023年11月10日commit
1. config中需要更换文件路径
2. utils中写了load_image,load_csv,load_id_list,load_all等函数
3. 只需要调用load_all就可以获取所有的image数据\[channels,width,height\]的列表以及所有csv中的数据

## 网络模型选择

经过对于yolov8网络模型性能图的分析，拟采用yolov8-s的网络(或yolov8-m)

![img](./assets/yolo-comparison-plots.png)

mAP在m处出现明显的拐点，使用再大的模型提升不大，速度在s处出现明显拐点，使用再大的模型提升不大，需要考虑模型的大小，可能s更适合