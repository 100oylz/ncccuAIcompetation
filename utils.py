import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import torch

import config
from typing import List, Tuple
import numpy as np
from torchvision import transforms
import yaml


def load_id_list(filepath: str) -> List[str]:
    """
    加载dataset下的train目录下的文件id
    :param filepath: dataset下的train目录的地址
    :type filepath: str
    :return: train目录下的文档id列表(不重复)
    :rtype: List[str]
    """
    files = tuple(os.walk(filepath))
    # print(files)
    id_list = list(sorted(list(set([filepath.split('.')[0] for filepath in files[0][2]]))))
    return id_list


def load_all(filepath: str, split: int = None) -> Tuple[List[np.ndarray], List[pd.DataFrame]]:
    """
    读取文件夹下全部数据
    :param filepath:文件夹路径
    :type filepath:str
    :return:图像数据列表，csv数据列表
    :rtype: Tuple[List[np.ndarray], List[pd.DataFrame]]
    """
    id_list = load_id_list(filepath)
    img_list = []
    csv_list = []
    if (split != None):
        id_list = id_list[:split]
    for fileid in id_list:
        image_path = os.path.join(config.DIRPATH, config.IMAGEFORMAT.format(fileid))
        csv_path = os.path.join(config.DIRPATH, config.CSVFORMAT.format(fileid))

        # Using OpenCV to read images
        img_data = cv2.imread(image_path)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        if (len(img_data.shape) != 3):
            print(fileid)
        img_list.append(img_data)

        # Using Pandas to read CSV
        csv_data = pd.read_csv(csv_path)
        csv_list.append(csv_data)

    return img_list, csv_list


def transform_image_to_tensor(image: np.ndarray):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((640, 640)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_tensor = transform(image)
    return img_tensor


def transform_csv_to_label(csv: pd.DataFrame) -> Tuple[int, float, float, float, float]:
    width = csv['width'].values
    height = csv['height'].values
    xmin = (csv['xmin'].values)
    ymin = (csv['ymin'].values)
    xmax = (csv['xmax'].values)
    ymax = (csv['ymax'].values)
    cls = np.array([config.classes[value] for value in csv['class'].values])
    x_center = (xmin + xmax) / width / 2
    y_center = (ymin + ymax) / height / 2
    width = (xmax - xmin) / width
    height = (ymax - ymin) / height
    '''
    (CLS,X_center/width,Y_center/height,(xmax-xmin)/width,(ymax-ymin)/height)
    '''
    return (cls, x_center, y_center, width, height)


if __name__ == '__main__':
    # 请确保你已经安装了PyYAML库，可以使用以下命令安装：
    # pip install pyyaml

    # 替换 'dataset/train_data.yaml' 为你的YAML文件路径
    yaml_file_path = 'dataset/train_data.yaml'

    # 使用PyYAML加载YAML文件
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    # 打印加载的数据
    print(data['names'])
