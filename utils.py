import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import torch

import config
from typing import List, Tuple
import numpy as np
from torchvision import transforms


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
    id_list = list(set([filepath.split('.')[0] for filepath in files[0][2]]))
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


def transform_image_to_tensor(img_list: List[np.ndarray]) -> (List[torch.Tensor], List[Tuple[int]]):
    """
    将图像转化为transform后的tensor，并且返回tensor列表和原始图像大小
    :param img_list: 图片列表
    :type img_list: List[np.ndarray]
    :return: 图像transform后的列表，图像原本的shape列表
    :rtype: (List[torch.Tensor], List[Tuple[int]])
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((640, 640)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_tensors = []
    for img in img_list:
        assert type(img) == np.ndarray, f"{type(img)} != numpy.ndarray"
        img_tensor = transform(img)
        img_tensors.append(img_tensor)
    return img_tensors


def transform_csv_to_label(csv: pd.DataFrame) -> Tuple[int, float, float, float, float]:
    width = csv['width'].values
    height = csv['height'].values
    xmin = (csv['xmin'].values)[0]
    ymin = (csv['ymin'].values)[0]
    xmax = (csv['xmax'].values)[0]
    ymax = (csv['ymax'].values)[0]
    cls = config.classe[csv['class'].values[0]]
    x_center = (xmin + xmax) / width
    y_center = (ymin + ymax) / height
    width = (xmax - xmin) / width
    height = (ymax - ymin) / height
    '''
    (CLS,X_center/width,Y_center/height,(xmax-xmin)/width,(ymax-ymin)/height)
    '''
    return (cls, x_center, y_center, width, height)


if __name__ == '__main__':
    img_list, csv_list = load_all(config.DIRPATH, split=1)
    for csv in csv_list:
        label = transform_csv_to_label(csv)
        print(label)
