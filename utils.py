import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import config
from typing import List, Tuple
import numpy as np


def load_id_list(filepath: str) -> List[str]:
    """
    加载dataset下的train目录下的文件id
    :param filepath: dataset下的train目录的地址
    :type filepath: str
    :return: train目录下的文档id列表(不重复)
    :rtype: List[str]
    """
    files = tuple(os.walk(filepath))
    id_list = list(set([filepath.split('.')[0] for filepath in files[0][2]]))
    return id_list


def load_image(filepath: str) -> np.ndarray:
    """
    读取图像数据
    :param filepath: 图像文件的路径
    :type filepath: str
    :return: 形状为[channels,height,weight]的矩阵
    :rtype: np.ndarray
    """
    img_data = cv2.imread(filepath)
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    return img_data


def load_csv(filepath: str) -> pd.DataFrame:
    """
    读取csv文件数据
    :param filepath:csv文件的路径
    :type filepath: str
    :return: csv文件的数据
    :rtype: pd.DataFrane
    """
    csv_data = pd.read_csv(filepath)
    return csv_data


def load_all(filepath: str) -> Tuple[List[np.ndarray], List[pd.DataFrame]]:
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
    for fileid in id_list:
        image_path = os.path.join(config.DIRPATh, config.IMAGEFORMAT.format(fileid))
        csv_path = os.path.join(config.DIRPATh, config.CSVFORMAT.format(fileid))

        csv = load_csv(csv_path)
        img = load_image(image_path)

        img_list.append(img)
        csv_list.append(csv)
    return img_list, csv_list


if __name__ == '__main__':
    load_all(config.DIRPATh)
