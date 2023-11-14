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
    # print(files)
    id_list = list(set([filepath.split('.')[0] for filepath in files[0][2]]))
    return id_list


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

def calculateIOU():
    pass

if __name__ == '__main__':
    load_all(config.DIRPATH)
