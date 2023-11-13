import numpy as np
import cv2
from typing import List, Tuple
from sklearn.model_selection import train_test_split
import os
import pandas as pd

import config


def add_gaussian_noise(image: np.ndarray, noise_strength: int = config.gaussian_noise_strength) -> np.ndarray:
    gaussian_noise = np.random.normal(0, noise_strength, image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = np.uint8(noisy_image)
    return noisy_image


def add_salt_and_pepper_noise(image: np.ndarray, salt_prob: float = config.salt_prob,
                              pepper_prob: float = config.pepper_prob) -> np.ndarray:
    noisy_image = image.copy()
    total_pixels = image.size

    # Add salt noise
    num_salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1], :] = 255

    # Add pepper noise
    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1], :] = 0

    return noisy_image


def add_uniform_noise(image: np.ndarray, noise_strength: int = config.uniform_noise_strength) -> np.ndarray:
    uniform_noise = np.random.uniform(-noise_strength, noise_strength, image.shape)
    noisy_image = image + uniform_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = np.uint8(noisy_image)
    return noisy_image


def add_poisson_noise(image: np.ndarray) -> np.ndarray:
    noisy_image = np.random.poisson(image).astype(np.uint8)
    return noisy_image


def add_impulse_noise(image: np.ndarray, prob: float = config.impulse_prob):
    noisy_image = image.copy()
    total_pixels = image.size

    # 生成随机噪声并将其应用于图像
    num_impulse = int(total_pixels * prob)
    impulse_coords = [np.random.randint(0, i - 1, num_impulse) for i in image.shape]
    noisy_image[impulse_coords[0], impulse_coords[1], :] = 255  # 设置为最大像素值（亮）
    noisy_image[impulse_coords[0], impulse_coords[1], :] = 0  # 设置为最小像素值（暗）

    return noisy_image


def random_add_all_noises(image_list: List[np.ndarray], csv_list: List[pd.DataFrame]) -> Tuple[
    List[np.ndarray], List[pd.DataFrame]]:
    """
    随机的给训练图像加上高斯噪声，盐和胡椒噪声，均匀噪声，泊松噪声，需要与原来的image和csv进行拼接
    :param image_list:训练图像的列表
    :type image_list: List[np.ndarray]
    :param csv_list: 训练csv的列表
    :type csv_list: List[pd.DataFrame]
    :return: noisy_image_list,noisy_csv_list
    :rtype: Tuple[List[np.ndarray], List[pd.DataFrame]]
    """
    image_list_temp_1, image_list_temp_2, csv_list_temp_1, csv_list_temp_2 = train_test_split(image_list, csv_list,
                                                                                              test_size=0.5,
                                                                                              shuffle=True)
    image_list_1, image_list_2, csv_list_1, csv_list_2 = train_test_split(image_list_temp_1, csv_list_temp_1,
                                                                          test_size=0.5,
                                                                          shuffle=True)
    image_list_3, image_list_4, csv_list_3, csv_list_4 = train_test_split(image_list_temp_2, csv_list_temp_2,
                                                                          test_size=0.5,
                                                                          shuffle=True)
    csv_list_new = []
    del image_list_temp_1, image_list_temp_2, csv_list_temp_1, csv_list_temp_2

    csv_list_new.extend(csv_list_1)
    csv_list_new.extend(csv_list_2)
    csv_list_new.extend(csv_list_3)
    csv_list_new.extend(csv_list_4)
    del csv_list_1, csv_list_2, csv_list_3, csv_list_4
    img_list_new = []
    for image_list, noise_func in zip((image_list_1, image_list_2, image_list_3, image_list_4), (
            add_impulse_noise, add_salt_and_pepper_noise, add_uniform_noise, add_gaussian_noise)):
        for img in image_list:
            if (len(img.shape) != 3):
                print(f"{img.shape}")
            img_list_new.append(noise_func(img))
    del image_list_1, image_list_2, image_list_3, image_list_4
    for img, csv in zip(img_list_new, csv_list_new):
        fileid = csv['filename'].values[0]
        img_file_path = os.path.join(config.NOISEDIRPATH, config.IMAGEFORMAT.format(fileid))
        csv_file_path = os.path.join(config.NOISEDIRPATH, config.CSVFORMAT.format(fileid))
        if (len(img.shape) != 3):
            print(fileid)
            print(img.shape)
            print(type(img))
        else:
            print(img.shape)
            cv2.imwrite(img_file_path, img.transpose((1, 2, 0)))
            csv.to_csv(csv_file_path)
    return img_list_new, csv_list_new


if __name__ == '__main__':
    from utils import load_all
    from config import DIRPATH

    img_list, csv_list = load_all(DIRPATH)
    print("Load All")
    original_image = img_list[0]
    noisy_image_list, csv_list_new = random_add_all_noises(img_list, csv_list)
    noise_image = noisy_image_list[0]

    cv2.imshow("Noisy_image", noise_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
