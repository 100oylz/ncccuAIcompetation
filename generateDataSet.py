from sklearn.model_selection import train_test_split
import config
from utils import *
import yaml
import os
import time
from dataUpdate import DataAugmentForObjectDetection


def generate_train_val_img_csv(path: str):
    def mkdir(*path):
        for p in path:
            if not os.path.exists(p):
                os.mkdir(p)

    def cat_path(*path):
        return os.path.abspath(os.path.join('./', *path))

    id_list = load_id_list(path)
    # 区分训练集和验证集
    train_id_list, val_id_list = train_test_split(id_list, test_size=0.2, train_size=0.8, shuffle=True,
                                                  random_state=int(time.time()))
    print(len(train_id_list), len(val_id_list))
    # 获取路径
    base_path = config.BASEPATH
    image_path = cat_path(config.BASEPATH, config.IMAGEPATH)
    label_path = cat_path(config.BASEPATH, config.LABELPATH)
    train_image_path = cat_path(config.BASEPATH, config.IMAGEPATH, config.TRAINPATH)
    val_image_path = cat_path(config.BASEPATH, config.IMAGEPATH, config.VALPATH)
    train_label_path = cat_path(config.BASEPATH, config.LABELPATH, config.TRAINPATH)
    val_label_path = cat_path(config.BASEPATH, config.LABELPATH, config.VALPATH)
    # 生成文件夹
    mkdir(base_path, image_path, label_path, train_image_path, train_label_path, val_image_path, val_label_path)
    print('Origin_Train')
    generate(train_id_list, train_image_path, train_label_path)
    print('Origin_Valid')
    generate(val_id_list, val_image_path, val_label_path)
    print("Aug_Train")
    generate(train_id_list, train_image_path, train_label_path, update=True)


def generate(id_list, image_path, label_path, update=False):
    if (update):
        need_aug_num = 5
        model = DataAugmentForObjectDetection()
    for id in id_list:

        imagepath = os.path.join(config.DIRPATH, config.IMAGEFORMAT.format(id))
        csvpath = os.path.join(config.DIRPATH, config.CSVFORMAT.format(id))
        img_data = cv2.imread(imagepath)
        # img_tensor = transform_image_to_tensor(img_data)
        csv_data = pd.read_csv(csvpath)

        if (update):
            xmin, ymin, xmax, ymax, cls = csv_data['xmin'].values, csv_data['ymin'].values, csv_data['xmax'].values, \
                csv_data['ymax'].values, csv_data['class'].values
            coords = np.stack((xmin, ymin, xmax, ymax), axis=1)
            labels = cls
            for i in range(1, need_aug_num + 1):
                auged_img, auged_bboxes = model.dataAugment(img_data, coords)
                height = auged_img.shape[0]
                width = auged_img.shape[1]
                auged_bboxes_int = np.array(auged_bboxes).astype(np.int32)
                aug_id = id + f'_{i}'
                cls = np.array([config.classes[value] for value in labels])
                xmin = auged_bboxes_int[:, 0]
                ymin = auged_bboxes_int[:, 1]
                xmax = auged_bboxes_int[:, 2]
                ymax = auged_bboxes_int[:, 3]
                assert width == auged_img.shape[1]
                assert height == auged_img.shape[0]
                x_center = (xmin + xmax) / width / 2
                y_center = (ymin + ymax) / height / 2
                block_width = (xmax - xmin) / width
                block_height = (ymax - ymin) / height
                print(aug_id)
                save(cls, block_height, aug_id, image_path, auged_img, label_path, block_width, x_center, y_center)
        else:
            print(id)
            cls, x_center, y_center, width, height = transform_csv_to_label(csv_data)
            save(cls, height, id, image_path, img_data, label_path, width, x_center, y_center)


def save(cls, height, id, image_path, img_data, label_path, width, x_center, y_center):
    cv2.imwrite(os.path.join(image_path, f"{id}.jpg"), img_data)
    with open(os.path.join(label_path, f'{id}.txt'), 'w') as f:
        for c, x, y, w, h in zip(cls, x_center, y_center, width, height):
            f.write(f'{c} {x} {y} {w} {h}\n')


if __name__ == '__main__':
    generate_train_val_img_csv(config.DIRPATH)
