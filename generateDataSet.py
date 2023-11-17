from sklearn.model_selection import train_test_split
import config
from utils import *
import yaml
import os


def generate_train_val_img_csv(path: str):
    def mkdir(*path):
        for p in path:
            if not os.path.exists(p):
                os.mkdir(p)

    def cat_path(*path):
        return os.path.abspath(os.path.join('./', *path))

    id_list = load_id_list(config.DIRPATH)
    # 区分训练集和验证集
    train_id_list, val_id_list = train_test_split(id_list, test_size=0.2, train_size=0.8, shuffle=True, random_state=1)
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
    generate(train_id_list, train_image_path, train_label_path)
    generate(val_id_list, val_image_path, val_label_path)


def generate(id_list, image_path, label_path):
    for id in id_list:
        print(id)
        imagepath = os.path.join(config.DIRPATH, config.IMAGEFORMAT.format(id))
        csvpath = os.path.join(config.DIRPATH, config.CSVFORMAT.format(id))

        img_data = cv2.imread(imagepath)

        # img_tensor = transform_image_to_tensor(img_data)
        cv2.imwrite(os.path.join(image_path, f"{id}.jpg"), img_data)
        csv_data = pd.read_csv(csvpath)

        cls, x_center, y_center, width, height = transform_csv_to_label(csv_data)
        with open(os.path.join(label_path, f'{id}.txt'), 'w') as f:
            for c, x, y, w, h in zip(cls, x_center, y_center, width, height):
                f.write(f'{c} {x} {y} {w} {h}\n')


if __name__ == '__main__':
    generate_train_val_img_csv(config.DIRPATH)
