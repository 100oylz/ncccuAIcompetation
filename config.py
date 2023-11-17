# 文件夹路径
DIRPATH = r"D:\dataset\objectDetectionyolo\train"
NOISEDIRPATH = r"D:\dataset\objectDetectionyolo\noise"
# Format字符串
CSVFORMAT = "{}.csv"
IMAGEFORMAT = "{}.jpg"
yolo_model_version = 'm'
classes = {'E2': 0, 'J20': 1, 'B2': 2, 'F14': 3, 'Tornado': 4, 'F4': 5, 'B52': 6, 'JAS39': 7, 'Mirage2000': 8}

# 加噪函数中的默认参数的配置
gaussian_noise_strength = 25
salt_prob = 0.01
pepper_prob = 0.01
uniform_noise_strength = 25
impulse_prob = 0.01

BASEPATH = 'dataset'
IMAGEPATH = 'image'
LABELPATH = 'label'
TRAINPATH = 'train'
VALPATH = 'val'
