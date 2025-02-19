# -*-coding:utf-8-*-
import numpy as np
import random
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from PIL import Image
from skimage import io
from utils import get_random_pos
import os

# Parameters
WINDOW_SIZE = (256, 256)  # Patch size
Stride_Size = 128  # Stride for train
STRIDE = 32  # Stride for testing
IN_CHANNELS = 3  # Number of input channels (e.g. RGB)
FOLDER = "./dataset/"
BATCH_SIZE = 10  # Number of samples in a mini-batch
LABELS = ["background", "lodging-wheats", "healthy-wheats"]  # Label names
N_CLASSES = len(LABELS)  # Number of classes
WEIGHTS = torch.ones(N_CLASSES)  # Weights for class balancing
CACHE = True  # Store the dataset in-memory  True

# 定义文件夹路径
folder_path = "./dataset/DOM_test"  # 替换为你的文件夹路径
tif_files = [f for f in os.listdir(folder_path) if f.startswith('DOM_') and f.endswith('.json')]
ids = [int(f.split('_')[1].split('.')[0]) for f in tif_files]
random.shuffle(ids)  # 随机打乱列表
# 按照7:3的比例分割训练集和测试集
total_count = len(ids)
train_count = int(total_count * 0.7)
test_count = total_count - train_count
train_ids = ids[:train_count]
test_ids = ids[train_count:]

# # 1_DOM_116 test_ids = [14, 53, 176, 177, 64, 50, 51, 52, 200, 196, 154, 174, 140, 55, 187, 173, 79, 104, 172, 185,
# 80, 138, 201, 127, 188, 141, 137, 175, 26, 152, 102, 20, 153, 15, 125, 164, 32, 202, 186, 139, 27, 163, 184, 75,
# 17, 114, 198, 18, 142, 100, 41, 123, 89, 111, 91, 116, 150, 16, 126, 178, 43, 162, 67, 117, 101, 199, 29, 149, 105,
# 115, 19, 113, 135, 65, 189, 136, 112, 78, 165, 69, 87] train_ids = [63, 54, 57, 68, 56, 39, 197, 128, 129, 99, 30,
# 76, 62, 44, 151, 148, 161, 90, 147, 160, 28, 124, 42, 77, 88, 190, 66, 40, 93, 81, 38, 103, 92, 166, 31] DOM_test
# train_ids = [39, 16, 38, 19, 29, 27, 20, 18, 32, 30, 15] test_ids = [14, 26, 31, 28, 17] # 2_DOM_28
# train_ids = [64, 60, 116, 67, 112, 89, 98, 87, 95, 96, 66, 91, 65, 94, 93, 88, 92, 97, 115]
# test_ids = [117, 113, 90, 114, 99, 61, 62, 86, 63]

DATA_FOLDER = FOLDER + 'DOM_test/DOM_{}.tif'
DSM_FOLDER = FOLDER + '1_DSM_172/resampled_DSM_{}.tif'
LABEL_FOLDER = FOLDER + 'DOM_test/DOM_{}_labeled_rgb.tif'
ERODED_FOLDER = FOLDER + 'DOM_test/DOM_{}_boundary_free_rgb.tif'

# DATA_FOLDER = MAIN_FOLDER + '3_DOM_28/DOM_{}.tif'  # 1_DOM_171
# DSM_FOLDER = MAIN_FOLDER + '2_DSM/resampled_DSM_{}.tif'
# LABEL_FOLDER = MAIN_FOLDER + '3_DOM_28/DOM_{}_labeled_rgb.tif'
# ERODED_FOLDER = MAIN_FOLDER + '3_DOM_28/DOM_{}_boundary_free_rgb.tif'

# DATA_FOLDER = MAIN_FOLDER + '2_DOM_672/DOM_{}.tif'  # 1_DOM_171
# DSM_FOLDER = MAIN_FOLDER + '2_DSM/resampled_DSM_{}.tif'
# LABEL_FOLDER = MAIN_FOLDER + '2_DOM_672/DOM_{}_labeled_rgb.tif'
# ERODED_FOLDER = MAIN_FOLDER + '2_DOM_672/DOM_{}_boundary_free_rgb.tif'

# DATA_FOLDER = MAIN_FOLDER + '1_DOM_116/DOM_{}.tif'  # 1_DOM_171
# DSM_FOLDER = MAIN_FOLDER + '1_DSM_172/resampled_DSM_{}.tif'
# LABEL_FOLDER = MAIN_FOLDER + '1_DOM_116/DOM_{}_labeled_rgb.tif'
# ERODED_FOLDER = MAIN_FOLDER + '1_DOM_116/DOM_{}_boundary_free_rgb.tif'

# DATA_FOLDER = MAIN_FOLDER + 'DOM_test/DOM_{}.tif'  # 1_DOM_171
# DSM_FOLDER = MAIN_FOLDER + '1_DSM_172/resampled_DSM_{}.tif'
# LABEL_FOLDER = MAIN_FOLDER + 'DOM_test/DOM_{}_labeled_rgb.tif'
# ERODED_FOLDER = MAIN_FOLDER + 'DOM_test/DOM_{}_boundary_free_rgb.tif'

# Let's define the standard  color palette
palette = {0: (255, 255, 255),  # Impervious surfaces (white)
           1: (255, 0, 0),  # Clutter (red)
           2: (0, 255, 0),  # Trees (green)
           3: (0, 0, 0)}  # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}


def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale 灰度 labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def save_img(tensor, name):
    tensor = tensor.cpu().permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')


class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                 cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache

        # List of files
        self.data_files = [DATA_FOLDER.format(id) for id in ids]  # 拼出数据的完整路径
        self.dsm_files = [DSM_FOLDER.format(id) for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.dsm_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        self.data_cache_ = {}
        self.dsm_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        # Default epoch size is 10 000 samples
        return BATCH_SIZE * 1000

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)  # 生成[a,b]范围内的随机整数

        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            data = io.imread(self.data_files[random_idx])[:, :, :3]  # 4个波段，取前3个组成RGB
            data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data

        if random_idx in self.dsm_cache_.keys():
            dsm = self.dsm_cache_[random_idx]
        else:
            # DSM is normalized in [0, 1]
            dsm = np.asarray(io.imread(self.dsm_files[random_idx]), dtype='float32')
            min = np.min(dsm)
            max = np.max(dsm)
            dsm = (dsm - min) / (max - min)
            if self.cache:
                self.dsm_cache_[random_idx] = dsm

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            # Labels are converted from RGB to their numeric values
            label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        dsm_p = dsm[x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        # Data augmentation
        data_p, dsm_p, label_p = self.data_augmentation(data_p, dsm_p, label_p)

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(dsm_p),
                torch.from_numpy(label_p))
