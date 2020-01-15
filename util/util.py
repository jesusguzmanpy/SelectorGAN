import os
import numpy as np


def tensor2im(input_image, imtype=np.uint8):
	print(input_image[0].shape)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)