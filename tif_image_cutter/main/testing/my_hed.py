import numpy as np
from skimage.color import rgb2hed
from scipy.signal import convolve2d
from matplotlib import pyplot as plt

from tif_image_cutter.main.create_dset import DsetCreator

if __name__ == '__main__':
    dc = DsetCreator("G:/_dataset_256_sent/", "")
    img, _ = dc.read_shuffled_img_from_txt_file(shulffle=True, how_many=1)
    img = np.reshape(img, [256, 256, 3])
    skimage_hed = rgb2hed(img)

    conv_arr = [
        [1.88, -0.07, -0.60],
        [-1.02, 1.13, -0.48],
        [-0.55, -0.13, 1.57]
    ]
    mul = convolve2d(img, conv_arr)
    plt.subplot(121)
    plt.imshow(np.reshape(skimage_hed[:, :, 0], [256, 256]), cmap="gray")
    plt.subplot(122)
    plt.imshow(np.reshape(mul[:, :, 0], [256, 256]), cmap="gray")
    plt.show()