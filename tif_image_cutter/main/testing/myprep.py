from scipy.signal import medfilt2d
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity

from tif_image_cutter.preprocessing.preprocessing import Preprocessing


def colors(x, y):
    return np.round(y * x).astype(float)/y


if __name__ == '__main__':
    for x in range(4):
        img = (plt.imread(f"testimg{x}.tif"))
        plt.subplot(420 + 2*x + 1)
        plt.imshow(img)
        #Preprocessing.HE_reinhard_blur_instance(img.astype(np.uint8), "G:/_dataset_256_sent/") #to tylko ustawia obraz
        #pre_img = Preprocessing.reinhard(img)
        #pre_img = rgb2hed(img)
        plt.subplot(420 + 2*x + 2)
        plt.imshow(Preprocessing.median_filter(img))
        #pre_img[:, :, 0] = rescale_intensity(pre_img[:, :, 0], (0, 1))

        #plt.imshow(np.reshape(pre_img[:, :, 0], [256, 256]), cmap="gray")
        #plt.subplot(143)
        #pre_img[:, :, 1] = rescale_intensity(pre_img[:, :, 1], (0, 1))
        #plt.imshow(np.reshape(pre_img[:, :, 1], [256, 256]), cmap="gray")
        #plt.subplot(144)
        #pre_img[:, :, 2] = rescale_intensity(pre_img[:, :, 2], (0, 1))
        #plt.imshow(np.reshape(pre_img[:, :, 2], [256, 256]), cmap="gray")

    plt.show()

