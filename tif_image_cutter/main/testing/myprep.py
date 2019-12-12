from scipy.signal import medfilt2d
import numpy as np
from matplotlib import pyplot as plt

from tif_image_cutter.preprocessing.preprocessing import Preprocessing


def colors(x, y):
    return np.round(y * x).astype(float)/y


if __name__ == '__main__':
    for x in range(6):
        img = (plt.imread(f"testimg{x}.tif")/255).astype(float)
        print(img.dtype)
        plt.subplot(141)
        plt.imshow(img)
        plt.subplot(142)
        Preprocessing.HE_reinhard_blur_instance(img.astype(np.uint8), "G:/_dataset_256_sent/") #to tylko ustawia obraz
        pre_img = Preprocessing.reinhard((img*255).astype(np.uint8))
        plt.imshow(pre_img)
        plt.subplot(143)
        nimg = colors((pre_img/255).astype(float), 3)
        plt.imshow(nimg)
        plt.subplot(144)
        print(np.shape(nimg))
        output = np.zeros(np.shape(nimg))
        for t in range(3):
            output[:, :, t] = medfilt2d(nimg[:, :, t], [3, 3])
        plt.imshow(output)
        plt.show()
