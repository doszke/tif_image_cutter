from skimage.color import rgb2hed
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    img = plt.imread("rainbow.jpg")
    conv = rgb2hed(img)
    plt.subplot(151)
    plt.imshow(img)
    plt.subplot(152)
    plt.imshow(conv)

    plt.subplot(153)
    plt.imshow(np.reshape(conv[:, :, 0], np.shape(img)[0:2]), cmap="gray")

    plt.subplot(154)
    plt.imshow(np.reshape(conv[:, :, 1], np.shape(img)[0:2]), cmap="gray")

    plt.subplot(155)
    plt.imshow(np.reshape(conv[:, :, 2], np.shape(img)[0:2]), cmap="gray")

    plt.show()