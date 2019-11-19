import numpy as np
import h5py
from matplotlib import pyplot as plt

if __name__ == "__main__":
    f = h5py.File("dataset_1000.h5", 'r')
    imgs = f["images"]
    masks = f["masks"]
    for x in range(np.shape(imgs)[0]):
        plt.subplot(121)
        plt.imshow(np.reshape(imgs[x, :, :, :], [256,256,3]))
        plt.subplot(122)
        plt.imshow(np.reshape(masks[x, :, :, :], [256, 256]))
        plt.show()
