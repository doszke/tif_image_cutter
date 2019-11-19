import h5py as h
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    dataset = h.File("dataset_64.hdf5", "r")
    train = dataset["train"]
    train_in = train["train_in"]
    train_out = train["train_out"]

    for x in range(3000, 3100):
        print("XD")
        plt.subplot(121)
        plt.imshow(train_in[x, :, :, :].reshape([64, 64, 3]))
        plt.subplot(122)
        plt.imshow(train_out[x, :, :, :].reshape([64, 64]), cmap="hot")
        plt.show()
