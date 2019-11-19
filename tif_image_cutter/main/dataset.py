import numpy as np
from matplotlib import pyplot as plt


class Dataset:

    def __init__(self, path):
        self.path = path

    def load(self):
        f = open("_dataset_256_names.txt", "r")
        names = f.readline().split(",")
        length = len(names) - 1  # ostatni znak to , stąd 1 więcej empty element
        imgs = np.zeros([length, 256, 256, 3])
        masks = np.zeros([length, 256, 256, 1])
        for x in range(length):
            y = np.reshape(plt.imread(self.path + names[x]), [256,256, 1])
            masks[x, :, :, :] = y
            y = plt.imread(self.path + names[x].replace("_annotated.tif", ".tif"))
            imgs[x, :, :, :] = y
        return imgs, masks
