import numpy.random as npr
from tif_image_cutter.main.create_dset import DsetCreator

import numpy as np
import h5py as h
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.models import model_from_json
from matplotlib import pyplot as plt

from tif_image_cutter.main import unet
from tif_image_cutter.preprocessing.preprocessing import Preprocessing

if __name__ == "__main__":
    model_name = "C:\\Users\\Jakub Siembida\\PycharmProjects\\inz\\tif_image_cutter\\main\\model_control"
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_name + ".h5")

    dc = DsetCreator("G:/_dataset_256_sent/", "")

    img, mask = dc.read_shuffled_img_from_txt_file(shulffle=True, how_many=2000, preprocessing_method=None)

    for x in range(30):
        nmbr = npr.randint(2000)
        i = np.reshape(img[nmbr, :, :, :], [256, 256, 3])
        m = np.reshape(mask[nmbr, :, :, :], [256, 256])
        plt.subplot(131)
        plt.imshow(i)
        plt.subplot(132)
        plt.imshow(m, cmap="hot")
        plt.subplot(133)
        plt.imshow(np.reshape(model.predict(np.reshape(i, [1, 256, 256, 3])), [256, 256]), cmap="hot")
        plt.show()