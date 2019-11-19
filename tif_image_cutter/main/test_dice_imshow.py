import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import *
from tensorflow.keras.models import model_from_json
import cv2

if __name__ == "__main__":
    model_name_1 = "test_dice"
    model_name_2 = "test_dice_2"

    json_file = open(model_name_1 + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model1 = model_from_json(loaded_model_json)
    model1.load_weights(model_name_1 + ".h5")

    json_file = open(model_name_2 + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model2 = model_from_json(loaded_model_json)
    model2.load_weights(model_name_2 + ".h5")

    img1 = np.reshape(plt.imread("G:\\_dataset_256_sent\\1\\1_507.tif"), [256, 256, 3])/255
    mask1 = np.reshape(plt.imread("G:\\_dataset_256_sent\\1\\1_507_annotated.tif"), [256, 256, 1])/255
    img2 = np.reshape(plt.imread("G:\\_dataset_256_sent\\1\\1_513.tif"), [256, 256, 3])/255
    mask2 = np.reshape(plt.imread("G:\\_dataset_256_sent\\1\\1_513_annotated.tif"), [256, 256, 1]) / 255

    imgs = np.zeros([2, 256, 256, 3])
    masks = np.zeros([2, 256, 256, 1])
    imgs[0, :, :, :] = img1
    imgs[1, :, :, :] = img2

    masks[0, :, :, :] = mask1
    masks[1, :, :, :] = mask2

    for x in range(2):
        plt.subplot(141)
        plt.imshow(np.reshape(imgs[x, :, :, :], [256, 256, 3]))
        plt.subplot(142)
        plt.imshow(np.reshape(masks[x, :, :, :], [256, 256]), cmap="hot")
        plt.subplot(143)
        plt.imshow(np.reshape(model1.predict(np.reshape(imgs[x, :, :, :], [1, 256, 256, 3])), [256, 256]), cmap="hot", vmin=0, vmax=1)
        plt.subplot(144)
        plt.imshow(np.reshape(model2.predict(np.reshape(imgs[x, :, :, :], [1, 256, 256, 3])), [256, 256]), cmap="hot", vmin=0, vmax=1)
        plt.show()