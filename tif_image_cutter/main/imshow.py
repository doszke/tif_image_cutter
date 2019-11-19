from create_dset import DsetCreator
import numpy as np
import h5py as h
from tensorflow.keras import *
from tensorflow.keras.models import model_from_json
from matplotlib import pyplot as plt


if __name__ == "__main__":
    model_name = "model_down_3"
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_name + ".h5")

    dc = DsetCreator("G:/_dataset_256_sent/", "")

    img, mask = dc.read_cut_img_names(4)

    for x in range(len(img)):
        fig = plt.figure()
        p1 = fig.add_subplot(131)
        p2 = fig.add_subplot(132)
        p3 = fig.add_subplot(133)
        image = plt.imread("{}{}/{}".format(dc.dir_path, 4, img[x]))
        maskk = plt.imread("{}{}/{}".format(dc.dir_path, 4, mask[x]))

        p1.imshow(np.reshape(image, [256, 256, 3]))
        p2.imshow(np.reshape(maskk, [256, 256]), cmap="hot")
        pred = model.predict(np.reshape(image, [1, 256, 256, 3]))
        print(pred)
        p3.imshow(np.reshape(pred, [256, 256]), cmap="hot", vmin=0, vmax=1)
        plt.show()
