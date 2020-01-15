from create_dset import DsetCreator
import numpy as np
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.models import model_from_json

from tif_image_cutter.main.model import unet
from tif_image_cutter.preprocessing.preprocessing import Preprocessing

if __name__ == "__main__":
    model_name = "model_down_1"
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_name + ".h5")

    dc = DsetCreator("G:/_dataset_256_sent/", "")
    Preprocessing.HE_reinhard_blur_instance(np.ones([256, 256, 3], dtype=np.uint8)*40, "G:/_dataset_256_sent/")
    img, mask = dc.read_shuffled_img_from_txt_file(how_many=1000, preprocessing_method=None, shulffle=True)
    img = img[900:1000, :, :, :]
    mask = mask[900:1000, :, :, :]
    arr = np.zeros([len(img)])
    for i in range(len(img)):
        #fig = plt.figure()
        #p1 = fig.add_subplot(141)
        #p2 = fig.add_subplot(142)
        #p3 = fig.add_subplot(143)
        #p4 = fig.add_subplot(144)
        image = img[i, :, :, :]
        maskk = mask[i, :, :, :]

        #p1.imshow(np.reshape(image, [256, 256, 3]))
        #p2.imshow(np.reshape(maskk, [256, 256]), cmap="hot")
        pred = model.predict(np.reshape(image, [1, 256, 256, 3]))
        val = unet.Unet().dice_coef(maskk, (np.round(pred).astype("int32")))
        xd = tf.keras.backend.eval(val)
        arr[i] = xd
        #p3.imshow(np.reshape(pred, [256, 256]), cmap="hot", vmin=0, vmax=1)
        #p4.imshow(np.round(np.reshape(pred, [256, 256])), cmap="hot", vmin=0, vmax=1)
        #plt.show()
    print(f"dice: {np.average(arr)}")
