import unet
from create_dset import DsetCreator as Dc
import numpy as np
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt

if __name__ == "__main__":
    model_name = "test_dice_2"
    u = unet.Unet()

    model = u.my_unet_model(size=256, down=4)

    model.summary()

    img1 = np.reshape(plt.imread("G:\\_dataset_256_sent\\1\\1_507.tif"), [256, 256, 3]) / 255
    mask1 = np.reshape(plt.imread("G:\\_dataset_256_sent\\1\\1_507_annotated.tif"), [256, 256, 1]) / 255
    img2 = np.reshape(plt.imread("G:\\_dataset_256_sent\\1\\1_513.tif"), [256, 256, 3]) / 255
    mask2 = np.reshape(plt.imread("G:\\_dataset_256_sent\\1\\1_513_annotated.tif"), [256, 256, 1]) / 255

    imgs = np.zeros([2, 256, 256, 3])
    masks = np.zeros([2, 256, 256, 1])
    imgs[0, :, :, :] = img1
    imgs[1, :, :, :] = img2

    masks[0, :, :, :] = mask1
    masks[1, :, :, :] = mask2

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[u.dice_coef])
    model.fit(imgs, masks, epochs=50, verbose=1, shuffle="batch")

    score = model.evaluate(imgs, masks, verbose=0)

    # zapis wag itd
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

    model_json = model.to_json()
    with open(f"{model_name}.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f"{model_name}.h5")
    print("Saved model to disk")