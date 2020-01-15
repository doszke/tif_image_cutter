import unet
from create_dset import DsetCreator as Dc
import numpy as np
import h5py as h
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":
    u = unet.Unet()

    dc = Dc("/home/doszke/", "")

    model = u.my_unet_model(size=256)

    imgs, masks = dc.to_dataset()

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[u.dice_coef])
    model.fit(imgs[0:900, :, :, :], masks[0:900, :, :, :], epochs=50, verbose=1, shuffle="batch")

    score = model.evaluate(imgs[900:1000, :, :, :], masks[900:1000, :, :, :], verbose=0)

    # zapis wag itd
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

    model_json = model.to_json()
    with open("model_256_on_10000.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_256_on_10000.h5")
    print("Saved model to disk")