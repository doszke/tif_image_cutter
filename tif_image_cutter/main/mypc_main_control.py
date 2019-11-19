import unet
from create_dset import DsetCreator as Dc
import numpy as np
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam


if __name__ == "__main__":
    model_name = "mypc_model_control"
    u = unet.Unet()

    dc = Dc("/home/doszke/", "")

    model = u.my_unet_model(size=256, down=4)

    model.summary()

    imgs, masks, order = dc.to_dataset()

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[u.dice_coef])

    model.fit(imgs[order[0:900], :, :, :], masks[order[0:900], :, :, :], validation_split=0.1, epochs=50, verbose=1, shuffle="batch")

    score = model.evaluate(imgs[order[900:1000], :, :, :], masks[order[900:1000], :, :, :], verbose=0)

    model_json = model.to_json()
    with open(f"{model_name}.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f"{model_name}.h5")
    print("Saved model to disk")

    # zapis wag itd
    print("%s: %.2f%%" % (model.metrics_names[0], score[0] * 100))
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

