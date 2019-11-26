import unet
from create_dset import DsetCreator as Dc
import numpy as np
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from preprocessing import Preprocessing

if __name__ == "__main__":
    model_name = "better_model_control"
    u = unet.Unet()
    begin_path = "/home/doszke/"
    dc = Dc(begin_path, "")

    model = u.my_unet_model(size=256, down=4)

    model.summary()

    reinhard = Preprocessing.get_reinhard_instance(begin_path)
    imgs, masks = dc.read_shuffled_img_from_txt_file(preprocessing_method=reinhard.transform)

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[u.dice_coef])

    model_json = model.to_json()
    with open(f"{model_name}.json", "w") as json_file:
        json_file.write(model_json)

    for x in range(5):
        print("ITERACJA " + str(x))
        model.fit(imgs[0:900, :, :, :], masks[0:900, :, :, :],  epochs=10, verbose=1, shuffle="batch")

        score = model.evaluate(imgs[900:1000, :, :, :], masks[900:1000, :, :, :], verbose=0)
        # serialize weights to HDF5
        model.save_weights(f"{model_name}_{x}.h5")
        print("Saved model to disk")

        # zapis wag itd
        print("%s: %.2f%%" % (model.metrics_names[0], score[0] * 100))
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

