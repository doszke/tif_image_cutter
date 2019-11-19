import unet
import numpy as np
import h5py as h
from tensorflow.keras.optimizers import Adam


if __name__ == "__main__":
    u = unet.Unet()
    dataset = h.File("dataset_64.hdf5")

    train = dataset["train"]
    validate = dataset["validate"]
    model = u.my_unet_model(size=64)

    print(np.shape(train["train_in"]))
    print(np.shape(train["train_out"]))
    print(np.shape(validate["validate_in"]))
    print(np.shape(validate["validate_out"]))

    model.compile(optimizer=Adam(1e-6), loss='binary_crossentropy', metrics=[u.dice_coef])
    model.fit(train["train_in"][2:3, :, :, :], train["train_out"][2:3, :, :, :], epochs=500, verbose=1, shuffle="batch")

    score = model.evaluate(validate["validate_in"][2:3, :, :, :], validate["validate_out"][2:3, :, :, :], verbose=0)

    # zapis wag itd
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

    model_json = model.to_json()
    with open("model_64_troubleshooting.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_64_troubleshooting.h5")
    print("Saved model to disk")
