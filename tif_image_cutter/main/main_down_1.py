import unet
from create_dset import DsetCreator as Dc
import numpy as np
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":
    model_name = "model_down_1"
    u = unet.Unet()

    dc = Dc("G:/_dataset_256_sent/", "")#"/home/doszke/", "")

    model = u.my_unet_model(size=256, down=1)

    model.summary()

    imgs, masks, order = dc.to_dataset()

    # todo zmień
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[u.dice_coef, u.hausdorff])
    model.fit(imgs[order[0:10], :, :, :], masks[order[0:10], :, :, :], epochs=1, verbose=1, shuffle="batch")

    score = model.evaluate(imgs[order[9:10], :, :, :], masks[order[9:10], :, :, :], verbose=0)

    # zapis wag itd
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

    model_json = model.to_json()
    with open(f"{model_name}.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f"{model_name}.h5")
    print("Saved model to disk")