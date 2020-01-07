import pandas as pd
import unet
from create_dset import DsetCreator as Dc
import numpy as np
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam


if __name__ == "__main__":
    model_name = "final_model_control"
    u = unet.Unet()

    dc = Dc("/home/doszke/", "")

    model = u.my_unet_model(size=256, down=4)

    model.summary()

    imgs, masks = dc.read_shuffled_img_from_txt_file(how_many=1000)

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[u.dice_coef])

    history = model.fit(imgs[0:900, :, :, :], masks[0:900, :, :, :],  epochs=100, verbose=1, shuffle="batch")

    score = model.evaluate(imgs[900:1000, :, :, :], masks[900:1000, :, :, :], verbose=0)

    model_json = model.to_json()
    with open(f"{model_name}.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f"{model_name}.h5")
    print("Saved model to disk")

    hist_df = pd.DataFrame(history.history)
    hist_json_file = f'{model_name}_history.json'
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    # zapis wag itd
    print("%s: %.2f%%" % (model.metrics_names[0], score[0] * 100))
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

