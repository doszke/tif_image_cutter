import unet
from create_dset import DsetCreator as Dc
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json

if __name__ == "__main__":
    dc = Dc("/home/doszke/", "")
    u = unet.Unet()
    model_name = "model_preprocessing_hed_3"

    json_file = open(f'{model_name}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    model.load_weights(f"{model_name}.h5")

    model.summary()

    imgs, masks = dc.read_shuffled_img_from_txt_file(how_many=1000)

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[u.dice_coef])
    #history = model.fit(imgs[0:900, :, :, :], masks[0:900, :, :, :], epochs=50, verbose=1, shuffle="batch")

    score = model.evaluate(imgs[900:1000, :, :, :], masks[900:1000, :, :, :], verbose=1)

    # zapis wag itd
    print("%s: %.2f%%" % (model.metrics_names[0], score[0] * 100))
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))


