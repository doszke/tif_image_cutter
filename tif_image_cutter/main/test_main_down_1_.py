import unet
from create_dset import DsetCreator as Dc
import numpy as np
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":
    model_name = "_mypc_model_down_1_"
    u = unet.Unet()
    path1 = "G:/_dataset_256_sent/"
    path2 = "/home/doszke/"
    dc = Dc(path2, "")

    model = u.my_unet_model(size=256, down=1)

    model.summary()

    imgs, masks = dc.read_shuffled_img_from_txt_file(how_many=100, shulffle=True)
    print(np.max(masks))

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[u.dice_coef])
    model.fit(imgs[0:90, :, :, :], masks[0:90, :, :, :], epochs=5, verbose=1, shuffle="batch")

    score = model.evaluate(imgs[90:100, :, :, :], masks[90:100, :, :, :], verbose=0)

    # zapis wag itd
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

    model_json = model.to_json()
    with open(f"{model_name}.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f"{model_name}.h5")
    print("Saved model to disk")