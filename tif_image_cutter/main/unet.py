# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 16:26:58 2018
@author: manjotms10
Źródło: https://github.com/manjotms10/U-Net-in-Keras
"""
import os
import numpy as np
import math
import h5py
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu
from scipy.spatial.distance import jaccard, directed_hausdorff as _hausdorff


class Unet:

    def hausdorff(self, a: tf, b: tf):
        sess = tf.Session()
        with sess:
            return _hausdorff(np.reshape(a.eval(sess), [256, 256]), np.reshape(b.eval(sess), [256, 256])) #, hausdorff(b, a))

    def prelu(self, x: tf):
        return relu(x) - 0.001 * x

    """Źródło: https://github.com/manjotms10/U-Net-in-Keras"""
    def down(self, input_layer, filters, pool=True, activation="softplus"):
        conv1 = Conv2D(filters, (3, 3), padding='same', activation=activation)(input_layer)
        residual = Conv2D(filters, (3, 3), padding='same', activation=activation)(conv1)
        if pool:
            max_pool = MaxPool2D()(residual)
            return max_pool, residual
        else:
            return residual

    """Źródło: https://github.com/manjotms10/U-Net-in-Keras"""
    def up(self, input_layer, residual, filters, activation="softplus"):
        filters = int(filters)
        upsample = UpSampling2D()(input_layer)
        upconv = Conv2D(filters, kernel_size=(2, 2), padding="same")(upsample)
        concat = Concatenate(axis=3)([residual, upconv])
        conv1 = Conv2D(filters, (3, 3), padding='same', activation=activation)(concat)
        conv2 = Conv2D(filters, (3, 3), padding='same', activation=activation)(conv1)
        return conv2

    def my_unet_model(self, filters=64, size=32, down=4, activation="softplus"):
        residuals = []
        shape = [size, size, 3]
        start_bound = shape[0]
        input_layer = Input(shape=shape)
        d = input_layer
        res = None
        for x in range(down):
            d, res = self.down(d, filters, activation=activation)
            residuals.append(res)
            filters *= 2
        d = self.down(d, filters, pool=False, activation=activation)
        print(np.shape(d))
        print(residuals)
        for x in range(down):
            d = self.up(d, residual=residuals[-x-1], filters=filters/2, activation=activation)
            filters /= 2
        out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(d)
        model = Model(input_layer, out)
        return model

    """Źródło: https://github.com/manjotms10/U-Net-in-Keras"""
    def get_unet_model(self, input_layer=Input(shape=[128, 128, 3]), filters=64):
        # Make a custom U-nets implementation.
        # input layer e.g: input_layer = Input(shape=[128, 128, 3])
        layers = [input_layer]
        residuals = []

        # Down 1, 128
        d1, res1 = self.down(input_layer, filters)
        residuals.append(res1)
        filters *= 2

        # Down 2, 64
        d2, res2 = self.down(d1, filters)
        residuals.append(res2)
        filters *= 2

        # Down 3, 32
        d3, res3 = self.down(d2, filters)
        residuals.append(res3)
        filters *= 2

        # Down 4, 16
        d4, res4 = self.down(d3, filters)
        residuals.append(res4)
        filters *= 2

        # Down 5, 8
        d5 = self.down(d4, filters, pool=False)

        # Up 1, 16
        up1 = self.up(d5, residual=residuals[-1], filters=filters / 2)
        filters /= 2

        # Up 2,  32
        up2 = self.up(up1, residual=residuals[-2], filters=filters / 2)
        filters /= 2

        # Up 3, 64
        up3 = self.up(up2, residual=residuals[-3], filters=filters / 2)
        filters /= 2

        # Up 4, 128
        up4 = self.up(up3, residual=residuals[-4], filters=filters / 2)

        out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(up4)

        model = Model(input_layer, out)
        # dot, svg,
        return model

    """Źródło: https://github.com/manjotms10/U-Net-in-Keras"""
    def dice_coef(self, y_true, y_pred):
        smooth = 1e-5

        y_true = tf.round(tf.reshape(y_true, [-1]))
        y_pred = tf.round(tf.reshape(y_pred, [-1]))

        isct = tf.reduce_sum(y_true * y_pred)

        return 2 * isct / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))


if __name__ == "__main__":
    u = Unet()
    dataset = h5py.File("dataset_64.hdf5")

    train = dataset["train"]
    validate = dataset["validate"]
    model = u.my_unet_model(size=64)

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[u.dice_coef])
    model.fit(train["train_in"], train["train_out"], epochs=5, verbose=1)

    score = model.evaluate(validate["validate_in"], validate["validate_out"], verbose=0)

    # zapis wag itd
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

    model_json = model.to_json()
    with open("empty.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("empty.h5")
    print("Saved model to disk")


