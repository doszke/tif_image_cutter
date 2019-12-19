import numpy as np

from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Concatenate, Input
from tensorflow.keras.models import Sequential, Model


class Unet:

    def down(self, input_layer, filters, pool=True, activation="softplus"):
        conv1 = Conv2D(filters, (3, 3), padding='same', activation=activation)(input_layer)
        residual = Conv2D(filters, (3, 3), padding='same', activation=activation)(conv1)
        if pool:
            max_pool = MaxPool2D()(residual)
            return max_pool, residual
        else:
            return residual

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
