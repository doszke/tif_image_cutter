from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from keras.utils import np_utils
from tensorflow.keras.models import model_from_json
import numpy as np

if __name__ == "__main__":
    json_file = open('mnist2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("mnist2.h5")

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print(X_train.shape)
    for x in range(np.shape(X_train)[0]):
        print("expected: ")
        print(y_train[x])
        print("got: ")
        print(model.predict(np.reshape(X_train[x, :, :], [1, 28, 28, 1])))
        print()
