import time
import numpy as np
from tensorflow.keras.activations import relu, softmax
import tensorflow.keras.backend as K
import tensorflow as tf
tf.get_logger().setLevel('INFO')

if __name__ == "__main__":
    input = np.ones([500, 256, 256], dtype=np.float)*3284.902
    t1 = time.time()
    K.softplus(input)
    t2 = time.time()
    print(f"shape: {np.shape(input)}")
    print(f"softplus: {t2 - t1}")

    t3 = time.time()
    K.relu(input)
    t4 = time.time()
    print(f"relu: {t4 - t3}")

    input = np.ones([25, 500, 256, 256], dtype=np.float) * 3284.902
    t1 = time.time()
    K.softplus(input)
    t2 = time.time()
    print(f"shape: {np.shape(input)}")
    print(f"softplus: {t2 - t1}")

    t3 = time.time()
    K.relu(input)
    t4 = time.time()
    print(f"relu: {t4 - t3}")

    input = np.ones([25*500*256*256], dtype=np.float) * 3284.902
    t1 = time.time()
    K.softplus(input)
    t2 = time.time()
    print(f"shape: {np.shape(input)}")
    print(f"softplus: {t2 - t1}")

    t3 = time.time()
    K.relu(input)
    t4 = time.time()
    print(f"relu: {t4 - t3}")

