import colorsys
import time
import os
import numpy as np
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt, colors
from scipy.signal import medfilt2d
from staintools.reinhard_normalization import ReinhardColorNormalizer as reinhard


class Preprocessing:

    pre = reinhard()

    @staticmethod
    def median_filter(image):
        median = cv2.medianBlur(image, 5)
        return median

    @staticmethod
    def get_mean_and_std(x):
        x_mean, x_std = cv2.meanStdDev(x)
        x_mean = np.hstack(np.around(x_mean, 2))
        x_std = np.hstack(np.around(x_std, 2))
        return x_mean, x_std

    @staticmethod
    def histogram_equalization(img):
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])

        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()

        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        return cdf[img]

    @staticmethod
    def get_reinhard_instance(beginpath: str):
        pre = reinhard()
        target = Preprocessing.histogram_equalization(
            plt.imread(f"{beginpath}59/59_139.tif"))  # 75/75_734.tif"))#28/28_6240.tif"))  # 75 734
        pre.fit(target=target)
        return pre


if __name__ == '__main__':
    t = time.time()
    f = open("C:\\Users\\Jakub Siembida\\PycharmProjects\\inz\\tif_image_cutter\\main\\_dataset_256_names.txt")
    x = f.readline().split(",")
    print(time.time() - t)

    norm = reinhard()
    x[0] = "23/23_420.tif"

    target = Preprocessing.histogram_equalization(plt.imread("G:/_dataset_256_sent/59/59_139.tif")) #75/75_734.tif"))#28/28_6240.tif"))  # 75 734
    norm.fit(target=target)
    for name in x:
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        name = name.replace("_annotated", "")
        img = plt.imread(f"G:/_dataset_256_sent/{name}")
        print(np.shape(img))
        plt.subplot(141)
        plt.imshow(img)
        plt.subplot(142)
        plt.imshow(Preprocessing.median_filter(img))
        plt.subplot(143)
        z = norm.transform(img)
        print(np.shape(z))
        plt.imshow(z)
        plt.subplot(144)
        plt.imshow(Preprocessing.median_filter(norm.transform(img)))
        plt.show()
