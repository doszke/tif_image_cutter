import colorsys
import time
import os
import numpy as np
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt, colors
from scipy.signal import medfilt2d


class Preprocessing:

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
    def reinhard_transform(image):
        s = image
        t = Preprocessing.histogram_equalization(image)
        s_mean, s_std = Preprocessing.get_mean_and_std(s)
        t_mean, t_std = Preprocessing.get_mean_and_std(t)

        height, width, channel = s.shape
        _s = s.copy()
        for i in range(0, height):
            for j in range(0, width):
                for k in range(0, channel):
                    x = s[i, j, k]
                    x = ((x - s_mean[k]) * (t_std[k] / s_std[k])) + t_mean[k]
                    # round or +0.5
                    x = round(x)
                    # boundary check
                    x = 0 if x < 0 else x
                    x = 255 if x > 255 else x
                    _s[i, j, k] = x

        s = cv2.cvtColor(_s, cv2.COLOR_LAB2RGB)
        return s


    @staticmethod
    def histogram_equalization(img):
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])

        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()

        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        return cdf[img]
        #plt.plot(cdf_normalized, color='b')
        #plt.hist(cdf[img].flatten(), 256, [0, 256], color='r')
        #plt.xlim([0, 256])
        #plt.legend(('cdf', 'histogram'), loc='upper left')
        #plt.show()
        #print(np.max(np.subtract(image, cdf[img])))


if __name__ == '__main__':
    t = time.time()
    f = open("C:\\Users\\Jakub Siembida\\PycharmProjects\\inz\\tif_image_cutter\\main\\_dataset_256_names.txt")
    x = f.readline().split(",")
    print(time.time() - t)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    for name in x:
        name = name.replace("_annotated", "")
        img = plt.imread(f"G:/_dataset_256_sent/{name}")
        print(np.shape(img))
        plt.subplot(141)
        plt.imshow(img)
        plt.subplot(142)
        plt.imshow(Preprocessing.median_filter(img))
        plt.subplot(143)
        plt.imshow(Preprocessing.reinhard_transform(img))
        plt.subplot(144)
        plt.imshow(Preprocessing.median_filter(Preprocessing.reinhard_transform(img)))
        plt.show()
