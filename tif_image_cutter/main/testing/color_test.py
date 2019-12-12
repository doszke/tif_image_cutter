from matplotlib import pyplot as plt
from skimage.color import rgb2hed
import numpy as np
from tif_image_cutter.main.create_dset import DsetCreator

if __name__ == '__main__':
    array = np.zeros([3])
    coll = np.zeros([256**3, 3])
    iter = 0
    # for b in range(256):
    #     print(b)
    #     for g in range(256):
    #         for r in range(256):
    #             coll[iter, :] = np.reshape(rgb2hed(array), [1, 3])
    #             array[2] += 1
    #         array[1] += 1
    #         array[2] = 0
    #     array[0] += 1
    #     array[1] = 0
    #     array[2] = 0

    # [-6.7153104852424015, 0.0, -4.894010178658725]
    # [0.0, 1.9608354107171047, 0.0]
    #-0.8388241431686116, 0.24493226518171235
    min = -6.7153104852424015
    max = 1.9608354107171047
    dc = DsetCreator("G:/_dataset_256_sent/", "")
    img, _ = dc.read_shuffled_img_from_txt_file(shulffle=True, how_many=100, preprocessing_method="my")

    for x in range(100):
        conv = np.reshape(img[x, :, :, :], [256, 256, 3])
        print(f"{np.min(conv)}, {np.max(conv)}")
        # conv = ((conv - min) /(max - min))*255
        print(f"{np.min(conv)}, {np.max(conv)}")
        plt.subplot(151)
        plt.imshow(np.reshape(conv[:, :, 0], [256, 256]), cmap="gray")
        plt.subplot(152)
        plt.imshow(np.reshape(conv[:, :, 1], [256, 256]), cmap="gray")
        plt.subplot(153)
        plt.imshow(np.reshape(conv[:, :, 2], [256, 256]), cmap="gray")
        plt.subplot(154)
        plt.imshow(conv)
        plt.subplot(155)
        plt.imshow(np.reshape(img[x, :, :, :], [256, 256, 3]))
        plt.show()