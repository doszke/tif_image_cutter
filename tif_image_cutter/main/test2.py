import numpy as np
from matplotlib import pyplot as plt
from tif_image_cutter.main.create_dset import DsetCreator
from skimage.color import rgb2hed
from skimage.exposure.exposure import rescale_intensity
from tif_image_cutter.preprocessing.preprocessing import Preprocessing as P

if __name__ == '__main__':
    dc = DsetCreator("G:/_dataset_256_sent/", "")
    img, mask = dc.read_shuffled_img_from_txt_file(shulffle=True, how_many=100, preprocessing_method="my")
    for x in range(100):
        mgr = plt.get_current_fig_manager()
        mgr.window.state('zoomed')
        pre_img = img[x, :, :, :]
        plt.subplot(151)
        plt.imshow(np.reshape(mask[x, :, :, :], [256, 256]), vmin=0, vmax=1, cmap="gray")
        plt.subplot(152)
        plt.imshow(pre_img[:, :, 0], cmap="gray")
        plt.subplot(153)
        plt.imshow(pre_img[:, :, 1], cmap="gray")
        plt.subplot(154)
        plt.imshow(pre_img[:, :, 2], cmap="gray")
        plt.subplot(155)
        plt.imshow(pre_img)
        plt.show()
