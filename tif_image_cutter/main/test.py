import numpy as np
from matplotlib import pyplot as plt
from tif_image_cutter.main.create_dset import DsetCreator
from skimage.color import rgb2hed
from skimage.exposure.exposure import rescale_intensity
from tif_image_cutter.preprocessing.preprocessing import Preprocessing as P

if __name__ == '__main__':
    dc = DsetCreator("G:/_dataset_256_sent/", "")
    inst = P.get_reinhard_instance("G:/_dataset_256_sent/")
    img, _ = dc.read_shuffled_img_from_txt_file(shulffle=True, how_many=20, preprocessing_method=inst.transform)
    for x in range(20):
        mgr = plt.get_current_fig_manager()
        mgr.window.state('zoomed')
        pre_img = rgb2hed(img[x, :, :, :])
        new_img = np.zeros([256, 256, 3], dtype=np.float)
        new_img[:, :, 0] = rescale_intensity(pre_img[:, :, 0], out_range=(0, 1))
        new_img[:, :, 1] = rescale_intensity(pre_img[:, :, 1], out_range=(0, 1))
        new_img[:, :, 2] = rescale_intensity(pre_img[:, :, 2], out_range=(0, 1))
        pre_img = new_img
        plt.subplot(151)
        plt.imshow(img[x, :, :, :])
        plt.subplot(152)
        plt.imshow(pre_img[:, :, 0], cmap="gray")
        print(np.max(pre_img[:, :, 0]))
        print(np.min(pre_img[:, :, 0]))
        plt.subplot(153)
        plt.imshow(pre_img[:, :, 1], cmap="gray")
        print(np.max(pre_img[:, :, 1]))
        print(np.min(pre_img[:, :, 1]))
        plt.subplot(154)
        plt.imshow(pre_img[:, :, 2], cmap="gray")
        plt.subplot(155)
        print(np.max(pre_img[:, :, 2]))
        print(np.min(pre_img[:, :, 2]))
        plt.imshow(pre_img)
        plt.show()
