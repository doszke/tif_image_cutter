import multiresolutionimageinterface as mir
from matplotlib import pyplot as plt
import numpy as np

from tif_image_cutter.main.net import Net

if __name__ == "__main__":
    net = Net()
    img = net.read_image(27, False)
    bounds = [23, 23167, 8051, 31819]
    xs, xe, ys, ye = bounds
    height = ye - ys
    width = xe - xs
    ds = img.getLevelDownsample(2)
    patch = img.getUCharPatch(int(xs*ds), int(ys*ds), height, width, level=2)
    print(np.shape(patch))
    plt.imshow(patch)
    plt.show()