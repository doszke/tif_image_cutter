import multiresolutionimageinterface as mir
from matplotlib import pyplot as plt

reader = mir.MultiResolutionImageReader()
mr_image = reader.open('G:/dataset_cz1/2.tif')
base = 230

xs, xe, ys, ye = [4079, 13417, 18419, 22225]

ds = mr_image.getLevelDownsample(2)
image_patch = mr_image.getUCharPatch(int(xs * ds), int(ys * ds), xe-xs, ye-ys, 2)
plt.imshow(image_patch)
    # wniosek: level odwrotnie proporcjonalny do powiększenia
plt.show()