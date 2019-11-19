import numpy as np
import multiresolutionimageinterface as mir
from matplotlib import pyplot as plt
import cv2

image = plt.imread("G:\\dataset_cz1_output 2019-10-07 15_29_46\\79_33.tif")
imagei = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
print(np.shape(image))
h = np.reshape(imagei[:, :, 0], [2048, 2048])
s = np.reshape(imagei[:, :, 1], [2048, 2048])
v = np.reshape(imagei[:, :, 2], [2048, 2048])

arr = []
for x in range(1000, 1100):
    for y in range(1000, 1100):
        arr.append(f"{h[x, y]}, {s[x, y]}, {v[x, y]}")

print("ciemny")
print(arr[12*100 + 28])

print("jasny-tkanka")
print(arr[38*100 + 87])

print("jasny nie-tkanka")
print(arr[92*100 + 63])


plt.imshow(image[1000:1100, 1000:1100, :])
plt.show()