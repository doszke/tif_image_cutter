import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import directed_hausdorff as hausdorff


def circle_a(x, y):
    if (x - 128)**2 + (y - 128)**2 < 100**2:
        return 1
    return 0


def circle_b(x, y):
    if (x - 128) ** 2 + (y - 128) ** 2 < 110**2:
        return 1
    return 0


if __name__ == '__main__':
    a = np.zeros([256, 256, 1])
    b = np.zeros([256, 256, 1])
    for x in range(256):
        for y in range(256):
            a[x, y, 0] = circle_a(x, y)
            b[x, y, 0] = circle_b(x, y)

    plt.subplot(121)
    plt.imshow(np.reshape(a, [256, 256]))
    plt.subplot(122)
    plt.imshow(np.reshape(b, [256, 256]))

    print(i)
    print(j)
    print(max(i[0], j[0]))
    plt.show()
