import math
import h5py
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
import random
import time

from tif_image_cutter.preprocessing.preprocessing import Preprocessing

random.seed(4678564)


class DsetCreator:

    def __init__(self, dir_path, output_path):
        if os.path.isdir(dir_path):
            self.dir_path = dir_path
            self.output_path = output_path
        else:
            raise Exception("Invalid directory path passed")


    def read_cut_img_names(self, idx):
        path = f"{self.dir_path}/{idx}"
        img = []
        mask = []
        if os.path.isdir(path):
            for r, d, f in os.walk(path):
                for file in f:
                    if "annotated.tif" in file:
                        mask.append(file)
                    elif ".tif" in file:
                        img.append(file)
                    else:
                        print(f"Omitted file {file}")
        else:
            raise Exception("Passed invalid image index")
        return img, mask

    def to_dataset(self):
        formatter = "{}{}/{}"
        length = 108104
        imgs = np.zeros([length, 256, 256, 3], dtype=np.uint8)
        masks = np.zeros([length, 256, 256, 1], dtype=np.uint8)
        counter = 0
        for i in range(1, 151):
            print(i)
            if i == 20:
                continue
            img, mask = self.read_cut_img_names(i)
            for x in range(len(img)):
                nimg = plt.imread(formatter.format(self.dir_path, i, img[x]))
                imgs[counter, :, :, :] = nimg

                nmask = cv2.imread(formatter.format(self.dir_path, i, mask[x]))
                masks[counter, :, :, :] = np.reshape(nmask[:, :, 1], [256, 256, 1]) / 255

                counter += 1
        order = self.shuffle2(imgs)
        return imgs, masks, order

    def to_h5_dataset(self, idx, img, mask):
        formatter = "{}{}/{}"

        dataset = h5py.File(f"{self.output_path}dataset_{idx}.h5", "w")
        imgs = np.zeros([len(img), 256, 256, 3], dtype=int)
        masks = np.zeros([len(img), 256, 256, 1], dtype=int)
        for i in range(len(img)):
            nimg = cv2.imread(formatter.format(self.dir_path, idx, img[i]))
            new_img = nimg
            new_img[:, :, 0] = nimg[:, :, 2]
            new_img[:, :, 2] = nimg[:, :, 0]
            imgs[i, :, :, :] = new_img
            nmask = cv2.imread(formatter.format(self.dir_path, idx, mask[i]))
            masks[i, :, :, :] = np.reshape(nmask[:, :, 1], [256, 256, 1])

        imgs, masks = self.shuffle(imgs, masks)
        dataset.create_dataset("images", data=imgs)
        dataset.create_dataset("masks", data=masks)


    # wektor indeksów zamiast obrazków
    # np slicing
    def shuffle(self, imgs, masks, whole=True):
        print("shuffling")
        nimgs = np.zeros(np.shape(imgs), dtype=np.uint8)
        nmasks = np.zeros(np.shape(masks), dtype=np.uint8)
        arg = np.linspace(0, np.shape(imgs)[0] - 1, np.shape(imgs)[0], dtype=int).tolist()
        narg = []
        while len(arg) != 0:
            narg.append(arg.pop(int(random.random()*(len(arg) - 1))))
        for x in range(len(narg)):
            nimgs[x, :, :, :] = np.reshape(imgs[narg[x], :, :, :], [256, 256, 3])
            nmasks[x, :, :, :] = np.reshape(masks[narg[x], :, :, :], [256, 256, 1])
        return nimgs, nmasks

    def shuffle2(self, imgs, whole=True):
        print("shuffling")
        arg = np.linspace(0, np.shape(imgs)[0] - 1, np.shape(imgs)[0], dtype=int).tolist()
        narg = []
        while len(arg) != 0:
            narg.append(arg.pop(int(random.random()*(len(arg) - 1))))
        return narg

    def percent_annotated(self, mask):
        return float(np.sum(np.reshape(mask, [256, 256]))) / 256 / 256 * 100

    def read_dataset(self, path):
        dset = h5py.File(path, "r")
        images = dset["images"]
        masks = dset["masks"]
        for x in range(np.shape(images)[0]):
            fig = plt.figure()
            s1 = fig.add_subplot(121)
            s1.imshow(np.reshape(images[x, :, :, :], [256, 256, 3]))
            s2 = fig.add_subplot(122)
            s2.imshow(np.reshape(masks[x, :, :, :], [256, 256]))
            plt.show()

    def generate_histogram_data(self):
        # init zmiennej
        labels = [
            "(0-10%)",
            "[20-10%)",
            "[20-30%)",
            "[30-40%)",
            "[40-50%)",
            "[50-60%)",
            "[60-70%)",
            "[70-80%)",
            "[80-90%)",
            "[90-100%)",
            "[100%]",
            "[0%]"
        ]
        files = {}
        count = {}
        vals = []
        for label in labels:
            files[label] = []
            count[label] = 0

        cntr = 0
        for x in range(1, 151):
            print(x)
            if x == 20:
                continue
            img, mask = self.read_cut_img_names(x)
            path = f"{self.dir_path}{x}//"
            for m in mask:
                c_mask = plt.imread(path + m)/255
                val = self.percent_annotated(c_mask)
                if val == 0.0:
                    files[labels[-1]].append(f"{x}/{m}")
                    count[labels[-1]] += 1
                elif val == 100.0:
                    files[labels[-2]].append(f"{x}/{m}")
                    count[labels[-2]] += 1
                else:  # w przypadku val == 100
                    vals.append(val)
                    idx = math.floor(val / 10)
                    files[labels[idx]].append(f"{x}/{m}")
                    count[labels[idx]] += 1
                    cntr += 1

        return files, count, vals

    def save_segregated(self, files, count, vals):
        file = open("files.csv", "w")
        for key in files:
            file.write(key)
            file.write(",")
            arr = files[key]
            for a in arr:
                file.write(a)
                file.write(",")
        file.close()

        file = open("count.csv", "w")
        for key in count:
            file.write(key)
            file.write(",")
            val = count[key]
            file.write(str(val))
        file.close()

        file = open("vals.csv", "w")
        for v in vals:
            file.write(str(v))
            file.write("\n")
        file.close()

    def to_normalized_dataset(self):
        files, count, _ = self.generate_histogram_data()
        least = 99999999999
        for key in count:
            if count[key] < least:
                least = count[key]
        counter = 0
        names = []
        for key in files:
            for x in range(least):
                item = files[key].pop(int(random.random()*(count[key] - x)))
                names.append(item)
                counter += 1
        return names

    def read_shuffled_img_from_txt_file(self, shulffle=True, how_many=-1, preprocessing_method=None):
        # filename = "_dataset_256_names.txt"
        filename = "_dataset_256_names.txt"
        f = open(filename, "r")
        names = f.readline().split(",")
        f.close()

        #  optional param parsing
        if how_many != -1:
            length = how_many
        else:
            length = len(names) - 1  # przecinek na końcu
        if shulffle:
            order = self.shuffle2(np.zeros(len(names) - 1))
        else:
            order = np.linspace(0, length - 1, length, dtype=int)  # mało wydajne ale moge mieszać lub nie
        imgs = np.zeros([length, 256, 256, 3], dtype=np.int)
        masks = np.zeros([length, 256, 256, 1], dtype=np.int)
        for idx in range(length):
            x = plt.imread(self.dir_path + names[order[idx]])
            masks[idx, :, :, :] = np.reshape(x, [256, 256, 1])/255  # 1 albo 0
            x = plt.imread(self.dir_path + names[order[idx]].replace("_annotated", ""))
            if type(preprocessing_method).__name__ == 'str':
                x = Preprocessing.HE_reinhard_blur_instance(x, self.dir_path)
            elif preprocessing_method is not None:
                x = preprocessing_method(x)
            imgs[idx, :, :, :] = x
        return imgs, masks


if __name__ == "__main__":
    dc = DsetCreator("G:/_dataset_256_sent/", "")#"/home/doszke/", "/home/doszke/")
    x = time.time()
    pre = Preprocessing.get_reinhard_instance("G:/_dataset_256_sent/")
    imgs, masks = dc.read_shuffled_img_from_txt_file(how_many=10, preprocessing=pre.transform)
    print(f"czas wczytywania datasetu: {time.time() - x}")
    print(np.shape(imgs))
    plt.subplot(151)
    plt.imshow(imgs[0, :, :, :])
    plt.subplot(152)
    plt.imshow(imgs[1, :, :, :])
    plt.subplot(153)
    plt.imshow(imgs[2, :, :, :])
    plt.subplot(154)
    plt.imshow(imgs[3, :, :, :])
    plt.subplot(155)
    plt.imshow(imgs[4, :, :, :])
    plt.show()