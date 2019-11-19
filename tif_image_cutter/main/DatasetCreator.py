import os
import math
import numpy as np
import h5py as h
from matplotlib import pyplot as plt

class DatasetCreator:

    def __init__(self, dir_path):
        self.path = dir_path

    def create_dataset(self, name):
        file = h.File(name, 'w')
        train = file.create_group("train")

        validate = file.create_group("validate")
        ti, to, vi, vo = self.fill_train_from_precut_images()
        train.create_dataset("train_in", data=ti)
        train.create_dataset("train_out", data=to)
        validate.create_dataset("validate_in", data=vi)
        validate.create_dataset("validate_out", data=vo)


    def fill_train_from_precut_images(self, ratio=0.75):
        files = self.get_itemlist()
        # nie przemyślałem kwestii nazewnictwa- teraz muszę posortować
        twos, threes, fours, fives = files

        size = 0
        for f in files:
            size += len(f)
        print(size)
        print(size/2)
        print()
        print()

        bound = math.floor(size/2*3/4)
        second_bound = math.floor(size/2-math.floor(size/2*3/4))

        img = plt.imread(self.path + twos[0])
        shape = np.shape(img)

        TRAIN_IN = np.zeros([bound, shape[0], shape[1], 3], dtype=int)
        TRAIN_OUT = np.zeros([bound, shape[0], shape[1], 1], dtype=int)

        VALIDATE_IN = np.zeros([second_bound, shape[0], shape[1], 3], dtype=int)
        VALIDATE_OUT = np.zeros([second_bound, shape[0], shape[1], 1], dtype=int)

        index_train = 0
        index_valid = 0
        lcounter = 0

        for z in range(len(files)):
            fileset = files[z]
            local_bound = math.floor(len(fileset)/2*ratio)
            for x in range(len(fileset)):
                file = fileset[x]
                img = plt.imread(self.path + file)
                lcounter += 1
                if lcounter % 8 == 1 or lcounter % 8 == 2:
                    print("VALIDATE")
                    if len(np.shape(img)) == 2:
                        VALIDATE_OUT[math.floor(index_valid)] = np.reshape(img, [shape[0], shape[1], 1])
                    else:
                        VALIDATE_IN[math.floor(index_valid)] = img
                    index_valid += 0.5
                else:
                    print("TRAIN")
                    if len(np.shape(img)) == 2:
                        TRAIN_OUT[math.floor(index_train)] = np.reshape(img, [shape[0], shape[1], 1])
                    else:
                        TRAIN_IN[math.floor(index_train)] = img
                    index_train += 0.5
                # xd
        return TRAIN_IN, TRAIN_OUT, VALIDATE_IN, VALIDATE_OUT

    def read_existing_dataset(self, name):
        file = h.File(name, 'r')
        return file

    def get_itemlist(self):
        # r=root, d=directories, f = files
        twos = []
        threes = []
        fours = []
        fives = []
        for r, d, f in os.walk(self.path):
            for file in f:
                if file[0] == '2':
                    twos.append(file)
                elif file[0] == '3':
                    threes.append(file)
                elif file[0] == '4':
                    fours.append(file)
                elif file[0] == '5':
                    fives.append(file)

        return [twos, threes, fours, fives]


if __name__ == "__main__":
    dataset_creator = DatasetCreator("G:\\dataset_64x64\\")
    f = dataset_creator.read_existing_dataset("dataset_64.hdf5")
    train = f["train"]
    data = train["train_in"]
    size = np.shape(data)
    cntr = 0
    for x in range(size[0]):
        if np.average(data[x, :, :, :]) == 0:
            cntr += 1
        plt.show()
    print(cntr)
    print(cntr/size[0]*100)

    validate = f["validate"]
    data = validate["validate_in"]
    size = np.shape(data)
    cntr = 0
    for x in range(size[0]):
        if np.average(data[x, :, :, :]) == 0:
            cntr += 1
        plt.show()
    print(cntr)
    print(cntr / size[0] * 100)