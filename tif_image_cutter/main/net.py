import os
import cv2
import imageio
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as color
import multiresolutionimageinterface as mir

from tensorflow.keras.optimizers import Adam
import h5py as h

from tif_image_cutter.main.unet import Unet


class Net:
    def __init__(self):
        self.reader = mir.MultiResolutionImageReader()
        self.img_path_prefix = "G:\\dataset_cz1\\"
        self.output_path_prefix = "G:\\dataset_cz1_output " + str(datetime.datetime.today())[0:19].replace(":", "_") + "\\"
        self.UPPER = 0
        self.LOWER = 1
        self.LEFT = 2
        self.RIGHT = 3

    def get_itemlist(self):
        m_files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(self.img_path_prefix):
            for file in f:
                if '.tif' in file:
                    m_files.append(os.path.join(r, file))

        return m_files

    def read_image(self, idx=1, annotated=False):
        path = self.img_path_prefix + str(idx)
        if annotated:
            path += "_tumor_annotations.tif"
        else:
            path += ".tif"
        mr_image = self.reader.open(path)
        return mr_image

    def cut_image(self, mr_image, level=2, x_begin=0, y_begin=0, width=128, height=128):
        ds = mr_image.getLevelDownsample(level)
        return mr_image.getUCharPatch(int(x_begin * ds), int(y_begin * ds), width, height, level)

    def read_and_cut_image(self, idx=1, read_annotated=False, level=2, x_begin=0, y_begin=0, width=128, height=128):
        mr_image = self.read_image(idx, read_annotated)
        return self.cut_image(mr_image, level, x_begin, y_begin, width, height)

    def get_width(self, img, level):
        i = 1
        width = 0
        while width == 0:
            v = self.cut_image(img, level, 0, 0, 1000 * i, 1)
            #  print(np.shape(v))
            v = np.reshape(v, [1000 * i, 3])
            if np.average(v[len(v) - 1]) == 0:
                for x in range(1, 1000):
                    if np.average(v[len(v) - x]) != 0:
                        width = len(v) - x + 1
                        break
            i += 1
        return width

    """
    Funkcja dająca współrzędne początku i końca adnotacji, z dokładnością do 1000 pikseli. 
    To ma na celu zlokalizowanie oznaczonego obrazu
    """
    def get_annotation_bounds(self, img, width, level, margin):
        y = 0
        xs, xe, ys, ye = [9999999, 0, 9999999, 0]
        iter = int(width / 1000)
        nmbr_of_black_in_row = 0
        while nmbr_of_black_in_row < 10:
            at_least_one_black = False
            v = self.cut_image(img, level, 0, y, width, 1)
            v = np.reshape(v, [width])
            for x in range(1, iter):
                if v[x * 1000] == 1:
                    nmbr_of_black_in_row = 0
                    at_least_one_black = True
                    if ys > y:
                        ys = y
                    if ye < y:
                        ye = y
                    if xs > x * 1000:
                        xs = x * 1000
                    if xe < x * 1000:
                        xe = x * 1000
            if ~at_least_one_black and xs < 9999999:
                nmbr_of_black_in_row += 1

            y += 100
        xs -= margin
        xe += margin
        ys -= margin
        ye += margin
        return xs, xe, ys, ye

    """
    Funkcja podająca położenie oznaczonego obrazu tkanki z dokładnością do 1 piksela
    """
    def get_patch_location(self, idx, level):
        print("Searching bounds")
        normal_img = self.read_image(idx, False)
        annotated_img = self.read_image(idx, True)

        width = self.get_width(normal_img, level)

        xs, xe, ys, ye = self.get_annotation_bounds(annotated_img, width, level, 0)  # bez marginesu

        ys = self.find_bound(normal_img, ys, level, width, self.UPPER)  # default 0, 0
        ye = self.find_bound(normal_img, ye, level, width, self.LOWER)
        xs = self.find_bound(normal_img, xs, level, width, self.LEFT, ys, ye)
        xe = self.find_bound(normal_img, xe, level, width, self.RIGHT, ys, ye)
        print(f"bounds found: {xs} {xe} {ys} {ye}")
        return [xs, xe, ys, ye]

    def find_bound(self, img, var, level, width, location, upper_bound=0, lower_bound=0):
        i = 1024  # var = ys
        # szukanie ys
        bound_set = False
        while not bound_set:
            print(f"location: {location}   var: {var}")
            if location == self.UPPER or location == self.LEFT:
                var -= i
            else:
                var += i
            # wczytuje co i-tą linijke
            if location == self.UPPER or location == self.LOWER:
                v = np.reshape(self.cut_image(img, level=level, x_begin=0, y_begin=var, width=width, height=1), [width, 3])
            elif location == self.LEFT or location == self.RIGHT:
                v = np.reshape(self.cut_image(img, level=level, x_begin=var, y_begin=upper_bound, width=1, height=lower_bound - upper_bound),
                               [lower_bound - upper_bound, 3])
            #szukam takiego wektora v, gdzie są tylko białe lub czarne piksele
            if self.contains_only_white(v):
                if location == self.UPPER or location == self.LEFT:
                    var += i
                else:
                    var -= i  # cofam sie do ostatnio napotkanego koloru
                bound = self.log2n_search_bound(img, i, level, var, location, upper_bound, lower_bound)  # nie biorą tu udziału
                bound_set = True
        if bound < 0:
            return 1
        return bound


    def contains_only_white(self, v):
        all_white = True
        for x in range(1, len(v)):
            if np.average(v[x]) != 255 and np.average(v[x]) != 0:
                all_white = False
        return all_white

    def log2n_search_bound(self, img, base, level, var, location, upper_bound, lower_bound):
        # lower_bound i upper_bound niezbędne są dla poszerzania
        width = self.get_width(img, level)
        # dla szukaniaw góre i w lewo odejmuje (wartość zmniejszam)
        if location == self.UPPER or location == self.LEFT:
            should_add = False
        else:  # dla pozostałych dodaje
            should_add = True
        base /= 2.0
        while base >= 1:  # wyszukanie binarne(?)
            if should_add:  # dodaje dla prawo i dół (gdy wynik pozytywny, chce ZWIĘKSZYĆ te zmienne)
                var += base
            else:  # odejmuje dla góry i lewo (gdy wynik pozytywny, chce ZMNIEJSZYĆ te zmienne)
                var -= base

            if var > width and location == self.RIGHT:
                var = width
                break
            elif var < 0 and location == self.LEFT:
                var = 0
                break

            if location == self.UPPER:  # wczytuje poziomy pasek
                v = np.reshape(self.cut_image(img, level=level, x_begin=0, y_begin=var, width=width, height=1),
                               [width, 3])
                should_add = (self.contains_only_white(v))
            elif location == self.LOWER:  # wczytuje poziomy pasek
                v = np.reshape(self.cut_image(img, level=level, x_begin=0, y_begin=var, width=width, height=1),
                               [width, 3])
                should_add = not self.contains_only_white(v)
            elif location == self.LEFT:  # wczytuje pionowy pasek
                v = np.reshape(self.cut_image(img, level=level, x_begin=var, y_begin=upper_bound, width=1, height=lower_bound - upper_bound),
                               [lower_bound - upper_bound, 3])
                should_add = self.contains_only_white(v)
            elif location == self.RIGHT:  # wczytuje pionowy pasek
                v = np.reshape(self.cut_image(img, level=level, x_begin=var, y_begin=upper_bound, width=1, height=lower_bound - upper_bound),
                               [lower_bound - upper_bound, 3])
                should_add = not self.contains_only_white(v)
            base /= 2.0

        return int(var)

    def cut_image_for_dataset(self, idx, bounds, side_length, output_path, level):
        xs, xe, ys, ye = bounds
        height = ye - ys
        width = xe - xs

        hor_iter = int(width / side_length)
        ver_iter = int(height / side_length)

        normal_img = self.read_and_cut_image(idx=idx, read_annotated=False, level=level, x_begin=xs, y_begin=ys, width=width, height=height)
        annotated_img = self.read_and_cut_image(idx=idx, read_annotated=True, level=level, x_begin=xs, y_begin=ys, width=width, height=height)

        image_format = "{}{}_{}.tif"
        annot_format = image_format.replace(".tif", "_annotated.tif")
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
            print(f"created folder at: {output_path}")
        print("cutting images begun:")
        for x in range(1, hor_iter):
            for y in range(1, ver_iter):
                image = normal_img[(x-1)*side_length:x*side_length, (y-1)*side_length:y*side_length, :]
                if self.validate_dataset_image(image) and np.shape(image)[0]*np.shape(image)[1] == side_length**2:  # pomijam zdjęcia z samym białym
                    imageio.imwrite(image_format.format(output_path, idx, x*hor_iter + 1 + y), image)
                    aimage = annotated_img[(x-1)*side_length:x*side_length, (y-1)*side_length:y*side_length, :]*255
                    imageio.imwrite(annot_format.format(output_path, idx, x*hor_iter + 1 + y), np.reshape(aimage, [side_length, side_length]))
        print("cutting images ended\n")

    def validate_dataset_image(self, image):
        avg = np.average(image)
        return avg <= 240
        

if __name__ == "__main__":
    net = Net()
    xs, xe, ys, ye = net.get_patch_location(idx=2, level=2)
    img = net.read_and_cut_image(idx=2, read_annotated=False, level=2, x_begin=xs, y_begin=ys, width=xe - xs,
                                 height=ye - ys)
    mask = net.read_and_cut_image(idx=2, read_annotated=True, level=2, x_begin=xs, y_begin=ys, width=xe - xs,
                                 height=ye - ys)
    plt.subplot(211)
    plt.imshow(img)
    plt.subplot(212)
    plt.imshow(np.reshape(mask, [ye-ys, xe-xs]), cmap="gray")
    plt.show()