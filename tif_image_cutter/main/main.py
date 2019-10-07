import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
import multiresolutionimageinterface as mir


class Main:
    def __init__(self):
        self.reader = mir.MultiResolutionImageReader()
        self.img_path_prefix = "G:\\dataset_cz1\\"
        self.output_path_prefix = self.img_path_prefix + "G:\\dataset_cz1_output " + str(datetime.datetime.today())[0:19] + "\\"
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
    Funkcja dająca współrzędne początku i końca adnotacji, z dokładnością do 100 pikseli. 
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
    Funkcja podająca położenie oznaczonego obrazu tkanki    
    """
    def get_patch_location(self, idx, level, annotation_bounds):
        normal_img = self.read_image(idx, False) 
        annotated_img = self.read_image(idx, True)

        width = self.get_width(normal_img, level)

        xs, xe, ys, ye = self.get_annotation_bounds(annotated_img, width, level, 0)  # bez marginesu

        i = 256
        # TODO złącz w 1 metode
        # szukanie ys
        bound_set = False
        while not bound_set:
            ys -= i
            # wczytuje co 256tą linijke
            v = np.reshape(self.cut_image(normal_img, level=level, x_begin=0, y_begin=ys, width=width, height=1), [width, 3])
            if self.contains_only_white(v):
                ys += i  # cofam sie do ostatnio napotkanego koloru
                ys = self.log2n_search_bound(normal_img, i, level, ys, self.UPPER, 0, 0)  # nie biorą tu udziału
                bound_set = True

        # szukanie ye
        bound_set = False
        while not bound_set:
            ye += i
            # wczytuje co 256tą linijke
            v = np.reshape(self.cut_image(normal_img, level=level, x_begin=0, y_begin=ye, width=width, height=1),
                           [width, 3])
            if self.contains_only_white(v):
                ye -= i  # cofam sie do ostatnio napotkanego koloru
                ye = self.log2n_search_bound(normal_img, i, level, ye, self.LOWER, 0, 0)
                bound_set = True

        #szukanie xs
        bound_set = False
        while not bound_set:
            xs -= i
            # wczytuje co 256tą linijke
            v = np.reshape(self.cut_image(normal_img, level=level, x_begin=xs, y_begin=ys, width=1, height=ye-ys),
                           [ye-ys, 3])
            if self.contains_only_white(v):
                xs += i  # cofam sie do ostatnio napotkanego koloru
                xs = self.log2n_search_bound(normal_img, i, level, xs, self.LEFT, ys, ye)
                bound_set = True

        # szukanie xe
        bound_set = False
        while not bound_set:
            xe += i
            # wczytuje co 256tą linijke
            v = np.reshape(self.cut_image(normal_img, level=level, x_begin=xe, y_begin=ys, width=1, height=ye - ys),
                           [ye-ys, 3])
            if self.contains_only_white(v):
                xe -= i  # cofam sie do ostatnio napotkanego koloru
                xe = self.log2n_search_bound(normal_img, i, level, xe, self.LEFT, ys, ye)
                bound_set = True

        return xs, xe, ys, ye  # TODO popraw potem by zwóciło 4

    def contains_only_white(self, v):
        all_white = True
        for x in range(1, len(v)):
            if np.average(v[x]) != 255: #0xff biały
                all_white = False
        return all_white

    def log2n_search_bound(self, img, base, level, var, location, upper_bound, lower_bound):
        # lower_bound i upper_bound niezbędne są dla poszerzania

        # dla szukaniaw góre i w lewo odejmuje (wartość zmniejszam)
        if location == self.UPPER or location == self.LEFT:
            should_add = False
        else:  # dla pozostałych dodaje
            should_add = True
        base /= 2.0
        while base >= 1:  # wyszukanie binarne(?)
            print(f"i: {base}, ys: {var}")
            if should_add:  # dodaje dla prawo i dół (gdy wynik pozytywny, chce ZWIĘKSZYĆ te zmienne)
                var += base
            else:  # odejmuje dla góry i lewo (gdy wynik pozytywny, chce ZMNIEJSZYĆ te zmienne)
                var -= base

            if location == self.UPPER:  # wczytuje poziomy pasek
                v = np.reshape(self.cut_image(img, level=level, x_begin=0, y_begin=var, width=width, height=1),
                               [width, 3])
                should_add = ~(self.contains_only_white(v))
            elif location == self.LOWER:  # wczytuje poziomy pasek
                v = np.reshape(self.cut_image(img, level=level, x_begin=0, y_begin=var, width=width, height=1),
                               [width, 3])
                should_add = self.contains_only_white(v)
            elif location == self.LEFT:  # wczytuje pionowy pasek
                v = np.reshape(self.cut_image(img, level=level, x_begin=var, y_begin=upper_bound, width=1, height=lower_bound - upper_bound),
                               [lower_bound - upper_bound, 3])
                should_add = self.contains_only_white(v)
            elif location == self.RIGHT:  # wczytuje pionowy pasek
                v = np.reshape(self.cut_image(img, level=level, x_begin=var, y_begin=upper_bound, width=1, height=lower_bound - upper_bound),
                               [lower_bound - upper_bound, 3])
                should_add = self.contains_only_white(v)

            base /= 2.0
        return int(var)


if __name__ == "__main__":
    main = Main()
    print(main.output_path_prefix)
    files = main.get_itemlist()

    level = 2


    # TEST
    #for f in files:
     #   print(f)
    #print(len(files) % 2 == 0)
    #print(len(files))
    #for x in range(1, 4):  # range(in, ex)
     #   print(x)

    mr_image = main.read_image()

    width = main.get_width(mr_image, level)

    n_img = main.read_image(3, True)

    #TO SĄ WSP DLA ADNOTACJI
    xs, xe, ys, ye = main.get_annotation_bounds(n_img, width, level, 100)

    # wyszukiwanie po 256

    xs, xe, ys, ye = main.get_patch_location(3, level, [xs, xe, ys, ye])

    print([xs, xe, ys, ye])

    whole_image_annotated = main.read_and_cut_image(idx=3, x_begin=xs, y_begin=ys, width=xe - xs, height=ye - ys, read_annotated=True, level=level)

    whole_image = main.read_and_cut_image(idx=3, x_begin=xs, y_begin=ys, width=xe - xs, height=ye - ys, read_annotated=False, level=level)
    plt.subplot(121)
    plt.imshow(whole_image)
    plt.subplot(122)
    plt.imshow(np.reshape(whole_image_annotated, [ye-ys, xe-xs]))
    plt.show()



#reader = mir.MultiResolutionImageReader()
#mr_image = reader.open('ACDC_challenge/Images/13111-1.tif')
#level = 2
#ds = mr_image.getLevelDownsample(level)
#image_patch = mr_image.getUCharPatch(int(568 * ds), int(732 * ds), 300, 200, level)


