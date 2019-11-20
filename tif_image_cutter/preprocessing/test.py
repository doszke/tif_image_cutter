from matplotlib import pyplot as plt
from tif_image_cutter.preprocessing import preprocessing as p

if __name__ == '__main__':
    f = open("C:\\Users\\Jakub Siembida\\PycharmProjects\\inz\\tif_image_cutter\\main\\_dataset_256_names.txt")
    x = f.readline().split(",")
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    img = plt.imread(f"G:/_dataset_256_sent/{x[0].replace('_annotated', '')}")
    plt.subplot(121)
    plt.imshow(img)

    plt.subplot(122)
    plt.imshow(p.Preprocessing.reinhard_transform(x))
    plt.show()