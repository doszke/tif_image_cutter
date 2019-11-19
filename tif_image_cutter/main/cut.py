import net
from matplotlib import pyplot as plt

if __name__ == "__main__":
    n = net.Net()
    side_length = 256
    # 10, 21, 27, 28, 32, 35, 36, 37!, 41, 44, 45!, 47, 49, 61, 62!, 64, 66, 67, 70, 72, 76!, 78!, 80, 81, 82!, 89,
    # 91, 92, 93, 96, 98,  sie wysypuje

    # 9 ma problem z maską przy level=2

    for idx in range(1, 51):  # zacznij od 30
        print(idx)
        output_path = f"G:\\_dataset_256\\{100+idx}\\"
        bounds = n.get_patch_location(idx=idx, level=2)
        n.cut_image_for_dataset(idx, bounds, side_length, output_path, 2)
    # ValueError: cannot reshape array of size 1408 into shape (64,64) 37
    # ValueError: cannot reshape array of size 128 into shape (64,64)  45
    # ValueError: cannot reshape array of size 3712 into shape (64,64) 62
    # ValueError: cannot reshape array of size 3200 into shape (64,64) 76
    # ValueError: cannot reshape array of size 2944 into shape (64,64) 78
    # ValueError: cannot reshape array of size 1664 into shape (64,64) 82
