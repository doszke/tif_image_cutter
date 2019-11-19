import create_dset
import h5py

if __name__ == "__main__":
    dc = create_dset.DsetCreator("G:\\_dataset_256_sent\\", "G:\\_dataset_256_sent")#"/home/doszke/", "/home/doszke/model_256/")
    imgs, masks = dc.to_dataset()
    dataset = h5py.File(f"{dc.output_path}dataset_1000.h5", "w")
    dataset.create_dataset("images", data=imgs[0:1000, :, :, :])
    dataset.create_dataset("masks", data=masks[0:1000, :, :, :])
