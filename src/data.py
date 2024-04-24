import h5py
import numpy as np
import csv

# <KeysViewHDF5 ['K', 'K0']>
# <HDF5 dataset "K": shape (64, 64), type "<f4">
# <HDF5 dataset "K0": shape (64, 64), type "<f4">


''' 
inspect jld2 file 
'''

with h5py.File("../data/K.jld2", "r") as f:
    # List all groups and datasets in the file
    def print_hierarchy(obj, name):
        print(name)
    f.visititems(print_hierarchy)

    for dataset_name in f.keys():
        print(dataset_name)
        # Load the dataset as a NumPy array
        dataset = np.array(f[dataset_name][()])
        print(dataset)
        csv_filename = f"np_{dataset_name}.csv"
        np.savetxt(csv_filename, dataset, delimiter=',', fmt='%s')





