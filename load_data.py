from loader import load_data, print_data, convert_data_to_matrices
import numpy as np

loaded_bins, loaded_flowers = load_data()
bins_matrices, flowers_pools = convert_data_to_matrices(loaded_bins, loaded_flowers)

print_data(loaded_bins, loaded_flowers)
