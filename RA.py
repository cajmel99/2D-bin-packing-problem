from loader import load_data, convert_data_to_matrices
import random
from load_data import bins_matrices, flowers_pools
import numpy as np
#from TS import evaluate_solution

oaded_bins, loaded_flowers = load_data()
bins_matrices, flowers_pools = convert_data_to_matrices()

def can_fit(large_matrix, start_row, start_col, block_rows, block_cols):
    """
    Check if the bin have enough free space to put flower
    """
    return np.all(large_matrix[start_row:start_row+block_rows, start_col:start_col+block_cols] == 0)

def put_flower(large_matrix, small_matrix):
    """
    Put flower into bin 
    """
    large_rows, large_cols = large_matrix.shape
    small_rows, small_cols = small_matrix.shape
    placed = False
    for i in range(large_rows - small_rows + 1):
        for j in range(large_cols - small_cols + 1):
            if can_fit(large_matrix, i, j, small_rows, small_cols):
                # Place the small matrix into the large matrix
                large_matrix[i:i+small_rows, j:j+small_cols] = small_matrix
                placed = True
                break
        if placed:
            break
    return placed, large_matrix

def random_algorith2(bins_matrices, flowers_matrices):
    remaining_flowers = flowers_matrices[:]

    while len(remaining_flowers) > 0:
        print("Number of remaining flowers", len(remaining_flowers))
        flower = random.choice(remaining_flowers)
        placed = False 
        bin = random.choice(bins_matrices)
        while not placed:
            placed, updated_bin = put_flower(bin, flower)
            print(f"Bin matrix shape: {updated_bin.shape}")
            print(f"Trying to place flower of shape: {flower.shape}")
            if placed:
                for index, flower2 in enumerate(remaining_flowers):
                    if np.array_equal(flower, flower2):
                        print(index)
                        remaining_flowers.pop(index)
                # Update bin
                for index, bin2 in enumerate(remaining_flowers):
                    if np.array_equal(bin, bin2):
                        print(index)
                        bins_matrices[index] = updated_bin
                print("Small matrix placed successfully:\n", updated_bin)
            else:
                bin = random.choice(bins_matrices)

    return bins_matrices

def evaluate_solution(bins_matrices):
    # Primary criterion: number of bins used
    num_bins_used = sum(1 for bin_matrix in bins_matrices if np.any(bin_matrix != 0))
    # Secondary criterion: free space in used bins (to prioritize efficient packing)
    total_free_space = sum(np.sum(bin_matrix == 0) for bin_matrix in bins_matrices if np.any(bin_matrix != 0))
    return num_bins_used, total_free_space

placed = random_algorith2(bins_matrices, flowers_pools)
print(placed)

results = evaluate_solution(bins_matrices)
print(results)