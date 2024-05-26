from loader import load_data, print_data
import random
from load_data import bins_matrices, flowers_pools
import numpy as np

loaded_bins, loaded_flowers = load_data()



# Function to check if the bin have enoungh free space to put flower
def can_fit(large_matrix, start_row, start_col, block_rows, block_cols):
    return np.all(large_matrix[start_row:start_row+block_rows, start_col:start_col+block_cols] == 0)

def random_algorith2(bins_matrices, flowers_matrices):
    remaining_flowers = flowers_matrices[:]
    print(len(remaining_flowers))

    while len(remaining_flowers) > 0:
        flower = random.choice(remaining_flowers)
        placed = False 
        bin = random.choice(bins_matrices)

        while not placed:
            placed = put_flower(bin, flower)
            print(f"Bin matrix shape: {bin.shape}")
            print(f"Trying to place flower of shape: {flower.shape}")

            if placed:
                for i, remaining_flowers in enumerate(remaining_flowers):
                    if np.array_equal(flower, remaining_flowers):
                        print(flower, i)
                        np.delete(remaining_flowers, i)
                        print(remaining_flowers)
                        break
                    print("Small matrix placed successfully:\n", bin)
                    print(len(remaining_flowers))

            else:
                bin = random.choice(bins_matrices)

    return bins_matrices

            


def put_flower(large_matrix, small_matrix):
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
    return placed

#placed = put_flower(bin_matrix, flower_matrix)
# Output the result
placed = random_algorith2(bins_matrices, flowers_pools)