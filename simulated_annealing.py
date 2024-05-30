from loader import load_data, convert_data_to_matrices
from load_data import bins_matrices, flowers_pools
from evaluate_fitness import evaluate
import random
import numpy as np


class PutFlower:
    def __init__(self, representation, x_coord, y_coord):
        self.representation = representation
        self.x_coord = x_coord
        self.y_coord = y_coord

class PackedBin:
    def __init__(self, index, representation):
        self.index = index
        self.representation = representation
        self.flowers = []


loaded_bins, loaded_flowers = load_data()
bins_matrices, flowers_pools = convert_data_to_matrices(loaded_bins, loaded_flowers)

packed_bins = []

for index, matrix in enumerate(bins_matrices):
    packed_bins.append(PackedBin(index, matrix))


def can_fit(large_matrix, start_row, start_col, block_rows, block_cols):
    """
    Check if the bin have enough free space to put flower
    """
    return np.all(large_matrix[start_row:start_row+block_rows, start_col:start_col+block_cols] == 0)

def put_flower(bin, small_matrix):
    """
    Put flower into bin
    """
    large_rows, large_cols = bin.representation.shape
    small_rows, small_cols = small_matrix.shape
    placed = False
    for i in range(large_rows - small_rows + 1):
        for j in range(large_cols - small_cols + 1):
            if can_fit(bin.representation, i, j, small_rows, small_cols):
                # Place the small matrix into the large matrix
                bin.representation[i:i+small_rows, j:j+small_cols] = small_matrix
                bin.flowers.append(PutFlower(small_matrix, j, i))
                placed = True
                break
        if placed:
            break
    return placed, bin

def random_search(packed_bins, flowers_matrices):
    remaining_flowers = flowers_matrices[:]

    while len(remaining_flowers) > 0:
        print("===================================")
        print("|-> Remaining flowers count: ", len(remaining_flowers))

        flower = random.choice(remaining_flowers)
        placed = False
        bin = random.choice(packed_bins)

        while not placed:
            placed, updated_bin = put_flower(bin, flower)

            print(f"|--> Bin matrix shape: {updated_bin.representation.shape}")
            print(f"|--> Trying to place flower of shape: {flower.shape}")

            if placed:
                for index, flower2 in enumerate(remaining_flowers):
                    if np.array_equal(flower, flower2):
                        print(f"|---> Used flower index: ", index)
                        remaining_flowers.pop(index)
                # Update bin
                for index, bin2 in enumerate(packed_bins):
                    if np.array_equal(bin.representation, bin2.representation):
                        print(f"|---> Used bin index: ", index)
                        packed_bins[index].representation = updated_bin.representation

                print("|--> Updated bin:\n", updated_bin.representation)
            else:
                bin = random.choice(packed_bins)

    return packed_bins


def move_flower(bins):
    for index, bin in enumerate(bins):
        # print(f"BIN[{index}] rep:\n", bin.representation.astype(int))
        for flower in bin.flowers:
            # print(f"\n\n> BIN[{index}] - Flower rep: {flower.representation.shape}, x: {flower.x_coord}, y: {flower.y_coord}")

            bin_xd = random.choice(bins)

            large_rows, large_cols = bin_xd.representation.shape
            small_rows, small_cols = flower.representation.shape

            found = False

            for i in range(large_rows - small_rows + 1):
                for j in range(large_cols - small_cols + 1):
                    if can_fit(bin_xd.representation, i, j, small_rows, small_cols):
                        # Place the small matrix into the large matrix
                        # print(f">> CAN_FIT! - BIN[{bin_xd.index}], x: {j}, y: {i}")
                        # print(f">>>> BEFORE:\n")
                        # print(f">>>>>>> SOURCE:\n", bin.representation.astype(int))
                        # print(f">>>>>>> DESTINATION:\n", bin_xd.representation.astype(int))

                        bin.representation[flower.y_coord:flower.y_coord+small_rows, flower.x_coord:flower.x_coord+small_cols] = 0
                        bin.flowers.remove(flower)
                        bin_xd.representation[i:i+small_rows, j:j+small_cols] = 1
                        flower.x_coord = j
                        flower.y_coord = i
                        bin_xd.flowers.append(flower)

                        # print(f">>>> AFTER:\n")
                        # print(f">>>>>>> SOURCE:\n", bin.representation.astype(int))
                        # print(f">>>>>>> DESTINATION:\n", bin_xd.representation.astype(int))

                        found = True
                        break

                if found:
                    break



##########################################################

results = random_search(packed_bins, flowers_pools)
print(f"Number of used bins: ", len(results))

fields = []
for bin in results:
    fields.append(bin.representation)


print(f">>>>>>>>>>>>> BEFORE MOVE <<<<<<<<<<<<<<")

for index, bin in enumerate(results):
    print(f"BIN[{index}] rep:\n", bin.representation.astype(int))
    for flower in bin.flowers:
        print(f"BIN[{index}] - Flower rep: {flower.representation.shape}, x: {flower.x_coord}, y: {flower.y_coord}")

final_fitness = evaluate(fields, True)
print("Final fitness: ", final_fitness)


move_flower(results)


print(f">>>>>>>>>>>>> AFTER MOVE <<<<<<<<<<<<<<")

for index, bin in enumerate(results):
    print(f"BIN[{index}] rep:\n", bin.representation.astype(int))
    for flower in bin.flowers:
        print(f"BIN[{index}] - Flower rep: {flower.representation.shape}, x: {flower.x_coord}, y: {flower.y_coord}")

final_fitness = evaluate(fields, True)
print("Final fitness: ", final_fitness)
