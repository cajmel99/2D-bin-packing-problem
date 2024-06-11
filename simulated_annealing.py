from loader import load_data, convert_data_to_matrices
from evaluate_fitness import evaluate, evaluate_bins
import random
import math
import numpy as np
import copy
import time
import matplotlib.pyplot as plt

#################################
DATA_PATH = 'data/'

BIN_SIZE_PATH = 'size50/'
BINS_PATH = BIN_SIZE_PATH + 'size50bins.csv'

FLOWERS_COUNT_PATH = BIN_SIZE_PATH + 'count150flowers/'
FLOWERS_PATH = FLOWERS_COUNT_PATH + 'count150flowers1.csv'

IMAGE_PATH = DATA_PATH + FLOWERS_COUNT_PATH + 'count150flowers1.pdf'

COOLING_RATE = 0.999
STARTING_TEMPERATURE = 1000000
PRINTING_STEP = 100
MAX_ITERATIONS = 50000
#################################


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


loaded_bins, loaded_flowers = load_data(data_folder='data', bins_filename=BINS_PATH, flowers_filename=FLOWERS_PATH)
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
    source_bin, target_bin = random.sample(bins, 2)
    no_progress = 0

    while True:
        while not source_bin.flowers:
            source_bin = random.choice(bins)
        flower = random.choice(source_bin.flowers)

        target_rows, target_cols = target_bin.representation.shape
        flower_rows, flower_cols = flower.representation.shape

        found = False

        if no_progress > 10:
            target_bin = random.choice(bins)
            no_progress = 0

        for i in range(target_rows - flower_rows + 1):
            for j in range(target_cols - flower_cols + 1):
                if can_fit(target_bin.representation, i, j, flower_rows, flower_cols):
                    source_bin.representation[flower.y_coord:flower.y_coord+flower_rows, flower.x_coord:flower.x_coord+flower_cols] = 0
                    source_bin.flowers.remove(flower)

                    target_bin.representation[i:i+flower_rows, j:j+flower_cols] = 1
                    flower.x_coord = j
                    flower.y_coord = i

                    target_bin.flowers.append(flower)
                    found = True
                    break
            if found:
                return bins

        no_progress += 1


temperatures = []
fitnesses = []

def simulated_annealing(max_iterations=MAX_ITERATIONS, starting_temperature=STARTING_TEMPERATURE, cooling_rate=COOLING_RATE):
    current_temperature = starting_temperature
    best_solution = random_search(packed_bins, flowers_pools)
    best_fitness, _ = evaluate_bins(best_solution)

    start = time.time()
    print("\n\n================= SIMULATED ANNEALING =================")
    print(f"\n|----> Starting fitness: {best_fitness}\n")

    for i in range(max_iterations):
        neighbour_solution = move_flower(copy.deepcopy(best_solution))
        neighbour_fitness, _ = evaluate_bins(neighbour_solution)

        fitness_diff = neighbour_fitness - best_fitness
        probability = random.uniform(0, 1)
        exponent = np.exp((fitness_diff) / current_temperature)

        if i % PRINTING_STEP == 0:
            print(f"|----> Iteration {i}, best fitness: {best_fitness}, neighbour fitness: {neighbour_fitness},", end=' ')
            print(f"diff: {fitness_diff}, probability: {probability}, exponent: {exponent}, temperature: {current_temperature}")

        if fitness_diff > 0:
            best_solution = neighbour_solution
            best_fitness = neighbour_fitness
        else:
            if probability < exponent:
                best_solution = neighbour_solution
                best_fitness = neighbour_fitness

        temperatures.append(current_temperature)
        fitnesses.append(best_fitness)

        current_temperature *= cooling_rate

    final_result, non_empty_bins = evaluate_bins(best_solution, True)
    end = time.time()
    elapsed = end - start

    print(f"|----> Final bins: {final_result}")
    for bin in best_solution:
        if np.count_nonzero(bin.representation == 1) != 0:
            print(f"BIN[{bin.index}]:\n{bin.representation}")
    print(f"\n|----> Final result: {final_result}, used bins: {non_empty_bins}, found in: {elapsed}s")
    print("\n\n=======================================================")

    return final_result


simulated_annealing()

plt.plot(temperatures, fitnesses)
plt.xscale('log')
plt.grid(True)
plt.gca().invert_xaxis()
plt.savefig(IMAGE_PATH)
