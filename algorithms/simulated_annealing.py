from .loader import load_data, convert_data_to_matrices
from .evaluate_fitness import evaluate, evaluate_bins
import random
import math
import numpy as np
import copy
import time
import matplotlib.pyplot as plt

#################################

BIN_SIZE = '10'
FLOWERS_COUNT = '50'

DATA_PATH = 'data/'

BIN_SIZE_DIR = 'size' + BIN_SIZE + '/'
DEFAULT_BINS_PATH = BIN_SIZE_DIR + 'size' + BIN_SIZE + 'bins.csv'

FLOWERS_COUNT_DIR = BIN_SIZE_DIR + 'count' + FLOWERS_COUNT + 'flowers/'
DEFAULT_FLOWERS_PATH = FLOWERS_COUNT_DIR + 'count' + FLOWERS_COUNT + 'flowers2.csv'

RESULT_FILE_PATH = 'sa_results/last_run.txt'
PLOT_PATH = 'sa_results/last_run_plot.pdf'

COOLING_RATE = 0.9995
STARTING_TEMPERATURE = 500000
MAX_ITERATIONS = 50000
PRINTING_STEP = 100

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

class PlotData:
    def __init__(self):
        self.temperatures = []
        self.fitnesses = []

#################################

def can_fit(large_matrix, start_row, start_col, block_rows, block_cols):
    return np.all(large_matrix[start_row:start_row+block_rows, start_col:start_col+block_cols] == 0)


def put_flower(bin, small_matrix):
    large_rows, large_cols = bin.representation.shape
    small_rows, small_cols = small_matrix.shape
    placed = False

    for i in range(large_rows - small_rows + 1):
        for j in range(large_cols - small_cols + 1):
            if can_fit(bin.representation, i, j, small_rows, small_cols):
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
        flower = random.choice(remaining_flowers)
        placed = False
        bin = random.choice(packed_bins)

        while not placed:
            placed, updated_bin = put_flower(bin, flower)

            if placed:
                for index, flower2 in enumerate(remaining_flowers):
                    if np.array_equal(flower, flower2):
                        remaining_flowers.pop(index)
                for index, bin2 in enumerate(packed_bins):
                    if np.array_equal(bin.representation, bin2.representation):
                        packed_bins[index].representation = updated_bin.representation
            else:
                bin = random.choice(packed_bins)

    return packed_bins

#################################

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

def simulated_annealing(
    plot_data, bins_path, flowers_path, max_iterations, starting_temperature, cooling_rate):

    loaded_bins, loaded_flowers = load_data(
        data_folder=DATA_PATH, bins_filename=bins_path, flowers_filename=flowers_path)
    bins_matrices, flowers_pools = convert_data_to_matrices(loaded_bins, loaded_flowers)

    packed_bins = []

    for index, matrix in enumerate(bins_matrices):
        packed_bins.append(PackedBin(index, matrix))

    current_temperature = starting_temperature
    best_solution = random_search(packed_bins, flowers_pools)
    best_fitness, _ = evaluate_bins(best_solution)

    start = time.time()
    print("\n\n================= SIMULATED ANNEALING =================")
    print(f"\n|-> Starting fitness: {best_fitness}\n")

    for i in range(max_iterations):
        neighbour_solution = move_flower(copy.deepcopy(best_solution))
        neighbour_fitness, _ = evaluate_bins(neighbour_solution)

        fitness_diff = neighbour_fitness - best_fitness
        probability = random.uniform(0, 1)
        exponent = np.exp((fitness_diff) / current_temperature)

        if i % PRINTING_STEP == 0:
            print(f"|-> Iteration {i}, best fitness: {best_fitness}, neighbour fitness: {neighbour_fitness},", end=' ')
            print(f"diff: {fitness_diff}, probability: {probability}, exponent: {exponent}, temperature: {current_temperature}")

        if fitness_diff > 0:
            best_solution = neighbour_solution
            best_fitness = neighbour_fitness
        else:
            if probability < exponent:
                best_solution = neighbour_solution
                best_fitness = neighbour_fitness

        plot_data.temperatures.append(current_temperature)
        plot_data.fitnesses.append(best_fitness)

        current_temperature *= cooling_rate

    final_result, non_empty_bins = evaluate_bins(best_solution, True)
    end = time.time()
    elapsed = end - start

    print(f"\n|--> Final result: {final_result}, used bins: {non_empty_bins}, found in: {elapsed}s")
    print("\n\n=======================================================")

    return final_result, non_empty_bins, elapsed

#################################

def solve_sa(
    bins_path=DEFAULT_BINS_PATH,
    flowers_path=DEFAULT_FLOWERS_PATH,
    max_iterations=MAX_ITERATIONS,
    starting_temperature=STARTING_TEMPERATURE,
    cooling_rate=COOLING_RATE):

    plot_data = PlotData()

    sim_result = simulated_annealing(
        plot_data, bins_path, flowers_path, max_iterations, starting_temperature, cooling_rate)
    file = open(RESULT_FILE_PATH, "a")
    file.write(f'{sim_result[0]} {sim_result[1]} {sim_result[2]}\n')
    file.close()

    plt.xlabel('Temperature')
    plt.ylabel('Fitness evaluation')
    plt.title('Last simulated annealing run')
    plt.plot(plot_data.temperatures, plot_data.fitnesses)
    plt.xscale('log')
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.savefig(PLOT_PATH)

    print(f"\nLast run results saved at sa_results/last_run.txt(last_run_plot.pdf)")

    return sim_result


if __name__ == "__main__":
    solve_sa()
