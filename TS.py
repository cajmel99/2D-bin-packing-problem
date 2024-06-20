import numpy as np
import random
from algorithms.loader import load_data, convert_data_to_matrices
from RA import random_algorith2, bins_matrices, flowers_pools, can_fit, put_flower

loaded_bins, loaded_flowers = load_data()
bins_matrices, flowers_pools = convert_data_to_matrices(loaded_bins, loaded_flowers)

def evaluate_solution(bins_matrices):
    # Primary criterion: number of bins used
    num_bins_used = sum(1 for bin_matrix in bins_matrices if np.any(bin_matrix != 0))
    # Secondary criterion: free space in used bins (to prioritize efficient packing)
    total_free_space = sum(np.sum(bin_matrix == 0) for bin_matrix in bins_matrices if np.any(bin_matrix != 0))
    return num_bins_used, total_free_space

def generate_neighbors(current_bins, flowers):
    neighbors = []
    for i in range(len(current_bins)):
        for flower in flowers:
            new_bin = current_bins[i].copy()
            placed, new_bin = put_flower(new_bin, flower)
            if placed:
                new_bins = current_bins[:i] + [new_bin] + current_bins[i+1:]
                neighbors.append(new_bins)
    return neighbors

def is_in_tabu_list(neighbor, tabu_list):
    for tabu_solution in tabu_list:
        if all(np.array_equal(neighbor_bin, tabu_bin) for neighbor_bin, tabu_bin in zip(neighbor, tabu_solution)):
            return True
    return False

def tabu_search(bins_matrices, flowers_matrices, max_iter=10, tabu_tenure=10):
    current_solution = bins_matrices.copy()
    best_solution = current_solution.copy()
    best_score = evaluate_solution(current_solution)

    tabu_list = []

    for iteration in range(max_iter):
        neighbors = generate_neighbors(current_solution, flowers_matrices)
        neighbors = [neighbor for neighbor in neighbors if not is_in_tabu_list(neighbor, tabu_list)]

        if not neighbors:
            break

        best_neighbor = min(neighbors, key=evaluate_solution)
        best_neighbor_score = evaluate_solution(best_neighbor)

        if best_neighbor_score < best_score:
            best_solution = best_neighbor.copy()
            best_score = best_neighbor_score

        current_solution = best_neighbor
        tabu_list.append(current_solution)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        print(f"Iteration {iteration}, Best score: {best_score}")

    return best_solution


best_solution = tabu_search(bins_matrices, flowers_pools)

print("Best solution found:\n", best_solution)
