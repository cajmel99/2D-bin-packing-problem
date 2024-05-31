import numpy as np

FILLED_FIELD_BASE_EVAL = 1000
EMPTY_FIELD_BASE_EVAL = 100

def evaluate(result_bins, debug = False):
    final_fitness = 0
    for index, result_bin in enumerate(result_bins):
        bins_count = len(result_bins)

        filled_field_multiplier = FILLED_FIELD_BASE_EVAL * ((bins_count - index) / bins_count)
        empty_field_multiplier = EMPTY_FIELD_BASE_EVAL * ((index + 1) / bins_count)

        filled_field_count = np.count_nonzero(result_bin == 1)
        empty_field_count = result_bin.size - filled_field_count

        fitness = \
            (filled_field_multiplier * filled_field_count) - \
            (empty_field_multiplier * empty_field_count)

        final_fitness += fitness

        if debug:
            print(f"|-> Bin[{index}] fitness: {fitness} | filled: {filled_field_count} | empty: {empty_field_count}")

    return final_fitness

def evaluate_bins(result_bins, debug = False):
    final_fitness = 0
    for result_bin in result_bins:
        bins_count = len(result_bins)

        filled_field_multiplier = FILLED_FIELD_BASE_EVAL * ((bins_count - result_bin.index) / bins_count)
        empty_field_multiplier = EMPTY_FIELD_BASE_EVAL * ((result_bin.index + 1) / bins_count)

        filled_field_count = np.count_nonzero(result_bin.representation == 1)
        empty_field_count = result_bin.representation.size - filled_field_count

        fitness = \
            (filled_field_multiplier * filled_field_count) - \
            (empty_field_multiplier * empty_field_count)

        final_fitness += fitness

        if debug:
            print(f"|-> Bin[{result_bin.index}] fitness: {fitness} | filled: {filled_field_count} | empty: {empty_field_count}")

    return final_fitness
