import csv
import os
import numpy as np

class Bin:
    def __init__(self, width, height):
        self.width = width
        self.height = height

class Flower:
    def __init__(self, width, height):
        self.width = width
        self.height = height

def load_data(data_folder='data', bins_filename='bins.csv', flowers_filename='flowers.csv'):
    bins = []
    flowers = []
    
    # Load bins from CSV
    bins_path = os.path.join(data_folder, bins_filename)
    with open(bins_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            width, height = int(row[0]), int(row[1])
            bins.append(Bin(width, height))
    
    # Load flowers from CSV
    flowers_path = os.path.join(data_folder, flowers_filename)
    with open(flowers_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            width, height = int(row[0]), int(row[1])
            flowers.append(Flower(width, height))
    
    return bins, flowers

def convert_data_to_matrices(loaded_bins, loaded_flowers):
    # Create the lists with height and width of flowers and bins
    bins_width = []
    bins_height = []
    for bin in loaded_bins:
        bins_width.append(bin.width)
        bins_height.append(bin.height)
    bins = [bins_width, bins_height]

    flowers_width = []
    flowers_height = []
    for flower in loaded_flowers:
        flowers_width.append(flower.width)
        flowers_height.append(flower.height)
    flowers = [flowers_width, flowers_height]

    # Create the pool
    bins_matrices = []
    flowers_pools = []

    for width, height in zip(bins_width, bins_height):
        bin = np.zeros((width, height))
        bins_matrices.append(bin)

    for width, height in zip(flowers_width, flowers_height):
        flower = np.ones((width, height))
        flowers_pools.append(flower)
    
    return bins_matrices, flowers_pools

def print_data(bins, flowers):
    print("\nLoaded Bins:")
    for i, bin in enumerate(bins):
        print(f"Bin {i+1}: Width = {bin.width}, Height = {bin.height}")

    print("\nLoaded Flowers:")
    for i, flower in enumerate(flowers):
        print(f"Flower {i+1}: Width = {flower.width}, Height = {flower.height}")