import random
import csv
import os

class Bin:
    def __init__(self, width, height):
        self.width = width
        self.height = height

class Flower:
    def __init__(self, width, height):
        self.width = width
        self.height = height

def generate_random_bins(num_bins, max_width, max_height):
    bins = []
    for _ in range(num_bins):
        width = max_width #random.randint(min_width, max_width)
        height = max_height #random.randint(min_height, max_height)
        bins.append(Bin(width, height))
    return bins

def generate_random_flowers(num_flowers, min_width, max_width, min_height, max_height):
    flowers = []
    for _ in range(num_flowers):
        width = random.randint(min_width, max_width)
        height = random.randint(min_height, max_height)
        flowers.append(Flower(width, height))
    return flowers

def generate_dataset(num_bins, bin_max_width, bin_max_height,
                     num_flowers, flower_min_width, flower_max_width, flower_min_height, flower_max_height):
    bins = generate_random_bins(num_bins, bin_max_width, bin_max_height)
    flowers = generate_random_flowers(num_flowers, flower_min_width, flower_max_width, flower_min_height, flower_max_height)
    return bins, flowers

def save_to_csv(bins, flowers, data_folder, bins_filename, flowers_filename):
    os.makedirs(data_folder, exist_ok=True)

    bins_path = os.path.join(data_folder, bins_filename)
    with open(bins_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Bin width', 'Bin height'])
        for bin in bins:
            writer.writerow([bin.width, bin.height])

    flowers_path = os.path.join(data_folder, flowers_filename)
    with open(flowers_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Flower width', 'Flower height'])
        for flower in flowers:
            writer.writerow([flower.width, flower.height])
