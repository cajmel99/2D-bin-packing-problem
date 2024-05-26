from loader import load_data, print_data
import numpy as np

loaded_bins, loaded_flowers = load_data()

print_data(loaded_bins, loaded_flowers)
#print(loaded_flowers.height)
print(loaded_bins)

# Create the lists with height and width of flowers and bins
bins_width = []
bins_height = []
for bin in loaded_bins:
    bins_width.append(bin.width)
    bins_height.append(bin.height)
bins = [bins_width, bins_height]
print(bins)

flowers_width = []
flowers_height = []
for flower in loaded_flowers:
    flowers_width.append(flower.width)
    flowers_height.append(flower.height)
flowers = [flowers_width, flowers_height]
print(flowers)

# Create the pool
bins_matrices = []
flowers_pools = []

for width, height in zip(bins_width, bins_height):
    bin = np.zeros((width, height))
    bins_matrices.append(bin)

for width, height in zip(flowers_width, flowers_height):
    flower = np.ones((width, height))
    flowers_pools.append(flower)

print(len(bins_matrices))
print("ppp",type(flowers_pools))