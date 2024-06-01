from dataset_generator import *

if __name__ == "__main__":
    num_bins = 15
    bin_min_width = 108
    bin_max_width = 108
    bin_min_height = 118
    bin_max_height = 118

    num_flowers = 80
    flower_min_width = 10
    flower_max_width = 50
    flower_min_height = 10
    flower_max_height = 50

    bins, flowers = generate_dataset(num_bins, bin_min_width, bin_max_width, bin_min_height, bin_max_height,
                                    num_flowers, flower_min_width, flower_max_width, flower_min_height, flower_max_height)

    # Save data to csv folder
    save_to_csv(bins, flowers)
