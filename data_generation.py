from dataset_generator import *

if __name__ == "__main__":
    num_bins = 100
    bin_max_width = 115
    bin_max_height = 118

    num_flowers = 80
    flower_min_width = 5
    flower_max_width = 60
    flower_min_height = 7
    flower_max_height = 50

    bins, flowers = generate_dataset(num_bins, bin_max_width, bin_max_height,
                                    num_flowers, flower_min_width, flower_max_width, flower_min_height, flower_max_height)

    # Save data to csv folder
    save_to_csv(bins, flowers, data_folder='data', bins_filename='bins_3_100.csv', flowers_filename='flowers_3_80.csv')