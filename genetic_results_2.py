import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os



path1 = "genetic_results/count100bins/size100/count100_flowers1"
path3= "genetic_results/count100bins/size100/count100_flowers2"
path2 = "genetic_results/count100bins/size100/count100_flowers3"

def list_all_folders(directory):
    dictionary = {}
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if all(s in root for s in ['count10bins', 'count100']):
                if any(s in root for s in ['flowers1', 'flowers2', 'flowers3']):
                    
                    dir_path = os.path.join(root, dir_name)
                    
                    for file in os.listdir(dir_path):
                        full_path = os.path.join(dir_path, file)
                        full_path_split = full_path.split('\\')
                        extension = full_path_split[-1].split('.')[-1]
                        if extension == 'csv':
                            
                            if full_path_split[-2] not in dictionary:
                                dictionary[full_path_split[-2]] = {}
                            if full_path_split[-3] not in dictionary[full_path_split[-2]]:
                                dictionary[full_path_split[-2]][full_path_split[-3]] = []
                            dictionary[full_path_split[-2]][full_path_split[-3]].append(full_path)
    return dictionary

d = list_all_folders("genetic_results")


for k in d:
    data_bins = []
    data_fitness = []
    for k2 in d[k]:
        for file_path in d[k][k2]:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=",")
                for i, line in enumerate(reader):
                    pass
                l = line[0].strip('()')    
                fitness, bins = l.split(', ')
                fitness = float(fitness)
                bins = int(bins)
                data_bins.append(bins)
                data_fitness.append(fitness)
    mean_bins = np.mean(data_bins)
    std_bins = np.std(data_bins)
    mean_fitness = np.mean(data_fitness)
    std_fitness = np.std(data_fitness)
    print(f"======== {k} ========")
    print(f"średnia: {mean_bins}, odch. stan.: {std_bins}")
    print(f"średnia: {mean_fitness}, odch. stan.: {std_fitness}")
    print(data_bins)
    print()
    

