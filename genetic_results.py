import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


filename = "count50bins_count150flowers_size50_size50bins.csv"


tournaments = [4, 6, 10]
tournaments_dict = {
    "2": [],
    "3": [],
    "4": [],
    "6": [],
    "10": [],
}
best_fitness_values = []
best_fitness_values_list = []
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

def create_heatmap(data, ax, title):
    df = pd.DataFrame(data, columns=['Mutation Rate', 'Crossover Rate', 'Fitness Z', 'Fitness Q'])

    x_values = [0.005, 0.01, 0.015, 0.02, 0.1, 0.15]
    y_values = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]

    heatmap_data = pd.DataFrame(0, index=y_values, columns=x_values, dtype=float)

    for _, row in df.iterrows():
        heatmap_data.at[row['Crossover Rate'], row['Mutation Rate']] = row['Fitness Z']

    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Fitness'}, ax=ax)
    ax.set_title(title)

with open(filename, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=";")
    for i, line in enumerate(reader):
        if i == 0:
            continue
        mutatnion = float(line[0])
        crossover = float(line[1])
        tournament = int(line[2])
        clean_fitness = line[-1].strip('()')
        clean_fitness = clean_fitness.split(', ')
        fitness = float(clean_fitness[0])
        bins = int(clean_fitness[-1])
        # print(f"{mutatnion}, {crossover}, {tournament}, {fitness}, {bins}")
        tournaments_dict["{}".format(str(tournament))].append((mutatnion, crossover, fitness, bins))
        best_fitness_values.append([fitness, mutatnion, crossover, tournament, bins])
# for tt, x in enumerate(['2','3','4','6','10']):
#     df = pd.DataFrame(tournaments_dict[x], columns=['Mutation Rate', 'Crossover Rate', 'Fitness Z', 'Fitness Q'])

#     # Unikalne wartości x i y
#     x_values = [0.005, 0.01, 0.015, 0.02, 0.1, 0.15]
#     y_values = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]

#     # Tworzenie pustej tabeli z zerami
#     heatmap_data = pd.DataFrame(0, index=y_values, columns=x_values, dtype=float)

#     # Wypełnianie tabeli wartościami z
#     for _, row in df.iterrows():
#         heatmap_data.at[row['Crossover Rate'], row['Mutation Rate']] = row['Fitness Z']

#     # Wyświetlanie tabeli
#     print(heatmap_data)

#     # Tworzenie heatmapy
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Fitness Z'})
#     plt.title('Heatmap of Fitness Z based on Mutation and Crossover Rates')
#     plt.xlabel('Mutation Rate')
#     plt.ylabel('Crossover Rate')
#     plt.show()
sorted_data = sorted(best_fitness_values, key=lambda x: x[0], reverse=True)
for i, element in enumerate(sorted_data):
    print(element)
    if i == 10:
        break
fig.delaxes(axes[2][1])
# Tworzenie heatmap dla poszczególnych zestawów danych
create_heatmap(tournaments_dict['2'], axes[0, 0], 'Tournament 2')
create_heatmap(tournaments_dict['3'], axes[0, 1], 'Tournament 3')
create_heatmap(tournaments_dict['4'], axes[1, 0], 'Tournament 4')
create_heatmap(tournaments_dict['6'], axes[1, 1], 'Tournament 6')
create_heatmap(tournaments_dict['10'], axes[2, 0], 'Tournament 10')
plt.tight_layout()
plt.show()
