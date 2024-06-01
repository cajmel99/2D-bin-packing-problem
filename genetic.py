import random
from loader import Bin, Flower, load_data
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import numpy as np
from evaluate_fitness import evaluate
import csv


class GeneticBin:
    def __init__(self, bin: Bin):
        self.width = bin.width
        self.height = bin.height
        self.matrix = np.zeros((bin.width, bin.height))
        self.debug_elements = []
        self.free_spaces = [(0,0)]

    def can_fit(self, item_width, item_height, x_bin, y_bin):
        return np.all(self.matrix[x_bin:x_bin+item_width, y_bin:y_bin+item_height] == 0) and ((x_bin+item_width) < self.width) and((y_bin+item_height < self.height))
    
    def update_free_spaces(self, flower:Flower, x, y):
        self.free_spaces = [(fx, fy) for fx, fy in self.free_spaces if not (fx >= x and fx < x + flower.width and fy >= y and fy < y + flower.height)]
        
        if x + flower.width < self.width:
            self.free_spaces.append((x + flower.width, y))
        if y + flower.height < self.height:
            self.free_spaces.append((x, y + flower.height))
        # print(self.free_spaces)

    def put_flower(self, flower:Flower, flower_number):
        for x, y in self.free_spaces:
                if self.can_fit(flower.width, flower.height, x, y):
                    # print(f"{x} - {x+flower.width}")
                    self.matrix[x:x+flower.width, y:y+flower.height] = (flower_number%9)+1
                    self.debug_elements.append((flower, flower_number))
                    self.update_free_spaces(flower, x, y)
                    return True
        return False

    def get_packing_density(self):
        used_area = sum(item[0] * item[1] for item in self.items)
        total_area = self.width * self.height
        return used_area / total_area
    
    def save_bin(self, filename):
        np.savetxt(filename, np.rot90(self.matrix), delimiter=",", fmt="%d")
        print("==========================")
        print(self.debug_elements)


# Klasa reprezentująca rozwiązanie (chromosom)
class Chromosome:
    def __init__(self, genes:list[Flower]):
        self.genes = genes
        self.fitness = random.randrange(100)

    def evaluate_fitness(self, bins):
        genetic_bins = [GeneticBin(bin) for bin in bins]
        for i, flower in enumerate(self.genes):
            placed = False
            for bin in genetic_bins:
                if bin.put_flower(flower, i):
                    placed = True
                    break
            if not placed:
                raise Exception("Nie można umieścić kwiatka :(")
        # for i, b in enumerate(genetic_bins):
        #     b.save_bin(f"file{i}.csv")
        # raise Exception("Nie można umieścić kwiatka koniec")

        self.fitness = evaluate([bin.matrix for bin in genetic_bins])
        # print(fitness)
        # raise Exception("Nie można umieścić kwiatka koniec")
        # used_bins = len(bins)
        # avg_packing_density = sum(b.get_packing_density() for b in bins) / used_bins
        # self.fitness = random.randrange(100)#1 / used_bins + avg_packing_density


class AlgorithmConfig:
    def __init__(self, population_size, generations, mutation_rate, crossover_rate, results_file, flowers_file, bins_file, tournament_size):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.results_file = results_file
        self.flowers_file = flowers_file
        self.bins_file = bins_file

class GeneticAlgorithm:
    def __init__(self, config: AlgorithmConfig):
        self.config = config
        self.flowers = []
        self.bins = []
        self.population = []
        self.results = []

    def load_data(self):
        self.bins, self.flowers = load_data(flowers_filename=self.config.flowers_file, bins_filename=self.config.bins_file)
    
    def initialize_population(self):
        for _ in range(self.config.population_size):
            flowers_copy = self.flowers.copy()
            random.shuffle(flowers_copy)
            self.population.append(Chromosome(flowers_copy))
        for chromosome in self.population:
            chromosome.evaluate_fitness(self.bins)


    def select_parents(self):
        best_candidate = self.population[random.randrange(len(self.population))]
        for _ in range(self.config.tournament_size):
            candidate = self.population[random.randrange(len(self.population))]
            if candidate.fitness > best_candidate.fitness:
                best_candidate = candidate
        return best_candidate
    
    def swap_mutate(self, chromosome: Chromosome):
        for i in range(len(chromosome.genes)):
            if random.random() < self.config.mutation_rate:
                j = random.choice([x for x in range(len(chromosome.genes)) if x != i])
                chromosome.genes[i], chromosome.genes[j] = chromosome.genes[j], chromosome.genes[i]

    def crossover(self, parent1:Chromosome, parent2: Chromosome):
        #TODO!!!!!!!!!!!!
        return parent1, parent2

    def evolve(self):
        children = []
        for _ in range(len(self.population) // 2):
            #selekcja
            parent1 = self.select_parents()
            parent2 = self.select_parents()
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)

            child1, child2 = self.crossover(parent1, parent2)
            self.swap_mutate(child1)
            self.swap_mutate(child2)

            #ewalucaja dzieci   
            child1.evaluate_fitness(self.bins)
            child2.evaluate_fitness(self.bins)

            #dodawanie dzieci do zbioru
            children.append(child1)
            children.append(child2)
        #zamiana pokoleń
        self.population = children

    def save_results(self):
        best = None
        worst = None
        mean = 0
        for x in self.population:
            if best:
                if x.fitness > best:
                    best = x.fitness
            else:
                best = x.fitness
            if worst:
                if x.fitness < worst:
                    worst = x.fitness
            else:
                worst = x.fitness
            mean += x.fitness
        mean = mean/len(self.population)
        self.results.append([best, worst, mean])

    def print_results(self):
        x = list(range(len(self.results)))
        y1 = [item[0] for item in self.results]
        y2 = [item[1] for item in self.results]
        y3 = [item[2] for item in self.results]

        plt.plot(x, y1, label='Y1', marker='o')
        plt.plot(x, y2, label='Y2', marker='s')
        plt.plot(x, y3, label='Y3', marker='^')

        plt.legend()

        plt.show()
    
    def run(self):
        self.initialize_population()

        self.save_results()
        for i in tqdm(range(self.config.generations)):
            self.evolve()
            self.save_results()
        
        self.print_results()
        with open('out.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.results)



genetic_config = AlgorithmConfig(
    population_size=50,
    generations=40,
    mutation_rate=0.05,
    crossover_rate=0.3,
    tournament_size = 2,
    flowers_file="flowers2.csv",
    results_file="genetic_results.csv",
    bins_file="bins2.csv",
)

genetic_algorithm = GeneticAlgorithm(genetic_config)
genetic_algorithm.load_data()
genetic_algorithm.run()
