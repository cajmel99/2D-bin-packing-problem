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
        return np.all(self.matrix[x_bin:x_bin+item_width, y_bin:y_bin+item_height] == 0) and ((x_bin+item_width) <= self.width) and((y_bin+item_height <= self.height))
    
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
                    self.matrix[x:x+flower.width, y:y+flower.height] = np.ones([flower.width, flower.height])
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
        binid = 0
        for i, flower in enumerate(self.genes):
            placed = False
            if genetic_bins[binid].put_flower(flower, i):
                placed = True
            else:
                binid+=1
                if genetic_bins[binid].put_flower(flower, i):
                    placed = True
                
                
            # for bin in genetic_bins:
            #     if bin.put_flower(flower, i):
            #         placed = True
            #         break
            if not placed:
                raise Exception("Nie można umieścić kwiatka :(")
        # for i, b in enumerate(genetic_bins):
        #     b.save_bin(f"file{i}.csv")
        # raise Exception("Nie można umieścić kwiatka koniec")

        self.fitness = evaluate([bin.matrix for bin in genetic_bins], False)
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
            if candidate.fitness[0] > best_candidate.fitness[0]:
                best_candidate = candidate
        return best_candidate
    
    def swap_mutate(self, chromosome: Chromosome):
        for i in range(len(chromosome.genes)):
            if random.random() < self.config.mutation_rate:
                j = random.choice([x for x in range(len(chromosome.genes)) if x != i])
                chromosome.genes[i], chromosome.genes[j] = chromosome.genes[j], chromosome.genes[i]

    def crossover(self, parent1: Chromosome, parent2: Chromosome, child1: Chromosome, child2: Chromosome):
        if random.random() < self.config.crossover_rate:
            size = len(parent1.genes)
            child1.genes, child2.genes = [None]*size, [None]*size

            start, end = sorted(random.sample(range(size), 2))
            child1.genes[start:end] = parent1.genes[start:end]
            child2.genes[start:end] = parent2.genes[start:end]



            def fill_child(child, parent):
                p = 0
                # print(len(child))
                # print(parent)
                # print(len(parent))
                for i in range(size):
                    # print(p)
                    if not child[i]:
                        
                        while parent[p] in child:
                            p += 1
                            if(p >=len(parent)):
                                p = 0

                        # print("kkkkkkk")
                        # print(child)
                        # print(parent)
                        # print(p)
                        child[i] = parent[p]
                # print(child)
                # print(parent)

            fill_child(child1.genes, parent2.genes)
            fill_child(child2.genes, parent1.genes)
            a = [d.width for d in child1.genes]
            b = [d.width for d in child2.genes]
            # if sum(a) != 26:
            #     exit()
            # if sum(b) != 26:
            #     exit()


    def evolve(self):
        children = []
        for _ in range(len(self.population) // 2):
            #selekcja
            parent1 = self.select_parents()
            parent2 = self.select_parents()
            # print("================================")
            # print(parent1.genes)
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)
            
            # print(child1.genes)
            self.crossover(parent1, parent2, child1, child2)
            self.swap_mutate(child1)
            # self.swap_mutate(child2)
            # print(len(set(child1.genes)))
            #ewalucaja dzieci   
            child1.evaluate_fitness(self.bins)
            child2.evaluate_fitness(self.bins)
            # print(child1.genes)

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
                if x.fitness[0] > best[0]:
                    best = x.fitness
            else:
                best = x.fitness
            if worst:
                if x.fitness[0] < worst[0]:
                    worst = x.fitness
            else:
                worst = x.fitness
            # print(x.fitness)
            mean += x.fitness[0]
        mean = mean/len(self.population)
        self.results.append([best, worst, mean])

    def print_results(self):
        x = list(range(len(self.results)))
        y1 = [item[0][0] for item in self.results]
        y2 = [item[1][0] for item in self.results]
        y3 = [item[2] for item in self.results]

        y11 = [item[0][1] for item in self.results]
        y22 = [item[1][1] for item in self.results]
        y33 = [item[2] for item in self.results]

        plt.plot(x, y1, label='Best')
        plt.plot(x, y2, label='Worst')
        plt.plot(x, y3, label='Mean')

        plt.legend()

        plt.show()
        print(self.results)
    
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

# loaded_bins, loaded_flowers = load_data(bins_filename='count100bins/size10/size10bins.csv', flowers_filename='count100bins/size10/count50flowers.csv')
if __name__ == "__main__":
    genetic_config = AlgorithmConfig(
        population_size=120,
        generations=100,
        mutation_rate=0.01,
        crossover_rate=0.25,
        tournament_size = 4,
        flowers_file="count100bins/size100/count200flowers.csv",
        results_file="genetic_results.csv",
        bins_file="count100bins/size100/size100bins.csv",
    )

    genetic_algorithm = GeneticAlgorithm(genetic_config)
    genetic_algorithm.load_data()
    genetic_algorithm.run()
