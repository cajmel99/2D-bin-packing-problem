import random
from .loader import Bin, Flower, load_data
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import numpy as np
from .evaluate_fitness import evaluate
import csv
import os
import json
import itertools

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

    def put_flower(self, flower:Flower, flower_number):
        for x, y in self.free_spaces:
                if self.can_fit(flower.width, flower.height, x, y):
                    self.matrix[x:x+flower.width, y:y+flower.height] = np.ones([flower.width, flower.height])
                    self.debug_elements.append((flower, flower_number))
                    self.update_free_spaces(flower, x, y)
                    return True
        return False
    
    def save_bin(self, filename):
        np.savetxt(filename, np.rot90(self.matrix), delimiter=",", fmt="%d")
        print("==========================")
        print(self.debug_elements)

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

            if not placed:
                raise Exception("Nie można umieścić kwiatka :(")

        self.fitness = evaluate([bin.matrix for bin in genetic_bins], False)


class AlgorithmConfig:
    def __init__(self, population_size, generations, mutation_rate, crossover_rate, results_file, results_file2, flowers_file, bins_file, tournament_size):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.results_file = results_file
        self.results_file2 = results_file2
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
                for i in range(size):
                    if not child[i]:
                        while parent[p] in child:
                            p += 1
                            if(p >=len(parent)):
                                p = 0
                        child[i] = parent[p]

            fill_child(child1.genes, parent2.genes)
            fill_child(child2.genes, parent1.genes)


    def evolve(self):
        children = []
        for _ in range(len(self.population) // 2):
            #selekcja
            parent1 = self.select_parents()
            parent2 = self.select_parents()

            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)
            
            # print(child1.genes)
            self.crossover(parent1, parent2, child1, child2)
            self.swap_mutate(child1)

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
            mean += x.fitness[0]
        mean = mean/len(self.population)
        self.results.append([best, worst, mean])

    def print_results(self):
        x = list(range(len(self.results)))
        y1 = [item[0][0] for item in self.results]
        y2 = [item[1][0] for item in self.results]
        y3 = [item[2] for item in self.results]

        plt.figure()
        plt.plot(x, y1, label='Best')
        plt.plot(x, y2, label='Worst')
        plt.plot(x, y3, label='Mean')

        plt.legend()

        plt.savefig(self.config.results_file2)
    
    def run(self):
        self.initialize_population()

        self.save_results()
        for i in tqdm(range(self.config.generations)):
            self.evolve()
            self.save_results()
        
        self.print_results()
        with open(self.config.results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.results)
        return self.results[-1][0]

class ConfTempl:
    def __init__(self, tour, cross, mut, bins, flowers, size, bins_file):
        self.conf_templ = {
            "mutation": mut,
            "crossover": cross,
            "tournament": tour,
            "bins": bins,
            "flowers": flowers,
            "size": size,
            "bins_file": bins_file
        }

    def get_conf_templ(self):
        return self.conf_templ

def run_genetic_algorithm(pop_size: int, gene: int, tour: int, cross: float, mut: float, flowers_file: str, bins_file: str, results_csv: str, results_jpg: str):
    genetic_config = AlgorithmConfig(
        population_size=pop_size,
        generations=gene,
        mutation_rate=mut,
        crossover_rate=cross,
        tournament_size = tour,
        flowers_file=flowers_file,
        results_file=results_csv,
        results_file2=results_jpg,
        bins_file=bins_file,
    )

    genetic_algorithm = GeneticAlgorithm(genetic_config)
    genetic_algorithm.load_data()
    results = genetic_algorithm.run()
    return results

if __name__ == "__main__":
    bins_flowers = [
        {
            "bins": "count10bins",
            "flowers": "count100flowers1",
            "size": "size10",
            "bins_file": "size10bins"
        },
        {
            "bins": "count10bins",
            "flowers": "count100flowers2",
            "size": "size10",
            "bins_file": "size10bins"
        },
        {
            "bins": "count10bins",
            "flowers": "count100flowers3",
            "size": "size10",
            "bins_file": "size10bins"
        },

    ]
    for b_f in bins_flowers:

        iterations = 6
        mutation_rates = [0.005, 0.01]
        crosover_rates = [0.25, 0.3, 0.35, 0.5, 0.6]
        tournament_sizes = [2, 3, 6, 10]
        tests_list = [tournament_sizes, crosover_rates, mutation_rates]

        tests = []

        for t in itertools.product(*tests_list):
            tests.append(t)
    
        # tests = [
        #     [6, 0.6, 0.005],
        #     [6, 0.5, 0.005],
        #     [10, 0.35, 0.01],
        #     [4, 0.35, 0.005],
        #     [6, 0.25, 0.015],
        # ]
        print(len(tests))
        # for x in tests:
        #     print(x)
        errors = []
        final_results = []
        confs = [ConfTempl(*t, *list(b_f.values())) for t in tests]
        alll_iters = len(confs)
        for ii, cont_templ in enumerate(confs):
            item = cont_templ.get_conf_templ()
            path = f"{item['tournament']}_{item['crossover']}_{item['mutation']}"
            path2 = f"genetic_results/{item['bins']}/{item['size']}/{item['flowers']}/{path}"
            if not os.path.exists(path2):
                os.makedirs(path2)
            for iteration in range(iterations):
                print("_".join(list(b_f.values())))
                print(f"{ii+1}/{alll_iters}")
                print(item)
                print(iteration)
                genetic_config = AlgorithmConfig(
                    population_size=150,
                    generations=250,
                    mutation_rate=item["mutation"],
                    crossover_rate=item["crossover"],
                    tournament_size = item["tournament"],
                    flowers_file=f"{item['size']}/{item['flowers'].strip('1').strip('2').strip('3')}/{item['flowers']}.csv",
                    results_file=f"genetic_results/{item['bins']}/{item['size']}/{item['flowers']}/{path}/{iteration}.csv",
                    results_file2=f"genetic_results/{item['bins']}/{item['size']}/{item['flowers']}/{path}/wykres.jpg",
                    bins_file=f"{item['size']}/{item['bins_file']}.csv",
                )

                genetic_algorithm = GeneticAlgorithm(genetic_config)
                genetic_algorithm.load_data()
                try:
                    res = genetic_algorithm.run()
                    final_results.append({**cont_templ.get_conf_templ(), **{"results": res}})

                except Exception as e:
                    errors.append(cont_templ.get_conf_templ())
        rsults_filename = "_".join(list(b_f.values()))

        with open(f'{rsults_filename}.csv', 'w', newline="", encoding="utf-8") as testfile:
            if(final_results):
                fieldnames = list(final_results[0].keys())
                print(final_results)
                print(fieldnames)
                print(final_results[0])
                
                writer = csv.writer(testfile, delimiter=";")
                writer.writerow(fieldnames)
                for item in final_results:
                    print(item)
                    writer.writerow([item[x] for x in item])

        with open(f'{rsults_filename}_ERRORS.json', 'w', newline="", encoding="utf-8") as testfile:
            json.dump(errors, testfile, ensure_ascii=False, indent=4)