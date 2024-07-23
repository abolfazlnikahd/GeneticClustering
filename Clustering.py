import random

import numpy as np

from Config import POPULATION_SIZE, GENE_LENGTH, MAX_GENERATIONS, MUTATION_RATE, MIN_MAX


class GeneticClustering:
    def __init__(self, db, *args, **kwargs):
        self.db = db
        if args is not None and len(args) > 0:
            if args["POPULATION_SIZE"]:
                POPULATION_SIZE = args["POPULATION_SIZE"]
            if args["GENE_LENGTH"]:
                GENE_LENGTH = args["GENE_LENGTH"]
            if args["MAX_GENERATIONS"]:
                MAX_GENERATIONS = args["MAX_GENERATIONs"]
            if args["MUTATION_RATE"]:
                MUTATION_RATE = args["MUTATION_RATE"]

    def _if2b(self, min_idx, max_idx):
        min = MIN_MAX[min_idx]
        max = MIN_MAX[max_idx]
        if "." in min and "." in max:
            min = float(min)
            max = float(max)
            randi = random.uniform(min, max)

            randi_li = str(randi).split(".")
            randi = str(randi_li[0]) + str(randi_li[1])

            return bin(int(randi))[2:] + "f" + str(len(randi_li[1]))
        else:
            randi = random.randrange(int(min), int(max))
            return bin(randi)[2:] + "f0"

    def if2b_parent(self, parent):
        response = []
        for i in range(16):
            if '.' in str(parent[i]):
                parent[i] = str(parent[i]).split('.')
                parent[i] = parent[i][0] + parent[i][1]
                response.append(bin(int(parent[i]))[2:] + "f" + str(len(parent[i][1])))
            else:
                response.append(bin(int(parent[i]))[2:] + "f0")

        return response

    def _generate_chromosome(self):
        min_idx = 0
        max_idx = 1
        response_list = []
        for repeat in range(16):
            response_list.append(self._if2b(min_idx, max_idx))
            min_idx += 2
            max_idx += 2
        return response_list

    def generate_population(self):
        return [self._generate_chromosome() for _ in range(POPULATION_SIZE)]

    def decode_chromosome(self, chromosome):
        decimal_list = []
        for gene in chromosome:
            gene_li = gene.split('f')
            if len(gene_li[0]) == int(gene_li[1]):
                decimal_list.append(float(f"0.{int(gene_li[0], 2)}"))
            elif gene_li[1] == "0":
                decimal_list.append(int(gene_li[0], 2))
            else:
                gene_li[0] = int(gene_li[0], 2)
                gene_li[1] = int(gene_li[1])
                decimal_list.append(float(str(gene_li[0])[:-gene_li[1]] + "." + str(gene_li[0])[-gene_li[1]:]))
        return decimal_list

    def encode_cluster(self, decimal_population, chromosomes):
        import math
        clustered = {}
        for idx in range(len(decimal_population)):
            clustered[idx] = []
        for row in chromosomes:
            best_distance = float("inf")
            best_chromosome = 0
            for chromosome_idx in range(len(decimal_population)):
                distance = sum((decimal_population[chromosome_idx][i] - row[i]) ** 2 for i in range(1, 16))
                squrt_distance = math.sqrt(distance)
                if squrt_distance < best_distance:
                    best_distance = squrt_distance
                    best_chromosome = chromosome_idx
            clustered[best_chromosome].append(np.array(row))
        return clustered

    def calculate_euclidean_distance(self, pair):
        from scipy.spatial.distance import euclidean
        return euclidean(pair[0], pair[1])

    def all_distance(self, chromosomes_li: list):
        import multiprocessing
        from itertools import combinations
        pairs = list(combinations(chromosomes_li, 2))
        # print(f"list of pares is : {pairs}")
        num_pairs = len(pairs)

        with multiprocessing.Pool() as pool:
            results = pool.map(self.calculate_euclidean_distance, pairs)

        return results

    def variance_of_clustered_population(self, clustered_population: dict):
        """
        input: clustered_population: A dictionary whose keys are the chromosomes (decimal numbers) and
                the values are a list of rows of decimal numbers in the same cluster.

        output: variance_dict: A dictionary whose keys are the chromosomes (decimal numbers) and
                the values are variance of data within a cluster.
        """
        import statistics
        variance_dict = {}
        for chromosome in clustered_population.keys():
            distances = self.all_distance(clustered_population[chromosome])
            if len(distances) == 0:
                distances.append(0)
                distances.append(0)

            variance_dict[chromosome] = statistics.variance(distances)
        return variance_dict

    def fitness(self, population):
        """
        input: List of lists containing binary numbers

        output: A dictionary whose keys are integer numbers of chromosomes (centers)
                and whose values are the variance of data within a cluster.
                {[265, ...,]: variance}
        """
        decimal_population = []
        # تبدیل کردن کروموزوم های باینری به دسیمال
        for chromosome in population:
            decimal_population.append(self.decode_chromosome(chromosome))

        # خوشه بندی کردن تمام دیتاها با توجه به مراکز
        clustered = self.encode_cluster(decimal_population, self.db)
        print("----------------------------clustered down---------------------------")
        for chromosome_idx in clustered:
            print(f"{chromosome_idx} : {len(clustered[chromosome_idx])}")

        # گرفتن واریانس تمام داده های داخل یک خوشه
        variances = self.variance_of_clustered_population(clustered)
        print(f"in fitness : {variances}")
        return variances

    def select(self, population):
        fitness = self.fitness(population)
        fitness_values = fitness.values()
        fitness_values = sorted(fitness_values)
        if 0 in fitness_values:
            for key in fitness.keys():
                if fitness[key] == 0:
                    fitness[key] = fitness_values[-1] + random.randrange(0, 1000)
        fitness_values = fitness.values()
        fitness_values = sorted(fitness_values)

        for new_key in fitness.keys():
            if fitness[new_key] == fitness_values[0]:
                chromosomes_1_idx = new_key
            elif fitness[new_key] == fitness_values[1]:
                chromosomes_2_idx = new_key
            elif fitness[new_key] == fitness_values[2]:
                chromosomes_3_idx = new_key
            elif fitness[new_key] == fitness_values[3]:
                chromosomes_4_idx = new_key
            elif fitness[new_key] == fitness_values[4]:
                chromosomes_5_idx = new_key



        return chromosomes_1_idx, chromosomes_2_idx, chromosomes_3_idx, chromosomes_4_idx, chromosomes_5_idx

    def clean_chromosomes_len(self, chromosome1: list, chromosome2: list):
        """
        input: list of decimal numbers
        output: list of binary strings
        """

        chromosome1_cleaned = []
        chromosome2_cleaned = []
        for idx in range(16):
            chromosome1_idx = chromosome1[idx].split("f")
            #print(chromosome1_idx)
            chromosome2_idx = chromosome2[idx].split("f")
            #print(chromosome2_idx)
            if int(chromosome1_idx[1]) == 0 and int(chromosome2_idx[1]) == 0:
                chromosome1_cleaned.append(int(chromosome1_idx[0], 2))
                chromosome2_cleaned.append(int(chromosome2_idx[0], 2))
            else:
                chromosome1_idx_decimal = str(int(chromosome1_idx[0][:int(chromosome1_idx[1])], 2)) + "." + str(int(chromosome1_idx[0][-int(chromosome1_idx[1]):], 2))
                chromosome2_idx_decimal = str(int(chromosome2_idx[0][:int(chromosome2_idx[1])], 2)) + "." + str(int(chromosome2_idx[0][-int(chromosome2_idx[1]):], 2))
                if len(chromosome1_idx_decimal) > len(chromosome2_idx_decimal):
                    chromosome1_idx_decimal = chromosome1_idx_decimal[:len(chromosome2_idx_decimal)]
                else:
                    chromosome2_idx_decimal = chromosome2_idx_decimal[:len(chromosome1_idx_decimal)]

                chromosome1_cleaned.append(float(chromosome1_idx_decimal))
                chromosome2_cleaned.append((float(chromosome2_idx_decimal)))

        return self.if2b_parent(chromosome1_cleaned), self.if2b_parent(chromosome2_cleaned)

    def crossover(self, parent1, parent2):
        parent1, parent2 = self.clean_chromosomes_len(parent1, parent2)

        point = random.randint(1, GENE_LENGTH - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mutate(self, chromosome):
        for i in range(len(chromosome)):
            if random.random() < MUTATION_RATE:
                x1, x2, x3, x4 = random.sample(range(0, 16), 4)
                chromosome[i][x1] = str(1 - int(chromosome[i][x1]))
                chromosome[i][x2] = str(1 - int(chromosome[i][x2]))
                chromosome[i][x3] = str(1 - int(chromosome[i][x3]))
                chromosome[i][x4] = str(1 - int(chromosome[i][x4]))
        return chromosome

    def run(self):
        population = self.generate_population()
        #print(population)
        for generation in range(MAX_GENERATIONS):
            print(f"Generation --------------------------------- > {generation}")
            new_population = []

            parent1, parent2, parent3, parent4, parent5 = self.select(population)
            #print(parent5)
            #print(f"p1 : {population[parent1]} \n", f"p2 : {population[parent2]} \n", f"p3 : {population[parent3]} \n", f"p4 : {population[parent4]}")
            #print(f"p5 : {population[parent5]} \n")
            child1, child2 = self.crossover(population[parent1], population[parent2])
            child3, child4 = self.crossover(population[parent1], population[parent3])
            child5, child6 = self.crossover(population[parent1], population[parent4])
            child7, child8 = self.crossover(population[parent1], population[parent5])

            new_population.append(child1)
            new_population.append(child2)
            new_population.append(child3)
            new_population.append(child4)
            new_population.append(child5)
            new_population.append(child6)
            new_population.append(child7)
            new_population.append(child8)

            population = new_population

        return [self.decode_chromosome(chromosome) for chromosome in population]


if __name__ == "__main__":
    import pandas as pd

    db = pd.read_excel("test_data.xlsx")
    q = GeneticClustering(db.values)
    q.run()