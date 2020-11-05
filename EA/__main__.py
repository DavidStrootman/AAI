from typing import List
import random
import time


class Phenotype:
    def __init__(self):
        self._chromosome: int = 0
        
    @property
    def _A(self):
        return self._chromosome & 0x3F
    
    @_A.setter
    def _A(self, v: int):     
        if v > 63:
            raise ValueError("given v bigger than 63")
        self._chromosome |= v << 0
    
    @property
    def _B(self):
        return (self._chromosome & 0x3F00) >> 8
    
    @_B.setter
    def _B(self, v: int):     
        if v > 63:
            raise ValueError("given v bigger than 63")
        self._chromosome |= v << 8

    @property
    def _C(self):   
        return (self._chromosome & 0x3F0000) >> 16
    
    @_C.setter
    def _C(self, v: int):       
        if v > 63:
            raise ValueError("given v bigger than 63")
        self._chromosome |= v << 16

    @property
    def _D(self):
        return (self._chromosome & 0x3F000000) >> 24
        
    @_D.setter
    def _D(self, v: int):   
        if v > 63:
            raise ValueError("given v bigger than 63")
        self._chromosome |= v << 24
    
    @property
    def fitness(self):
        return (self._A - self._B)**2 + (self._C + self._D)**2 - (self._A - 30)**3 - (self._C - 40)**3

    def randomize_chromosome(self):
        random_values = [random.randrange(0, 63) for i in range(4)]
        self.encode_chromosome(random_values)
    
    @property
    def decoded_chromosome(self):
        return [self._A, self._B, self._C, self._D]
    
    def encode_chromosome(self, parameters: List[int]):      
        self._A = parameters[0]
        self._B = parameters[1]
        self._C = parameters[2]
        self._D = parameters[3]

    def mutate(self, propability: float):
        if propability > random.uniform(0, 1):
            pos_to_mutate = random.randrange(0, 24)
            index_to_mutate = (int(pos_to_mutate / 6)) * 2 + pos_to_mutate
            self._chromosome = self._chromosome ^ (1 << index_to_mutate)

    def repopulate(self, mate: 'Phenotype', nr_of_children: int) -> List['Phenotype']:         
        children = []
        for i in range(nr_of_children):
            new_child = Phenotype()
            child_parameters = []
            parameters = self.decoded_chromosome
            parameters[1] = mate.decoded_chromosome[1]
            parameters[3] = mate.decoded_chromosome[3]
            new_child.encode_chromosome(parameters)
            children.append(new_child)
        return children


def brute_force():
    max_lift = 0
    out_values = []
    for A in range(64):
        for B in range(64):
            for C in range(64):
                for D in range(64):
                    lift = (A - B)**2 + (C + D)**2 - (A - 30)**3 - (C - 40)**3
                    if lift > max_lift:
                        max_lift = lift
                        out_values = [A, B, C, D]
    return out_values, max_lift


def generate_population(size) -> List[Phenotype]:
    population = []
    for _ in range(size):
        individual = Phenotype()
        individual.randomize_chromosome()
        population.append(individual)
    return population


def evaluate(population, nr_of_survivors = None, nr_of_failures = None) -> List[Phenotype]:
    population.sort(key=lambda phenotype: phenotype.fitness, reverse=True)
    return population[:nr_of_survivors]


def mutate_population(population, probability):
    for individual in population:
            individual.mutate(probability)


def crossover(population: List[Phenotype], nr_of_children_per_couple):
    children = []
    random.shuffle(population)
    males = population[:int(len(population)/2)]
    females = population[int(len(population)/2):]
    couples = zip(males, females)
    for couple in couples:
        children = children + couple[0].repopulate(couple[1], nr_of_children_per_couple)
    return children


def main():   
    population = generate_population(300)
    mutate_population(population, 1)
    
    time_start = time.time_ns()
    for i in range(30):
        children = crossover(population, 1)
        mutate_population(children, 0.5)
        population.extend(children)
        population = evaluate(population, nr_of_survivors=150)
        print("============================================ generation:", i)
        for individual in population[:3]:
            print(individual.fitness, individual.decoded_chromosome)
    time_end = time.time_ns() - time_start
    print("EA takes:", time_end / 1000000000)

    time_start = time.time_ns()
    brute_force()
    time_end = time.time_ns() - time_start
    print("Brute force takes:", time_end / 1000000000)


if __name__ == "__main__":
    main()
