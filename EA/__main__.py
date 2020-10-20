from typing import List
import random

class Phenotype:
    """Phenotype
    
    00 DDDDDD 00 CCCCCC 00 BBBBBB 00 AAAAAA
    """
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
       return ((self._A - self._B)**2 + (self._C + self._D)**2 - (self._A - 30)**2 * (self._A - 30) - (self._C - 40)**3)


    def randomize_chromosome(self):
        random_values = [random.randrange(0, 19) for i in range(4)]
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
            index_to_mutate = (int(pos_to_mutate / 6) + 1) * 2 + pos_to_mutate 
            self._chromosome = self._chromosome ^ (1 << index_to_mutate)
    
    def repopulate(self, mate: 'Phenotype', nr_of_children: int) -> List['Phenotype']:
        children = []
        for _i in range(nr_of_children):
            new_child = Phenotype()
            child_parameters = []
            print("-------------------")
            for j, mate_parameter in enumerate(mate.decoded_chromosome):
                split = random.randrange(1, 6)
                mask = 0b0
                for k in range(split):
                    mask = mask ^ (1 << k)
                parameter = (self.decoded_chromosome[j] & mask) | (mate_parameter & mask ^ 0x3f)
                print("{0:b}".format(parameter), "{0:b}".format((self.decoded_chromosome[j] & mask)), "{0:b}".format((mate_parameter & mask ^ 0x3f)))
                child_parameters.append(parameter)
            new_child.encode_chromosome(child_parameters)
            children.append(new_child)
        return children


def brute_force():
    max_lift = 0
    out_values = []
    for A in range(64):
        for B in range(64):
            for C in range(64):
                for D in range(64):
                    lift = (A - B)*(A - B) + (C + D) * (C + D) - (A - 30) * (A - 30) * (A - 30) - (C - 40) * (C - 40) * (C - 40)
                    if lift > max_lift:
                        max_lift = lift
                        out_values = [A, B, C, D]
    return (out_values, max_lift)

def generate_population(size) -> List[Phenotype]:
    population = []
    for _ in range(size):
        individual = Phenotype()
        individual.randomize_chromosome()
        population.append(individual)
    return population

def evaluate(population, nr_of_survivors) -> List[Phenotype]:
    population.sort(key=lambda phenotype: phenotype.fitness, reverse=False)
    return population[:nr_of_survivors]

def mutate_population(population, probability):
    for individual in population:
            individual.mutate(probability)
    
def orgy(population: List[Phenotype], nr_of_children_per_couple):
    children = []
    random.shuffle(population)
    males = population[:int(len(population)/2)]
    females = population[int(len(population)/2):]
    couples = zip(males, females)
    for couple in couples:
        children = children + couple[0].repopulate(couple[1], nr_of_children_per_couple)
    return children


def main():
    population = generate_population(3)
    for i in range(1):
        surviving_population = evaluate(population, 3)
        population = orgy(surviving_population, 3)
        mutate_population(population, 0.3)
        print("============================================", i)
        for individual in population:
            print(individual.fitness, individual.decoded_chromosome)
    
    


    
    # for var in brute_force()[0]:
    #     print(var)
    #     print()


if __name__ == "__main__":
    main()