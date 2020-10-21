from typing import List
import random
import time
class Phenotype:
    """Phenotype
    
    00 DDDDDD 00 CCCCCC 00 BBBBBB 00 AAAAAA
    """
    def __init__(self):
        """[summary]
        """
        
    @property
    def _A(self):
        """Returns A

        Returns:
            [type]: returns value for A
        """
    
    @_A.setter
    def _A(self, v: int):
        """Setter

        Args:
            v (int): value

        Raises:
            ValueError: value higher than 63
        """      
    
    @property
    def _B(self):
        """Returns B

        Returns:
            [type]: returns value for B
        """
    
    @_B.setter
    def _B(self, v: int):
        """Setter

        Args:
            v (int): value

        Raises:
            ValueError: value higher than 63
        """      

    @property
    def _C(self):
        """Returns C

        Returns:
            [type]: returns value for C
        """       
    
    @_C.setter
    def _C(self, v: int):
        """Setter

        Args:
            v (int): value

        Raises:
            ValueError: value higher than 63
        """            

    @property
    def _D(self):
        """Returns D

        Returns:
            [type]: returns value for D
        """ 
        
    @_D.setter
    def _D(self, v: int):
        """Setter

        Args:
            v (int): value

        Raises:
            ValueError: value higher than 63
        """        
    
    @property
    def fitness(self):
        """returns the fitness of the phenotype"""


    def randomize_chromosome(self):
        """randomnizes the chromosome"""

    
    @property
    def decoded_chromosome(self):
        """Returns decoded chromosome

        Returns:
            [List]: List of chromosome parameters.
        """
        return [self._A, self._B, self._C, self._D]
    
    def encode_chromosome(self, parameters: List[int]):
        """Encodes a list of parameters.

        Args:
            parameters (List[int]): List of chromosome parameters.
        """        

    def mutate(self, propability: float):
        """
        Description of mutate

        Args:
            self (undefined):
            propability (float):

        """

    def repopulate(self, mate: 'Phenotype', nr_of_children: int) -> List['Phenotype']:
        """
        Generates a list of X childs with parameters of the self and target mate.

        Args:
            mate ('Phenotype'): Partner
            nr_of_children (int): number of children

        Returns:
            List['Phenotype']: a list of childs.

        """            

def brute_force():
    """[summary]
    solve the problem with little programming effort.

    Returns:
        The max lift with correpsonding A, B, C & D values.
    """

def generate_population(size) -> List[Phenotype]:
    """Generates a population 

    Args:
        size ([type]): size of the population

    Returns:
        List[Phenotype]: the generated population
    """

def evaluate(population, nr_of_survivors = None, nr_of_failures = None) -> List[Phenotype]:
    """Sorts a population soley based on fitness and return the top nr_of_survivors population

    Args:
        population ([type]): A population of phenotypes
        nr_of_survivors ([type], optional): Number of survivors. Defaults to None.

    Returns:
        List[Phenotype]: [description]
    """

def mutate_population(population, probability):
    """[summary]

    Args:
        population ([type]): [description]
        probability ([type]): [description]
    """
    
def crossover(population: List[Phenotype], nr_of_children_per_couple):
    """Performs a crossover with all the phenotypes in the population randomly

    Args:
        population (List[Phenotype]): a list of phenotypes
        nr_of_children_per_couple ([type]): the amounbt of children every couple should get

    Returns:
        [type]: A list of phenotype children resulted from the crossover
    """

def main():   
