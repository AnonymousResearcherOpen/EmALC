from typing import Iterable
import numpy as np
from functools import reduce

def hard_negation(x):
    return 1-x

def godel_tnorm(x, y):
    return np.minimum(x, y)

def godel_snorm(x, y):
    return np.maximum(x, y)

def product_tnorm(x, y):
    return np.multiply(x, y)

def product_snorm(x, y):
    if isinstance(x, Iterable):
        return np.array([a+b-a*b for a,b in zip(x, y)])
    else:
        return x+y-x*y

def godel_implication(a, b):
    if a<=b:
        return 1
    else:
        return b

def product_implication(a, b):
    if a<=b:
        return 1
    else:
        return b/a

def godel_forall(r, x):
    return np.array([np.min([godel_implication(r[i][j], x[j]) for j in range(len(x))]) for i in range(len(x))])

def godel_exist(r, x):
    return np.array([np.max([godel_tnorm(r[i][j], x[j]) for j in range(len(x))]) for i in range(len(x))])

def product_forall(r, x):
    return np.array([np.prod([product_implication(r[i][j], x[j]) for j in range(len(x))]) for i in range(len(x))])

def product_exist(r, x):
    return np.array([reduce(lambda a,b: a+b-a*b, [product_tnorm(r[i][j], x[j]) for j in range(len(x))]) for i in range(len(x))])

class EmbeddingComputer(object):

    @staticmethod
    def negate(x):
        pass

    @staticmethod
    def intersect(x, y):
        pass

    @staticmethod
    def unify(x, y):
        pass

    @staticmethod
    def forall(r, x):
        pass

    @staticmethod
    def exists(r, x):
        pass

    @staticmethod
    def imply(a, b):
        pass

    def __init__(self, mode):
        self.mode = mode

        if self.mode in ['godel', 'crisp']:
            self.negate = hard_negation
            self.intersect = godel_tnorm
            self.unify = godel_snorm
            self.forall = godel_forall
            self.exists = godel_exist
            self.imply = godel_implication
        elif self.mode in ['product']:
            self.negate = hard_negation
            self.intersect = product_tnorm
            self.unify = product_snorm
            self.forall = product_forall
            self.exists = product_exist
            self.imply = product_implication
        else:
            assert 0, "Unrecognized mode " + self.mode

    def compute_subclassof_truth_degree(self, x, y):
        # P subsetof Q   ====>  forall x P(x) \to Q(x)
        return reduce(self.intersect, [self.imply(a,b) for a,b in zip(x, y)])

if __name__ == '__main__':
    computer = EmbeddingComputer(mode='godel')
    print(computer.negate(np.array([0.3])))

    print(computer.compute_subclassof_truth_degree(
        np.array([0.3, 0.4]),
        np.array([0.7, 0.1])
    ))
    computer = EmbeddingComputer(mode='crisp')
    print(computer.negate(np.array([0])))
    print(computer.compute_subclassof_truth_degree(
        np.array([1, 1]),
        np.array([0, 1])
    ))
    computer = EmbeddingComputer(mode='product')
    print(computer.negate(np.array([0.3])))
    print(computer.compute_subclassof_truth_degree(
        np.array([0.3, 0.4]),
        np.array([0.7, 0.1])
    ))