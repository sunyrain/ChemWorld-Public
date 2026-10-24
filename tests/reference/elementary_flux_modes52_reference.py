"""Independent small exact determinants/minors; no ChemWorld dependency or RREF."""

from fractions import Fraction
from itertools import combinations, permutations


def determinant(matrix):
    n = len(matrix)
    if not n:
        return Fraction(1)
    result = Fraction(0)
    for order in permutations(range(n)):
        sign = (-1) ** sum(order[i] > order[j] for i in range(n) for j in range(i + 1, n))
        product = Fraction(sign)
        for row, column in enumerate(order):
            product *= Fraction(matrix[row][column])
        result += product
    return result


def rank(matrix, width):
    for size in range(min(len(matrix), width), 0, -1):
        for rows in combinations(range(len(matrix)), size):
            for columns in combinations(range(width), size):
                if determinant([[matrix[i][j] for j in columns] for i in rows]):
                    return size
    return 0


def minimal_balanced(matrix, vector):
    vector = tuple(Fraction(v) for v in vector)
    support = [i for i, v in enumerate(vector) if v]
    if not support or any(
        sum(Fraction(v) * x for v, x in zip(row, vector, strict=True)) for row in matrix
    ):
        return False
    for removed in support:
        subset = [i for i in support if i != removed]
        if rank([[row[i] for i in subset] for row in matrix], len(subset)) != len(subset):
            return False
    return True


def exchanges(matrix, vector):
    return [
        str(sum((Fraction(a) * Fraction(b) for a, b in zip(row, vector, strict=True)), Fraction(0)))
        for row in matrix
    ]
