"""Independent finite CME and multinomial laws. No ChemWorld imports."""

from math import comb

import numpy as np
from scipy.linalg import expm
from scipy.stats import binom, multinomial


class Law:
    def __init__(self, case, duration):
        self.total = sum(case["initial"].values())
        self.size = len(case["species"])
        self.order = case.get("order", 1)
        self.probability = self.support = self.p = None
        if case["kind"] == "linear":
            self.p = expm(np.asarray(case["q"]) * duration)[0]
            self.mean = self.total * self.p
            self.covariance = self.total * (np.diag(self.p) - np.outer(self.p, self.p))
            self.product_variance = np.zeros((self.size, self.size))
            for i, p in enumerate(self.p):
                for j, q in enumerate(self.p):
                    if i == j:
                        v = self.covariance[i, i]
                        fourth = self.total * p * (1-p) * (1-6*p*(1-p)) + 3*v*v
                    else:
                        cumulant = self.total * (-p*q + 2*p*p*q + 2*p*q*q - 6*p*p*q*q)
                        fourth = cumulant + self.covariance[i, i]*self.covariance[j, j]
                        fourth += 2*self.covariance[i, j]**2
                    self.product_variance[i, j] = fourth - self.covariance[i, j]**2
        else:
            count = self.total // self.order + 1
            generator = np.zeros((count, count))
            for b in range(count):
                if b+1 < count:
                    generator[b, b+1] = case["rows"][0][2] * comb(
                        self.total-self.order*b, self.order)
                if b:
                    generator[b, b-1] = .4*b
                generator[b, b] = -sum(generator[b])
            self.probability = expm(generator*duration)[0][::-1]
            self.support = np.asarray([(self.total-self.order*b, b)
                                       for b in reversed(range(count))])
            self.mean = self.probability @ self.support
            centered = self.support-self.mean
            self.covariance = (centered*self.probability[:, None]).T @ centered
            products = centered[:, :, None]*centered[:, None, :]
            self.product_variance = np.sum(
                self.probability[:, None, None]*(products-self.covariance)**2, axis=0)
        probability = self.p if self.p is not None else self.probability
        if abs(sum(probability)-1) > 1e-10 or np.min(probability) < -1e-14:
            raise ArithmeticError("invalid independent probability law")

    def valid(self, rows):
        weights = np.ones(self.size, dtype=int)
        if self.order > 1:
            weights[1] = self.order
        return bool(np.all(rows >= 0) and np.all(rows @ weights == self.total))

    def cdf_distance(self, rows):
        support, frequency = np.unique(rows, axis=0, return_counts=True)
        empirical = np.cumsum(frequency)/len(rows)
        if self.p is None:
            indices = (support[:, 0] // self.order).astype(int)
            right = np.cumsum(self.probability)[indices]
            point = self.probability[indices]
        elif self.size == 2:
            right = binom.cdf(support[:, 0], self.total, self.p[0])
            point = binom.pmf(support[:, 0], self.total, self.p[0])
        else:
            first = support[:, 0]
            right = binom.cdf(first-1, self.total, self.p[0])
            right += binom.pmf(first, self.total, self.p[0]) * binom.cdf(
                support[:, 1], self.total-first, self.p[1]/(1-self.p[0]))
            point = multinomial.pmf(support, self.total, self.p)
        return float(max(np.max(abs(empirical-right)),
                         np.max(abs(empirical-frequency/len(rows)-(right-point)))))
