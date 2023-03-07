import numpy as np


class UncertainValue:
    '''
    A class to represent a physical measurement and
    its associated uncertainty.

    Parameters:
            value : float
            the value of the measurement.

            sigma : float
            the uncertainty associated with the value.
            this is in absolute standard deviation, not relative. percentage.
    '''

    def __init__(self, num):
        self.value, self.sigma = num

    def inplace_add(self, added):
        '''
        In-place addition for an uncertain value.
        '''


def multiply(v1: UncertainValue, v2: UncertainValue) -> UncertainValue:
    product = v1.value * v2.value
    return UncertainValue(product,
                          product * np.sqrt(
                              (v1.sigma / v1.value)**2
                              + (v2.sigma / v2.value)**2))


def add(v1: UncertainValue, v2: UncertainValue) -> UncertainValue:
    total = v1.value - v2.value
    return UncertainValue(total, (v2.sigma**2 + v1.sigma**2))


def weightedAverage(values: list[UncertainValue]) -> UncertainValue:
    weights = np.array([1 / val.sigma**2 for val in values])
    vals = np.array([val.value for val in values])
    sigma_f = 1 / np.sqrt(np.sum(weights))
    return UncertainValue(np.sum(weights * vals) / np.sum(weights), sigma_f)


vals = [
    (7.4819, 0.5154),
    (9.6613, 0.6571),
    (8.9978, 1.1081),
    (7.7491, 0.2368),
    (7.7871, 0.2644), ]

uncvals = [UncertainValue(*v) for v in vals]

print(weightedAverage(uncvals).value, weightedAverage(uncvals).sigma)
