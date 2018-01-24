import numpy as np
import sys

class CrossValidation:
    def __init__(self):
        self.LeastMSE = float('inf')
        self.Weights = []