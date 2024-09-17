import numpy as np
from itertools import cycle

class ContinuumBasis:
    pass

class NoContinuum(ContinuumBasis):
    
    @property
    def num_parameters(self):
        return 0

    def design_matrix(self, λ):
        return 0

class Sinusoids(ContinuumBasis):    
        
    def __init__(self, P=7, L=None):
        self.P = P
        self.L = L
        return None
    
    @property
    def num_parameters(self):
        return self.P
        
    def default_L(self, λ):
        return 2 * np.ptp(λ)
    
    def design_matrix(self, λ):
        if self.L is None:
            L = self.default_L(λ)
        elif isinstance(self.L, (float, int)):
            L = self.L
        else:
            L = self.L(λ)
            
        scale = (np.pi * λ) / L
        A = np.ones((λ.size, self.P), dtype=float)
        for j, f in zip(range(1, self.P), cycle((np.sin, np.cos))):
            A[:, j] = f(scale * (j + (j % 2)))        
        return A
            
        scale = 2 * (np.pi / L)
        return np.vstack(
            [
                np.ones_like(λ).reshape((1, -1)),
                np.array(
                    [
                        [np.cos(o * scale * λ), np.sin(o * scale * λ)]
                        for o in range(1, self.deg + 1)
                    ]
                ).reshape((2 * self.deg, λ.size)),
            ]
        ).T


class Polynomial(ContinuumBasis):
    
    def __init__(self, deg=2):
        self.deg = deg
        return None
    
    @property
    def num_parameters(self):
        return self.deg + 1
    
    def design_matrix(self, λ):
        return np.vander(λ - np.mean(λ), self.deg + 1)
