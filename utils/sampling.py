
import numpy as np
from scipy.stats import qmc


#%%
##################################
'''
# Methods
'''
##################################

#%%
# Random sampling (uniform distribution)
def Random_sampling(dimension, sample_size, random_seed):
    np.random.seed(random_seed)
    xdoe = np.random.rand(sample_size, dimension)
    return xdoe

# Sobol sequence
def Sobol_sampling(dimension, sample_size, random_seed):
    sampler = qmc.Sobol(d=dimension, scramble=False, seed=random_seed)
    # sample = sampler.random_base2(m=3)
    sample = sampler.random(n=sample_size)
    return sample

# Halton sequence
def Halton_sampling(dimension, sample_size, random_seed):
    sampler = qmc.Halton(d=dimension, scramble=False, seed=random_seed)
    sample = sampler.random(n=sample_size)
    return sample

# Latin Hypercube
def LHS_sampling(dimension, sample_size, random_seed):
    sampler = qmc.LatinHypercube(d=dimension, centered=False, optimization=None, strength=1, seed=random_seed)
    sample = sampler.random(n=sample_size)
    return sample


#%%
##################################
'''
# Sampling
'''
##################################

#%%
# Initialize a class for DOE sampling pipeline
class sampling:
    def __init__(self,
                 method: str,
                 n: int,
                 lower_bound: list,
                 upper_bound: list,
                 round_off: int or bool = False,
                 random_seed: int = 0,
                 verbose: bool = True
                 ):
        """
        The base class for DOE sampling.
        ----------
        Parameters
        ----------
        method: str
            Sampling method to create DOE, e.g., 'sobol', 'lhs', 'random', etc.
        n: int
            Number of sample points in design space.
        lower_bound: list
            Lower boundary of design space.
        upper_bound: list
            Upper boundary of design space.
        round_off
            Round off the DOE to nearest digit, by default False.
        random_seed: int, optional
            Random seed to initialize generator, by default seed 0.
        verbose: bool, optional
            The verbosity, by default True.
        """
        self.method: str = method
        self.n: int = n
        self.lower_bound: list = lower_bound
        self.upper_bound: list = upper_bound
        self.round_off: int or bool = round_off
        self.random_seed: int = random_seed
        self.verbose: bool = verbose
        
        # initialize sampling methods
        self.dict_sampling = {'sobol': Sobol_sampling,
                              'halton': Halton_sampling,
                              'lhs': LHS_sampling,
                              'random': Random_sampling}
        if (self.method not in self.dict_sampling.keys()):
            raise ValueError(f'{self.method} is not defined, use only {list(self.dict_sampling.keys())}.')
        if (self.n <= 0):
            raise ValueError('Sample size must be positive and non-zero.')
        if (len(self.lower_bound) != len(self.upper_bound)):
            raise ValueError(f'Lower and upper bound must of same length: Lower {len(self.lower_bound)}; Upper {len(self.upper_bound)}.')
        for lower, upper in zip(self.lower_bound, self.upper_bound):
            assert upper > lower
        self.dimension = len(self.lower_bound)
        
    #%%
    def __call__(self):
        doe_base = self.dict_sampling[self.method](self.dimension, self.n, self.random_seed)
        doe_rescale = qmc.scale(doe_base, self.lower_bound, self.upper_bound, reverse=False)
        if (self.round_off):
            doe_rescale = doe_rescale.round(self.round_off)
        return doe_rescale
    # END DEF
# END CLASS