

import ioh
import warnings
import numpy as np
import pandas as pd
from utils.ceoela_pipeline import ceoela_pipeline
from utils.sampling import sampling


#%%
def setup():
    dim = 2
    X = sampling('sobol',
                 n=200,
                 lower_bound=[-5.0]*dim,
                 upper_bound=[5.0]*dim,
                 round_off=False,
                 random_seed=0,
                 verbose=True)()
    f = ioh.get_problem(1, 1, X.shape[1])
    y = np.array(list(map(f, X)))
      
    ela_pipeline = ceoela_pipeline(pd.DataFrame(X, columns=[f'DV{i+1}' for i in range(dim)]),
                                   pd.DataFrame(y, columns=['y']),
                                   lower_bound = [-5.0]*dim,
                                   upper_bound = [5.0]*dim,
                                   normalize_x = True,
                                   normalize_x_lower = [-5.0]*dim,
                                   normalize_x_upper = [5.0]*dim,
                                   normalize_y = True,
                                   problem_label = 'example_bbob',
                                   path_output = '',
                                   bootstrap = True,
                                   bs_ratio = 0.8,
                                   bs_repeat = 2,
                                   bs_seed = 1,
                                   list_fid = [i+1 for i in range(1)],
                                   list_iid = [i+1 for i in range(5)],
                                   genf_number = 10,
                                   genf_eval_max = None,
                                   genf_seed = 42,
                                   np_ela = 1,
                                   verbose = True,
                                   )
    ela_pipeline(ela_problem=True, ela_bbob=True, ela_genf=True)
# END

#%%
def main():
    setup()
# END DEF

#%%
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    main()
# END IF