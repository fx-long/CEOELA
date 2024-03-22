from typing import Optional
import os
import ast
import ioh
import copy
import numpy as np
import pandas as pd
from functools import partial
from itertools import product
from .utils import data_rescaling, runParallelFunction
from .compute_ela import compute_ela, bootstrap_ela
from .func_generator import func_generator
from .analyze_ela import analyze_ela



#%%
class ceoela_pipeline:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        lower_bound: list,
        upper_bound: list,
        normalize_x: Optional[bool] = True,
        normalize_x_lower: Optional[list] = None,
        normalize_x_upper: Optional[list] = None,
        problem_label: Optional[str] = 'problem',
        path_output: Optional[str] = None,
        bootstrap: Optional[bool] = True,
        bs_ratio: Optional[float] = 0.8,
        bs_repeat: Optional[int] = 2,
        list_fid: Optional[list] = [i+1 for i in range(24)],
        list_iid: Optional[list] = [1],
        genf_number: Optional[int] = 5,
        seed: Optional[int] = 42,
        n_jobs: Optional[int] = 1,
        verbose: Optional[bool] = True,
    ):
        """CEOELA

        Args:
            X (pd.DataFrame): 
                DoE X samples. Each row represents a sample and each column represents a design variable.
            y (pd.DataFrame): 
                DoE y samples. Each row represents a sample and each column represents a response function.
            lower_bound (list): 
                Lower bound of the design space.
            upper_bound (list): 
                Upper bound of the design space.
            normalize_x (Optional[bool], optional): 
                Whether to normalize the design space. Defaults to True.
            normalize_x_lower (Optional[list], optional): 
                Lower bound of design space after normalization. Used only when normalize_x is True. Defaults to None.
            normalize_x_upper (Optional[list], optional): 
                Upper bound of design space after normalization. Used only when normalize_x is True. Defaults to None.
            problem_label (Optional[str], optional): 
                Labelling of the problem instance. Defaults to 'problem'.
            path_output (Optional[str], optional): 
                Path to save output files. Defaults to None.
            bootstrap (Optional[bool], optional): 
                Whether to use bootstrapping when computing ELA features. Defaults to True.
            bs_ratio (Optional[float], optional): 
                Ratio of initial DoE sample size considered for bootstrapping. Used only when bootstrap is True. Defaults to 0.8.
            bs_repeat (Optional[int], optional): 
                Repetition of bootstrapping. Used only when bootstrap is True. Defaults to 2.
            list_fid (Optional[list], optional): 
                List of BBOB functions to be considered. Defaults to [i+1 for i in range(24)].
            list_iid (Optional[list], optional): 
                List of BBOB instances to be considered. Defaults to [1].
            genf_number (Optional[int], optional): 
                Number of feasible random functions to be generated. Defaults to 5.
            seed (Optional[int], optional): 
                Random seed. Defaults to 42.
            n_jobs (Optional[int], optional): 
                Number of parallel execution. Defaults to 1.
            verbose (Optional[bool], optional): 
                Verbosity. Defaults to True.
        """
        
        # basic information
        self.X = X
        self.y = y
        self.lower_bound: list = lower_bound
        self.upper_bound: list = upper_bound
        self.normalize_x: bool = normalize_x
        self.normalize_x_lower: list = normalize_x_lower
        self.normalize_x_upper: list = normalize_x_upper
        self.problem_label: str = problem_label
        
        # bootstrapping
        self.bootstrap: bool = bootstrap
        self.bs_ratio: float = bs_ratio
        self.bs_repeat: int = bs_repeat
        
        # BBOB functions
        self.list_fid: list = list_fid
        self.list_iid: list = list_iid
        
        # generated functions
        self.genf_number: int = genf_number
        
        # misc
        path_base = path_output if isinstance(path_output, str) else os.getcwd()
        self.path_output = os.path.join(path_base, 'results_ela', f'ela_{self.problem_label}')
        if not (os.path.isdir(self.path_output)):
            os.makedirs(self.path_output)
        self.seed: int = seed
        self.n_jobs: int = n_jobs
        self.verbose: bool = verbose
        
        # condition
        assert len(self.lower_bound) == len(self.upper_bound)
        assert len(self.X) == len(self.y)
        assert len(self.lower_bound) == self.X.shape[1]
        if (self.normalize_x):
            assert isinstance(self.normalize_x_lower, list)
            assert isinstance(self.normalize_x_upper, list)
            assert self.normalize_x_lower and self.normalize_x_upper
        
    #%%
    def preprocess(self):
        # drop duplicated sample points
        df_data = pd.concat([self.X, self.y], axis=1)
        df_data.drop_duplicates(subset=list(self.X.keys()), keep='first', inplace=True, ignore_index=True)
        self.X_filter = df_data[list(self.X.keys())]
        self.y_filter = df_data[list(self.y.keys())]
        if (self.verbose):
            print(f'[CEOELA] {len(self.X)-len(self.X_filter)} duplicated dropped. Final {len(self.X_filter)} sample points.')
        
        # re-scaling design variables
        self.X_normalize = copy.deepcopy(self.X_filter)
        if (self.normalize_x):
            for i_dv, dv in enumerate(self.X.keys()):
                bound_min = float(self.lower_bound[i_dv])
                bound_max = float(self.upper_bound[i_dv])
                target_min = float(self.normalize_x_lower[i_dv])
                target_max = float(self.normalize_x_upper[i_dv])
                self.X_normalize[dv] = data_rescaling(self.X_normalize[dv], bound_min, bound_max, target_min, target_max)
            if (self.verbose):
                print('[CEOELA] Doe samples X are re-scaled.')
    # END DEF
        
    #%%
    def __call__(self, ela_problem=True, ela_bbob=True, ela_genf=True):
        self.preprocess()
        X_ = np.array(self.X_normalize)
        dict_bs = {'bootstrap': self.bootstrap,
                   'lower_bound': self.lower_bound,
                   'upper_bound': self.upper_bound,
                   'normalize_y': True,
                   'bs_ratio': self.bs_ratio,
                   'bs_repeat': self.bs_repeat,
                   'bs_seed': self.seed,
                   'path_output': self.path_output}
        
        # problem instance
        if (ela_problem):
            list_y = list(self.y.keys())
            ela_ = partial(computeELA_problem_parallel, X=X_, y=self.y_filter, dict_bs=dict_bs)
            runParallelFunction(ela_, list_y, np=self.n_jobs)
        
        # bbob
        if (ela_bbob):
            list_bbob = list(product(self.list_fid, self.list_iid))
            ela_ = partial(computeELA_bbob_parallel, X=X_, dict_bs=dict_bs)
            runParallelFunction(ela_, list_bbob, np=self.n_jobs)
        
        # generated functions
        if (ela_genf):
            func_generator(X_,
                           f_number = self.genf_number,
                           f_seed = self.seed,
                           path_output = self.path_output,
                           verbose = True)()
            filepath = os.path.join(dict_bs['path_output'], 'results_rgf', 'rgf_genf.csv')
            df_func = pd.read_csv(filepath)
            ela_ = partial(computeELA_rgf_parallel, X=X_, ydata=df_func, dict_bs=dict_bs)
            runParallelFunction(ela_, df_func.index.tolist(), np=self.n_jobs) 
            
        if (self.verbose):
            print('[CEOELA] ELA done.')
    # END DEF
    
    def analyzeELA(self):
        analyze_ela(self.path_output, list(self.y.keys()), self.list_fid, self.list_iid, self.genf_number, label=self.problem_label)
# END CLASS

#%%
def computeELA(X_, y_, dict_bs, label=''):
    bootstrap = dict_bs['bootstrap']
    lower_bound = dict_bs['lower_bound']
    upper_bound = dict_bs['upper_bound']
    normalize_y = dict_bs['normalize_y']
    bs_ratio = dict_bs['bs_ratio']
    bs_repeat = dict_bs['bs_repeat']
    bs_seed = dict_bs['bs_seed']
    path_output = dict_bs['path_output']
    
    if (bootstrap):
        ela_ = bootstrap_ela(X_, y_, lower_bound=lower_bound, upper_bound=upper_bound, normalize_y=normalize_y,
                             bs_ratio=bs_ratio, bs_repeat=bs_repeat, bs_seed=bs_seed)
    else:
        ela_ = compute_ela(X_, y_, lower_bound=lower_bound, upper_bound=upper_bound, normalize_y=normalize_y)
    filepath = os.path.join(path_output, f'ela_{label}.csv')
    ela_.to_csv(filepath, index=False)
    print(f'[CEOELA] Features {label} computed.')
# END DEF

#%%
def computeELA_problem_parallel(problem, X, y, dict_bs):
    y_ = np.array(y[problem])
    try:
        computeELA(X, y_, dict_bs, label=f'{problem}')
    except Exception as e:
        print(e)
# END DEF

#%%
def computeELA_bbob_parallel(bbob, X, dict_bs):
    fid = bbob[0]
    iid = bbob[1]
    f = ioh.get_problem(fid, iid, X.shape[1])
    y = np.array(list(map(f, X)))
    try:
        computeELA(X, y, dict_bs, label=f'bbob_f{fid}_ins{iid}')
    except Exception as e:
        print(e)
# END DEF

#%%
def computeELA_rgf_parallel(ind, X, ydata, dict_bs):
    y = np.array(ast.literal_eval(ydata['y'].iloc[ind]))
    try:
        computeELA(X, y, dict_bs, label=f'rgf{ind+1}')
    except Exception as e:
        print(e)
# END DEF