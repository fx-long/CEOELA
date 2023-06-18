
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


#%%
# Initialize CEOELA pipeline
class ceoela_pipeline:
    def __init__(self,
                 X,
                 y,
                 lower_bound: list,
                 upper_bound: list,
                 normalize_x: bool = True,
                 normalize_x_lower: list = [],
                 normalize_x_upper: list = [],
                 normalize_y: bool = True,
                 problem_label: str = 'problem',
                 path_output: str = '',
                 bootstrap: bool = True,
                 bs_ratio: float = 0.8,
                 bs_repeat: int = 2,
                 bs_seed: int = 42,
                 list_fid: list = [i+1 for i in range(24)],
                 list_iid: list = [1],
                 genf_number: int = 1,
                 genf_eval_max: int = None,
                 genf_seed: int = 42,
                 np_ela: int = 1,
                 verbose: bool = True,
                 ):
        # basic information
        self.X = X
        self.y = y
        self.lower_bound: list = lower_bound
        self.upper_bound: list = upper_bound
        self.normalize_x: bool = normalize_x
        self.normalize_x_lower: list = normalize_x_lower
        self.normalize_x_upper: list = normalize_x_upper
        self.normalize_y: bool = normalize_y
        self.problem_label: str = problem_label
        
        # bootstrapping
        self.bootstrap: bool = bootstrap
        self.bs_ratio: float = bs_ratio
        self.bs_repeat: int = bs_repeat
        self.bs_seed: int = bs_seed
        
        # BBOB functions
        self.list_fid: list = list_fid
        self.list_iid: list = list_iid
        
        # generated functions
        self.genf_number: int = genf_number
        self.genf_eval_max: int = genf_eval_max if genf_eval_max else np.inf
        self.genf_seed: int = genf_seed
        
        # misc
        path_base = path_output if path_output else os.getcwd()
        self.path_output = os.path.join(path_base, 'results_ela', f'ela_{self.problem_label}')
        if not (os.path.isdir(self.path_output)):
            os.makedirs(self.path_output)
        self.np_ela: int = np_ela
        self.verbose: bool = verbose
        
        # condition
        assert len(self.lower_bound) == len(self.upper_bound)
        assert len(self.X) == len(self.y)
        assert len(self.lower_bound) == self.X.shape[1]
        if (self.normalize_x):
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
                orig_min = float(self.lower_bound[i_dv])
                orig_max = float(self.upper_bound[i_dv])
                target_min = float(self.normalize_x_lower[i_dv])
                target_max = float(self.normalize_x_upper[i_dv])
                self.X_normalize[dv] = data_rescaling(self.X_normalize[dv], orig_min, orig_max, target_min, target_max)
            if (self.verbose):
                print('[CEOELA] Doe samples X are re-scaled.')
    # END DEF
        
    #%%
    def __call__(self, ela_problem=True, ela_bbob=True, ela_genf=True):
        self.preprocess()
        X_ = np.array(self.X_normalize)
        dict_bs = {'bootstrap': self.bootstrap,
                   'lower_bound': self.normalize_x_lower,
                   'upper_bound': self.normalize_x_upper,
                   'normalize_y': self.normalize_y,
                   'bs_ratio': self.bs_ratio,
                   'bs_repeat': self.bs_repeat,
                   'bs_seed': self.bs_seed,
                   'path_output': self.path_output}
        
        # problem instance
        if (ela_problem):
            list_y = list(self.y.keys())
            ela_ = partial(computeELA_problem_parallel, X=X_, y=self.y_filter, dict_bs=dict_bs)
            runParallelFunction(ela_, list_y, np=self.np_ela)
        
        # bbob
        if (ela_bbob):
            list_bbob = list(product(self.list_fid, self.list_iid))
            ela_ = partial(computeELA_bbob_parallel, X=X_, dict_bs=dict_bs)
            runParallelFunction(ela_, list_bbob, np=self.np_ela)
        
        # generated functions
        if (ela_genf):
            func_generator(X_,
                           max_eval = self.genf_eval_max,
                           f_number = self.genf_number,
                           f_seed = self.genf_seed,
                           path_output = self.path_output,
                           verbose = True)()
            filepath = os.path.join(dict_bs['path_output'], 'results_rfg', 'rfg_genf.csv')
            df_func = pd.read_csv(filepath)
            ela_ = partial(computeELA_rfg_parallel, X=X_, ydata=df_func, dict_bs=dict_bs)
            runParallelFunction(ela_, df_func.index.tolist(), np=self.np_ela) 
            
        if (self.verbose):
            print('[CEOELA] ELA done.')
    # END DEF
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
    
    if (normalize_y):
        y_ = (y_-min(y_))/(max(y_)-min(y_))
    if (bootstrap):
        ela_ = bootstrap_ela(X_, y_, lower_bound=lower_bound, upper_bound=upper_bound,
                             bs_ratio=bs_ratio, bs_repeat=bs_repeat, bs_seed=bs_seed)
    else:
        ela_ = compute_ela(X_, y_, lower_bound=lower_bound, upper_bound=upper_bound)
    filepath = os.path.join(path_output, f'ela_{label}.csv')
    ela_.to_csv(filepath, index=False)
    print(f'[CEOELA] Features {label} computed.')
# END DEF

#%%
def computeELA_problem_parallel(problem, X, y, dict_bs):
    y_ = np.array(y[problem])
    computeELA(X, y_, dict_bs, label=f'{problem}')
# END DEF

#%%
def computeELA_bbob_parallel(bbob, X, dict_bs):
    fid = bbob[0]
    iid = bbob[1]
    f = ioh.get_problem(fid, iid, X.shape[1])
    y = np.array(list(map(f, X)))
    computeELA(X, y, dict_bs, label=f'bbob_f{fid}_ins{iid}')
# END DEF

#%%
def computeELA_rfg_parallel(ind, X, ydata, dict_bs):
    y = np.array(ast.literal_eval(ydata['y'].iloc[ind]))
    computeELA(X, y, dict_bs, label=f'rfg{ind+1}')
# END DEF