
import os
import numpy as np
import pandas as pd
from .rfg import rfg_generate_tree as genTree
from .rfg import rfg_tree2func as genTree2func


#%%
class func_generator:
    def __init__(self,
                 doe_x,
                 max_eval: int = None,
                 f_number: int = 1,
                 f_seed: int = 0,
                 path_output: str = '',
                 verbose: bool = True
                 ):
        self.doe_x = doe_x
        self.max_eval = max_eval if max_eval else np.inf
        self.f_number = f_number
        self.f_seed = f_seed
        path_base = path_output if path_output else os.getcwd()
        self.path_output = os.path.join(path_base, 'results_rfg')
        self.verbose = verbose
        if not (os.path.isdir(self.path_output)):
            os.makedirs(self.path_output)
    
    #%%
    def __call__(self):
        np.seterr(all='ignore')
        np.random.seed(self.f_seed)
        i_run = 0
        i_invalid = 0
        i_bad_y = 0
        self.df_func = pd.DataFrame()
        while (len(self.df_func) < self.f_number) and (i_run < self.max_eval):
            i_run += 1
            tree = genTree.generate_tree(8, 12)
            f_, str_f = genTree2func.tree2func(tree, 1, self.doe_x.shape[1])
            
            # skip if function generation failed
            try:
                list_y = []
                for i in range(len(self.doe_x)):
                    y = f_(self.doe_x[i])
                    list_y.append(np.mean(y))
                y = np.array(list_y)
                y[abs(y) < 1e-20] = 0.0
            except:
                i_invalid += 1
                continue
            
            # skip if objective values are invalid
            if (np.isnan(y).any() or np.isinf(y).any() or np.var(y)<1e-10):
                i_bad_y += 1
                continue
            
            df_f = pd.DataFrame.from_dict({'str_f': [str_f], 'y': [str(list(y))], 'label': f'rfg{len(self.df_func)+1}'})
            self.df_func = pd.concat([self.df_func, df_f], axis=0, ignore_index=True)
        filepath = os.path.join(self.path_output, 'rfg_genf.csv')
        self.df_func.to_csv(filepath, index=False)
        if (self.verbose):
            print(f'[RFG] Done. Valid: {len(self.df_func)}; Invalid: {i_invalid}; Bad y-values: {i_bad_y}; Total runs: {i_run}; Max eval: {self.max_eval}; Success rate: {round(len(self.df_func)/i_run, 2)}')
# END DEF