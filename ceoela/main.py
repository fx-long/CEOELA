import os
import warnings
import pandas as pd
from ceoela import ceoela_pipeline, excel2doe


#%%
def main():
    # read doe data
    filepath = os.path.join(os.getcwd(), 'DOE_example.xlsx')
    X, y, lb, ub = excel2doe(filepath)
    dim = X.shape[1]
    
    # initialize CEOELA pipeline
    ela_pipeline = ceoela_pipeline(
        X,
        y,
        lower_bound = lb,
        upper_bound = ub,
        normalize_x = True,
        normalize_x_lower = [-5.0]*dim,
        normalize_x_upper = [5.0]*dim,
        problem_label = 'example',
        path_output = '',
        bootstrap = True,
        bs_ratio = 0.8,
        bs_repeat = 2,
        list_fid = [i+1 for i in range(10)],
        list_iid = [i+1 for i in range(1)],
        genf_number = 5,
        seed = 1,
        n_jobs = 1,
        verbose = True,
    )
    # compute ELA features
    ela_pipeline(ela_problem=True, ela_bbob=True, ela_genf=True)
    
    # analyze ELA features
    ela_pipeline.analyzeELA()
# END DEF


#%%
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    main()
# END IF