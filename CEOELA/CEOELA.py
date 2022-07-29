# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 18:36:49 2022

@author: Q521100
"""


import os
import copy
import random
import math
import json
import shutil
import pandas as pd
import numpy as np
from itertools import product

from sklearn.utils import resample
from .utils import (get_script_dir, data_rescaling, readFile2Dict, dict2DF, create_linkageMat, plot_dendrogram, create_Rscript, 
                    get_similar_func, write_similarF, write_bs_AF)
from . import bbobbenchmarks as bbob
from .modulesRandFunc import generate_tree as genTree
from .modulesRandFunc import generate_tree2exp as genTree2exp
from .modulesRandFunc import generate_exp2fun as genExp2fun




__authors__ = ["Fu Xing Long"]






#%%
##################################
'''
# ELA pipeline
'''
##################################

#%%
# Initialize a class for CEOELA pipeline
class CEOELA_pipeline:
    def __init__(self,
                 filepath_excel: str,
                 list_sheetname: list = [],
                 problem_label: str = '',
                 filepath_save: str = '',
                 bootstrap: bool = True,
                 bootstrap_size: float or int = 0.8,
                 bootstrap_repeat: int = 2,
                 bootstrap_seed: int = 0,
                 BBOB_func: list = ['F'+str(i+1) for i in range(24)],
                 BBOB_instance: list = [1],
                 BBOB_seed: int = 0,
                 AF_number: int = 1,
                 AF_seed: int = 0,
                 np_ela: int = 1,
                 os_system: str = 'windows',
                 purge: bool = False,
                 verbose: bool = True,
                 ):
        """
        The base class for ELA pipeline.
        ----------
        Parameters
        ----------
        filepath_excel: str
            Excel file to be read, containing x- and y-data of crash instance.
            The Excel file must have the following sheets (with exact name):
            'KPI': Defining the design and output variables and their respective new labels.
            'Bounds': Defining the upper and lower boundary of design variables.
            At least 1 sheet for DOE data.
            Use the 'DOE_template.xlsx' template as referance.
        list_sheetname: list, optional
            Sheet inside Excel file to be read, by default None.
        problem_label: str, optional
            Re-name the problem instance, by default None.
        filepath_save: str, optional
            Path to save the output files, by default None.
        bootstrap: bool, optional
            Compute the ELA features in bootstrapping manner, by default True.
        bootstrap_size: float or int, optional
            Bootstrap size, by default 80% of the original sample size.
        bootstrap_repeat: int, optional
            Bootstrap repitition, by default 2 repititions.
        bootstrap_seed: int, optional
            Random seed to initialize boot-strapping, by default seed 0.
        BBOB_func: list, optional
            List of noiseless BBOB functions to be considered, by default F1 to F24.
        BBOB_instance: list, optional
            List of BBOB instances to be considered, by default instance 1.
        BBOB_seed: int, optional
            Random seed to initialize BBOB, by default seed 0.
        AF_number: int, optional
            Number of AF to be generated, by default 1 function.
        AF_seed: int, optional
            Random seed to initialize AF generator, by default seed 0.
        np_ela: int, optional
            Number of processor for multiprocessing of ELA computation, by default 1. (works only in Lunix)
        os_system: str, optional
            Operating system, either 'windows' or 'linux', by default windows.
        purge: bool, optional
            Remove previous result directory, by default False.
        verbose: bool, optional
            The verbosity, by default True.
        """
        
        # basic information
        self.filepath_excel: str = filepath_excel
        self.list_sheetname: list = list_sheetname
        self.problem_label: str = problem_label if problem_label else 'Problem'
        self.filepath_save: str = filepath_save
        self.purge: bool = purge
        
        # problem instance
        self.bootstrap: bool = bootstrap
        self.bootstrap_size: float or int = bootstrap_size
        self.bootstrap_repeat: int = bootstrap_repeat
        self.bootstrap_seed: int = bootstrap_seed
        
        # BBOB functions
        self.BBOB_func: list = BBOB_func
        self.BBOB_instance: list = BBOB_instance
        self.BBOB_seed: int = BBOB_seed
        
        # artificial functions
        self.AF_number: int = AF_number
        self.AF_seed: int = AF_seed
        
        # misc
        self.np_ela: int = np_ela
        self.path_dir_base = get_script_dir(follow_symlinks=True, directory='parent')
        self.os_system: str = os_system
        self.verbose: bool = verbose
        
        #%%
        # read excel file
        if (os.path.isfile(self.filepath_excel)):
            # check sheet names
            list_sheetname_excel = pd.ExcelFile(self.filepath_excel).sheet_names
            if (('KPI' not in list_sheetname_excel) or ('Bounds' not in list_sheetname_excel)):
                raise ValueError('Excel sheet KPI and/or Bounds is missing.')
            if (len(list_sheetname_excel) <= 2):
                raise ValueError('Excel sheet with DOE data is missing.')
            # read excel file
            if (self.list_sheetname):
                self.list_sheetname = ['KPI', 'Bounds'] + self.list_sheetname
            else:
                self.list_sheetname = list_sheetname_excel
            if (verbose):
                print(f'[CEOELA] Following sheets are included {self.list_sheetname}.')
            dict_base = readFile2Dict(self.filepath_excel, self.list_sheetname, list_sheet_type='select', header=0)
        else:
            raise ValueError(f'Excel file {self.filepath_excel} is missing.')
        # END IF
        dict_main = copy.deepcopy(dict_base['excel'])
        self.list_input = list(dict_main['KPI']['input'].dropna())
        self.list_input_rename = list(dict_main['KPI']['input_rename'].dropna())
        self.list_output = list(dict_main['KPI']['output'].dropna())
        self.list_output_rename = list(dict_main['KPI']['output_rename'].dropna())
        self.list_total = self.list_input + self.list_output
        self.df_boundary = dict_main['Bounds']
        
        # check input name
        if (not len(self.list_input_rename)):
            self.list_input_rename = self.list_input
            if (self.verbose):
                print('[CEOELA] No re-naming for input variables.')
        elif (len(self.list_input) != len(self.list_input_rename)):
            raise ValueError('Variable list_input and list_input_rename must have same length.')
        # END IF
            
        # check output name
        if (not len(self.list_output_rename)):
            self.list_output_rename = self.list_output
            if (self.verbose):
                print('[CEOELA] No re-naming for output responses.')
        elif (len(self.list_output) != len(self.list_output_rename)):
            raise ValueError('Variable list_output and list_output_rename must have same length.')
        # END IF
        
        # check boundary values
        if (self.df_boundary['lower'].isnull().any()):
            self.df_boundary['lower'].fillna(0.0, inplace=True)
            if (verbose):
                print('[CEOELA] Missing lower boundary values are replaced with default 0.')
        if (self.df_boundary['upper'].isnull().any()):
            self.df_boundary['upper'].fillna(100.0, inplace=True)
            if (verbose):
                print('[CEOELA] Missing upper boundary values are replaced with default 100.')
            # END IF
        # END IF
        
        # create input dataframe
        dict_main.pop('KPI', None)
        dict_main.pop('Bounds', None)
        df_data_main = pd.DataFrame()
        for key in dict_main.keys():
            df_key = copy.deepcopy(dict_main[key].loc[:, dict_main[key].columns.isin(self.list_total)]) # ignore missing column keys
            df_data_main = pd.concat([df_data_main, df_key], axis=0, ignore_index=True)
        # END FOR
        self.df_data_main = df_data_main
        
        # create folder for results
        if (self.filepath_save):
            self.filepath_save = os.path.join(self.filepath_save, 'CEOELA_results', self.problem_label)
        else:
            self.filepath_save = os.path.join(self.path_dir_base, 'CEOELA_results', self.problem_label)
        if (self.purge and os.path.isdir(self.filepath_save)):
            shutil.rmtree(self.filepath_save)
        if not (os.path.isdir(self.filepath_save)):
            os.makedirs(self.filepath_save)
        # END IF
        
        # check os system
        if not ((self.os_system=='windows') or (self.os_system=='linux')):
            raise ValueError(f'Operating system {self.os_system} is undefined. Use only \'windows\' or \'linux\'.')
        # END IF
            
        # check boot-strap inputs
        if (self.bootstrap):
            if (type(self.bootstrap_size) is float):
                if ((self.bootstrap_size <= 0.0) or (self.bootstrap_size >= 1.0)):
                    raise ValueError('Boostrap size float must be between 0 and 1.')
                # END IF
            else:
                if (self.bootstrap_size <= 0):
                    raise ValueError('Boostrap size int must be non-zero and positive.')
                if (self.bootstrap_size >= len(self.df_data_main)):
                    raise ValueError('Boostrap size int must be smaller than total sample size.')
                # END IF
            # END IF
        # END IF
        
        # check number of processor
        if (self.os_system=='windows'):
            self.np_ela = 1
        if (self.np_ela < 1):
            raise ValueError('Number of processor must be at least 1.')
        # END IF
        
        # set-up R-script
        filepath_base = os.path.join(self.path_dir_base, 'CEOELA', 'CEOELA_computeELA_base.R')
        filepath_new = os.path.join(self.path_dir_base, 'CEOELA_computeELA.R')
        if not (os.path.isfile(filepath_base)):
            raise ValueError(f'R-script {filepath_base} is missing.')
        if (os.path.isfile(filepath_new)):
            os.remove(filepath_new)
        create_Rscript(filepath_base, filepath_new, os_system=self.os_system)
        if (self.verbose):
            print('[CEOELA] R-script is created.')
        # END IF
        
        #%%
        if (self.verbose):
            print('[CEOELA] ELA pipeline is initialized.')
        # END IF
    # END DEF     
        
        
        
        
        
    #%%
    ##################################
    '''
    # Data pre-processing
    '''
    ##################################
    
    #%%
    def DataPreProcess(self):
        """
        Pre-processing the input data, e.g. filtering, re-scaling, boot-strapping etc.
        """
        # remove failed FE simulations
        self.df_data_filter = copy.deepcopy(self.df_data_main)
        self.df_data_filter.fillna('EXPRESSION_ERROR', inplace=True)
        for i_output in self.list_output:
            self.df_data_filter = self.df_data_filter[self.df_data_filter[i_output] != 'EXPRESSION_ERROR']
        # END FOR
        self.df_data_filter = self.df_data_filter.astype(float)
        self.df_data_filter.reset_index(drop=True, inplace=True)
        if (self.verbose):
            print(f'[CEOELA] Total {len(self.df_data_main)-len(self.df_data_filter)} failed FE simulations are filtered.')
        # END IF
        
        # drop duplicated sample points
        row_dups = self.df_data_filter.duplicated(subset=self.list_input, keep='first')
        if (row_dups.any()):
            initial_size = len(self.df_data_filter)
            self.df_data_filter.drop_duplicates(subset=self.list_input, keep='first', inplace=True)
            self.df_data_filter.reset_index(drop=True, inplace=True)
            if (self.verbose):
                print(f'[CEOELA] {initial_size-len(self.df_data_filter)} duplicated dropped. Final {len(self.df_data_filter)} sample points.')
            # END IF
        else:
            if (self.verbose):
                print(f'[CEOELA] No duplicated. Final {len(self.df_data_filter)} sample points.')
            # END IF
        # END IF
        
        # re-name inputs
        for input_original, input_rename in zip(self.list_input, self.list_input_rename):
            self.df_data_filter.rename(columns={input_original: input_rename}, inplace=True)
        # re-name responses
        for output, output_rename in zip(self.list_output, self.list_output_rename):
            self.df_data_filter.rename(columns={output: output_rename}, inplace=True)
        if (self.verbose):
            print(f'[CEOELA] Total {len(self.list_input_rename)} input variables re-named to {self.list_input_rename}.')
            print(f'[CEOELA] Total {len(self.list_output_rename)} output responses re-named to {self.list_output_rename}.')
        self.df_data_filter = self.df_data_filter.astype(float)
        
        #%%
        # re-scaling design variables
        self.df_data_rescale = copy.deepcopy(self.df_data_filter)
        for dv in self.list_input:
            df_row_temp = self.df_boundary.loc[self.df_boundary['design variable'] == dv]
            bound_min = float(df_row_temp['lower'].iloc[-1])
            bound_max = float(df_row_temp['upper'].iloc[-1])
            target_min = -5.0
            target_max = 5.0
            self.df_data_rescale[dv] = data_rescaling(self.df_data_rescale[dv], bound_min, bound_max, target_min, target_max)
        # END FOR
        
        # check x-data range after re-scaling
        if (max(list(self.df_data_rescale[self.list_input].max())) > 5.0) or (min(list(self.df_data_rescale[self.list_input].min())) < -5.0):
            raise ValueError('[CEOELA] Re-scaled x-data are not within [-5,5].')
        else:
            if (self.verbose):
                print('[CEOELA] Re-scaling crash design space to [-5,5] done.')
            # END IF
        # END IF
        self.array_x_original = np.array(self.df_data_filter[self.list_input])
        self.array_x_rescale = np.array(self.df_data_rescale[self.list_input])
        
        #%%
        # generate BBOB functions
        dict_bbob = {}
        for i_ins in self.BBOB_instance:
            df_BBOB = copy.deepcopy(self.df_data_rescale[self.list_input])
            for i_id in self.BBOB_func:
                np.random.seed(self.BBOB_seed)
                func_bbob = getattr(bbob, i_id)(i_ins)
                array_bbob_y = func_bbob(self.array_x_rescale)
                df_BBOB[i_id] = list(array_bbob_y)
            # END FOR
            dict_bbob[str(i_ins)] = df_BBOB
        # END FOR
        self.dict_BBOB = dict_bbob
        if (self.verbose):
            print(f'[CEOELA] BBOB functions {self.BBOB_func} of instance {self.BBOB_instance} are generated.')
        # END IF
        
        #%%
        # generate artificial functions
        np.seterr(all='ignore')
        i_AF_seed = self.AF_seed
        i_run = 0
        i_fail = 0
        i_invalid = 0
        self.dict_AF_func = {}
        self.df_AF = copy.deepcopy(self.df_data_filter[self.list_input])
        dict_AF = {}
        
        while (len(self.dict_AF_func) < self.AF_number):
            i_run += 1
            random.seed(i_AF_seed)
            
            # create an artificial function
            tree = genTree.generate_tree(8, 12)
            exp = genTree2exp.generate_tree2exp(tree)
            fun = genExp2fun.generate_exp2fun(exp, len(self.array_x_original), self.array_x_original.shape[1])
            
            # skip if function generation failed
            try:
                array_x = self.array_x_original
                array_y = eval(fun)
            except:
                i_fail += 1
                i_AF_seed += 1
                continue
            # END TRY
            
            # skip if function values are invalid
            # missing value or infinity or extremely low and high value or small variance or ydoe is multidimensional
            if (np.isnan(array_y).any() or np.isinf(array_y).any() or np.any(abs(array_y)<1e-8) or np.any(abs(array_y)>1e8) 
                or np.var(array_y)<1.0 or array_y.ndim!=1):
                i_invalid += 1
                i_AF_seed += 1
                continue
            # END IF
            
            name_temp_base = 'AF_' + str(len(self.dict_AF_func)+1)
            self.dict_AF_func[name_temp_base] = [fun]
            dict_AF[name_temp_base] = list(array_y)
            i_AF_seed += 1
        # END WHILE
        df_AF_temp = pd.DataFrame.from_dict(dict_AF)
        self.df_AF = pd.concat([self.df_AF, df_AF_temp], axis=1)
        if (self.verbose):
            print(f'[CEOELA] AF are generated. Valid: {len(self.dict_AF_func)}; Fail: {i_fail}; Invalid: {i_invalid}; Total runs: {i_run}')
        # END IF
        
        #%%
        # bootstrapping
        self.dict_bs_crash_original = {}
        self.dict_bs_crash_rescale = {}
        self.dict_bs_AF = {}
        if (self.bootstrap):
            if (type(self.bootstrap_size) is int):
                num_sample = self.bootstrap_size
            else:
                num_sample = int(math.ceil(len(self.df_data_filter) * self.bootstrap_size))
            # END IF
            for i_bs in range(self.bootstrap_repeat):
                i_bs_seed = i_bs + self.bootstrap_seed
                # crash original
                df_bs_crash_orig = resample(self.df_data_filter, replace=False, n_samples=num_sample, random_state=i_bs_seed, stratify=None)
                df_bs_crash_orig.reset_index(drop=True, inplace=True)
                self.dict_bs_crash_original[str(i_bs+1)] = df_bs_crash_orig
                
                # crash rescale
                df_bs_crash_rescale = resample(self.df_data_rescale, replace=False, n_samples=num_sample, random_state=i_bs_seed, stratify=None)
                df_bs_crash_rescale.reset_index(drop=True, inplace=True)
                self.dict_bs_crash_rescale[str(i_bs+1)] = df_bs_crash_rescale
                
                # AF
                df_bs_AF = resample(self.df_AF, replace=False, n_samples=num_sample, random_state=i_bs_seed, stratify=None)
                df_bs_AF.reset_index(drop=True, inplace=True)
                self.dict_bs_AF[str(i_bs+1)] = df_bs_AF
            # END FOR
            if (self.verbose):
                print(f'[CEOELA] Boot-strapping size {self.bootstrap_size} and repitition {self.bootstrap_repeat} done.')
            # END IF
        else:
            self.dict_bs_crash_original['full'] = self.df_data_filter
            self.dict_bs_crash_rescale['full'] = self.df_data_rescale
            self.dict_bs_AF['full'] = self.df_AF
            if (self.verbose):
                print('[CEOELA] Compute ELA with full data. Boot-strapping is deactivated.')
            # END IF
        # END IF            
        
        #%%
        # save results
        # BBOB functions
        filename = self.problem_label + '_BBOB.xlsx'
        filepath_out = os.path.join(self.filepath_save, filename)
        with pd.ExcelWriter(filepath_out) as writer:
            for i_sheet in self.dict_BBOB.keys():
                self.dict_BBOB[i_sheet].to_excel(writer, sheet_name='ins_'+i_sheet, index=False)
            # END FOR
        # END WITH
        
        if (self.bootstrap):
            # problem instance (original)
            filename = self.problem_label + '_original.xlsx'
            filepath_out = os.path.join(self.filepath_save, filename)
            with pd.ExcelWriter(filepath_out) as writer:
                for i_sheet in self.dict_bs_crash_original.keys():
                    self.dict_bs_crash_original[i_sheet].to_excel(writer, sheet_name='bs_'+i_sheet, index=False)
            
            # problem instance (re-scale)
            filename = self.problem_label + '_rescale.xlsx'
            filepath_out = os.path.join(self.filepath_save, filename)
            with pd.ExcelWriter(filepath_out) as writer:
                for i_sheet in self.dict_bs_crash_rescale.keys():
                    self.dict_bs_crash_rescale[i_sheet].to_excel(writer, sheet_name='bs_'+i_sheet, index=False)
            
            # artificial functions    
            # TODO: parallel computing
            for i_bs in self.dict_bs_AF.keys():
                write_bs_AF(self.filepath_save, self.problem_label, i_bs, self.dict_bs_AF)
            # END FOR  
        else:
            # problem instance (original)
            filename = self.problem_label + '_original.xlsx'
            filepath_out = os.path.join(self.filepath_save, filename)
            with pd.ExcelWriter(filepath_out) as writer:
                self.dict_bs_crash_original['full'].to_excel(writer, sheet_name='full', index=False)
            
            # problem instance (re-scale)
            filename = self.problem_label + '_rescale.xlsx'
            filepath_out = os.path.join(self.filepath_save, filename)
            with pd.ExcelWriter(filepath_out) as writer:
                self.dict_bs_crash_rescale['full'].to_excel(writer, sheet_name='full', index=False)
            
            # artificial functions
            filename = self.problem_label + '_AF.xlsx'
            filepath_out = os.path.join(self.filepath_save, filename)
            with pd.ExcelWriter(filepath_out) as writer:
                self.dict_bs_AF['full'].to_excel(writer, sheet_name='full', index=False)
        # END IF            
        
        # save AF expression
        self.df_AF_func = pd.DataFrame.from_dict(self.dict_AF_func)
        self.df_AF_func = self.df_AF_func.T
        self.df_AF_func.reset_index(drop=False, inplace=True)
        self.df_AF_func.rename(columns={'index': 'label', 0: 'func'}, inplace=True)
        filename = self.problem_label + '_AF_func.xlsx'
        filepath_out = os.path.join(self.filepath_save, filename)
        self.df_AF_func.to_excel(filepath_out, sheet_name='func', index=False)
        
        if (self.verbose):
            print(f'[CEOELA] Results are saved to {self.filepath_save}.')
        if (self.verbose):
            print('[CEOELA] Data pre-processing done.')
        # END IF
    # END DEF
    
    
    
    #%%
    ##################################
    '''
    # Computation of ELA features
    '''
    ##################################
    
    #%%
    def ComputeELA(self, 
                   ELA_problem: bool = True, 
                   ELA_BBOB: bool = True, 
                   ELA_AF: bool = True):
        """
        Computation of ELA features.
        ----------
        Parameters
        ----------
        ELA_problem: bool, optional
            Compute ELA features on problem instances, by default True.
        ELA_BBOB: bool, optional
            Compute ELA features on BBOB functions, by default True.
        ELA_AF: bool, optional
            Compute ELA features on artificial functions, by default True.
        """
        # initialize
        self.ELA_problem: bool = ELA_problem
        self.ELA_BBOB: bool = ELA_BBOB
        self.ELA_AF: bool = ELA_AF
        
        # export meta-data for communication between python and R
        dict_meta = {}
        dict_meta['path_dir_base'] = self.path_dir_base
        dict_meta['problem_label'] = self.problem_label
        dict_meta['filepath_save'] = self.filepath_save
        dict_meta['list_input'] = self.list_input
        dict_meta['list_output'] = self.list_output_rename
        dict_meta['bootstrap'] = self.bootstrap
        dict_meta['bootstrap_size'] = self.bootstrap_size
        dict_meta['bootstrap_repeat'] = ['bs_'+str(bs+1) for bs in range(self.bootstrap_repeat)]
        dict_meta['BBOB_func'] = self.BBOB_func
        dict_meta['BBOB_instance'] = ['ins_'+str(ins) for ins in self.BBOB_instance]
        dict_meta['AF_number'] = ['AF_'+str(AF_num+1) for AF_num in range(self.AF_number)]
        dict_meta['ELA_problem'] = self.ELA_problem
        dict_meta['ELA_BBOB'] = self.ELA_BBOB
        dict_meta['ELA_AF'] = self.ELA_AF
        dict_meta['np_ela'] = self.np_ela
        dict_meta['os_system'] = self.os_system
                         
        filename_meta_base = 'CEOELA_metadata.json'
        filepath_meta = os.path.join(self.path_dir_base, filename_meta_base)
        with open(filepath_meta, "w") as outfile: 
            json.dump(dict_meta, outfile)
        # END WITH
        
        # execute R script
        if (self.os_system=='windows'):
            # Set R_HOME
            os.environ['R_HOME'] = r"C:\ProgramData\Anaconda3\envs\rstudio\lib\R"
            import rpy2.robjects as robjects
            # Defining the R script and loading the instance in Python
            r = robjects.r
            r['source']('CEOELA_computeELA.R')
        else:
            import subprocess
            subprocess.call("Rscript --version 4.0.5 CEOELA_computeELA.R", shell=True)
        # END IF
        
        if (self.verbose):
            print('[CEOELA] Computation of ELA features done.')
        # END IF
    # END DEF

        
        
    #%%
    ##################################
    '''
    # Processing of ELA features
    '''
    ##################################
    
    #%%    
    def ProcessELA(self, list_obj_hl=[], list_obj_ignore=[], corr_thres=0.95, corr_ignore=[]):
        """
        Processing the ELA features for hierarchical clustering.
        ----------
        Parameters
        ----------
        list_obj_hl: list, optional
            List of labels in clustering plot to be visually highlighted, by default None.
        list_obj_ignore: list, optional
            List of labels to be ignored in clustering, by default None.
        corr_thres: float, optional
            Pearson coefficient considered as threshold value for dropping ELA features, by default 0.95.
        corr_ignore: list, optional
            List of ELA features to be ignored when dropping highly correlated ELA features (enforced for clustering), by default None. 
        """
        # check input
        if ((corr_thres<0.0) or (corr_thres>1.0)):
            raise ValueError('Correlation threshold must be between 0.0 and 1.0.')
        # END IF
        
        # read ELA results
        # BBOB functions
        if (self.ELA_BBOB):
            filepath = os.path.join(self.filepath_save, 'featELA_' + self.problem_label + '_BBOB.xlsx')
            dict_result_BBOB_base = readFile2Dict(filepath, ['ins_'+str(ins) for ins in self.BBOB_instance], 
                                                  list_sheet_type='select', header=0)
            self.df_ELA_BBOB_base = dict2DF(dict_result_BBOB_base)
        
        if (self.bootstrap):
            # crash original
            if (self.ELA_problem):
                filepath = os.path.join(self.filepath_save, 'featELA_' + self.problem_label + '_original.xlsx')
                dict_result_crash_orig_base = readFile2Dict(filepath, ['bs_'+str(bs+1) for bs in range(self.bootstrap_repeat)], 
                                                            list_sheet_type='select', header=0)
                self.df_ELA_crash_original_base = dict2DF(dict_result_crash_orig_base)
            
            # crash re-scale
            if (self.ELA_problem and self.ELA_BBOB):
                filepath = os.path.join(self.filepath_save, 'featELA_' + self.problem_label + '_rescale.xlsx')
                dict_result_crash_rescale_base = readFile2Dict(filepath, ['bs_'+str(bs+1) for bs in range(self.bootstrap_repeat)], 
                                                               list_sheet_type='select', header=0)
                self.df_ELA_crash_rescale_base = dict2DF(dict_result_crash_rescale_base)
            
            # Artificial functions
            # TODO: parallel computing
            if (self.ELA_AF):
                dict_result_AF_base = {}
                for bs in range(self.bootstrap_repeat):
                    sheetname = 'bs_' + str(bs+1)
                    filepath = os.path.join(self.filepath_save, 'featELA_' + self.problem_label + '_AF_' + sheetname + '.xlsx')
                    dict_result_AF_base[sheetname] = readFile2Dict(filepath, [sheetname], list_sheet_type='select', header=0)
                # END FOR
                self.df_ELA_AF_base = dict2DF(dict_result_AF_base)
        else:
            # crash original
            if (self.ELA_problem):
                filepath = os.path.join(self.filepath_save, 'featELA_' + self.problem_label + '_original.xlsx')
                dict_result_crash_orig_base = readFile2Dict(filepath, ['full'], list_sheet_type='select', header=0)
                self.df_ELA_crash_original_base = dict2DF(dict_result_crash_orig_base)
            
            # crash re-scale
            if (self.ELA_problem and self.ELA_BBOB):
                filepath = os.path.join(self.filepath_save, 'featELA_' + self.problem_label + '_rescale.xlsx')
                dict_result_crash_rescale_base = readFile2Dict(filepath, ['full'], list_sheet_type='select', header=0)
                self.df_ELA_crash_rescale_base = dict2DF(dict_result_crash_rescale_base)
                
            # Artificial functions
            if (self.ELA_AF):
                filepath = os.path.join(self.filepath_save, 'featELA_' + self.problem_label + '_AF.xlsx')
                dict_result_AF_base = readFile2Dict(filepath, ['full'], list_sheet_type='select', header=0)
                self.df_ELA_AF_base = dict2DF(dict_result_AF_base)
        # END IF
        
        #%%
        # create linkage matrix
        # crash re-scale with BBOB functions
        if (self.ELA_problem and self.ELA_BBOB):
            list_df = [self.df_ELA_crash_rescale_base, self.df_ELA_BBOB_base]
            self.linkageMat_BBOB, self.df_data_standard_BBOB, self.df_data_cluster_BBOB = create_linkageMat(list_df, 
                                                                                                            list_obj_hl=list_obj_hl, 
                                                                                                            list_obj_ignore=list_obj_ignore,
                                                                                                            corr_thres=corr_thres, 
                                                                                                            corr_ignore=corr_ignore, 
                                                                                                            verbose=self.verbose)
        # crash original with Artificial functions
        if (self.ELA_problem and self.ELA_AF):
            list_df = [self.df_ELA_crash_original_base, self.df_ELA_AF_base]
            self.linkageMat_AF, self.df_data_standard_AF, self.df_data_cluster_AF = create_linkageMat(list_df, 
                                                                                                      list_obj_hl=list_obj_hl, 
                                                                                                      list_obj_ignore=list_obj_ignore,
                                                                                                      corr_thres=corr_thres, 
                                                                                                      corr_ignore=corr_ignore, 
                                                                                                      verbose=self.verbose)
        #%%
        if (self.verbose):
            print('[CEOELA] Processing of ELA features done.')
        # END IF
    # END DEF
    
   
    
    #%%
    ##################################
    '''
    # Comparison of ELA features
    '''
    ##################################
    
    #%% 
    def CompareELA(self):
        # plot dendrogram for clustering
        filepath_AF_func = os.path.join(self.filepath_save, self.problem_label+'_AF_func.xlsx')
        
        # BBOB functions
        if (self.ELA_problem and self.ELA_BBOB):
            # save similar BBOB in excel
            self.dict_similarf_BBOB = get_similar_func(self.linkageMat_BBOB, self.df_data_standard_BBOB, self.list_output_rename)
            write_similarF(self.filepath_save, self.dict_similarf_BBOB, self.problem_label, 'BBOB', filepath_AF_func)
            
            # plot dendrogram
            plot_dendrogram(self.linkageMat_BBOB, 
                            xlabel='', 
                            ylabel='Euclidean distance', 
                            rot_angle=90, label_ha='center', fontsize=8,
                            titel='', 
                            dir_out=self.filepath_save, 
                            cfigname='plot_cluster_BBOB_' + self.problem_label, 
                            figformat='.png', 
                            figsize=(6,3), 
                            dpi=300, show=False, labels=list(self.df_data_standard_BBOB.index), truncate_mode=None, p=2)
            
        # Artificial functions
        if (self.ELA_problem and self.ELA_AF):
            # save similar AF in excel
            self.dict_similarf_AF = get_similar_func(self.linkageMat_AF, self.df_data_standard_AF, self.list_output_rename)
            write_similarF(self.filepath_save, self.dict_similarf_AF, self.problem_label, 'AF', filepath_AF_func)
            
            # plot dendrogram
            plot_dendrogram(self.linkageMat_AF, 
                            xlabel='', 
                            ylabel='Euclidean distance', 
                            rot_angle=90, label_ha='center', fontsize=8,
                            titel='', 
                            dir_out=self.filepath_save, 
                            cfigname='plot_cluster_AF_' + self.problem_label, 
                            figformat='.png', 
                            figsize=(20+self.df_data_standard_AF.shape[0]*0.1,3), 
                            dpi=300, show=False, labels=list(self.df_data_standard_AF.index), truncate_mode=None, p=2)
        if (self.verbose):
            print('[CEOELA] Comparison of ELA features done.')
        # END IF
    # END DEF
# END CLASS            




#%%











#%%









