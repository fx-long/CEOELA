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
import pandas as pd
import numpy as np

from sklearn.utils import resample
from .utils import get_script_dir, data_rescaling, readFile2Dict, dict2DF, create_linkageMat, plot_dendrogram
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
# Initialize a class for ELA pipeline
class ELA_pipeline:
    def __init__(self,
                 filepath_excel: str,
                 list_sheetname: list = [],
                 crash_label: str = '',
                 filepath_save: str = '',
                 bootstrap_size: float or int = 0.8,
                 bootstrap_repeat: int = 2,
                 bootstrap_seed: int = 0,
                 BBOB_func: list = ['F'+str(i+1) for i in range(24)],
                 BBOB_instance: list = [1],
                 BBOB_seed: int = 0,
                 AF_number: int = 1,
                 AF_seed: int = 0,
                 os_system: str = 'windows',
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
            Use the 'doe_template.xlsx' template as referance.
        list_sheetname: list, optional
            Sheet inside Excel file to be read, by default None.
        crash_label: str, optional
            Re-name the crash instance, by default None.
        filepath_save: str, optional
            Path to save the output files, by default None.
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
        os_system: str, optional
            Operating system, either 'windows' or 'linux', by default windows.
        verbose: bool, optional
            The verbosity, by default True.
        """
        
        # basic information
        self.filepath_excel: str = filepath_excel
        self.list_sheetname: list = list_sheetname
        self.crash_label: str = crash_label if crash_label else 'Instance'
        self.filepath_save: str = filepath_save
        
        # crash problem instance
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
            # END IF
            dict_base = readFile2Dict(self.filepath_excel, self.list_sheetname, list_sheet_type='select', header=0)
        else:
            raise ValueError(f'Excel file {self.filepath_excel} is missing.')
        # END IF
        dict_main = copy.deepcopy(dict_base['excel'])
        self.list_input = list(dict_main['KPI']['input'].dropna())
        self.list_output = list(dict_main['KPI']['output'].dropna())
        self.list_output_rename = list(dict_main['KPI']['output_rename'].dropna())
        self.list_total = self.list_input + self.list_output
        self.df_boundary = dict_main['Bounds']
        
        # check output name
        if (len(self.list_output) != len(self.list_output_rename)):
            raise ValueError('Variable list_output and list_output_rename must have same length.')
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
            self.filepath_save = os.path.join(self.filepath_save, 'results_ELA', self.crash_label)
        else:
            self.filepath_save = os.path.join(self.path_dir_base, 'results_ELA', self.crash_label)
        if not (os.path.isdir(self.filepath_save)):
            os.makedirs(self.filepath_save)
        # END IF
        
        # check os system
        if not ((self.os_system=='windows') or (self.os_system=='linux')):
            raise ValueError(f'Operating system {self.os_system} is undefined. Use only \'windows\' or \'linux\'.')
        # END IF
            
        # check boot-strap inputs
        if (type(self.bootstrap_size) is float):
            if ((self.bootstrap_size <= 0.0) or (self.bootstrap_size >= 1.0)):
                raise ValueError('Boostrap size float must be between 0 and 1.')
            # END IF
        else:
            if (self.bootstrap_size <= 0):
                raise ValueError('Boostrap size int must be non-zero and positive.')
            if (self.bootstrap_size > len(self.df_data_main)):
                raise ValueError('Boostrap size int must be smaller than total sample size.')
            # END IF
        # END IF
        
        #%%
        if (self.verbose):
            print('[ELA] ELA pipeline is initialized.')
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
            print(f'Total {len(self.df_data_main)-len(self.df_data_filter)} failed FE simulations filtered.')
        # END IF
        
        # drop duplicated sample points
        row_dups = self.df_data_filter.duplicated(subset=self.list_input, keep='first')
        if (row_dups.any()):
            initial_size = len(self.df_data_filter)
            self.df_data_filter.drop_duplicates(subset=self.list_input, keep='first', inplace=True)
            if (self.verbose):
                print(f'{initial_size-len(self.df_data_filter)} duplicated dropped. Final {len(self.df_data_filter)} sample points.')
            # END IF
        else:
            if (self.verbose):
                print(f'No duplicated. Final {len(self.df_data_filter)} sample points.')
            # END IF
        # END IF
        
        # re-name responses
        for output, output_rename in zip(self.list_output, self.list_output_rename):
            self.df_data_filter.rename(columns={output: output_rename}, inplace=True)
        # END FOR
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
            raise ValueError('[ELA] Re-scaled x-data are not within [-5,5].')
        else:
            if (self.verbose):
                print('[ELA] Re-scaling crash design space to [-5,5] done.')
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
            print(f'BBOB functions {self.BBOB_func} of instance {self.BBOB_instance} are generated.')
        # END IF
        
        #%%
        # generate artificial functions
        np.seterr(all='ignore')
        i_AF_seed = 0 + self.AF_seed
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
            print(f'[ELA] AF are generated. Valid: {len(self.dict_AF_func)}; Fail: {i_fail}; Invalid: {i_invalid}; Total runs: {i_run}')
        # END IF
         
        #%%
        # bootstrapping
        if (type(self.bootstrap_size) is int):
            num_sample = self.bootstrap_size
        else:
            num_sample = int(math.ceil(len(self.df_data_filter) * self.bootstrap_size))
        # END IF
        self.dict_bs_crash_original = {}
        self.dict_bs_crash_rescale = {}
        self.dict_bs_AF = {}
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
            print(f'[ELA] Boot-strapping size {self.bootstrap_size} and repitition {self.bootstrap_repeat} done.')
        # END IF
        
        #%%
        # save results
        # crash original
        filename = self.crash_label + '_crash_original.xlsx'
        filepath_out = os.path.join(self.filepath_save, filename)
        with pd.ExcelWriter(filepath_out) as writer:
            for i_sheet in self.dict_bs_crash_original.keys():
                self.dict_bs_crash_original[i_sheet].to_excel(writer, sheet_name='bs_'+i_sheet, index=False)
            # END FOR
        # END WITH
        
        # crash rescale
        filename = self.crash_label + '_crash_rescale.xlsx'
        filepath_out = os.path.join(self.filepath_save, filename)
        with pd.ExcelWriter(filepath_out) as writer:
            for i_sheet in self.dict_bs_crash_rescale.keys():
                self.dict_bs_crash_rescale[i_sheet].to_excel(writer, sheet_name='bs_'+i_sheet, index=False)
            # END FOR
        # END WITH
        
        # BBOB functions
        filename = self.crash_label + '_BBOB.xlsx'
        filepath_out = os.path.join(self.filepath_save, filename)
        with pd.ExcelWriter(filepath_out) as writer:
            for i_sheet in self.dict_BBOB.keys():
                self.dict_BBOB[i_sheet].to_excel(writer, sheet_name='ins_'+i_sheet, index=False)
            # END FOR
        # END WITH
        
        # artificial functions
        for i_bs in self.dict_bs_AF.keys():
            filename = self.crash_label + '_AF_bs_' + str(i_bs) + '.xlsx'
            filepath_out = os.path.join(self.filepath_save, filename)
            with pd.ExcelWriter(filepath_out) as writer:
                self.dict_bs_AF[i_bs].to_excel(writer, sheet_name='bs_'+str(i_bs), index=False)
            # END WITH
        # END FOR
        
        # save AF expression
        self.df_AF_func = pd.DataFrame.from_dict(self.dict_AF_func)
        self.df_AF_func = self.df_AF_func.T
        self.df_AF_func.reset_index(drop=False, inplace=True)
        self.df_AF_func.rename(columns={'index': 'label', 0: 'func'}, inplace=True)
        filename = self.crash_label + '_AF_func.xlsx'
        filepath_out = os.path.join(self.filepath_save, filename)
        self.df_AF_func.to_excel(filepath_out, sheet_name='func', index=False)
        
        if (self.verbose):
            print(f'[ELA] Results are saved to {self.filepath_save}.')
        if (self.verbose):
            print('[ELA] Data pre-processing done.')
        # END IF
    # END DEF
    
    
    
    #%%
    ##################################
    '''
    # Computation of ELA features
    '''
    ##################################
    
    #%%
    def ComputeELA(self, ELA_crash=True, ELA_BBOB=True, ELA_AF=True):
        """
        Computation of ELA features.
        ----------
        Parameters
        ----------
        ELA_crash: bool, optional
            Compute ELA features on crash responses, by default True.
        ELA_BBOB: bool, optional
            Compute ELA features on BBOB functions, by default True.
        ELA_AF: bool, optional
            Compute ELA features on artificial functions, by default True.
        """
        # export meta-data for communication between python and R
        dict_meta = {}
        dict_meta['path_dir_base'] = self.path_dir_base
        dict_meta['crash_label'] = self.crash_label
        dict_meta['filepath_save'] = self.filepath_save
        dict_meta['list_input'] = self.list_input
        dict_meta['list_output'] = self.list_output_rename
        dict_meta['bootstrap_size'] = self.bootstrap_size
        dict_meta['bootstrap_repeat'] = ['bs_'+str(bs+1) for bs in range(self.bootstrap_repeat)]
        dict_meta['BBOB_func'] = self.BBOB_func
        dict_meta['BBOB_instance'] = ['ins_'+str(ins) for ins in self.BBOB_instance]
        dict_meta['AF_number'] = ['AF_'+str(AF_num+1) for AF_num in range(self.AF_number)]
        dict_meta['ELA_crash'] = ELA_crash
        dict_meta['ELA_BBOB'] = ELA_BBOB
        dict_meta['ELA_AF'] = ELA_AF
        dict_meta['os_system'] = self.os_system
                         
        filename_meta_base = 'ELA_metadata.json'
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
            r['source']('ELA_compute_ELA.R')
        else:
            import subprocess
            subprocess.call("Rscript --version 4.0.5 ELA_compute_ELA.R", shell=True)
        # END IF
        if (self.verbose):
            print('[ELA] Computation of ELA features done.')
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
        # crash original
        filepath = os.path.join(self.filepath_save, 'featELA_' + self.crash_label + '_crash_original.xlsx')
        dict_result_crash_orig_base = readFile2Dict(filepath, ['bs_'+str(bs+1) for bs in range(self.bootstrap_repeat)], 
                                                    list_sheet_type='select', header=0)
        self.df_ELA_crash_original_base = dict2DF(dict_result_crash_orig_base)
        
        # crash re-scale
        filepath = os.path.join(self.filepath_save, 'featELA_' + self.crash_label + '_crash_rescale.xlsx')
        dict_result_crash_rescale_base = readFile2Dict(filepath, ['bs_'+str(bs+1) for bs in range(self.bootstrap_repeat)], 
                                                       list_sheet_type='select', header=0)
        self.df_ELA_crash_rescale_base = dict2DF(dict_result_crash_rescale_base)
        
        # BBOB functions
        filepath = os.path.join(self.filepath_save, 'featELA_' + self.crash_label + '_BBOB.xlsx')
        dict_result_BBOB_base = readFile2Dict(filepath, ['ins_'+str(ins) for ins in self.BBOB_instance], 
                                              list_sheet_type='select', header=0)
        self.df_ELA_BBOB_base = dict2DF(dict_result_BBOB_base)
        
        # Artificial functions
        dict_result_AF_base = {}
        for bs in range(self.bootstrap_repeat):
            sheetname = 'bs_' + str(bs+1)
            filepath = os.path.join(self.filepath_save, 'featELA_' + self.crash_label + '_AF_' + sheetname + '.xlsx')
            dict_result_AF_base[sheetname] = readFile2Dict(filepath, [sheetname], list_sheet_type='select', header=0)
        # END FOR
        self.df_ELA_AF_base = dict2DF(dict_result_AF_base)
        
        #%%
        # create linkage matrix
        # TODO: check cluster_hl
        # crash re-scale with BBOB functions
        list_df = [self.df_ELA_crash_rescale_base, self.df_ELA_BBOB_base]
        self.linkageMat_BBOB, self.dict_cluster_BBOB, self.dict_hl_BBOB, self.df_data_standard_BBOB, self.df_data_cluster_BBOB = create_linkageMat(list_df, 
                                                                                                                                                   list_obj_hl=list_obj_hl, 
                                                                                                                                                   list_obj_ignore=list_obj_ignore,
                                                                                                                                                   corr_thres=corr_thres, 
                                                                                                                                                   corr_ignore=corr_ignore, 
                                                                                                                                                   verbose=self.verbose)
        # crash original with Artificial functions
        list_df = [self.df_ELA_crash_original_base, self.df_ELA_AF_base]
        self.linkageMat_AF, self.dict_cluster_AF, self.dict_hl_AF, self.df_data_standard_AF, self.df_data_cluster_AF = create_linkageMat(list_df, 
                                                                                                                                         list_obj_hl=list_obj_hl, 
                                                                                                                                         list_obj_ignore=list_obj_ignore,
                                                                                                                                         corr_thres=corr_thres, 
                                                                                                                                         corr_ignore=corr_ignore, 
                                                                                                                                         verbose=self.verbose)
        #%%
        if (self.verbose):
            print('[ELA] Processing of ELA features done.')
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
        # BBOB functions
        plot_dendrogram(self.linkageMat_BBOB, 
                        xlabel='', 
                        ylabel='Euclidean distance', 
                        rot_angle=90, label_ha='center', fontsize=8,
                        titel='', 
                        dir_out=self.filepath_save, 
                        cfigname='plot_cluster_BBOB_' + self.crash_label, 
                        figformat='.png', 
                        figsize=(6,3), dpi=300, show=False, labels=list(self.df_data_standard_BBOB.index), truncate_mode=None, p=2)
        
        # Artificial functions
        plot_dendrogram(self.linkageMat_AF, 
                        xlabel='', 
                        ylabel='Euclidean distance', 
                        rot_angle=90, label_ha='center', fontsize=8,
                        titel='', 
                        dir_out=self.filepath_save, 
                        cfigname='plot_cluster_AF_' + self.crash_label, 
                        figformat='.png', 
                        figsize=(140,3), dpi=300, show=False, labels=list(self.df_data_standard_AF.index), truncate_mode=None, p=2)
        if (self.verbose):
            print('[ELA] Comparison of ELA features done.')
        # END IF
    # END DEF
# END CLASS            
            
            








#%%











#%%









