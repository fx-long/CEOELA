# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 12:28:46 2022

@author: Q521100
"""



from CEOELA.CEOELA import CEOELA_pipeline





#%%
# initliaze
ceoela_pipeline = CEOELA_pipeline(r"C:\Users\Q521100\Desktop\Workspace_local\AutoOpti_pipeline\CEOELA_pipeline\crash_data\DOE_BBOB_Crash.xlsx",
                                  list_sheetname = [],
                                  problem_label = '',
                                  filepath_save = '',
                                  bootstrap = True,
                                  bootstrap_size = 0.8,
                                  bootstrap_repeat = 2,
                                  bootstrap_seed = 0,
                                  BBOB_func = ['F1', 'F2'],
                                  BBOB_instance = [1, 2],
                                  BBOB_seed = 0,
                                  AF_number = 2,
                                  AF_seed = 0,
                                  np_ela = 1,
                                  purge = True,
                                  verbose = True,
                                  )

#%%
# data pre-processing
ceoela_pipeline.DataPreProcess()

#%%
# computation of ELA features
ceoela_pipeline.ComputeELA(ELA_problem=True, ELA_BBOB=True, ELA_AF=True)

#%%
# processing of ELA features
ceoela_pipeline.ProcessELA(list_obj_hl=[], list_obj_ignore=[], corr_thres=0.95, corr_ignore=[])

#%%
# comparison of ELA features
ceoela_pipeline.CompareELA()










#%%












