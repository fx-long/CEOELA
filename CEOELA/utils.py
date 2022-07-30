# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 18:46:27 2022

@author: Q521100
"""



import os
import sys
import inspect
import copy
import pandas as pd
import numpy as np
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from multiprocessing import Pool, cpu_count




#%%
##################################
'''
# Get directory path
'''
##################################

#%%
# Determine the current work directory
def get_script_dir(follow_symlinks=True, directory=''):
    """
    Returns the current directory path.
    ----------
    Returns
    ----------
    path: str
    """
    if getattr(sys, 'frozen', False): # py2exe, PyInstaller, cx_Freeze
        path = os.path.abspath(sys.executable)
    else:
        path = inspect.getabsfile(get_script_dir)
    if (follow_symlinks):
        path = os.path.realpath(path)
    if (directory == 'parent'):
        return os.path.dirname(os.path.dirname(path))
    else:
        return os.path.dirname(path)
    # END IF
# END DEF





#%%
##################################
'''
# Data Cleaning
'''
##################################

#%%
def dataCleaning(df_data,
                 replace_nan=None,
                 col_allnan=False, 
                 col_anynan=False, 
                 row_nan=False, 
                 col_null_var=True, 
                 var_thres=None, 
                 row_dupli=True, 
                 filter_key=[], 
                 reset_index=True,
                 verbose=True):
    """
    Clean a data frame.
    ----------
    Parameters
    ----------
    df_data: DataFrame
    replace_nan: Boolean
    col_allnan: Boolean
    col_anynan: Boolean
    row_nan: Boolean
    col_null_var: Boolean
    var_thres: Float
    row_dupli: Boolean
    filter_key: List
    reset_index: Boolean
    verbose: Boolean
    ----------
    Returns
    ----------
    df_data_clean: DataFrame
    """
    
    df_data_clean = copy.deepcopy(df_data)
    
    # replace all nan with a value
    if (replace_nan):
        df_data_clean.fillna(replace_nan)
        if (verbose):
            print(f'All missing values have been replaced with {replace_nan}.\n')
        # END IF
    # END IF
    
    # drop columns with key of a specific string    
    if (filter_key):
        for key in filter_key:
            df_data_clean = df_data_clean[df_data_clean.columns.drop(list(df_data_clean.filter(regex=key)))]
        # END FOR
        
        list_filter_key = []
        for key in df_data.keys():
            if (key not in df_data_clean.keys()):
                list_filter_key.append(key)
            # END IF
        # END FOR
        
        if (list_filter_key and verbose):
            print(f'Following {len(list_filter_key)} features with filter keyword have been dropped: {list_filter_key}.\n')
        # END IF
    # END IF
    
    # drop columns with only missing value or NAN
    if (col_allnan):
        col_to_drop = []
        for col in df_data_clean.columns:
            df_temp = df_data_clean[[col]]
            
            if (df_temp.isnull().values.all()):
                col_to_drop.append(col)
            # END IF
        # END FOR
        
        if (col_to_drop and verbose):
            print(f'Following {len(col_to_drop)} columns with ONLY NAN have been dropped: {col_to_drop}.\n')
            df_data_clean.drop(col_to_drop, axis=1, inplace=True)
        # END IF
    # END IF
    
    # drop columns with any missing value or NAN
    if (col_anynan):
        col_to_drop = []
        for col in df_data_clean.columns:
            df_temp = df_data_clean[[col]]
            
            if (df_temp.isnull().values.any()):
                col_to_drop.append(col)
            # END IF
        # END FOR
        
        if (col_to_drop and verbose):
            print(f'Following {len(col_to_drop)} columns with ANY NAN have been dropped: {col_to_drop}.\n')
            df_data_clean.drop(col_to_drop, axis=1, inplace=True)
        # END IF
    # END IF
    
    # drop row with any missing value or NAN
    if (row_nan):
        row_to_drop = []
        for ind in range(len(df_data_clean)):
            df_temp = df_data_clean.iloc[ind]
            
            if (df_temp.isnull().values.any()):
                row_to_drop.append(df_data_clean.index[ind])
            # END IF
        # END FOR
        
        if (row_to_drop and verbose):
            print(f'Following {len(row_to_drop)} rows with ANY NAN have been dropped: {row_to_drop}.\n')
            df_data_clean.drop(row_to_drop, axis=0, inplace=True)
        # END IF
    # END IF
    
    # drop columns with no variance or 0 variance
    if (col_null_var):
        
        # variance threshold not possible
        counts = df_data_clean.nunique()
        # record columns to delete
        to_del = [counts.index[i] for i,v in enumerate(counts) if v == 1]
        if (to_del and verbose):
            print(f'Following {len(to_del)} features with 0 variance have been dropped: {to_del}.\n')
            # drop useless columns
            df_data_clean.drop(to_del, axis=1, inplace=True)
        # END IF
    # END IF
    
    # drop columns with variance smaller than threshold
    if (var_thres):
        pass
        # variance threshold possible, but setting threshold value is challenging
        '''
        TODO
        transformer = VarianceThreshold(threshold=var_thres)
        df_data_clean = transformer.fit_transform(df_data_clean)
        '''
    # END IF
    
    # drop duplicated rows
    if (row_dupli):
        dups = df_data_clean.duplicated()
        # report if there are any duplicates
        if (dups.any() and verbose):
            print('Duplicated rows have been dropped.\n')
            df_data_clean.drop_duplicates(inplace=True)
        # END IF
    # END IF

    if (reset_index):
        df_data_clean.reset_index(drop=True, inplace=True)
        if (verbose):
            print('Index of dataframe has been reset.\n')
        # END IF
    # END IF
    
    return df_data_clean
# END DEF





#%%
##################################
'''
# Function Feature Correlation
'''
##################################

#%%
def dropFeatCorr(df_data, 
                 corr_thres=0.9, 
                 corr_method='pearson', 
                 mode='pair', 
                 ignore_keys=[],
                 verbose=True):
    """
    Drop highly correlated feature pair. 
    ----------
    Parameters
    ----------
    df_data: DataFrame
    corr_thres: Float
    corr_method: Str {'pearson', 'kendall', 'spearman'}
    mode: Str {'pair', 'direct'}
    ignore_keys: List
    verbose: Boolean
    ----------
    Returns
    ----------
    df_data_orig: DataFrame
    """
    
    df_data_orig = copy.deepcopy(df_data)
    
    df_data_corr = df_data_orig.corr(method=corr_method)
    
    df_data_corr.index.names = ['feat_name']
    
    # get upper triangle form of the correlation matrix
    df_upper_tri = df_data_corr.where(np.triu(np.ones(df_data_corr.shape),k=1).astype(np.bool))
        
    if (mode == 'pair'):
        # remove feature with correlation exceeding threshold AND higher mean correlation wirh rest features
        # get all correlated pairs exceed correlation threshold (positive and negative correlation)
        list_corr_pair = []
        for column in df_upper_tri.columns:
            df_temp = abs(df_upper_tri[column]) >= corr_thres
            list_feat = list(df_temp.index[df_temp].values)
            
            for feat in list_feat:
                list_pair = []
                list_pair.append(column)
                list_pair.append(feat)
                list_corr_pair.append(list_pair)
            # END FOR
        # END FOR
        
        # from a correlated pair, get and drop the feature with a higher correlation with other remaining features
        list_featA = []
        list_featB = []
        list_featA_corr = []
        list_featB_corr = []
        list_corr = []
        
        col_to_drop = []
        df_corr_temp = copy.deepcopy(df_data_corr)
        df_corr_temp.reset_index(inplace=True)
        for corr_pair in list_corr_pair:
            featA = corr_pair[0]
            featB = corr_pair[-1]
            
            # first feature A
            # ignore self-correlation and correlation with feature B
            df_corr_remainA = df_corr_temp.loc[df_corr_temp['feat_name'] == featA, (df_corr_temp.columns != featA) & (df_corr_temp.columns != featB)]
            df_corr_remainA.drop(['feat_name'], axis=1, inplace=True)
            corr_remainA = abs(df_corr_remainA).mean(axis=1).values[-1]
            
            # second feature B
            # ignore self-correlation and correlation with feature A
            df_corr_remainB = df_corr_temp.loc[df_corr_temp['feat_name'] == featB, (df_corr_temp.columns != featB) & (df_corr_temp.columns != featA)]
            df_corr_remainB.drop(['feat_name'], axis=1, inplace=True)
            corr_remainB = abs(df_corr_remainB).mean(axis=1).values[-1]
            
            if (corr_remainA > corr_remainB):
                col_to_drop.append(featA)
            else:
                col_to_drop.append(featB)
            # END IF
            
            # save results
            list_featA.append(featA)
            list_featB.append(featB)
            list_featA_corr.append(corr_remainA)
            list_featB_corr.append(corr_remainB)
            list_corr.append(df_corr_temp.loc[df_corr_temp['feat_name'] == featA, (df_corr_temp.columns == featB)].values[-1][-1])    
        # END FOR
        
        # drop duplicated features
        set_col_to_drop = list(set(col_to_drop))
        list_to_drop = copy.deepcopy(set_col_to_drop)
        
        # ignore certain features
        if (ignore_keys):
            list_to_ignore = []
            for col2drop in set_col_to_drop:
                for key2ignore in ignore_keys:
                    if (key2ignore in col2drop):
                        list_to_drop.remove(col2drop)
                        list_to_ignore.append(col2drop)
                        break
                    # END IF
                # END FOR
            # END FOR
            list_to_drop = list(set(list_to_drop))
            if (verbose):
                print(f'Following {len(list_to_ignore)} features have been enforced/ignored: {list_to_ignore}.\n')
            # END IF
        # END IF                
        
        if (verbose):
            print(f'Following {len(list_to_drop)} highly correlated features > {corr_thres} have been dropped: {list_to_drop}.\n')
        # END IF
        
        dict_featpair = {}
        dict_featpair['featA'] = list_featA
        dict_featpair['featB'] = list_featB
        dict_featpair['corrA_mean'] = list_featA_corr
        dict_featpair['corrB_mean'] = list_featB_corr
        dict_featpair['corr_pair'] = list_corr
        df_featpair = pd.DataFrame.from_dict(dict_featpair)
        df_data_orig = df_data_orig.drop(list_to_drop, axis=1)
        return df_data_orig, df_featpair
    
    elif (mode == 'direct'):
        # remove feature with correlation exceeding threshold
        # get all correlated pairs exceed correlation threshold (positive and negative correlation)
        list_corr_pair = [column for column in df_upper_tri.columns if any(df_upper_tri[column] > corr_thres)]
        
        if (verbose):
            print(f'Following {len(list_corr_pair)} highly correlated features > {corr_thres} have been dropped: {list_corr_pair}.\n')
        # END IF
        
        # Drop features 
        df_data_orig.drop(list_corr_pair, axis=1, inplace=True)
        return df_data_orig
        
    else:
        raise ValueError(f'Mode {mode} is undefined!')
    # END IF    
# END DEF




#%%
##################################
'''
# Re-scaling design variables
'''
##################################

#%%
# scale data range into -5 to 5 to match BBOB functions
def data_rescaling(df_data, 
                   bound_min, 
                   bound_max, 
                   target_min, 
                   target_max):
    """
    Calculate area under curve. 
    ----------
    Parameters
    ----------
    df_data: DataFrame
    bound_min: Float 
    bound_max: Float 
    target_min: Float 
    target_max: Float 
    ----------
    Returns
    ----------
    df_rescale: DataFrame
    """
    
    df_rescale = (df_data-bound_min) / (bound_max-bound_min) * (target_max-target_min) + target_min
    return df_rescale
# END DEF






#%%
##################################
'''
# Function Read Excel --> Dict
'''
##################################

#%%
def readFile2Dict(filepath,
                  list_sheet=[], 
                  list_sheet_type='ignore', 
                  header=0):
    """
    Convert excel (each sheet as a dataframe) and return a dictionary. 
    ----------
    Parameters
    ----------
    filepath: Str
    list_sheet: List 
    list_sheet_type: Str {'ignore', 'select'} 
    header: Int
    ----------
    Returns
    ----------
    data_dict: Dictionary
    """
    
    # check if file exists
    if not (os.path.isfile(filepath)):
        raise ValueError(f'File {filepath} is missing!')
    # END IF
    
    data_dict = {}
    
    # read file and convert into dataframe
    if (filepath.endswith('.xlsx')):
        xls = pd.ExcelFile(filepath)
        sheet_to_df = {}
        for sheet_name in xls.sheet_names:
            # ignore filter key
            if (list_sheet_type == 'ignore'):
                if not any(key in sheet_name for key in list_sheet):
                    sheet_to_df[sheet_name] = xls.parse(sheet_name, header=header)
                # END IF
            # add selected key
            elif (list_sheet_type == 'select'):
                if sheet_name in list_sheet:
                    sheet_to_df[sheet_name] = xls.parse(sheet_name, header=header)
                # END IF
            else:
                raise ValueError(f'Command {list_sheet_type} is undefined!')
            # END IF
        # END FOR
        data_dict['excel'] = sheet_to_df
               
    elif (filepath.endswith('.csv')):
        sheet_to_df = pd.read_csv(filepath)
        data_dict['csv'] = sheet_to_df
    else:
        raise ValueError(f'File type {filepath} is not defined!')
    # END IF
    
    return data_dict   
# END FUNCTION
    

 




#%%
##################################
'''
# Function Recursive Dict --> DataFrame
'''
##################################

#%%
def recursiveDict(obj, dict_):
    for key in dict_.keys():
        if not isinstance(dict_[key], dict):
            df_temp = copy.deepcopy(dict_[key])
            df_temp.set_index('ELA_feat', inplace=True)
            df_temp = df_temp.T
            df_temp.reset_index(drop=False, inplace=True)
            df_temp['sheetname'] = key
            obj.df_main = pd.concat([obj.df_main, df_temp], axis=0, ignore_index=True)
        else:
            recursiveDict(obj, dict_[key])
        # END IF
    # END FOR
# END DEF
    
class dict2DF_obj:
    def __init__(self):
        self.df_main = pd.DataFrame()
    # END DEF
# END CLASS
        
def dict2DF(dict_main):
    """
    Recursively flatten a dictionary and combine all dataframes.
    ----------
    Parameters
    ----------
    dict_main: Dict
    ----------
    Returns
    ----------
    df_main: DataFrame
    """
    obj = dict2DF_obj()
    recursiveDict(obj, dict_main)
    return obj.df_main
# END FUNCTION







#%%
##################################
'''
# Function Get Similar Function from Clusters
'''
##################################

#%%
# To flatten a linkage matrix and get cluster id
def flat_linkageMat(row_clusters, labels, id_item):
    clusters = {}
    list_clust = []
    for row in range(row_clusters.shape[0]):
        cluster_n = row + len(labels)
        # which clusters / labels are present in this row
        glob1, glob2 = row_clusters[row, 0], row_clusters[row, 1]

        # if this is a cluster, pull the cluster
        this_clust = []
        for glob in [glob1, glob2]:
            if glob > (len(labels)-1):
                this_clust += clusters[glob]
            # if it isn't, add the label to this cluster
            else:
                this_clust.append(glob)
        # END FOR
                
        if (id_item in this_clust):
            if (list_clust):
                if (len(list_clust) > len(this_clust)):
                    list_clust = this_clust
            else:
                list_clust = this_clust
        clusters[cluster_n] = this_clust
    # END FOR
    list_clust = [int(i) for i in list_clust]
    return clusters, list_clust
# END FUNCTION

def get_similar_func(linkage_matrix, df_standard, list_func, metric='euclidean'):
    dict_func = {}
    dict_dist = {}
    df_data = copy.deepcopy(df_standard)
    df_data.reset_index(drop=False, inplace=True)
    for func in list_func:
        id_item = df_data.index[df_data['label'] == func].tolist()[0]
        clusters, list_clust = flat_linkageMat(linkage_matrix, list(df_data.index), id_item)
        list_clust.remove(id_item)
        list_label = []
        for item in list_clust:
            list_label.append(df_data.iloc[item]['label'])
        dict_func[func] = list_label
        
        # pairwise distance
        df_a = df_standard.loc[func]
        list_dist = []
        for label in list_label:
            df_b = df_standard.loc[label]
            if (metric == 'euclidean'):
                dist = np.linalg.norm(df_a-df_b)
                list_dist.append(dist)
            else:
                raise ValueError(f'{metric} is not defined.')
        dict_dist[func] = list_dist
    # END FOR
    return dict_func, dict_dist
# END FUNCTION







#%%
##################################
'''
# Function Create Linkage Matrix
'''
##################################

#%%
# To create linkage matrix 
def create_linkageMat(list_df, list_obj_hl=[], list_obj_ignore=[], corr_thres=0.95, corr_ignore=[], verbose=True):
    
    df_data_comb = pd.DataFrame()
    for df_data in list_df:
        df_temp = copy.deepcopy(df_data)
        df_temp.drop(['sheetname'], axis=1, inplace=True)
        if (list_obj_ignore):
            for obj_ignore in list_obj_ignore:
                df_temp = df_temp[df_temp['index'] != obj_ignore]
            # END FOR
        # END IF
        df_temp = df_temp.groupby(by=['index']).mean()
        df_temp.index.name = 'label'
        df_data_comb = pd.concat([df_data_comb, df_temp], axis=0)
    # END FOR
    
    # data cleaning
    df_data_clean = dataCleaning(df_data_comb, col_allnan=True, col_anynan=True, row_nan=False, col_null_var=True, 
                                 row_dupli=True, filter_key=['costs_fun_evals', 'costs_runtime', 'basic'], reset_index=False, verbose=verbose)
    
    # highly correlated
    df_data_cluster, df_featpair = dropFeatCorr(df_data_clean, corr_thres=0.95, corr_method='pearson', mode='pair', ignore_keys=corr_ignore,
                                                verbose=verbose)
    
    # print all remaining column keys
    if (verbose):
        print(f'Following {len(df_data_cluster.keys())} features are remaining: {list(df_data_cluster.keys())}.\n')
    # END IF
    
    # standardization
    scaler = StandardScaler()
    df_data_standard = scaler.fit_transform(df_data_cluster)
    df_data_standard = pd.DataFrame(df_data_standard, index=df_data_cluster.index, columns=df_data_cluster.columns)
    
    # calculate linkage matrix
    linkage_matrix = linkage(df_data_standard, method='ward', metric='euclidean')
    
    return linkage_matrix, df_data_standard, df_data_cluster
# END FUNCTION







#%%
##################################
'''
# Function Plot Dendrogram
'''
##################################

#%%
def plot_dendrogram(linkage_matrix, xlabel='', ylabel='', rot_angle=None, label_ha='center', fontsize=8,
                    titel='', dir_out='', cfigname='', figformat='', figsize=None, dpi=300, show=False, **kwargs):
    
    # initialize
    font = FontProperties()
    font.set_size(fontsize)
    font.set_name('Arial')
    fig,ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    if (rot_angle is not None):
        plt.setp(ax.get_xticklabels(), rotation=rot_angle, horizontalalignment=label_ha, size=fontsize)
    
    # diagramm properties
    ax.tick_params(axis='x', which='major', labelsize=fontsize)
    ax.tick_params(axis='y', which='major', labelsize=fontsize)
    ax.set_xlabel(xlabel,fontproperties=font)
    ax.set_ylabel(ylabel,fontproperties=font)
    if (titel):
        ax.set_title(titel,fontproperties=font)
    fig.tight_layout()
    if dir_out:
        if figformat:
            fig_ending = figformat
        else:
            fig_ending = '.pdf'
        filename_save = cfigname + fig_ending
        plt.savefig(os.path.join(dir_out, filename_save), bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
# END FUNCTION







#%%
##################################
'''
# Function Create R-script
'''
##################################

#%%
def create_Rscript(filepath_base, filepath_new, os_system='windows'):
    # create temporary file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(filepath_base) as old_file:
            if (os_system == 'windows'):
                new_file.write('\nenv_windows <- TRUE\n')
                new_file.write('\npath_split <- "'"\\\\"'"\n')
            elif (os_system == 'linux'):
                new_file.write('\n.libPaths(c("/proj/cae_muc/q521100/83_Miniconda/r4.0.5/library/", .libPaths()))\n')
                new_file.write('\nenv_windows <- FALSE\n')
                new_file.write('\npath_split <- "'"/"'"\n')
            for line in old_file:
                new_file.write(line)
            # END FOR
        # END WITH
    # END WITH
    copymode(filepath_base, abs_path)
    move(abs_path, filepath_new)
# END FUNCTION







#%%
##################################
'''
# Function Save Similar Functions in Clusters
'''
##################################

#%%
def write_similarF(filepath_save, dict_similarF, dict_dist, problem_label, type_func, filepath_AF_func):
    filename = 'similarF_' + problem_label + f'_{type_func}.xlsx'
    filepath_out = os.path.join(filepath_save, filename)
    
    if (type_func == 'BBOB'):
        with pd.ExcelWriter(filepath_out) as writer:
            for key in dict_similarF.keys():
                df_temp = pd.DataFrame({f'similar {type_func}': dict_similarF[key],
                                        'dist': dict_dist[key]})
                df_temp.sort_values(by=['dist'], axis=0, ascending=True, inplace=True)
                df_temp.to_excel(writer, sheet_name=key, index=False)
    
    elif (type_func == 'AF'):
        df_AF_func = pd.read_excel(filepath_AF_func, sheet_name='func')
        with pd.ExcelWriter(filepath_out) as writer:
            for key in dict_similarF.keys():
                list_func = []
                for item in dict_similarF[key]:
                    if ('AF_' in item):
                        list_func.append(df_AF_func.iloc[df_AF_func[df_AF_func['label'] == item].index.tolist()[0]]['func'])
                    else:
                        list_func.append('')
                df_temp = pd.DataFrame({f'similar {type_func}': dict_similarF[key],
                                        'dist': dict_dist[key],
                                        'func': list_func})
                df_temp.sort_values(by=['dist'], axis=0, ascending=True, inplace=True)
                df_temp.to_excel(writer, sheet_name=key, index=False)
    else:
        raise ValueError(f'{type_func} is not defined.')
    # END IF
# END FUNCTION







#%%
##################################
'''
# Function Parallel Computing
'''
##################################

#%%
def runParallelFunction(runFunction, arguments):
    """
        Return the output of runFunction for each set of arguments,
        making use of as much parallelization as possible on this system

        :param runFunction: The function that can be executed in parallel
        :param arguments:   List of tuples, where each tuple are the arguments
                            to pass to the function
        :return:
    """
    arguments = list(arguments)
    p = Pool(min(cpu_count(), len(arguments)))
    results = p.map(runFunction, arguments)
    p.close()
    return results
# END FUNCTION

def write_bs_AF(filepath_save, problem_label, i_bs, dict_bs_AF):
    filename = problem_label + '_AF_bs_' + str(i_bs) + '.xlsx'
    filepath_out = os.path.join(filepath_save, filename)
    dict_bs_AF[i_bs].to_excel(filepath_out, sheet_name='bs_'+str(i_bs), index=False)
# END FUNCTION
                    
                    
                    
                    
#%%













