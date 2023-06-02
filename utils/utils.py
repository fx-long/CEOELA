
import os
import sys
import copy
import pickle
import importlib
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count


#%%
def write_file(filepath, list_msg):
    with open(filepath, 'w+') as file:
        for msg in list_msg:
            file.write(msg + '\n')
# END DEF

#%%
def write_af(path_output, label_af, str_af):
    filename = f"genf_{label_af}"
    filepath = os.path.join(path_output, f"{filename}.py")
    list_msg = ['import numpy as np\n',
                f'def {filename}(x):',
                f'    return {str_af}']
    write_file(filepath, list_msg)
# END DEF

#%%
def eval_af(path_output, label_af):
    filename = f"af_{label_af}"
    filepath = os.path.join(path_output, f"{filename}.py")
    if not (os.path.isfile(filepath)):
        raise ValueError(f'File {filepath} is missing.')
    sys.path.append(path_output)
    module_ = importlib.import_module(filename)
    importlib.reload(module_)
    func_ = getattr(module_, filename)
    return func_
# END DEF

#%%
# export pickle
def export_pickle(filepath, data):
    with open(filepath, 'wb') as handle:
        pickle.dump(data, handle)
# END DEF

#%%
# read pickle
def read_pickle(filepath):
    with open(filepath, 'rb') as handle:
        data = pickle.load(handle)
    return data
# END DEF

#%%
# scale data range
def data_rescaling(df_data, bound_min, bound_max, target_min, target_max):
    df_rescale = (df_data-bound_min) / (bound_max-bound_min) * (target_max-target_min) + target_min
    return df_rescale
# END DEF

#%%
def dataCleaning(df_data,
                 replace_nan=None,
                 inf_as_nan=False,
                 col_allnan=False, 
                 col_anynan=False, 
                 row_anynan=False, 
                 col_null_var=True,
                 row_dupli=True, 
                 filter_key=[], 
                 reset_index=True,
                 verbose=True):
    df_data_clean = copy.deepcopy(df_data)
    # replace all inf as nan
    if (inf_as_nan):
        df_data_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        if (verbose):
            print('All -inf and inf have been replaced with NaN.\n')
            
    # replace all nan with a value
    if (replace_nan):
        df_data_clean.fillna(replace_nan)
        if (verbose):
            print(f'All missing values have been replaced with {replace_nan}.\n')
            
    # drop columns with key of a specific string    
    if (filter_key):
        for key in filter_key:
            df_data_clean = df_data_clean[df_data_clean.columns.drop(list(df_data_clean.filter(regex=key)))]
        list_filter_key = []
        for key in df_data.keys():
            if (key not in df_data_clean.keys()):
                list_filter_key.append(key)
        if (list_filter_key and verbose):
            print(f'Following {len(list_filter_key)} features with filter keyword have been dropped: {list_filter_key}.\n')
    
    # drop columns with all missing value or NAN
    if (col_allnan):
        col_to_drop = []
        for col in df_data_clean.columns:
            df_temp = df_data_clean[[col]]
            if (df_temp.isnull().values.all()):
                col_to_drop.append(col)
        if (col_to_drop):
            df_data_clean.drop(col_to_drop, axis=1, inplace=True)
            if (verbose):
                print(f'Following {len(col_to_drop)} columns with ONLY NAN have been dropped: {col_to_drop}.\n')
    
    # drop columns with any missing value or NAN
    if (col_anynan):
        col_to_drop = []
        for col in df_data_clean.columns:
            df_temp = df_data_clean[[col]]
            if (df_temp.isnull().values.any()):
                col_to_drop.append(col)
        if (col_to_drop):
            df_data_clean.drop(col_to_drop, axis=1, inplace=True)
            if (verbose):
                print(f'Following {len(col_to_drop)} columns with ANY NAN have been dropped: {col_to_drop}.\n')
    
    # drop row with any missing value or NAN
    if (row_anynan):
        row_to_drop = []
        for ind in range(len(df_data_clean)):
            df_temp = df_data_clean.iloc[ind]
            
            if (df_temp.isnull().values.any()):
                row_to_drop.append(df_data_clean.index[ind])
        if (row_to_drop):
            df_data_clean.drop(row_to_drop, axis=0, inplace=True)
            if (verbose):
                print(f'Following {len(row_to_drop)} rows with ANY NAN have been dropped: {row_to_drop}.\n')
    
    # drop columns with no variance or 0 variance
    if (col_null_var):
        counts = df_data_clean.nunique()
        to_del = [counts.index[i] for i,v in enumerate(counts) if v == 1]
        if (to_del):
            df_data_clean.drop(to_del, axis=1, inplace=True)
            if (verbose):
                print(f'Following {len(to_del)} features with 0 variance have been dropped: {to_del}.\n')
    
    # drop duplicated rows
    if (row_dupli):
        dups = df_data_clean.duplicated()
        if (dups.any()):
            df_data_clean.drop_duplicates(inplace=True)
            if (verbose):
                print('Duplicated rows have been dropped.\n')

    if (reset_index):
        df_data_clean.reset_index(drop=True, inplace=True)
        if (verbose):
            print('Index of dataframe has been reset.\n')
    if (verbose):
        print(f'Final {len(df_data_clean.keys())} features {list(df_data_clean.keys())} remain.')
    return df_data_clean
# END DEF

#%%
def runParallelFunction(runFunction, arguments, np=1):
    assert np > 0
    p = Pool(min(cpu_count(), len(arguments), np))
    results = p.map(runFunction, arguments)
    p.close()
    return results
# END DEF

#%%
def dropFeatCorr(df_data, corr_thres=0.9, corr_method='pearson', mode='pair', ignore_keys=[], verbose=True):
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
            list_to_drop = list(set(list_to_drop))
            if (verbose):
                print(f'Following {len(list_to_ignore)} features have been enforced/ignored: {list_to_ignore}.\n')
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
        df_data_orig.drop(list_corr_pair, axis=1, inplace=True)
        return df_data_orig
    else:
        raise NotImplementedError
# END DEF