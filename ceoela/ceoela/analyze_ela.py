import os
import numpy as np
import pandas as pd
from math import ceil
from copy import deepcopy
from sklearn.manifold import TSNE
from .utils import dropFeatCorr, dataCleaning
from sklearn.preprocessing import StandardScaler
from .visualization import plot_ela_space, plot_dendrogram
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import pairwise_distances



#%%
def read_ela(path_base, list_response, list_bbob_fid, list_bbob_iid, genf_number, label='problem'):
    # problem
    df_problem = pd.DataFrame()
    for response in list_response:
        filename = f'ela_{response}.csv'
        df_ = pd.read_csv(os.path.join(path_base, filename))
        df_['label'] = f'problem_{response}'
        df_problem = pd.concat([df_problem, df_], axis=0, ignore_index=True)
        
    # bbob   
    df_bbob = pd.DataFrame()
    for fid in list_bbob_fid:
        for iid in list_bbob_iid:
            filename = f'ela_bbob_f{fid}_ins{iid}.csv'
            df_ = pd.read_csv(os.path.join(path_base, filename))
            df_['label'] = f'bbob_f{fid}_ins{iid}'
            df_bbob = pd.concat([df_bbob, df_], axis=0, ignore_index=True)
    
    # rgf
    df_rgf = pd.DataFrame()
    for rgf in range(genf_number):
        filename = f'ela_rgf{rgf+1}.csv'
        try:
            df_ = pd.read_csv(os.path.join(path_base, filename))
            df_['label'] = f'rgf{rgf+1}'
            df_rgf = pd.concat([df_rgf, df_], axis=0, ignore_index=True)
        except:
            pass
    
    # filter bad samples   
    df_ela = pd.concat([df_problem, df_bbob, df_rgf], axis=0, ignore_index=True)
    df_clean = dataCleaning(df_ela, replace_nan=False, inf_as_nan=True, col_allnan=False, col_anynan=False, row_anynan=False, col_null_var=False, 
                            row_dupli=False, filter_key=['pca.expl_var.cov_x', 'pca.expl_var.cor_x', 'pca.expl_var_PC1.cov_x', 'pca.expl_var_PC1.cor_x', 
                                                         'pca.expl_var.cov_init', 'pca.expl_var.cor_init'], 
                            reset_index=True, verbose=True)
    df_problem = df_clean[df_clean['label'].str.contains('problem')]
    df_bbob = df_clean[df_clean['label'].str.contains('bbob')]
    df_rgf = df_clean[df_clean['label'].str.contains('rgf')]
    return df_problem, df_bbob, df_rgf
# END DEF
    

#%%
def standardize_ela(ela_problem, ela_bbob, ela_rgf):
    ela_bbob_ = deepcopy(ela_bbob)
    ela_bbob_[['label', 'fid', 'iid']] = ela_bbob_['label'].str.split('_', expand=True)
    ela_bbob_ = ela_bbob_.drop(columns=['label', 'iid'])
    ela_bbob_ = ela_bbob_.groupby(['fid'], as_index=True).mean()
    ela_bbob_ = dataCleaning(ela_bbob_, replace_nan=False, inf_as_nan=True, col_allnan=True, col_anynan=True, row_anynan=False, col_null_var=True,
                             row_dupli=False, filter_key=[], reset_index=False, verbose=True)
    bbob_corr, bbob_pair = dropFeatCorr(ela_bbob_, corr_thres=0.95, corr_method='pearson', mode='pair', ignore_keys=[], verbose=True)
    scaler = StandardScaler()
    bbob_normalize = scaler.fit_transform(bbob_corr)
    bbob_normalize = pd.DataFrame(bbob_normalize, columns=list(bbob_corr.keys()))
    bbob_normalize['type'] = ela_bbob_.index.to_list()
    
    # problem
    ela_problem_ = deepcopy(ela_problem)
    ela_problem_ = ela_problem_.groupby(['label'], as_index=False).mean()
    ela_problem_ = ela_problem_[list(bbob_corr.keys())]
    problem_normalize = scaler.transform(ela_problem_)
    problem_normalize = pd.DataFrame(problem_normalize, columns=list(bbob_corr.keys()))
    problem_normalize['type'] = ela_problem['label']
    
    # rgf
    ela_rgf_ = deepcopy(ela_rgf)
    ela_rgf_ = ela_rgf_.groupby(['label'], as_index=False).mean()
    ela_rgf_ = ela_rgf_[list(bbob_corr.keys())]
    ela_rgf_ = dataCleaning(ela_rgf_, replace_nan=False, inf_as_nan=False, col_allnan=False, col_anynan=False, row_anynan=True, col_null_var=False,
                            row_dupli=False, filter_key=[], reset_index=False, verbose=True)
    rgf_normalize = scaler.transform(ela_rgf_)
    rgf_normalize = pd.DataFrame(rgf_normalize, columns=list(bbob_corr.keys()))
    rgf_normalize['type'] = 'rgf'
    return problem_normalize, bbob_normalize, rgf_normalize


#%%
def clustering_ela(path_base, problem_normalize, bbob_normalize, label='problem'):
    # bbob inter-cluster
    bbob_triu = deepcopy(bbob_normalize)
    bbob_triu.set_index('type', inplace=True)
    bbob_triu = np.triu(pairwise_distances(bbob_triu, metric='euclidean'), k=1)
    bbob_triu[bbob_triu == 0.0] = np.nan
    bbob_intercluster_dist = np.nanmean(bbob_triu)
    
    # clustering
    df_ela = pd.concat([bbob_normalize, problem_normalize], axis=0)
    df_ela = dataCleaning(df_ela, replace_nan=False, inf_as_nan=False, col_allnan=False, col_anynan=True, row_anynan=False, col_null_var=False,
                          row_dupli=False, filter_key=[], reset_index=False, verbose=True)
    df_ela.set_index('type', inplace=True)
    linkage_matrix = linkage(df_ela, method='ward', metric='euclidean')
    path_plot = os.path.join(path_base, 'results_plot')
    plot_dendrogram(linkage_matrix, path_dir=path_plot, label=label, labels=list(df_ela.index), color_threshold=bbob_intercluster_dist, truncate_mode=None, p=2)
# END DEF


#%%
def embedding_tsne_ela(path_base, problem_normalize, bbob_normalize, rgf_normalize, label='problem'):
    df_ela = pd.concat([bbob_normalize, problem_normalize, rgf_normalize], axis=0, ignore_index=True)
    df_ela.set_index('type', inplace=True)
    df_ela = dataCleaning(df_ela, replace_nan=False, inf_as_nan=False, col_allnan=False, col_anynan=True, row_anynan=False, col_null_var=False,
                          row_dupli=False, filter_key=[], reset_index=False, verbose=True)
    perplexity = min(ceil(df_ela.shape[0]/5), 30.0)
    df_embbed = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(df_ela)
    df_embbed = pd.DataFrame(df_embbed, columns=['x0','x1'])
    df_embbed['type'] = list(df_ela.index)
    df_problem = df_embbed[df_embbed['type'].str.contains('problem')]
    df_bbob = df_embbed[df_embbed['type'].str.contains('bbob')]
    df_rgf = df_embbed[df_embbed['type'].str.contains('rgf')]
    path_plot = os.path.join(path_base, 'results_plot')
    plot_ela_space(df_problem, df_bbob, df_rgf, path_dir=path_plot, label=label)
# END DEF



#%%
def analyze_ela(path_base, list_response, list_bbob_fid, list_bbob_iid, list_rgf, label='problem'):
    # preprocessing
    ela_problem, ela_bbob, ela_rgf = read_ela(path_base, list_response, list_bbob_fid, list_bbob_iid, list_rgf, label=label)
    problem_normalize, bbob_normalize, rgf_normalize = standardize_ela(ela_problem, ela_bbob, ela_rgf)
    
    # clustering
    clustering_ela(path_base, problem_normalize, bbob_normalize, label=label)
    
    # ELA feature space
    # TODO: visualize the feature space
    # embedding_tsne_ela(path_base, problem_normalize, bbob_normalize, rgf_normalize, label=label)
# END DEF