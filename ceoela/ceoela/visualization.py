import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram


#%%
def plot_dendrogram(linkage_matrix, path_dir, label='', **kwargs):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(6,3) ,dpi=300)
    plt.axhline(y=kwargs['color_threshold'], color='k', linestyle='--')
    dendrogram(linkage_matrix, **kwargs)
    plt.xticks(rotation=90, ha='center')
    if (label):
        ax.set_title(label)
    ax.set_ylabel('Euclidean')
    if not (os.path.isdir(path_dir)):
        os.makedirs(path_dir)
    filepath = os.path.join(path_dir, f'plot_cluster_{label}.pdf')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
# END DEF


#%%
def plot_ela_space(crash, bbob, rfg, path_dir, label=''):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(5,5) ,dpi=300)
    sns.scatterplot(data=rfg, x='x0', y='x1', s=75, color='gray', alpha=0.2)
    g = sns.scatterplot(data=bbob, x='x0', y='x1', s=75, color=sns.color_palette('tab10', 1), alpha=0.9)
    crash.rename(columns={'type': 'function'}, inplace=True)
    sns.scatterplot(data=crash, x='x0', y='x1', style='function', s=100, color=sns.color_palette('tab10', 2)[1], alpha=1.0, markers=['X', 's', 'P', '^', '*'])
    if (label):
        ax.set_title(label)
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    if not (os.path.isdir(path_dir)):
        os.makedirs(path_dir)
    filepath = os.path.join(path_dir, f'plot_scatter_{label}.pdf')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
# END DEF