
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from scipy.cluster.hierarchy import dendrogram


#%%
def plot_contour(X, Y, Z, path_dir, label=''):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(1,1,dpi=300)
    cp = ax.contourf(X, Y, Z)
    fig.colorbar(cp)
    if not (os.path.isdir(path_dir)):
        os.makedirs(path_dir)
    filepath = os.path.join(path_dir, f'plot_contourf_{label}_target.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
# END DEF

#%%
def plot_surface(X, Y, Z, path_dir, label=''):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    cp = ax.plot_surface(X, Y, Z, cmap='viridis')
    if not (os.path.isdir(path_dir)):
        os.makedirs(path_dir)
    filepath = os.path.join(path_dir, f'plot_contour3d_{label}_target.png')
    ax.view_init(45, -135)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
# END DEF

#%%
def plot_heatmap(X, label=''):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams.update({'font.size': 5})
    fig, ax = plt.subplots(figsize=(40, 40) ,dpi=300)
    ax = sns.heatmap(X, linewidth=0.1, cmap='viridis')
    path_dir = os.path.join(os.getcwd(), 'plots', f'{label}')
    if not (os.path.isdir(path_dir)):
        os.makedirs(path_dir)
    filepath = os.path.join(path_dir, f'plot_heatmap_{label}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
# END DEF

#%%
def plot_barplot(data, path_dir, label=''):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(18, 3) ,dpi=300)
    plt.axhline(y=data['dist'].mean(), color='k', linestyle='--')
    sns.boxplot(data=data, x='label', y='dist', palette=sns.color_palette('tab10', 1))
    if not (os.path.isdir(path_dir)):
        os.makedirs(path_dir)
    filepath = os.path.join(path_dir, f'plot_barplot_{label}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
# END DEF

#%%
def plot_scatterplot(crash, bbob, rfg, path_dir, label=''):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(5,5) ,dpi=300)
    sns.scatterplot(data=rfg, x='x0', y='x1', s=75, color='gray', alpha=0.2)
    sns.scatterplot(data=bbob, x='x0', y='x1', s=75, color=sns.color_palette('tab10', 1), alpha=.9)
    crash['type'] = crash['type'].str.replace('crash_', '')
    crash['type'] = crash['type'].str.replace('Mass_total', '$M$')
    crash['type'] = crash['type'].str.replace('Force_pole', '$F_{max}$')
    crash['type'] = crash['type'].str.replace('Intrusion', '$Intr$')
    crash['type'] = crash['type'].str.replace('Energy_total', '$EA$')
    # crash['type'] = crash['type'].str.replace('Mass', '$EA$')
    crash.rename(columns={'type': 'objective'}, inplace=True)
    sns.scatterplot(data=crash, x='x0', y='x1', style='objective', s=100, color=[sns.color_palette('tab10', 2)[1]], alpha=1., markers=['X', 's', 'P', '^'])
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    if not (os.path.isdir(path_dir)):
        os.makedirs(path_dir)
    filepath = os.path.join(path_dir, f'plot_scatter_{label}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
# END DEF

#%%
def plot_parallel_coordinate(data, path_dir, label=''):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(18,6) ,dpi=300)
    parallel_coordinates(data, 'type', ax=ax, color=('grey', 'blue', 'orange', 'green', 'red'))
    plt.xticks(rotation=90, ha='center')
    if not (os.path.isdir(path_dir)):
        os.makedirs(path_dir)
    filepath = os.path.join(path_dir, f'plot_parallelcoord_{label}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
# END DEF

#%%
def plot_dendrogram(linkage_matrix, path_dir, label='', **kwargs):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(6,3) ,dpi=300)
    kwargs['labels'] = [label.replace('crash_', '') for label in kwargs['labels']]
    kwargs['labels'] = [label.replace('Mass_total', 'Mass') for label in kwargs['labels']]
    kwargs['labels'] = [label.replace('Force_pole', 'Force') for label in kwargs['labels']]
    kwargs['labels'] = [label.replace('Intrusion', 'Intrusion') for label in kwargs['labels']]
    # kwargs['labels'] = [label.replace('Rotation', '$Rot$') for label in kwargs['labels']]
    kwargs['labels'] = [label.replace('Energy_total', 'Energy') for label in kwargs['labels']]
    dendrogram(linkage_matrix, color_threshold=.1, **kwargs)
    plt.xticks(rotation=90, ha='center')
    for i in range(len(ax.get_xticklabels())):
        key_label = ax.get_xticklabels()[i].get_text()
        if (key_label in ['Mass', 'Force', 'Intrusion', '$Rot$', 'Energy']):
            ax.get_xticklabels()[i].set_bbox(dict(boxstyle='round', pad=.2, fc='orange', alpha=.3))
    ax.set_ylabel('Euclidean')
    if not (os.path.isdir(path_dir)):
        os.makedirs(path_dir)
    filepath = os.path.join(path_dir, f'plot_dendrogram_{label}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
# END DEF