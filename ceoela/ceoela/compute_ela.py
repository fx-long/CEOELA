import math
import pandas as pd
from sklearn.utils import resample
import pflacco.classical_ela_features as pflacco_ela
from .utils import dataCleaning




#%%
def compute_ela(X, y, lower_bound=-5.0, upper_bound=5.0, normalize_y=True):
    if (normalize_y):
        y = (max(y) - y) / (max(y)-min(y))
    # Calculate ELA features
    ela_meta = pflacco_ela.calculate_ela_meta(X, y)
    ela_distr = pflacco_ela.calculate_ela_distribution(X, y)
    try:
        ela_level = pflacco_ela.calculate_ela_level(X, y)
    except Exception as e:
        print(e)
        ela_level = {}
    pca = pflacco_ela.calculate_pca(X, y)
    limo = pflacco_ela.calculate_limo(X, y, lower_bound, upper_bound)
    nbc = pflacco_ela.calculate_nbc(X, y)
    disp = pflacco_ela.calculate_dispersion(X, y)
    ic = pflacco_ela.calculate_information_content(X, y, seed=100)
    ela_ = {**ela_meta, **ela_distr, **ela_level, **pca, **limo, **nbc, **disp, **ic}
    df_ela = pd.DataFrame([ela_])
    df_clean = dataCleaning(df_ela, replace_nan=False, inf_as_nan=False, col_allnan=False, col_anynan=False, row_anynan=False, col_null_var=False, 
                            row_dupli=False, filter_key=['.costs_runtime'], reset_index=False, verbose=False)
    return df_clean
# END DEF

#%%
def bootstrap_ela(X, y, lower_bound=-5.0, upper_bound=5.0, normalize_y=True, bs_ratio=0.8, bs_repeat=2, bs_seed=42):
    assert 0.0 < bs_ratio < 1.0
    assert bs_repeat > 0
    num_sample = int(math.ceil(len(X) * bs_ratio))
    df_ela = pd.DataFrame()
    for i_bs in range(bs_repeat):
        i_bs_seed = i_bs + bs_seed
        X_, y_ = resample(X, y, replace=False, n_samples=num_sample, random_state=i_bs_seed, stratify=None)
        ela_ = compute_ela(X_, y_, lower_bound=lower_bound, upper_bound=upper_bound, normalize_y=normalize_y)
        df_ela = pd.concat([df_ela, ela_], axis=0, ignore_index=True)
    return df_ela
# END DEF