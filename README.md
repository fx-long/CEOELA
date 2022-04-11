# CEOELA
Characterizing Engineering Optimization with Exploratory Landscape Analysis

## Exemplary Use Case

```python
from modulesAutoOpti.ELA_pipeline import ELA_pipeline

# initliaze
ela_pipeline = ELA_pipeline(r"C:\Users\Q521100\Desktop\Workspace_local\Landscape_Analysis\CEO-ELA_pipeline\crash_data\DOE_BBOB_Crash.xlsx",
                            list_sheetname = [],
                            crash_label = 'D4_P2',
                            filepath_save = '',
                            bootstrap_size = 0.8,
                            bootstrap_repeat = 2,
                            bootstrap_seed = 0,
                            BBOB_func = ['F1'], # ['F'+str(i+1) for i in range(24)],
                            BBOB_instance = [1],
                            BBOB_seed = 0,
                            AF_number = 300,
                            AF_seed = 0,
                            os_system = 'windows',
                            verbose = True
                            )

# data pre-processing
ela_pipeline.DataPreProcess(preanalysis=False, preanalysis_plot=False)

# computation of ELA features
ela_pipeline.ComputeELA(ELA_crash=True, ELA_BBOB=True, ELA_AF=True)

# processing of ELA features
ela_pipeline.ProcessELA(list_obj_hl=[], list_obj_ignore=[], corr_thres=0.95, corr_ignore=[])

# comparison of ELA features
ela_pipeline.CompareELA()
```

## Cite Us

When using CEOELA for your research, please cite us as follows:

```bibtex
@inproceedings{van2019automatic,
    title={Automatic Configuration of Deep Neural Networks with Parallel Efficient Global Optimization},
    author={van Stein, Bas and Wang, Hao and B{\"a}ck, Thomas},
    booktitle={2019 International Joint Conference on Neural Networks (IJCNN)},
    pages={1--7},
    year={2019},
    organization={IEEE}
}
```
