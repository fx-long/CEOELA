# CEOELA
Characterizing Engineering Optimization with Exploratory Landscape Analysis

## Exemplary Use Case

```python
from CEOELA.CEOELA import ELA_pipeline

# initliaze
ela_pipeline = ELA_pipeline(r"C:\Users\Q521100\Desktop\Workspace_local\Landscape_Analysis\CEO-ELA_pipeline\crash_data\DOE_BBOB_Crash.xlsx",
                            list_sheetname = [],
                            crash_label = 'D4_P2',
                            filepath_save = '',
                            bootstrap_size = 0.8,
                            bootstrap_repeat = 2,
                            bootstrap_seed = 0,
                            BBOB_func = ['F1'], 
                            BBOB_instance = [1],
                            BBOB_seed = 0,
                            AF_number = 300,
                            AF_seed = 0,
                            os_system = 'windows',
                            verbose = True
                            )


# data pre-processing
ela_pipeline.DataPreProcess()

# computation of ELA features
ela_pipeline.ComputeELA(ELA_crash=True, ELA_BBOB=True, ELA_AF=True)

# processing of ELA features
ela_pipeline.ProcessELA(list_obj_hl=[], list_obj_ignore=[], corr_thres=0.95, corr_ignore=[])

# comparison of ELA features
ela_pipeline.CompareELA()
```

## Cite Us
 
When using CEOELA for your research, please cite us as follows:

Fu Xing Long, Bas van Stein, Moritz Frenzel, Peter Krause, Markus Gitterle, and Thomas Bäck. 2022. Learning the Characteristics of Engineering Optimization Problems with Applications in Automotive Crash. In Genetic and Evolutionary Computation Conference (GECCO ’22), July 9–13, 2022, Boston, MA, USA. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3512290.3528712

