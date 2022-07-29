[![DOI](https://zenodo.org/badge/480536772.svg)](https://zenodo.org/badge/latestdoi/480536772)

# CEOELA
Characterizing Engineering Optimization with Exploratory Landscape Analysis (CEOELA) is an automated pipeline developed for characterizing real-world black-box optimization problems based on Benchmarking Black-Box Optimization (BBOB) problem set and artificially generated function set.

## Dependencies

* `Python 3.8` and `R 3.6.1`
* `rpy2`
* `flacco`, available at [here](https://github.com/kerschke/flacco).

## Exemplary Use Case

To use this pipeline, you need to prepare an initial DOE, consisting of the x- and y-data of your optimization problem. To facilitate our workflow working with the commercial software HyperStudy, we store our DOE data set in form of an Excel file. The Excel template is available [here](https://github.com/fx-long/CEOELA/blob/main/CEOELA/doe_template.xlsx).

In this example we treat the BBOB F1 Sphere function as our real-world optimization problem instance, which will be characterized using one BBOB function and 2 artificially generated functions.

```python
import os
from CEOELA.CEOELA import ELA_pipeline


# define path
filepath_excel = os.path.join(os.getcwd(), r'CEOELA\DOE_example.xlsx')

# initliaze
ela_pipeline = ELA_pipeline(filepath_excel = filepath_excel,
                            list_sheetname = [],
                            instance_label = 'result_example',
                            filepath_save = '',
                            bootstrap_size = 0.8,
                            bootstrap_repeat = 2,
                            bootstrap_seed = 0,
                            BBOB_func = ['F1'], 
                            BBOB_instance = [1],
                            BBOB_seed = 0,
                            AF_number = 2,
                            AF_seed = 0,
                            r_home = r'C:\ProgramData\Anaconda3\envs\rstudio\lib\R',
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

## Citation
 
When using CEOELA for your research, please cite us as follows:

Fu Xing Long, Bas van Stein, Moritz Frenzel, Peter Krause, Markus Gitterle, and Thomas Bäck. 2022. Learning the Characteristics of Engineering Optimization Problems with Applications in Automotive Crash. In Genetic and Evolutionary Computation Conference (GECCO ’22), July 9–13, 2022, Boston, MA, USA. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3512290.3528712

## Contact

If you have any suggestions or ideas, or if you encounter any problems while running the code, please use the issue tracker or send me an e-mail (f.x.long@liacs.leidenuniv.nl).

## Reference
* Tian, Y., Peng, S., Zhang, X., Rodemann, T., Tan, K. C., & Jin, Y. (2020). A recommender system for metaheuristic algorithms for continuous optimization based on deep recurrent neural networks. IEEE Transactions on Artificial Intelligence, 1(1), 5-18.
* Kerschke, P., & Dagefoerde, J. (2017). flacco: Feature-based landscape analysis of continuous and constrained optimization problems. R-package version, 1.
