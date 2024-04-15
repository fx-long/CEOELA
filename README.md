[![DOI](https://zenodo.org/badge/480536772.svg)](https://zenodo.org/badge/latestdoi/480536772)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10397153.svg)](https://doi.org/10.5281/zenodo.10397153)

# CEOELA
Characterizing Engineering Optimization with Exploratory Landscape Analysis (CEOELA) is an automated pipeline developed for characterizing real-world black-box optimization problems based on Benchmarking Black-Box Optimization (BBOB) problem set and artificially generated function set.

## Dependencies

* `pflacco`, avaiable at [here](https://github.com/Reiyan/pflacco).
* Refer to `requirements.txt`.

## Exemplary Use Case

To use this pipeline, you need to prepare an initial DOE, consisting of the x- and y-data of your optimization problem, e.g., in an Excel file similar to `DOE_example.xlsx`.

In this example we assume the BBOB F1 Sphere function as our real-world optimization problem instance, which will be characterized using 10 BBOB functions and 5 randomly generated functions (RGF).

```python
import os
import pandas as pd
from ceoela import ceoela_pipeline, excel2doe


#%%
def main():
    # read doe data
    filepath = os.path.join(os.getcwd(), 'DOE_example.xlsx')
    X, y, lb, ub = excel2doe(filepath)
    dim = X.shape[1]
    
    # initialize CEOELA pipeline
    ela_pipeline = ceoela_pipeline(
        X,
        y,
        lower_bound = lb,
        upper_bound = ub,
        normalize_x = True,
        normalize_x_lower = [-5.0]*dim,
        normalize_x_upper = [5.0]*dim,
        problem_label = 'example',
        path_output = '',
        bootstrap = True,
        bs_ratio = 0.8,
        bs_repeat = 2,
        list_fid = [i+1 for i in range(10)],
        list_iid = [i+1 for i in range(1)],
        genf_number = 5,
        seed = 1,
        n_jobs = 1,
        verbose = True,
    )
    # compute ELA features
    ela_pipeline(ela_problem=True, ela_bbob=True, ela_genf=True)
    
    # analyze ELA features
    ela_pipeline.analyzeELA()
# END DEF


#%%
if __name__ == '__main__':
    main()
# END IF
```

## Citation
 
If our CEOELA pipeline is utlized in your research, please cite us as follows:

Fu Xing Long, Bas van Stein, Moritz Frenzel, Peter Krause, Markus Gitterle, and Thomas Bäck. 2024. Generating Cheap Representative Functions for Expensive Automotive Crashworthiness Optimization. ACM Trans. Evol. Learn. Optim. Just Accepted (February 2024). https://doi.org/10.1145/3646554

```
@article{10.1145/3646554,
author = {Long, Fu Xing and van Stein, Bas and Frenzel, Moritz and Krause, Peter and Gitterle, Markus and B\"{a}ck, Thomas},
title = {Generating Cheap Representative Functions for Expensive Automotive Crashworthiness Optimization},
year = {2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3646554},
doi = {10.1145/3646554},
note = {Just Accepted},
journal = {ACM Trans. Evol. Learn. Optim.},
month = {feb},
keywords = {automotive crashworthiness, black-box optimization, single-objective, exploratory landscape analysis, representative functions.}
}
```

## Contact

If you have any suggestions or ideas, or if you encounter any problems while running the code, please use the issue tracker or send me an e-mail (f.x.long@liacs.leidenuniv.nl).

## Reference
* Fu Xing Long, Bas van Stein, Moritz Frenzel, Peter Krause, Markus Gitterle, and Thomas Bäck. 2022. Learning the Characteristics of Engineering Optimization Problems with Applications in Automotive Crash. In Genetic and Evolutionary Computation Conference (GECCO ’22), July 9–13, 2022, Boston, MA, USA. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3512290.3528712
* Tian, Y., Peng, S., Zhang, X., Rodemann, T., Tan, K. C., & Jin, Y. (2020). A recommender system for metaheuristic algorithms for continuous optimization based on deep recurrent neural networks. IEEE Transactions on Artificial Intelligence, 1(1), 5-18.
* Kerschke, P., & Dagefoerde, J. (2017). flacco: Feature-based landscape analysis of continuous and constrained optimization problems. R-package version, 1.
