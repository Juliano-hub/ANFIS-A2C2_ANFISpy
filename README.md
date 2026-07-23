# ANFIS-A2C2

Implementation of **ANFIS-A2C2** based on the paper:

> [ANFISpy: a python package for neuro-fuzzy models](https://link.springer.com/article/10.1007/s00521-026-11914-6)

This implementation extends the original ANFIS implementation available at:

https://github.com/mZaiam/ANFISpy


# ANFISpy 
A Python implementation of **Adaptive Neuro-Fuzzy Inference Systems (ANFIS)**, combining neural networks and fuzzy logic for interpretable machine learning. The implementation is based on the original [ANFIS](https://ieeexplore.ieee.org/abstract/document/256541?casa_token=bWStLllx3e8AAAAA:Z7Tj7kk-7lHlGSIEVJZfJVtRi_IVpig2ANbVv6qou4Ok32c7X7Yfh8SsvIUUBjALl3dfHRgFRJs3) paper, adapting the model to perform both regression and classification tasks with customizable membership functions. A Recurrent ANFIS (RANFIS), a Gated Recurrent Unit ANFIS (GRU-ANFIS) and Long-Short Term memmory ANFIS (LSTM-ANFIS) are also implemented, suited for time series regression and classification. A Coactive ANFIS (CANFIS) is also included, for deeper neuro-fuzzy models.

# Key Features
- **Regression and Classification**
- **Time Series Analysis with RANFIS, GRU-ANFIS and LSTM-ANFIS**
- **Deep Neuro-Fuzzy models with CANFIS**
- **Visualization and Interpretability** via `.print_rules()`, `.plot_var()`, `.plot_rules()`, `.rule_activations()`  
- **Various Membership Functions** (`GaussianMF`, `BellMF`, `TriangularMF`, `SigmoidMF`) **and T-Norms** (`MinAND`, `ProdAND`, `HamacherAND`, `FrankAND`, `LukasiewiczAND`)
- **PyTorch Integration** (GPU acceleration, optimizers, ...) 

# Repository Organization
The repository is organized in the following directories:
- **ANFISpy**: has the implementation of the ANFIS's based models;
- **examples**: has jupyter-notebooks with examples of how to use the models;
- **tests**: has testing files for managing the code behaviour.

# Installation
The installation of the package can be done using `pip` in a `bash` terminal:

```bash
pip install anfispy
```
Then, the package can be imported in Python using:

```python
from ANFISpy import ANFIS
from ANFISpy import RANFIS
```

# Quick Example
The ANFIS model can be used to perform both regression and classification, as explained in [anfis_example.ipynb](https://github.com/mZaiam/ANFISpy/blob/main/examples/anfis_example.ipynb). To instantiate a regression model, set the value of `n_classes` in the `output` to 1.

```python
from ANFISpy import ANFIS

n_vars = 3
mf_names = [['L', 'M', 'H']]

variables = {
    'inputs': {
        'n_sets': [3, 3, 3],
        'uod': n_vars * [(0, 1)],
        'var_names': ['var1', 'var2', 'var3'],
        'mf_names': n_vars * mf_names,
    },
    'output': {
        'var_names': 'out',
        'n_classes': 1,
    },
}

anfis_regression = ANFIS(variables, 'gaussian')
```

To create a clasification model, set the value of `n_classes` in the `output` to a number of classes greater or equal to 2. 

```python
from ANFISpy import ANFIS

n_vars = 3
mf_names = [['L', 'M', 'H']]

variables = {
    'inputs': {
        'n_sets': [3, 3, 3],
        'uod': n_vars * [(0, 1)],
        'var_names': ['var1', 'var2', 'var3'],
        'mf_names': n_vars * mf_names,
    },
    'output': {
        'var_names': 'out',
        'n_classes': 3,
    },
}

anfis_classification = ANFIS(variables, 'bell')
```