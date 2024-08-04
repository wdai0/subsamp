# subsamp Feature Selection with Subsampling Winner Algorithm

Subsampling Winner Algorithm (SWA)

SubsampWinner is a Python package that implements the Subsampling Winner Algorithm (SWA) for feature selection in high-dimensional datasets.
It includes a robust double assurance procedure to enhance stability and reliability in feature selection.

## Features

- Subsampling Winner Algorithm (SWA) for efficient feature selection;
- Double Assurance procedure for improved stability;
- Support for both homoskedastic and heteroskedastic data;
- Parallel processing capabilities for improved performance;
- Flexible parameter tuning and multiple testing correction methods.

## Installation

You can install SubsampWinner using pip:

```bash
pip install subsamp
```

## Quick Start

```python
import numpy as np
from subsamp import subsamp, SubsampDoubleAssurance
from GenerateData import generate_heteroskedastic_data

# Generate sample data
n, p = 100, 80
beta0 = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
beta0_index = np.arange(len(beta0))
beta = np.zeros(p)
beta[beta0_index] = beta0
gamma = np.zeros(p)

X, y, _, _ = generate_heteroskedastic_data(n, p, hetero_func=lambda x: 3, 
    beta=beta, gamma=gamma, type='diagonal')

# Initialize and run SWA
swa = subsamp(s=20, m=1000, qnum=15)
swa.fit(X, y)

print("Selected features:", swa.finalists)

# Run Double Assurance procedure
sda = SubsampDoubleAssurance(m=1000)
results = sda.double_assurance(X, y, s0=26, T=0.9, I_max=20, init_range=0.3, r=0.75)
```
