# eigen-derivatives

A Python package for computing higher-order derivatives of eigenvalues and eigenvectors, supporting both dense NumPy
arrays and SciPy sparse matrices.

This project is an optimized Python translation of original MATLAB library eigen_space:
https://github.com/david-ebert/eigen_derivative

## Key Features

- Efficient calculation of
  - derivatives of eigenpairs for non-degenerate eigenpairs
  - derivatives of eigenpairs with respect to the eigenspace for degenerate eigenpairs
  - polarization matrix and derivatives for degenerate eigenpairs
- `DerivativesList` container object for passing of Sequences of Derivatives and efficient Taylor approximation (Horner)
- Handles NumPy and SciPy matrices.

## Installation

```bash
pip install -e .
```

## Usage

```python
import numpy as np
import eigen_derivatives as ed

eigenvalues, eigenvectors = np.linalg.eigh(K0)
stiffness_derivatives = ed.DerivativesList([K0, K1, K2])

dL, dQ = ed.eigenpair_derivatives(eigenvalues[0], eigenvectors[:,0], stiffness_derivatives)
```
