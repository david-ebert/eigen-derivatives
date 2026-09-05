# eigen-derivatives

A Python package for computing higher-order derivatives of eigenvalues and eigenvectors, supporting both dense NumPy
arrays and SciPy sparse matrices.

This project is an optimized Python translation of the original MATLAB library `eigen_derivatives`:
https://github.com/david-ebert/eigen_derivatives

## Key Features

- Efficient calculation of
  - derivatives of eigenpairs for non-degenerate eigenpairs
  - derivatives of eigenpairs with respect to the eigenspace for degenerate eigenpairs
  - polarization matrix and derivatives for degenerate eigenpairs
- `DerivativeSeries` container object for passing of Sequences of Derivatives and efficient Taylor approximation via
  Horner, with `truncate` and `pad_with_zeros` to adjust the highest order
- Handles NumPy and SciPy matrices. The derivatives are always returned as dense NumPy arrays, regardless of the input type, since the eigenvector derivatives of a sparse matrix are generally dense.

## Installation

Currently only local installation available.
```bash
pip install -e .
```

## Usage

### Non-degenerate eigenvalue

```python
import numpy as np
import eigen_derivatives as ed

# evaluation and derivatives of the parametrized matrix at the expansion point
K0 = np.array([[1.0, 0.0], [0.0, 2.0]])
K1 = np.array([[1.0, -1.0], [-1.0, -1.0]])
K2 = np.zeros((2, 2))
stiffness_ds = ed.DerivativeSeries((K0, K1, K2))

# eigenvalue problem of the unperturbed matrix
unperturbed = np.linalg.eigh(stiffness_ds[0])

# derivatives up to the order the input allows
eigenvalue_ds, eigenvector_ds = ed.eigenpair_derivatives(
    unperturbed.eigenvalues[0], unperturbed.eigenvectors[:, 0:1], stiffness_ds
)

print(eigenvalue_ds[1])                    # [[1.0]], the first derivative, shaped by the multiplicity
print(eigenvalue_ds[2])                    # [[-2.0]], the second derivative
print(eigenvalue_ds.evaluate_taylor(0.1))  # [[1.09]], exact value 1.0877
```

### Degenerate eigenvalues

Example from Friswell (1996). Both eigenvalues coincide at the expansion point and separate in first order, so the
eigenvectors have to be polarized before they can be expanded.

```python
import numpy as np
import eigen_derivatives as ed

K0 = 2.0 * np.eye(2)
K1 = np.array([[1.0, -1.0], [-1.0, -1.0]])
K2 = np.zeros((2, 2))
stiffness_ds = ed.DerivativeSeries((K0, K1, K2))

unperturbed = np.linalg.eigh(stiffness_ds[0])

# the whole eigenspace is passed, not a single eigenvector
eigenvalue_ds, eigenvector_ds = ed.eigenpair_derivatives(
    unperturbed.eigenvalues[0], unperturbed.eigenvectors, stiffness_ds
)

# polarization, and the order at which each pair of eigenvalues separates
initial_polarization, polarization_order = ed.polarize(eigenvalue_ds)
print(polarization_order)  # [[1., 1.], [1., 1.]], pairs that never separate are inf

polarization_ds, polarized_eigenvalue_ds = ed.polarization_derivatives(
    eigenvalue_ds, eigenvector_ds, initial_polarization, polarization_order
)
polarized_eigenvector_ds = ed.polarized_eigenvectors(eigenvector_ds, polarization_ds)

print(polarized_eigenvalue_ds.evaluate_taylor(1.0))  # [0.5858, 3.4142], the exact eigenvalues 2 -+ sqrt(2)

print(polarized_eigenvector_ds.evaluate_taylor(0.1))              # NaN throughout
print(polarized_eigenvector_ds.truncate(1).evaluate_taylor(0.1))  # matches eigh(K0 + 0.1 * K1)
```

The highest orders of the polarization derivatives are not determined by the supplied eigenvalue derivatives and are
returned as `NaN`, which is why the last line truncates the series to the orders the input supports before evaluating
it.

M. I. Friswell, *The Derivatives of Repeated Eigenvalues and Their Associated Eigenvectors*,
Journal of Vibration and Acoustics 118(3), 1996. https://doi.org/10.1115/1.2888195
