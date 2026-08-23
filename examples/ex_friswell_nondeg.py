import time

import numpy as np

import eigen_derivatives as ed


def example_matrix(x:np.ndarray):
    """
    Example of parameterized matrix with non-degenerate eigenvalues.
    """
    evaluation = np.array(
        [[1, 0],
         [0, 2]]
    )
    derivative1 = np.array(
        [[x[0], -x[1]],
         [-x[1], -x[0]]]
    )
    derivative2 = np.zeros((2, 2))
    return ed.DerivativesList([evaluation, derivative1, derivative2])


if __name__ == '__main__':
    # unperturbed eigenvalues
    example = example_matrix(np.array([0, 0]))
    orig_eigenvalues, orig_eigenvectors = np.linalg.eigh(example[0])

    print(f"Unperturbed eigenvalues:\n{orig_eigenvalues}\n")
    print(f"Unperturbed eigenvectors (columns):\n{orig_eigenvectors}\n")

    # perturbation
    perturbation = np.array([1, 1])
    perturbed_matrix = example_matrix(perturbation)

    mass_matrix_derivatives = ed.DerivativesList([np.eye(2), np.zeros((2,2)), np.zeros((2,2))])

    start_time = time.perf_counter()

    # derivatives with respect to eigenspace (mass matrix optional here)
    eigenvalue_derivatives, eigenvector_derivatives = ed.eigenpair_derivatives(
        orig_eigenvalues[0], orig_eigenvectors[:,0:1], perturbed_matrix, mass_matrix_derivatives
    )

    print(f"Eigenvalue derivatives:\n{eigenvalue_derivatives}\n")
    print(f"Approximation of eigenvalue:\n{eigenvalue_derivatives.evaluate_taylor(dx = 1.).item()}\n")

    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Example took {execution_time:.6f} seconds to run.")
