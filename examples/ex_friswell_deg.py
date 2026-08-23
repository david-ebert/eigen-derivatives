import time

import numpy as np

import eigen_derivatives as ed


def example_matrix_friswell(x:np.ndarray):
    """
    Example of parameterized matrix with degenerate eigenvalues from
    M. I. Friswell, The Derivatives of Repeated Eigenvalues and Their Associated Eigenvectors,
    Journal of Vibration and Acoustics, 1996
    """
    evaluation = np.array(
        [[2, 0],
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
    example = example_matrix_friswell(np.array([0, 0]))
    orig_eigenvalues, orig_eigenvectors = np.linalg.eigh(example[0])

    print(f"Unperturbed eigenvalues:\n{orig_eigenvalues}\n")
    print(f"Unperturbed eigenvectors (columns):\n{orig_eigenvectors}\n")

    # perturbation
    perturbation = np.array([1, 1])
    perturbed_matrix = example_matrix_friswell(perturbation)

    mass_matrix_derivatives = ed.DerivativesList([np.eye(2), np.zeros((2,2)), np.zeros((2,2))])

    start_time = time.perf_counter()

    # derivatives with respect to eigenspace (mass matrix optional here)
    eigenvalue_derivatives, eigenvector_derivatives = ed.eigenpair_derivatives(
        orig_eigenvalues[0], orig_eigenvectors, perturbed_matrix, mass_matrix_derivatives
    )

    # polarization
    initial_polarization, polarization_order = ed.polarize(eigenvalue_derivatives)

    # derivatives of polarization
    polarization_derivatives, polarized_eigenvalues_derivatives = ed.polarization_derivatives(eigenvalue_derivatives,
                                                                                              eigenvector_derivatives,
                                                                                              initial_polarization,
                                                                                              polarization_order)

    print(f"Polarized eigenvalue derivatives:\n{polarized_eigenvalues_derivatives}\n")
    print(f"Approximation of polarization eigenvalues:\n{polarized_eigenvalues_derivatives.evaluate_taylor(dx = 1.)}\n")

    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Example took {execution_time:.6f} seconds to run.")
