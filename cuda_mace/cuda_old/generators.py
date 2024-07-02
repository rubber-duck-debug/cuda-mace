import numpy as np
import math

def su2_clebsch_gordan(j1: float, j2: float, j3: float) -> np.ndarray:
    """Calculates the Clebsch-Gordon matrix."""
    assert isinstance(j1, (int, float))
    assert isinstance(j2, (int, float))
    assert isinstance(j3, (int, float))
    mat = np.zeros((int(2 * j1 + 1), int(2 * j2 + 1), int(2 * j3 + 1)))
    if int(2 * j3) in range(int(2 * abs(j1 - j2)), int(2 * (j1 + j2)) + 1, 2):
        for m1 in (x / 2 for x in range(-int(2 * j1), int(2 * j1) + 1, 2)):
            for m2 in (x / 2 for x in range(-int(2 * j2), int(2 * j2) + 1, 2)):
                if abs(m1 + m2) <= j3:
                    mat[int(j1 + m1), int(j2 + m2), int(j3 + m1 + m2)] = _su2_cg(
                        (j1, m1), (j2, m2), (j3, m1 + m2)
                    )
    return mat / math.sqrt(2 * j3 + 1)

def su2_generators(j: float) -> np.ndarray:
    """Return the generators of the SU(2) group of dimension `2j+1`."""
    m = np.arange(-j, j)
    raising = np.diag(-np.sqrt(j * (j + 1) - m * (m + 1)), k=-1)

    m = np.arange(-j + 1, j + 1)
    lowering = np.diag(np.sqrt(j * (j + 1) - m * (m - 1)), k=1)

    m = np.arange(-j, j + 1)
    return np.stack(
        [
            0.5 * (raising + lowering),  # x (usually)
            np.diag(1j * m),  # z (usually)
            -0.5j * (raising - lowering),  # -y (usually)
        ],
        axis=0,
    )


def change_basis_real_to_complex(l: int) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    q = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.complex128)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / np.sqrt(2)
        q[l + m, l - abs(m)] = -1j / np.sqrt(2)
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** m / np.sqrt(2)
        q[l + m, l - abs(m)] = 1j * (-1) ** m / np.sqrt(2)
    return (
        -1j
    ) ** l * q  # Added factor of 1j**l to make the Clebsch-Gordan coefficients real


def clebsch_gordan(l1: int, l2: int, l3: int) -> np.ndarray:
    r"""The Clebsch-Gordan coefficients of the real irreducible representations of :math:`SO(3)`.

    Args:
        l1 (int): the representation order of the first irrep
        l2 (int): the representation order of the second irrep
        l3 (int): the representation order of the third irrep

    Returns:
        np.ndarray: the Clebsch-Gordan coefficients
    """
    C = su2_clebsch_gordan(l1, l2, l3)
    Q1 = change_basis_real_to_complex(l1)
    Q2 = change_basis_real_to_complex(l2)
    Q3 = change_basis_real_to_complex(l3)
    C = np.einsum("ij,kl,mn,ikn->jlm", Q1, Q2, np.conj(Q3.T), C)

    assert np.all(np.abs(np.imag(C)) < 1e-5)
    return np.real(C)


def generators(l: int) -> np.ndarray:
    r"""The generators of the real irreducible representations of :math:`SO(3)`.

    Args:
        l (int): the representation order of the irrep

    Returns:
        np.ndarray: the generators
    """
    X = su2_generators(l)
    Q = change_basis_real_to_complex(l)
    X = np.conj(Q.T) @ X @ Q

    assert np.all(np.abs(np.imag(X)) < 1e-5)
    return np.real(X)