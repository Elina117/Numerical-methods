import numpy as np

def tridiag_eigval(A):
    n = len(A)
    eigvals, eigvecs = np.linalg.eigh(A)
    min_eigval_index = np.abs(eigvals).argmin()
    min_eigval = eigvals[min_eigval_index]
    min_eigvec = eigvecs[:, min_eigval_index]
    return min_eigval, min_eigvec

def generate_tridiag_matrix(n):
    main_diag = 2 * np.ones(n)
    off_diag = -1 * np.ones(n - 1)
    A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    return A

n_values = [3, 4, 5, 6]

for n in n_values:
    A = generate_tridiag_matrix(n)
    min_eigval, min_eigvec = tridiag_eigval(A)
    print(f"n = {n}:")
    print("Трехдиагональная матрица:")
    print(A)
    print("Численное минимальное собственное значение:", min_eigval)
    exact_eigval = 2 - 2 * np.cos(np.pi / (n + 1))
    print("Точное минимальное собственное значение:", exact_eigval)
