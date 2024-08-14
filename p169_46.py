import numpy as np

def cyclic_LU_solve(A, f):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # LU-разложение
    for k in range(n - 1):
        L[k, k] = 1
        U[k, k] = A[k, k]
        L[k + 1, k] = A[k + 1, k] / U[k, k]
        U[k, k + 1] = A[k, k + 1]
        A[k + 1, k + 1] -= L[k + 1, k] * U[k, k + 1]

    L[-1, -1] = 1
    U[-1, -1] = A[-1, -1] - L[-1, -2] * U[-2, -1]

    # Решение системы Ly = f
    y = np.zeros(n)
    for i in range(n):
        y[i] = f[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]
        y[i] /= L[i, i]

    # Решение системы Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]

    return x


n = 10
h = 1 / n
def cyclic_tridiagonal_matrix(n, h):
    a_ii = 2 + h**2
    a_i_iminus1 = -1
    a_i_iplus1 = -1
    A = np.zeros((n, n))
    for i in range(n):
        A[0,0]=-1
        A[i, i] = a_ii
        A[i, (i + 1) % n] = a_i_iplus1
        A[i, (i - 1) % n] = a_i_iminus1
        A[-1, -1] = -1
    print(A)
    return A

f = np.array([(1 + 4 / h ** 2 * np.sin(np.pi * h) ** 2) * np.sin(2 * np.pi * (i) * h) for i in range(n)])
A = cyclic_tridiagonal_matrix(n, h)
x = cyclic_LU_solve(A, f)
print("Решение системы Ax = f:")
print(x)

