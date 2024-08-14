import numpy as np

def LU_decomposition_with_pivoting(A):
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n - 1):
        max_row = np.argmax(np.abs(U[k:, k])) + k
        if max_row != k:
            U[[k, max_row]] = U[[max_row, k]]
            P[[k, max_row]] = P[[max_row, k]]


        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

    return L, U, P


def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros_like(b, dtype=np.double)  #размерность вектора y, такой же размерности что и б
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y


def backward_substitution(U, y):
    n = U.shape[0]
    x = np.zeros_like(y, dtype=np.double)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x


def inverse_matrix(A):
    L, U, P = LU_decomposition_with_pivoting(A)
    n = A.shape[0]
    inv_A = np.zeros((n, n))

    for i in range(n):
        b = np.zeros(n)
        b[i] = 1
        y = forward_substitution(L, np.dot(P.T, b))
        x = backward_substitution(U, y)
        inv_A[:, i] = x

    return inv_A

def lehmer_matrix(n):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = min(i + 1, j + 1) / max(i + 1, j + 1)
    return A


def inverse_lehmer_matrix(n):
    A = lehmer_matrix(n)
    inv_A = inverse_matrix(A)
    return A,inv_A, check_tridiagonal(inv_A)

def check_tridiagonal(matrix):
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and matrix[i, j] != 0:
                return False
    return True

for n in range(2, 6):
    print(f"n = {n}:")
    A,inv_A, is_tridiagonal = inverse_lehmer_matrix(n)
    print("Матрица Лемера:")
    print(A)
    print("Обратная матрица:")
    print(inv_A)
    print("Трехдиагональная:",is_tridiagonal)
    print()