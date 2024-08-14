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
    y = np.zeros_like(b, dtype=np.double)
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
            A[i, j] = min(i + 1, j + 1)
    return A

def condition_number(A):
    inv_A = inverse_matrix(A)
    norm_1_A = np.linalg.norm(A, ord=1)
    norm_2_A = np.linalg.norm(A)
    norm_A_Fro = np.linalg.norm(A, ord='fro') #используем Фробениусову норму, вычисляем как корень из суммы квадратов ее элемента
    norm_1_inv_A = np.linalg.norm(inv_A, ord=1)
    norm_2_inv_A = np.linalg.norm(inv_A)
    norm_Fro_inv_A = np.linalg.norm(inv_A, ord='fro')
    condition_number_1 = norm_1_A * norm_1_inv_A
    condition_number_2 = norm_2_A * norm_2_inv_A
    condition_number_Fro = norm_A_Fro * norm_Fro_inv_A
    return condition_number_1, condition_number_2, condition_number_Fro


def check_tridiagonal(matrix):
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and matrix[i, j] != 0:
                return False
    return True


for n in range(2, 6):
    print(f"n = {n}:")
    A = lehmer_matrix(n)
    cond_1, cond_2, cond_Fro = condition_number(A)
    print("Число обусловленности для 1-нормы:", cond_1)
    print("Число обусловленности для Евклидовой нормы(беск):", cond_2)
    print("Число обусловленности для нормы Фробениуса:", cond_Fro)
    # print("Число обусловленности:", condition_number(A))
    inv_A = inverse_matrix(A)
    print("Обратная матрица:")
    print(inv_A)
    print("Является ли трехдиагональной:", check_tridiagonal(inv_A))
    print()