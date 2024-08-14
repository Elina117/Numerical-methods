import numpy as np
import math


def partial_pivot_lu_decomposition(matrix):
    n = len(matrix)
    LU = np.copy(matrix)
    P = np.eye(n)  # Матрица перестановок

    for k in range(n - 1):
        # Выбор главного элемента
        max_index = np.argmax(np.abs(LU[k:, k])) + k
        if max_index != k:
            # Перестановка строк в матрице
            LU[[k, max_index]] = LU[[max_index, k]]
            # Перестановка строк в матрице перестановок
            P[[k, max_index]] = P[[max_index, k]]
        for i in range(k + 1, n):
            LU[i, k] /= LU[k, k]
            for j in range(k + 1, n):
                LU[i, j] -= LU[i, k] * LU[k, j]
    return LU, P


def det(matrix):
    return np.prod(np.diag(matrix))


def pascal_matrix(n):
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P[i][j] = math.factorial(i + j) / (math.factorial(i) * math.factorial(j))
    return P


def main():
    n = 5  # Размерность матрицы Паскаля
    A = pascal_matrix(n)
    A_1 = np.linalg.inv(A)
    print("Матрица Паскаля:")
    print(A)
    print()

    # LU-разложение с выбором главного элемента
    LU, P = partial_pivot_lu_decomposition(A)
    print("LU-разложение с выбором главного элемента:")
    print("Матрица LU:")
    print(LU)
    print("Матрица перестановок P:")
    print(P)
    print("Определитель матрицы A:", det(A))
    print()
    print(det(P))
    print(det(A_1))

main()