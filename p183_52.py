5.2

import numpy as np


def relaxation_method(A, f, alpha, t, max_iter=1000, tol=1e-6):
    n = len(f)
    x = np.zeros(n)
    x_new = np.zeros(n)
    iter_count = 0
    residual = tol + 1

    while residual > tol and iter_count < max_iter:
        for i in range(n):
            if i == 0:
                x_new[i] = (1 - t) * x[i] + t * (f[i] - A[i, i + 1] * x[i + 1]) / A[i, i]
            elif i == n - 1:
                x_new[i] = (1 - t) * x[i] + t * (f[i] - A[i, i - 1] * x_new[i - 1]) / A[i, i]
            else:
                x_new[i] = (1 - t) * x[i] + t * ((f[i] - A[i, i - 1] * x_new[i - 1] - A[i, i + 1] * x[i + 1]) / A[i, i])

        residual = np.linalg.norm(A.dot(x_new) - f)
        x = x_new.copy()
        iter_count += 1

    return x, iter_count


def generate_tridiagonal_matrix(n, alpha):
    A = np.zeros((n, n))
    A[0, 0] = 2
    A[0, 1] = -1 - alpha
    A[-1, -1] = 2
    A[-1, -2] = -1 + alpha
    for i in range(1, n - 1):
        A[i, i] = 2
        A[i, i - 1] = -1 + alpha
        A[i, i + 1] = -1 - alpha
    return A


def generate_rhs_vector(n, alpha):
    f = np.zeros(n)
    f[0] = 1 - alpha
    f[-1] = 1 + alpha
    return f


# Параметры
n_values = [5, 10, 15]  # Размерность матрицы
alpha_values = [0.1, 0.5, 1.0]  # Значения параметра alpha
t_values = [1.2, 1.5, 1.9]  # Значения параметра t

# Исследование зависимости скорости сходимости
for n in n_values:
    print(f"Размерность матрицы: {n}")
    for alpha in alpha_values:
        print(f"Параметр alpha: {alpha}")
        A = generate_tridiagonal_matrix(n, alpha)
        f = generate_rhs_vector(n, alpha)
        for t in t_values:
            x, iterations = relaxation_method(A, f, alpha, t)
            print(f"Параметр t: {t}, Количество итераций: {x}")