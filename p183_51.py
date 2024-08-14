5.1

import numpy as np

def jacobi_iteration(A, b, x0, tol=1e-6, max_iter=1000):
    n = len(b)
    x = x0.copy()
    x_new = np.zeros_like(x)
    for k in range(max_iter):
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]
            x_new[i] = (b[i] - sigma) / A[i, i]
        if np.linalg.norm(x_new - x) < tol:
            break
        x[:] = x_new
    return x

# Функция для создания трехдиагональной матрицы A и вектора b для данного n и альфа
def create_tridiagonal_matrix(n, alpha):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 2
        if i > 0:
            A[i, i - 1] = -1 + alpha
        if i < n - 1:
            A[i, i + 1] = -1 - alpha
    b = np.zeros(n)
    b[0] = 1 - alpha
    b[-1] = 1 + alpha
    return A, b

# Пример использования
n = 10
alpha = 1
A, b = create_tridiagonal_matrix(n, alpha)
x0 = np.zeros(n)  # Начальное приближение
solution = jacobi_iteration(A, b, x0)

print("Матрица A:")
print(A)
print("\nВектор b:")
print(b)
print("\nSolution:", solution)