import numpy as np

def inverse_iteration(A, mu, b0, num_values=3, max_iter=1000, tol=1e-6):
    n = A.shape[0]
    I = np.eye(n)
    eigenvalues = []
    eigenvectors = []

    for _ in range(num_values):
        b_k = b0 / np.linalg.norm(b0)

        for _ in range(max_iter):
            inverse_matrix = np.linalg.inv(A - mu * I)
            b_k_plus_1 = np.dot(inverse_matrix, b_k)
            b_k_plus_1 = b_k_plus_1 / np.linalg.norm(b_k_plus_1)

            eigenvalue_estimate = np.dot(b_k, np.dot(A, b_k)) / np.dot(b_k, b_k)

            if np.linalg.norm(b_k_plus_1 - b_k) < tol:
                break

            b_k = b_k_plus_1

        eigenvalues.append(eigenvalue_estimate)
        eigenvectors.append(b_k)

        # Добавим сдвиг для поиска следующего минимального собственного значения
        mu = eigenvalue_estimate + 0.1

    return eigenvalues, eigenvectors

def pascal_matrix(n):
    pascal = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pascal[i, j] = np.math.factorial(i + j) // (np.math.factorial(i) * np.math.factorial(j))
    return pascal

# Пример использования
n = 8
A = pascal_matrix(n)
mu = 0.5  # Начальное приближение собственного значения
b0 = np.random.rand(n)

eigenvalues, eigenvectors = inverse_iteration(A, mu, b0)
print("Первые три минимальных по модулю собственных значения:")
for i in range(len(eigenvalues)):
    print("Собственное значение:", eigenvalues[i])
    print("Собственный вектор:", eigenvectors[i])
    print()
