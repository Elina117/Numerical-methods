import numpy as np


def neville(x, x_values, y_values):
    n = len(x_values)
    p = np.zeros((n, n))
    for i in range(n):
        p[i, 0] = y_values[i]

    for j in range(1, n):
        for i in range(1, j + 1):
            p[i, j] = (((x - x_values[j - i]) * p[i, j - 1] - (x - x_values[j]) * p[i - 1, j - 1]) /
                       (x_values[j - i] - x_values[j]))

    return p[n - 1, n - 1]


# Функция для интерполяции
def f(x):
    return 1 / np.arctan(1 + x ** 2)


# Заданные параметры
x_values = np.linspace(-3, 3, 4)  # Равномерно распределенные узлы
x_interp = [1.5, 2.5]  # Точки для интерполяции

# Интерполяция
for p in [4, 6, 10]:
    print(f"Интерполяция с p = {p}:")
    y_values = f(x_values)
    for x in x_interp:
        interp_value = neville(x, x_values, y_values)
        print(f"f({x}) = {interp_value}")
