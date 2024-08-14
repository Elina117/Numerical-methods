import numpy as np

def kernel(x, s):
    """Ядро интегрального уравнения."""
    return (x - s)

def f(x):
    """Функция правой части интегрального уравнения."""
    return 3 - 2 * x

def simpson_rule(f, a, b, N):
    """Квадратурная формула Симпсона для вычисления интеграла на отрезке [a, b] с N узлами."""
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    integral = h / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2])
    return integral

def fredholm_eq_second_kind(N):
    """Численное решение интегрального уравнения Фредгольма второго рода."""
    a = 0  # Нижний предел интегрирования
    b = 1  # Верхний предел интегрирования
    integral = 0
    for i in range(N):
        for j in range(N):
            integral += kernel((i + 0.5) / N, (j + 0.5) / N) * f((j + 0.5) / N)
    return integral / N**2

# Задаем различные значения числа частичных отрезков для анализа точности
N_values = [10, 20, 50, 100, 200]

# Вычисляем численные решения для каждого значения N
for N in N_values:
    approx_solution = fredholm_eq_second_kind(N)
    sum_value = approx_solution + (3 - 2 * ((N + 0.5) / N))
    print(f"При N = {N}, численное решение уравнения Фредгольма: {approx_solution}")
