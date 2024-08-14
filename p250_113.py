import numpy as np

def kernel(x, s):
    """Ядро интегрального уравнения."""
    return np.exp(x - s)

def f(x):
    """Функция правой части интегрального уравнения."""
    return 1

def trapezoidal_rule(f, a, b, N):
    """Квадратурная формула трапеций для вычисления интеграла на отрезке [a, b] с N узлами."""
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    integral = h * (np.sum(y) - 0.5 * (y[0] + y[-1]))
    return integral

def volterra_eq_second_kind(N):
    """Численное решение интегрального уравнения Вольтерра второго рода."""
    a = 0  # Нижний предел интегрирования
    b = 1  # Верхний предел интегрирования
    integral = 0
    for i in range(N):
        integral += kernel(b, i / N) * f(i / N)
    return integral

# Задаем различные значения числа частичных отрезков для анализа точности
N_values = [10, 20, 50, 100, 200]

# Вычисляем численные решения для каждого значения N
for N in N_values:
    approx_solution = volterra_eq_second_kind(N)
    print(f"При N = {N}, численное решение уравнения Вольтерра: {approx_solution}")
