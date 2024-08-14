import numpy as np
from scipy.optimize import curve_fit

# Функция для аппроксимации
def func(x, a, b):
    return a * np.exp(b * x)

# Заданные параметры
n = 10  # Количество точек на сетке
h = 1 / n  # Шаг сетки

# Вычисляем значения Xi и Vi
x_values = np.arange(0, 1 + h, h)
y_values = 1 - np.cos(x_values)

# Аппроксимация функции
popt, pcov = curve_fit(func, x_values, y_values)

# Получение параметров a и b
a, b = popt

# Вывод результатов аппроксимации
print("Параметры аппроксимации:")
print("a =", a)
print("b =", b)
