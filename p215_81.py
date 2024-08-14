import numpy as np
from scipy.interpolate import interp1d


# Определяем функцию f(x)
def f(x):
    return x ** 2 - 50 * np.sin(x)


# Задаем интервал [a, b]
a = 0.1
b = 10

# Количество отрезков для интерполяции
segments = 100

# Разбиваем интервал на равные отрезки
x_values = np.linspace(a, b, segments)
y_values = f(x_values)

# Находим минимум на каждом отрезке с положительными значениями x
min_values = []
for i in range(segments - 2):
    # Проверяем, что все значения x в текущем сегменте положительные
    if x_values[i] >= 0 and x_values[i + 1] >= 0 and x_values[i + 2] >= 0:
        # Выбираем точки на текущем отрезке
        x_segment = x_values[i:i + 3]
        y_segment = y_values[i:i + 3]

        # Выполняем интерполяцию полиномом второго порядка
        poly_interp = interp1d(x_segment, y_segment, kind='quadratic')

        # Находим минимум интерполяционного полинома на текущем отрезке
        min_value = np.min(poly_interp(x_segment))
        min_values.append(min_value)

# Находим индексы минимальных значений в массиве y_values
min_indices = [np.argmin(y_values) for _ in range(2)]

# Получаем соответствующие значения x, при которых достигаются минимальные значения y
x_local_mins = x_values[min_indices]

# Находим индексы минимальных значений в массиве y_values
min_indices = np.argsort(y_values)[:2]  # находим индексы двух минимальных значений
unique_x_indices = np.unique(min_indices)  # находим уникальные индексы

# Получаем соответствующие значения x, при которых достигаются минимальные значения y
x_local_mins = x_values[unique_x_indices]
y_local_mins = y_values[unique_x_indices]
# Выводим результаты
for i in range(len(x_local_mins)):
    print("Локальный минимум функции:", y_local_mins[i])
    print("x, при котором достигается локальный минимум:", x_local_mins[i])
    print()