import numpy as np

def f(x):
    return x**2 - 10 * np.sin(x)

def df(x):
    return 2 * x - 10 * np.cos(x)

def newton_method(f, df, x0, tol=1e-6, max_iter=100):
    x = x0
    iteration = 0
    while abs(f(x)) > tol and iteration < max_iter:
        x = x - f(x) / df(x)
        iteration += 1
    if iteration == max_iter:
        print("Метод Ньютона не сошелся за максимальное число итераций.")
    return x

x0 = 2.0

# Решение уравнения
solution = newton_method(f, df, x0)

print("Положительные корень уравнения x^2 - 10sin(x) = 0:")
print("x =", solution)
