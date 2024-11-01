import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from PyQt5.QtCore import QLoggingCategory
QLoggingCategory.setFilterRules("qt.*=false")

np.set_printoptions(linewidth=np.inf, threshold=sys.maxsize)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def f(x):
    return np.tan(x)


start = -0.5
end = 0.5
num_points = 30
x_values = np.linspace(start, end, num_points)
tan_values = f(x_values)

n = len(x_values)
div_diff = np.zeros((n, n))
div_diff[:, 0] = tan_values

for j in range(1, n):
    for i in range(n - j):
        div_diff[i][j] = (div_diff[i + 1][j - 1] - div_diff[i][j - 1]) / (x_values[i + j] - x_values[i])

def newton_polynomial(x, x_values, div_diff):
    n = len(x_values)
    result = div_diff[0, 0]
    product_term = 1.0
    
    for i in range(1, n):
        product_term *= (x - x_values[i - 1])
        result += div_diff[0, i] * product_term
    return result

def newton_polynom_to_str(n, div_diff):
    polynomial_terms = {}

    for i in range(1, n):
        coeff = div_diff[0, i]
        
        key = f"x^{i}"
        
        if key in polynomial_terms:
            polynomial_terms[key] += coeff
        else:
            polynomial_terms[key] = coeff

    simplified_polynomial = " + ".join(f"{coef:.4f} * {key}" for key, coef in polynomial_terms.items())
    return simplified_polynomial

display_start =0.5 - 0.5e-13
display_end = 0.5
x_plot = np.linspace(display_start, display_end, 10000)  
y_interpolated = [newton_polynomial(x, x_values, div_diff) for x in x_plot]
y_true = f(x_plot)

absolute_error = np.abs(y_true - y_interpolated)

div_diff_table = pd.DataFrame(div_diff, columns=[f'f(x0, ..., x{i})' for i in range(n)])
div_diff_table.insert(0, 'x', x_values)
print("Таблиця розділених різниць:")
print(div_diff_table)

polynomial_representation = newton_polynom_to_str(x_values.size, div_diff)
print(f"Поліном у вигляді рядка: {polynomial_representation}")

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)  
plt.plot(x_plot, y_interpolated, label='Інтерполяційний поліном Ньютона', color='blue')
plt.plot(x_plot, y_true, label='Графік f(x) = tan(x)', color='green')
plt.title("Порівняння f(x) = tan(x) та інтерполяційного полінома Ньютона")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)


plt.subplot(2, 1, 2)  
plt.ylim(0, 1e-15)  
plt.plot(x_plot, absolute_error, label='Абсолютна похибка', color='red')
plt.title("Абсолютна похибка між tan(x) та інтерполяційним поліном")
plt.xlabel('x')
plt.ylabel('Абсолютна похибка')
plt.grid(True)

plt.tight_layout()
plt.show()
