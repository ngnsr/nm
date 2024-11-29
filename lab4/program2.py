import numpy as np
import pandas as pd
from PyQt5.QtCore import QLoggingCategory
QLoggingCategory.setFilterRules("qt.*=false")
import sys

np.set_printoptions(linewidth=np.inf, threshold=sys.maxsize)

pd.set_option('display.width', 120) 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def f(x):
    return np.tan(x)

start = -0.5
end = 0.5
num_points = 30
x_values = np.linspace(start, end, num_points)
f_values = f(x_values)

# Swap x & f(x) for reverse interpolation we can do it because tan is continuos function
x_values, f_values = f_values, x_values

n = len(x_values)
div_diff = np.zeros((n, n))
div_diff[:, 0] = f_values

for j in range(1, n):
    for i in range(n - j):
        div_diff[i][j] = (div_diff[i + 1][j - 1] - div_diff[i][j - 1]) / (x_values[i + j] - x_values[i])

full_newton_polynomial_str = f"{div_diff[0, 0]:.5f}"

product_terms = ""  
for j in range(1, n):
    if(x_values[j - 1] > 0):
        product_terms += f"(x - {x_values[j - 1]:.4f})"
    else:
        product_terms += f"(x - ({x_values[j - 1]:.4f}))"

    full_newton_polynomial_str += f" + ({div_diff[0, j]:.4f}) * {product_terms}"

print(full_newton_polynomial_str)

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

div_diff_table = pd.DataFrame(div_diff, columns=[f'f(x0, ..., x{i})' for i in range(n)])
div_diff_table.insert(0, 'x', x_values)
print("Таблиця розділених різниць:")
print(div_diff_table)

print()

polynomial_representation = newton_polynom_to_str(x_values.size, div_diff)
print(f"Поліном у вигляді рядка: {polynomial_representation}")

print()

x = 0.18

print(newton_polynomial(x, x_values, div_diff))
