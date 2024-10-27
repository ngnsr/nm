import numpy as np

def f(x):
    return np.tan(x)

start = -0.5
end = 0.5
num_points = 30
x_values = np.linspace(start, end, num_points)
tan_values = f(x_values)

y_target = 0.18 

def newton_inverse_interpolation(y_target, x_values, tan_values):
    idx = np.searchsorted(tan_values, y_target)
    if idx == 0 or idx == len(tan_values):
        raise ValueError("y_target знаходиться за межами діапазону функції.")
    
    x0 = x_values[idx - 1]
    x1 = x_values[idx]
    
    for _ in range(100):
        f0 = f(x0) - y_target
        f1 = f(x1) - y_target
        
        x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
        
        x0, x1 = x1, x_new
        
        if abs(f(x1) - y_target) < 1e-6:
            break
    
    return x1

x_result = newton_inverse_interpolation(y_target, x_values, tan_values)
print(f"Значення x для y = {y_target}: {x_result}")
print(f"Перевірка: f({x_result}) = {f(x_result)}")
