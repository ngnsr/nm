import numpy as np
from numpy.linalg import inv

np.set_printoptions(linewidth=np.inf, precision=16)
def iterative_method(f, x0, tol=1e-4, max_iterations=50):
    x = x0

    print(f"{'Iteration':^10} | {'x_new':^40} | {'Norm':^10}")
    print("-" * 70)

    print(f"{0:<10} | {np.array2string(x, precision=6, floatmode='fixed', suppress_small=True):<40} | -")

    for iteration in range(max_iterations):
        f_x_T = np.transpose(f(x))
        x_new = x - C(x) @ f_x_T

        norm = np.linalg.norm(x_new - x, ord=np.inf)

        print(f"{iteration + 1:<10} | {np.array2string(x_new, precision=6, floatmode='fixed', suppress_small=True):<40} | {norm:<10.9f}")

        if norm < tol:
            print(f"\nЗбіжність досягнута після {iteration + 1} ітерацій.")
            return x_new

        x = x_new

    raise ValueError("Метод не збігається після максимальної кількості ітерацій.")


def modified_newton_method(f, A, x0, tol=1e-6, max_iterations=1000):
    x = x0
    print(A(np.transpose(x)))
    invA = inv(A(np.transpose(x)))
    print(invA)

    print(f"{'Iteration':^10} | {'x_new':^40} | {'Norm':^10}")
    print("-" * 70)
    print(f"{0:<10} | {np.array2string(x, precision=6, floatmode='fixed', suppress_small=True):<40} | -")
    for iteration in range(max_iterations):
        x_new = x - invA @ f(x)

        norm = np.linalg.norm(x_new - x, ord=np.inf)

        print(f"{iteration + 1:<10} | {np.array2string(x_new, precision=6, floatmode='fixed', suppress_small=True):<40} | {norm:<10.9f}")

        if norm < tol:
            print(f"\nЗбіжність досягнута після {iteration + 1} ітерацій.")
            return x_new

        x = x_new

    raise ValueError("Метод не збігається після максимальної кількості ітерацій.")

def f(x):
    return np.array([
        (3*x[0] - np.sin(x[1]) - 1/np.exp(x[2])),   # 3*x1 - sin(x2) - 1/e^x3   = 0
        (5*x[1] + np.cos(x[0]) - x[2]**2),          # 5*x2 + cos(x1) - x3^2     = 0
        (4*x[2] + x[0]**2 - x[1])                   # 4*x3 + x1^2    - x2       = 0
    ])

def C(x):
    return np.array([
        [0.1 + x[0]/1000, 0, 0],
        [0, 0.1 + x[1]/1000, 0],
        [0, 0, 0.1 + x[2]/1000]
    ])

def A(x):
    x1, x2, x3 = x
    
    J = np.array([
        [3, -np.cos(x2), np.exp(x3)],		    # df1/dx1, df1/dx2, df1/dx3
        [-np.sin(x1), 5, -2 * x3],		    # df2/dx1, df2/dx2, df2/dx3
        [2 * x1, -1, 4] 		            # df3/dx1, df3/dx2, df3/dx3
    ])
    
    return J

x0 = np.array([0.5,-0.5, -0.5])

print("Iterative method")
result = iterative_method(f, x0)
print("Розв'язок:", result)
print(f"f(result) = {f(result)}")

print()

print("Modified Newton method")
result = modified_newton_method(f, A, x0)
print("Розв'язок:", result)
print(f"f(result) = {f(result)}")
