import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=np.inf)

x_nodes = np.linspace(-0.5, 0.5, 15)
print(f"x: {x_nodes}")
y_nodes = np.tan(x_nodes)
print(f"y: {y_nodes}")

n = len(x_nodes) - 1

h = np.diff(x_nodes)
print(f"h: {h}\n")

C = np.zeros((n - 1, n - 1))
for i in range(n - 1):
    if i == 0:
        C[i, i] = (h[i] + h[i + 1]) / 3
        C[i, i + 1] = h[i + 1] / 6
    elif i == n - 2:
        C[i, i - 1] = h[i] / 6
        C[i, i] = (h[i] + h[i + 1]) / 3
    else:
        C[i, i - 1] = h[i] / 6
        C[i, i] = (h[i] + h[i + 1]) / 3
        C[i, i + 1] = h[i + 1] / 6

print(f"C:\n{C}\n")

H = np.zeros((n - 1, n + 1))
for i in range(n - 1):
    H[i, i] = 1 / h[i]
    H[i, i + 1] = -1 / h[i] - 1 / h[i + 1]
    H[i, i + 2] = 1 / h[i + 1]

print(f"H:\n{H}\n")

b = H @ y_nodes
print(f"b: {b}\n")

m = np.zeros(n + 1)  # M_0 = M_n = 0
alpha = np.zeros(n - 1)
beta = np.zeros(n - 1)

alpha[0] = -C[0, 1] / C[0, 0]
beta[0] = b[0] / C[0, 0]
for i in range(1, n - 1):
    denom = C[i, i] + C[i, i - 1] * alpha[i - 1]
    alpha[i] = -C[i, i + 1] / denom if i < n - 2 else 0
    beta[i] = (b[i] - C[i, i - 1] * beta[i - 1]) / denom

m[-2] = beta[-1]
for i in range(n - 3, -1, -1):
    m[i + 1] = alpha[i] * m[i + 2] + beta[i]

print(f"m: {m}\n")

A = y_nodes[:-1] - m[:-1] * h ** 2 / 6
print(f"A:\n{A}\n")
B = y_nodes[1:] - m[1:] * h ** 2 / 6
print(f"B:\n{B}\n")

def spline(x):
    for i in range(n):
        if x_nodes[i] <= x <= x_nodes[i + 1]:
            xi, xi1 = x_nodes[i], x_nodes[i + 1]
            hi = xi1 - xi
            return (
                ((xi1 - x) ** 3 / (6 * hi)) * m[i] +
                ((x - xi) ** 3 / (6 * hi)) * m[i + 1] +
                A[i] * (xi1 - x)/hi +
                B[i] * (x - xi)/hi
            )
    return None

def printSimplifiedSpline():
    print("Spline :\n")
    for i in range(n):
        xi, xi1 = x_nodes[i], x_nodes[i + 1]
        hi = xi1 - xi

        a3 = (m[i + 1] - m[i]) / (6 * hi)
        a2 = m[i] / 2
        a1 = (y_nodes[i + 1] - y_nodes[i]) / hi - (2 * hi * m[i] + hi * m[i + 1]) / 6
        a0 = y_nodes[i] - (m[i] * hi ** 2) / 6

        polynomial = f"{a3:.6f} * x**3 + {a2:.6f} * x**2 + {a1:.6f} * x + {a0:.6f}"
        print(f"{polynomial}, [{xi:.6f}, {xi1:.6f}]")
    print()
printSimplifiedSpline()

x_dense = np.linspace(-0.5, 0.5, 1000)
y_original = np.tan(x_dense)
y_spline = [spline(x) for x in x_dense]

def spline_first_derivative(x):
    for i in range(n):
        if x_nodes[i] <= x <= x_nodes[i + 1]:
            xi, xi1 = x_nodes[i], x_nodes[i + 1]
            hi = xi1 - xi
            return (
                -((xi1 - x) ** 2 / (2 * hi)) * m[i] +
                ((x - xi) ** 2 / (2 * hi)) * m[i + 1] +
                (-A[i] / hi) +
                (B[i] / hi)
            )
    return None

def spline_second_derivative(x):
    for i in range(n):
        if x_nodes[i] <= x <= x_nodes[i + 1]:
            xi, xi1 = x_nodes[i], x_nodes[i + 1]
            hi = xi1 - xi
            return (
                (xi1 - x) / hi * m[i] +
                (x - xi) / hi * m[i + 1]
            )
    return None

y_spline_derivative1 = [spline_first_derivative(x) for x in x_dense]
y_spline_derivative2 = [spline_second_derivative(x) for x in x_dense]

plt.figure(figsize=(10, 6))
plt.plot(x_dense, y_original, label="Оригінальна функція (tan(x))", color="blue")
plt.plot(x_dense, y_spline, label="Сплайн", color="green", linestyle="--")
plt.scatter(x_nodes, y_nodes, label="Вузли інтерполяції", color="red")
plt.title("Природний кубічний сплайн")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x_dense, y_spline_derivative1, label="Перша похідна сплайну", color="orange")
plt.title("Перша похідна сплайну")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x_dense, y_spline_derivative2, label="Друга похідна сплайну", color="purple")
plt.title("Друга похідна сплайну")
plt.legend()
plt.grid()
plt.show()
