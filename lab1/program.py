import math


def f(x):
    return x**4 + x**3 - 6*x**2 + 20*x - 16


def phi(x):
    # psi = (1/(x-24))
    return x + (x**4 + x**3 - 6*x**2 + 20*x - 16) * (1/(x-24))


def ao(q, xn, xp):
    return (q / (1 - q)) * abs(xn - xp)


def iterative_method():
    a = 0
    b = 1.4
    x0 = (a + b)/2
    e = 10**-4
    q = 0.4
    n = int((math.log(abs(phi(x0) - x0) / ((1 - q) * e))) / (math.log(1/q))) + 1
    x = x0
    print(f"{'n':<10} | {'x':<25} | {'f(x)':<25} | {'ao':<25}")
    print('-' * 95)
    print(f"{0:<10} | {x:<25} | {f(x):<25} | {'-':<25}")
    for i in range(n):
        xp = x
        x = phi(x)
        ao_value = ao(q, x, xp)
        print(f"{i+1:<10} | {x:<25} | {f(x):<25} | {ao_value:<25}")


def relaxation_method():
    a = 0
    b = 1.4
    x0 = (a + b)/2
    z0 = 0.7
    e = 10**-4
    m1 = 14
    M1 = 20
    q = (M1 - m1)/(M1 + m1)
    t = 2 / (m1 + M1)
    n = int(math.log(z0 / e)/math.log(1/q)) + 1

    x = x0
    print(f"{'n':<10} | {'x':<25} | {'f(x)':<25} | {'z':<25}")
    print('-' * 95)
    print(f"{0:<10} | {x:<25} | {f(x):<25} | {'-':<25}")
    for i in range(n):
        x = x - t * f(x)
        z = q**(i + 1) * z0
        print(f"{i+1:<10} | {x:<25} | {f(x):<25} | {z:<25}")


while (True):
    mode = input(
        "Select algoritm to solve x^4 + x^3 - 6x^2 + 20x - 16 = 0 : \n 1: Iterative method\n 2: Relaxation method\n > ")
    if (mode != "1" and mode != "2"):
        print("Try again\n")
        continue

    if (mode == "1"):
        iterative_method()
    elif (mode == "2"):
        relaxation_method()

    break
