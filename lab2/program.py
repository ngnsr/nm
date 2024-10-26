import numpy as np

def generate_augmented_matrix(size=4, seed=None):
    if seed is not None:
        np.random.seed(seed)

    matrix = np.random.randint(-9, 9, (size, size))
    vector = np.random.randint(-30, 30, size)
    for i in range(size):
        sum_of_row = sum(abs(matrix[i][j]) for j in range(size)) - matrix[i][i]
        if abs(matrix[i][i]) <= sum_of_row:
            matrix[i][i] = sum_of_row + np.random.randint(1, 5)
    augmented_matrix = np.column_stack((matrix, vector))
    return augmented_matrix


def generate_tridiagonal_augmented_matrix(size=4, seed=None):
    if seed is not None:
        np.random.seed(seed)

    lower_diag = np.random.randint(-9, 9, size-1)
    upper_diag = np.random.randint(-9, 9, size-1)

    matrix = np.zeros((size, size))
    np.fill_diagonal(matrix, 0)
    np.fill_diagonal(matrix[1:], lower_diag)
    np.fill_diagonal(matrix[:, 1:], upper_diag)

    for i in range(size):
        row_sum = sum(abs(matrix[i][j]) for j in range(size))
        if abs(matrix[i][i]) <= row_sum:
            matrix[i][i] = row_sum + np.random.randint(1, 5)

    vector = np.random.randint(-30, 30, size)

    augmented_matrix = np.column_stack((matrix, vector))

    return augmented_matrix

def gauss(augmented_matrix):
    n = len(augmented_matrix)
    matrix = np.array(augmented_matrix, dtype=np.double)

    for k in range(n):
        if matrix[k][k] == 0:
            for i in range(k + 1, n):
                if matrix[i][k] != 0:
                    matrix[[k, i]] = matrix[[i, k]]
                    break
            else:
                raise ValueError(f"Error, can't find row with non zero diagonal element")

        matrix[k] = matrix[k] / matrix[k][k]
        # print(matrix)
        
        for i in range(k + 1, n):
            factor = matrix[i][k]
            matrix[i] = matrix[i] - factor * matrix[k]
            # print(matrix)

    for k in range(n - 1, -1, -1):
        for i in range(k - 1, -1, -1):
            factor = matrix[i][k]
            matrix[i] = matrix[i] - factor * matrix[k]
            # print(matrix)

    m = np.array([row[:-1] for row in augmented_matrix], dtype=float)
    # note: we use whole numbers for matrices
    print("determinant: " , int(np.linalg.det(m)))
    print("inv:\n", np.linalg.inv(m))

    return matrix[:, -1]

def check_solution(augmented_matrix, solution):
    print(solution)
    n = len(augmented_matrix)
    for i in range(n):
        equation_sum = sum(augmented_matrix[i][j] * solution[j] for j in range(n))
        print(f"Рівняння {i + 1}: {equation_sum} = {augmented_matrix[i][-1]}")

# def seidel(augmented_matrix, tol=10e-5, max_iterations=1000):
#     for k in range(max_iterations):
#         iteration_count += 1
#         x_old = x.copy()
#
#         for i in range(len(b)):
#             sigma = 0
#             for j in range(len(b)):
#                 if j != i:
#                     sigma += A[i, j] * x[j]
#
#             x[i] = (b[i] - sigma) / A[i, i]
#
#         if np.linalg.norm(x - x_old, ord=np.inf) < tolerance:
#             break

def seidel(augmented_matrix, tol=10e-5, max_iterations=1000):
    A = np.array([row[:-1] for row in augmented_matrix], dtype=float)
    b = np.array([row[-1] for row in augmented_matrix], dtype=float)

    n = len(A)
    x = np.zeros_like(b, dtype=np.double)

    print(f"{'Iteration':^10} | {'x_new':^50} | {'Norm':^10}")
    print("-" * 90)

    for iteration in range(max_iterations):
        x_new = np.copy(x)

        for i in range(n):
            sigma = 0
            for j in range(len(b)):
                if j != i:
                    sigma += A[i,j] * x[j]
            x_new[i] = (b[i] - sigma) / A[i][i]

        norm = np.linalg.norm(x_new - x, ord=np.inf)

        print(f"{iteration + 1:<10} | {np.array2string(x_new, precision=6, floatmode='fixed'):<50} | {norm:<10}")

        if norm < tol:
            print(f"\nЗбіжність досягнута після {iteration + 1} ітерацій.")
            return x_new

        x = x_new

    raise ValueError("Метод не збігається після максимальної кількості ітерацій.")


def tridiagonal_matrix_algorithm(matrix):
    c = np.array([row[:-1] for row in matrix], dtype=float)
    f = np.array([row[-1] for row in matrix], dtype=float)
    n = len(f)
    a = np.zeros((n));
    b = np.zeros((n));

    a[0] = -c[0,1]/c[0,0]
    b[0] = f[0]/c[0,0]

    for i in range(1, n-1):
        a[i] = -(c[i, i + 1]/(a[i-1] * c[i,i-1] + c[i,i]))
        b[i] = (f[i] - c[i,i-1] * b[i-1])/(a[i-1] * c[i,i-1] + c[i,i])

    b[-1] = (f[-1] - c[-1,-1-1] * b[-1-1])/(a[-1-1] * c[-1,-1-1] + c[-1,-1])

    print("a: ", a)
    print("b: ", b)

    x = np.zeros(n)
    x[-1] = b[-1]

    for i in range(n-2, -1, -1):
        x[i] = a[i] * x[i+1] + b[i]

    return x

seed = 69

while (True):
    # # print(matrix)
    #
    # augmented_matrix = generate_augmented_matrix(seed=seed)
    # # print("----")
    # print(augmented_matrix)
    # # result = gauss(matrix)
    # # print("gauss: ", result)
    # # print("----")
    # result = seidel(augmented_matrix)
    # print("seidel: ", result)
    # # coefficients = np.array([[2, 4, 0],
    # #                      [4, 1, 5],
    # #                      [0, 5, 2]], dtype=float)
    # # f = np.array([18, 33, 30], dtype=float)
    #
    # # matrix = generate_tridiagonal_augmented_matrix(seed=seed)
    # # matrix = np.column_stack((coefficients, f))
    # # print(matrix)
    # #
    # # print("----")
    # #
    # # result = tridiagonal_matrix_algorithm(matrix)
    # # print(result)
    #
    # # check_solution(matrix, result)
    
    augmented_matrix = generate_augmented_matrix(seed=seed)
    mode = input("Select algorighm:\n1. Gauss\n2. Siedel\n3. Thomas\n > ")

    if (mode != "1" and mode != "2" and mode != "3"):
        print("Try again\n")
        continue

    if (mode == "1"):
        print(augmented_matrix)
        result = gauss(augmented_matrix)
        print("gauss: ", result)
    elif (mode == "2"):
        print(augmented_matrix)
        tol = float(input("tol : "))
        result = seidel(augmented_matrix, tol=tol)
        print("seidel: ", result)
    elif (mode == "3"):
        matrix = generate_tridiagonal_augmented_matrix(seed=seed)
        print(matrix)
        result = tridiagonal_matrix_algorithm(matrix)
        print("Thomas: ", result)

    break
