import numpy as np


def cost_function(x, y, w, b):
    m = x.shape[0]
    total_error = 0.0
    for i in range(m):
        f_wb = w * x[i] + b
        total_error += (f_wb - y[i]) ** 2

    return total_error / (2 * m)


def grad_compute(x, y, w, b):
    m = x.shape[0]
    w_grad = 0.0
    b_grad = 0.0
    for i in range(m):
        f_wb = w * x[i] + b
        w_grad += (1.0 / m) * (f_wb - y[i]) * x[i]
        b_grad += (1.0 / m) * (f_wb - y[i])

    return w_grad, b_grad


def grad_descent(x, y, w_start, b_start, alpha, process_iterations):
    w = w_start
    b = b_start
    cost_history = []
    for i in range(process_iterations):
        cost_history.append(cost_function(x, y, w, b))
        w_grad, b_grad = grad_compute(x, y, w, b)
        w = w - alpha * w_grad
        b = b - alpha * b_grad

    return w, b, cost_history


# fitting line only when w and b is known.
def compute_regression(x, y, w_start, b_start, alpha, process_iterations):
    size = x.shape[0]
    f_wb = np.zeros(size)
    w, b, cost_history = grad_descent(x, y, w_start, b_start, alpha, process_iterations)
    print(f'w: {w:.4f} b: {b:.4f}')
    
    for i in range(size):
        f_wb[i] = w * x[i] + b

    return f_wb
