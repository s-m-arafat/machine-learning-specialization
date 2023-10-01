# single variable linear regression with gradient descent

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from defs import compute_regression

# plot style configuration
plt.style.use('ggplot')
matplotlib.use('Agg')

# dataset
x_train = np.array([1.0 , 2.0])
y_train = np.array([300.0 , 500.0])
w_start = 0
b_start = 0
alpha = 0.001
process_iterations = 10000000

f_wb = compute_regression(x_train, y_train, w_start, b_start, alpha, process_iterations)

# plot data
plt.plot(x_train, f_wb, c='r', label='Model')
plt.scatter(x_train, y_train, marker='.', c='g', label='Actual data')
plt.title('Housing price data')
plt.xlabel('Area in square feet')
plt.ylabel('Price in 1000$')
plt.savefig('plot.png')