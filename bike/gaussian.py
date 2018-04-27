from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import numpy as np
import matplotlib.pylab as plt

def exponential_cov(x, y, params):
    print (x, y, params, "he")
    return params[0] * np.exp( -0.5 * params[1] * np.subtract.outer(x, y)**2)


def conditional(x_new, x, y, params):
    B = exponential_cov(x_new, x, params)
    C = exponential_cov(x, x, params)
    A = exponential_cov(x_new, x_new, params)
    
    mu = np.linalg.inv(C).dot(B.T).T.dot(y)
    sigma = A - B.dot(np.linalg.inv(C).dot(B.T))

    return (mu.squeeze(), sigma.squeeze())

def predict(x, data, kernel, params, sigma, t):
    k = [kernel(x, y, params) for y in data]
    Sinv = np.linalg.inv(sigma)
    y_pred = np.dot(k, Sinv).dot(t)
    sigma_new = kernel(x, x, params) - np.dot(k, Sinv).dot(k)

kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) \
       + WhiteKernel(noise_level=1)

theta = [1, 10]
sigma_theta = exponential_cov(0, 0, theta)
xpts = np.arange(-3, 3, step=0.01)
plt.errorbar(xpts, np.zeros(len(xpts)), yerr=sigma_theta, capsize=0)

x = [1.]
y = [np.random.normal(scale=sigma_theta)]
print (y)

sigma_1 = exponential_cov(x, x, theta)

x_pred = np.linspace(-3, 3, 1000)
predictions = [predict(i, x, exponential_cov, 0, sigma_1, y) for i in x_pred]

y_pred, sigmas = np.transpose(predictions)
plt.errorbar(x_pred, y_pred, yerr=sigmas, capsize=0)
plt.plot(x, y, "ro")