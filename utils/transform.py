import numpy as np
import pandas as pd

from utils.bounds import S_bound, K_bound, T_bound, r_bound, sigma_bound, value_bound

class Transform:
    def __init__(self, use_boxcox=False):
        self.min_vals = [S_bound[0], K_bound[0], T_bound[0], r_bound[0], sigma_bound[0], value_bound[0]]
        self.max_vals = [S_bound[1], K_bound[1], T_bound[1], r_bound[1], sigma_bound[1], value_bound[1]]
        self.lmda = 0.1
        self.use_boxcox = use_boxcox

    def transform_x(self, x):
        x_new = np.copy(x)
        for i in range(x_new.shape[1]):
            x_new[:, i] = (x_new[:, i] - self.min_vals[i]) / (self.max_vals[i] - self.min_vals[i])
        return x_new

    def transform_y(self, y):
        y_new = np.copy(y)
        y_new = (y_new - self.min_vals[-1]) / (self.max_vals[-1] - self.min_vals[-1])
        if self.use_boxcox:
            y_new = self.boxcox(y_new)
        return y_new

    def inverse_transform_x(self, x):
        x_new = np.copy(x)
        for i in range(x_new.shape[1]):
            x_new[:, i] = x_new[:, i] * (self.max_vals[i] - self.min_vals[i]) + self.min_vals[i]
        return x_new

    def inverse_transform_y(self, y):
        y_new = np.copy(y)
        if self.use_boxcox:
            y_new = self.inv_boxcox(y_new)
        y_new = y_new * (self.max_vals[-1] - self.min_vals[-1]) + self.min_vals[-1]
        return y_new
    
    def boxcox(self, x: np.ndarray):
        if self.lmda == 0:
            return np.log(x)
        else:
            return (np.power(x, self.lmda) - 1) / self.lmda

    def inv_boxcox(self, x: np.ndarray):
        if self.lmda == 0:
            return np.exp(x)
        else:
            return np.power((x * self.lmda + 1), 1 / self.lmda)