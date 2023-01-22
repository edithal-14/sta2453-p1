import numpy as np
import pandas as pd

class Transform:
    def __init__(self, use_boxcox=False):
        df = pd.read_csv("dataset/training_data.csv")
        df = df[["S", "K", "T", "r", "sigma", "value"]]
        df = df.to_numpy()
        self.min_vals = []
        self.max_vals = []
        self.lmda = 0.1
        self.use_boxcox = use_boxcox
        for i in range(df.shape[1]):
            self.min_vals.append(np.min(df[:, i]))
            self.max_vals.append(np.max(df[:, i]))

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