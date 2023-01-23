import pandas as pd
import numpy as np

from utils.black_scholes import black_scholes_put


# Domain boundaries
S_bound = (0.0, 200.0)
K_bound = (50.0, 150.0)
T_bound = (0.0, 5.0)
r_bound = (0.001, 0.05)
sigma_bound = (0.05, 1.5)


def generate_data(n):
    # TODO: modify me!
    return np.random.rand(n, 5)


def generate_black_scholes_put_data(n):
    x = generate_data(n)

    S_delta = S_bound[1] - S_bound[0]
    K_delta = K_bound[1] - K_bound[0]
    T_delta = T_bound[1] - T_bound[0]
    r_delta = r_bound[1] - r_bound[0]
    sigma_delta = sigma_bound[1] - sigma_bound[0]

    deltas = np.array([S_delta, K_delta, T_delta, r_delta, sigma_delta])
    l_bounds = np.array([S_bound[0], K_bound[0], T_bound[0], r_bound[0], sigma_bound[0]])

    x = x * deltas + l_bounds
    y = black_scholes_put(S=x[:, 0], K=x[:, 1], T=x[:, 2], r=x[:, 3], sigma=x[:, 4]).reshape(-1, 1)

    # Generate data points along the edges of the domain
    # Take 10 points along each axis
    eps = np.nextafter(np.float32(0), np.float32(1))
    x_new = np.zeros((10**5, 5))
    i = 0
    for s in np.linspace(S_bound[0] + eps, S_bound[1], num=10):
        for k in np.linspace(K_bound[0], K_bound[1], num=10):
            for t in np.linspace(T_bound[0] + eps, T_bound[1], num=10):
                for r in np.linspace(r_bound[0], r_bound[1], num=10):
                    for sigma in np.linspace(sigma_bound[0], sigma_bound[1], num=10):
                        x_new[i,:] = np.array([s, k, t, r, sigma])
                        i+=1
    y_new = black_scholes_put(S=x_new[:, 0], K=x_new[:, 1], T=x_new[:, 2], r=x_new[:, 3], sigma=x_new[:, 4]).reshape(-1, 1)
    x = np.concatenate((x, x_new))
    y = np.concatenate((y, y_new))

    return np.append(x, y, axis=1)


def main():
    training_size = 1000000
    validation_size = 10000
    testing_size = 10000

    xy = generate_black_scholes_put_data(training_size)
    xy_df = pd.DataFrame(xy, columns=["S", "K", "T", "r", "sigma", "value"])
    xy_df.to_csv("dataset/training_data.csv")

    xy = generate_black_scholes_put_data(validation_size)
    xy_df = pd.DataFrame(xy, columns=["S", "K", "T", "r", "sigma", "value"])
    xy_df.to_csv("dataset/validation_data.csv")

    xy = generate_black_scholes_put_data(testing_size)
    xy_df = pd.DataFrame(xy, columns=["S", "K", "T", "r", "sigma", "value"])
    xy_df.to_csv("dataset/testing_data.csv")

    xy = generate_black_scholes_put_data(testing_size)
    xy_df = pd.DataFrame(xy, columns=["S", "K", "T", "r", "sigma", "value"])
    xy_df.to_csv("dataset/testing_data_1.csv")

    xy = generate_black_scholes_put_data(testing_size)
    xy_df = pd.DataFrame(xy, columns=["S", "K", "T", "r", "sigma", "value"])
    xy_df.to_csv("dataset/testing_data_2.csv")


if __name__ == "__main__":
    main()
