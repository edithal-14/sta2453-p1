### Notes

- Use batch size, or stochastic gradient

- take 1 mil for training

- take less data for validation

- take less data for testing, may be same as validation

- use scaling
  - range scaler: 0 to 1

- neural networks learn best for normally distributed data

- validation will be on a skewed transform

- use the analytical pricer to guide your network
    - a.k.a sampling rate

- implement dropout
- implement SGD
- implement batch normalization
- training data: 20000
- validation data: 20000
- test data: 20000
- epochs: 100000
- model: 5 -> 20 -> 20 -> 20 -> 100