- Scenario
    - Pension fund calls trader
    - Trader runs the pricing model
    - Trader waits for couple of days
    - Trader quotes the price to pension fund

- Riskfuel is a company which accelerates pricing model

- Put option
    - Seller of the put option has the obligation to buy the asset at the strike price
    - Buyer of the  put option can exercise the right to sell the asset at the strike price
    - You buy put option if you bet the price of asset will go down

- Call option
    - Seller of the call option has the obligation to sell the asset at the strike price
    - Buyer of the call option can exercise the right to buy the asset at the strike price
    - You buy call option if you bet the price of the asset will go up

- BSM (Black - Scholes model)
    - Invented by Martin Scholes from Mc Master University

- Pytorch is recommended

- Grade
    - Level 1: Max error 50 - 100
    - Level 2: Max error 5 - 50
    - Level 3: Max error 0.01 - 5
    - Level 4: Max error < 0.01

- Any ML model can be used such as gradient boosted trees (e.g. XGB)
    - Not necessary to use Neural Nets

- Grading will be done on secret testing data which might have skewed distribution

- Tips
    - Use scaling techniques
        - Range scaler (e.g. (x - min) / (max - min)) to transform values between 0 to 1
        - Neural nets work best with normal distribution data, use batch normalization?
    - Use analytical pricer to guide your predictions
    - Increase width and depth of the network