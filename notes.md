- Scenario
    - Pension fund calls trader
    - Trader runs the pricing model
    - Trader waits for couple of days
    - Trader quotes the price to pension fund

- Riskfuel is a company which accelerates pricing model

- Put option
    - Right to sell the stock at a strike price after maturity
    - Bet the stock price will decrease

- Call option
    - Right to buy the stock at a strike price after maturity
    - Bet the stock price will increase

- BSM (black - scholes model)
    - Invented by Martin Scholes from Mc Master University

- Pytorch is recommended

- Grade
    - Level 4: Max error less than a cent
    - Level 3: less than a dollar
    - Passing grade (Level 2): less than 10 dollars

- `riskfuel_test.py` is the file used for evaluation

- Any ML model can be used such as gradient boosted trees (e.g. XGB)
    - Not necessary to use Neural Nets

- Grading will be done on secret testing data which might have skewed distribution

- Tips
    - Add dropout layers to prevent overfitting
    - Use different testing, validation and testing dataset
    - Use scaling techniques
        - Range scaler (e.g. (x - min) / (max - min)) to transform values between 0 to 1
        - Neural nets work best with normal distribution data, use batch normalization?
        - Need to inverse tranform the predictions
    - Use analytical pricer to guide your predictions

- My ideas
    - Use batch size
    - Increase training data size
        - Maybe 1 million?
    - Increase width and depth of the network