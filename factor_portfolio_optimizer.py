import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import scipy.optimize as optimization
import statsmodels.api as sm
from pandas_datareader import data as pdr

# Declare constants
NUM_TRADING_DAYS = 252
NUM_PORTFOLIOS = 10000

# Select stocks
stocks = ['AAPL', 'GE', 'WMT', 'AMZN', 'TSLA', 'DB']

# Select start and end dates
start_date = '2012-01-01'
end_date = '2017-01-01'


# Download stock data
def download_data():
    data = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        data[stock] = ticker.history(start=start_date, end=end_date)['Close']

    return pd.DataFrame(data)


# Download Fama-French Factors
def download_factors():
    # Get Fama-French factors (Market, SMB, HML, Risk-free rate)
    # Suppress the FutureWarning by setting date_format instead of using the default parser
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ff_factors = pdr.get_data_famafrench('F-F_Research_Data_Factors_daily', start=start_date, end=end_date)[0]

    # Rename columns for clarity
    ff_factors.columns = ['MKT', 'SMB', 'HML', 'RF']

    # Convert from percentage to decimal
    ff_factors = ff_factors / 100

    # Ensure index is datetime and timezone-naive
    ff_factors.index = pd.to_datetime(ff_factors.index).tz_localize(None)

    return ff_factors


# Visualise data
def show_data(data):
    data.plot(figsize=(12, 8))
    plt.show()


# Calculate Returns
def calculate_returns(data):
    log_returns = np.log(data / data.shift(1))
    return log_returns[1:]


# Calculate factor loadings (betas) for each stock
def calculate_factor_loadings(returns, factors):
    # Convert both indexes to timezone-naive DatetimeIndex for proper alignment
    returns.index = returns.index.tz_localize(None)
    if factors.index.tz is not None:
        factors.index = factors.index.tz_localize(None)

    # Align the dates of returns and factors
    aligned_data = returns.join(factors, how='inner')

    # Calculate excess returns by subtracting risk-free rate
    for stock in stocks:
        aligned_data[f'{stock}_excess'] = aligned_data[stock] - aligned_data['RF']

    # Calculate factor loadings for each stock
    factor_loadings = {}
    for stock in stocks:
        # Prepare dependent and independent variables
        Y = aligned_data[f'{stock}_excess']
        X = aligned_data[['MKT', 'SMB', 'HML']]
        X = sm.add_constant(X)

        # Fit the model
        model = sm.OLS(Y, X).fit()

        # Store coefficients (alpha, beta_mkt, beta_smb, beta_hml)
        factor_loadings[stock] = model.params.values

    return factor_loadings, aligned_data


# Performance calculator using Fama-French
def performance_calculator_ff(weights, returns, factors, factor_loadings):
    # Calculate weighted factor exposures
    total_alpha = 0
    total_mkt_exposure = 0
    total_smb_exposure = 0
    total_hml_exposure = 0

    for i, stock in enumerate(stocks):
        alpha, beta_mkt, beta_smb, beta_hml = factor_loadings[stock]
        total_alpha += weights[i] * alpha
        total_mkt_exposure += weights[i] * beta_mkt
        total_smb_exposure += weights[i] * beta_smb
        total_hml_exposure += weights[i] * beta_hml

    # Calculate expected return using Fama-French model
    # E(R) = RF + β_mkt * MKT + β_smb * SMB + β_hml * HML + α
    expected_return = (factors['RF'].mean() +
                       total_mkt_exposure * factors['MKT'].mean() +
                       total_smb_exposure * factors['SMB'].mean() +
                       total_hml_exposure * factors['HML'].mean() +
                       total_alpha) * NUM_TRADING_DAYS

    # Calculate portfolio volatility (standard deviation of returns)
    portfolio_returns = np.sum(returns.values * weights, axis=1)
    portfolio_volatility = np.std(portfolio_returns) * np.sqrt(NUM_TRADING_DAYS)

    return expected_return, portfolio_volatility, total_mkt_exposure, total_smb_exposure, total_hml_exposure


# Traditional performance calculator
def performance_calculator(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))
    return portfolio_return, portfolio_volatility


# Generate portfolios with Fama-French evaluation
def generate_portfolios_ff(returns, factors, factor_loadings):
    portfolio_weights = []
    portfolio_returns = []
    portfolio_volatilities = []
    portfolio_sharpe_ratios = []
    portfolio_mkt_exposures = []
    portfolio_smb_exposures = []
    portfolio_hml_exposures = []

    for _ in range(NUM_PORTFOLIOS):
        w = np.random.rand(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)

        # Calculate performance using Fama-French model
        portfolio_return, portfolio_volatility, mkt_exp, smb_exp, hml_exp = performance_calculator_ff(
            w, returns, factors, factor_loadings
        )

        portfolio_returns.append(portfolio_return)
        portfolio_volatilities.append(portfolio_volatility)
        portfolio_sharpe_ratios.append(portfolio_return / portfolio_volatility)
        portfolio_mkt_exposures.append(mkt_exp)
        portfolio_smb_exposures.append(smb_exp)
        portfolio_hml_exposures.append(hml_exp)

    return (np.array(portfolio_weights), np.array(portfolio_returns), np.array(portfolio_volatilities),
            np.array(portfolio_sharpe_ratios), np.array(portfolio_mkt_exposures),
            np.array(portfolio_smb_exposures), np.array(portfolio_hml_exposures))


# Visualise portfolios
def show_portfolios(returns, volatilities, sharpe_ratios, opt, log_returns):
    plt.figure(figsize=(12, 8))
    plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', marker='o', alpha=0.5)
    plt.title('Portfolio Performance')
    plt.xlabel('Risk')
    plt.ylabel('Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.grid(True)

    # Plot optimal portfolio
    portfolio_return, portfolio_volatility = performance_calculator(opt['x'].round(3), log_returns)
    plt.scatter(portfolio_volatility, portfolio_return, color='red', marker='*', s=120, label='Optimum Portfolio')
    plt.legend()
    plt.show()


# Visualize portfolios with factor exposures
def show_factor_exposures(returns,exposures, color_metric, exposure_name, cmap):
    # plt.figure(figsize=(12, 8))
    # plt.scatter(exposures, returns, alpha=0.5)
    # plt.title(f'Portfolio Return vs {label} Exposure')
    # plt.xlabel(f'{label} Exposure')
    # plt.ylabel('Expected Return')
    # plt.grid(True)
    # plt.show()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(exposures, returns, c=color_metric, cmap=cmap, alpha=0.7, edgecolor='k')
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.xlabel(f'{exposure_name} Exposure')
    plt.ylabel('Expected Return')
    plt.title(f'Return vs. {exposure_name} Exposure (Colored by Sharpe)')
    plt.grid(True)
    plt.show()


# Min negative sharpe function for optimization
def min_sharpe_function(weights, returns, factors, factor_loadings):
    expected_return, volatility, _, _, _ = performance_calculator_ff(weights, returns, factors, factor_loadings)
    return -expected_return / volatility


# Optimize portfolio using Fama-French model
def optimize_portfolio_ff(returns, factors, factor_loadings):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    initial_guess = np.array([1 / len(stocks)] * len(stocks))  # Equal weights as initial guess

    optimum = optimization.minimize(
        fun=min_sharpe_function,
        x0=initial_guess,
        method='SLSQP',
        args=(returns, factors, factor_loadings),
        bounds=bounds,
        constraints=constraints
    )
    return optimum


# Print optimized portfolio details
def print_optimum_portfolio_ff(opt, returns, factors, factor_loadings):
    weights = opt['x'].round(3)
    print('Optimum portfolio weights:')
    for i, stock in enumerate(stocks):
        print(f'{stock}: {weights[i]:.3f}')

    expected_return, volatility, mkt_exp, smb_exp, hml_exp = performance_calculator_ff(
        weights, returns, factors, factor_loadings
    )

    print('\nPortfolio Performance:')
    print(f'Expected Return: {expected_return:.4f}')
    print(f'Volatility: {volatility:.4f}')
    print(f'Sharpe Ratio: {expected_return / volatility:.4f}')

    print('\nFactor Exposures:')
    print(f'Market Beta: {mkt_exp:.4f}')
    print(f'Size (SMB) Exposure: {smb_exp:.4f}')
    print(f'Value (HML) Exposure: {hml_exp:.4f}')

def show_factor_exposures_with_optimum(exposures, returns, label, cmap, opt_weights, factors, returns_data, factor_loadings):
    # Calculate the optimum portfolio return and exposure
    expected_return, _, mkt_exp, smb_exp, hml_exp = performance_calculator_ff(
        opt_weights, returns_data, factors, factor_loadings
    )

    # Pick the correct exposure
    if label == "Market":
        opt_exposure = mkt_exp
    elif label == "Size (SMB)":
        opt_exposure = smb_exp
    elif label == "Value (HML)":
        opt_exposure = hml_exp
    else:
        raise ValueError("Unknown factor label")

    # Plotting
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(exposures, returns, c=returns, cmap=cmap, alpha=0.6)
    plt.colorbar(scatter, label='Expected Return')
    plt.scatter(opt_exposure, expected_return, color='red', marker='*', s=200, label='Optimum Portfolio')
    plt.title(f'Portfolio Return vs {label} Exposure')
    plt.xlabel(f'{label} Exposure')
    plt.ylabel('Expected Return')
    plt.grid(True)
    plt.legend()
    plt.show()


# Main method
if __name__ == '__main__':
    # Download data
    print("Downloading stock data...")
    dataset = download_data()

    print("Downloading Fama-French factors...")
    ff_factors = download_factors()

    # Calculate stock returns
    log_daily_returns = calculate_returns(dataset)

    # Calculate factor loadings
    print("Calculating factor loadings...")
    factor_loadings, aligned_data = calculate_factor_loadings(log_daily_returns, ff_factors)

    # Generate random portfolios with Fama-French evaluation
    print("Generating and evaluating portfolios...")
    (p_weights, p_returns, p_volatilities, p_sharpe_ratios,
     p_mkt_exposures, p_smb_exposures, p_hml_exposures) = generate_portfolios_ff(
        log_daily_returns, ff_factors, factor_loadings
    )

    # Optimize portfolio using Fama-French model
    print("Optimizing portfolio...")
    optimum_portfolio = optimize_portfolio_ff(log_daily_returns, ff_factors, factor_loadings)

    # Print optimal portfolio details
    print_optimum_portfolio_ff(optimum_portfolio, log_daily_returns, ff_factors, factor_loadings)

    # Visualize results
    show_portfolios(p_returns, p_volatilities, p_sharpe_ratios, optimum_portfolio, log_daily_returns)

    # # Visualize factor exposures
    # show_factor_exposures(p_returns, p_mkt_exposures, p_sharpe_ratios, "Market", 'coolwarm')
    # show_factor_exposures(p_returns, p_smb_exposures, p_sharpe_ratios, "Size (SMB)", "cividis")
    # show_factor_exposures(p_returns, p_hml_exposures, p_sharpe_ratios, "Value (HML)", "plasma")

    # Visualize factor exposures
    show_factor_exposures_with_optimum(p_mkt_exposures, p_returns, "Market", cmap='plasma',
                                       opt_weights=optimum_portfolio['x'],
                                       factors=ff_factors,
                                       returns_data=log_daily_returns,
                                       factor_loadings=factor_loadings)

    show_factor_exposures_with_optimum(p_smb_exposures, p_returns, "Size (SMB)", cmap='cividis',
                                       opt_weights=optimum_portfolio['x'],
                                       factors=ff_factors,
                                       returns_data=log_daily_returns,
                                       factor_loadings=factor_loadings)

    show_factor_exposures_with_optimum(p_hml_exposures, p_returns, "Value (HML)", cmap='BuGn',
                                       opt_weights=optimum_portfolio['x'],
                                       factors=ff_factors,
                                       returns_data=log_daily_returns,
                                       factor_loadings=factor_loadings)

    # Show factor contribution to optimal portfolio
    weights = optimum_portfolio['x'].round(3)
    expected_return, volatility, mkt_exp, smb_exp, hml_exp = performance_calculator_ff(
        weights, log_daily_returns, ff_factors, factor_loadings
    )

    # Plot factor contributions
    plt.figure(figsize=(10, 6))
    factors = ['Alpha', 'Market', 'Size (SMB)', 'Value (HML)']

    # Calculate alpha contribution
    total_alpha = 0
    for i, stock in enumerate(stocks):
        total_alpha += weights[i] * factor_loadings[stock][0]

    # Calculate factor contributions to expected return
    contributions = [
        total_alpha * NUM_TRADING_DAYS,
        mkt_exp * ff_factors['MKT'].mean() * NUM_TRADING_DAYS,
        smb_exp * ff_factors['SMB'].mean() * NUM_TRADING_DAYS,
        hml_exp * ff_factors['HML'].mean() * NUM_TRADING_DAYS
    ]

    plt.bar(factors, contributions)
    plt.title('Factor Contributions to Portfolio Expected Return')
    plt.ylabel('Contribution to Expected Return')
    plt.grid(True, axis='y')
    plt.show()