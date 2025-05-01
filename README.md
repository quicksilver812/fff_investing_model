# Factor-Based Portfolio Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)

An advanced portfolio optimization tool implementing the Fama-French three-factor model to build optimal investment portfolios based on market risk, size, and value factors.

## Overview

This tool extends traditional Modern Portfolio Theory by incorporating the Fama-French three-factor model to provide deeper insights into portfolio risk exposures. It:

1. Downloads historical stock data using Yahoo Finance
2. Retrieves Fama-French factors data (Market, SMB, HML)
3. Calculates factor loadings (betas) for each stock
4. Simulates thousands of portfolios with different asset allocations
5. Visualizes portfolio performance across different factor exposures
6. Uses optimization techniques to find the portfolio with the maximum risk-adjusted return

## Features

- **Multi-factor Analysis**: Incorporates Market, Size (SMB), and Value (HML) factors
- **Factor Loadings**: Calculates individual stock exposures to each factor
- **Advanced Visualization**: Plots factor exposures with optimized portfolio highlights
- **Factor Attribution**: Shows how each factor contributes to portfolio returns
- **Robust Optimization**: Finds optimal weights considering factor exposures

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/factor-portfolio-optimization.git
cd factor-portfolio-optimization

# Install required packages
pip install numpy pandas matplotlib yfinance scipy statsmodels pandas-datareader
```

## Usage

The script comes with predefined stocks and date ranges that you can modify in the code:

```python
# Select stocks
stocks = ['AAPL', 'GE', 'WMT', 'AMZN', 'TSLA', 'DB']

# Select start and end dates
start_date = '2012-01-01'
end_date = '2017-01-01'
```

Run the script:

```bash
python factor_portfolio_optimizer.py
```

## Output

The program will:
1. Download stock data and Fama-French factors
2. Calculate factor loadings for each stock
3. Generate and evaluate random portfolios
4. Find the optimal portfolio allocation
5. Display a series of visualizations showing:
   - Portfolio risk-return tradeoff (efficient frontier)

     <img width="650" alt="Image" src="https://github.com/user-attachments/assets/8a2c340e-cbc0-43aa-89bd-a6f684b6458c" />
     
   - Market factor exposure vs. returns

     <img width="650" alt="Image" src="https://github.com/user-attachments/assets/716f79a0-f52b-4290-8f43-f6406223d162" />
     
   - Size factor (SMB) exposure vs. returns

     <img width="650" alt="Image" src="https://github.com/user-attachments/assets/8218aae6-2caa-43a6-9f6c-9b0e466cdb20" />
     
   - Value factor (HML) exposure vs. returns

     <img width="650" alt="Image" src="https://github.com/user-attachments/assets/03de81ec-c8b2-4c16-9e8e-9fc6503cc6f5" />
     
   - Factor contributions to the optimal portfolio's expected return

     <img width="650" alt="Image" src="https://github.com/user-attachments/assets/aa3f53e5-0525-47b4-95a7-5e1d5d6834c4" />
     
6. Terminal Output
```Downloading stock data...
Downloading Fama-French factors...
Calculating factor loadings...
Generating and evaluating portfolios...
Optimizing portfolio...
Optimum portfolio weights:
AAPL: 0.138
GE: 0.383
WMT: 0.000
AMZN: 0.316
TSLA: 0.163
DB: 0.000

Portfolio Performance:
Expected Return: 0.2355
Volatility: 0.1939
Sharpe Ratio: 1.2146

Factor Exposures:
Market Beta: 1.1353
Size (SMB) Exposure: -0.0825
Value (HML) Exposure: -0.4594
```

## How It Works

### Theory

This implementation extends Modern Portfolio Theory with the Fama-French three-factor model, which suggests that:
- Stock returns are explained by exposure to the overall market, size premium, and value premium
- The three factors are:
  - Market risk (excess return of the market over risk-free rate)
  - Size (SMB - Small Minus Big) capturing the return premium of small-cap companies
  - Value (HML - High Minus Low) capturing the return premium of high book-to-market companies

### Key Functions

* `download_data()`: Retrieves historical stock prices from Yahoo Finance
* `download_factors()`: Retrieves Fama-French factor data 
* `calculate_returns()`: Computes logarithmic returns from price data
* `calculate_factor_loadings()`: Estimates factor betas for each stock using regression
* `performance_calculator_ff()`: Calculates expected return and volatility using the factor model
* `generate_portfolios_ff()`: Creates random portfolios and evaluates their factor exposures
* `optimize_portfolio_ff()`: Finds weights that maximize the risk-adjusted return considering factor exposures
* `show_factor_exposures_with_optimum()`: Visualizes portfolio factor exposures with the optimal portfolio highlighted

## Requirements

- Python 3.6+
- NumPy
- Pandas
- Matplotlib
- yfinance
- SciPy
- statsmodels
- pandas-datareader

## Advanced Customization

You can customize the analysis by:
- Changing the stocks in the portfolio
- Modifying the date range for historical analysis
- Adjusting the number of simulated portfolios (NUM_PORTFOLIOS constant)
- Adding constraints to the optimization (e.g., sector exposure limits)

## Contributing

Contributions are welcome! Areas for potential improvement include:
- Adding more factors (e.g., momentum, profitability)
- Implementing portfolio constraints
- Adding transaction costs and rebalancing considerations
- Creating an interactive dashboard

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Eugene Fama and Kenneth French for developing the three-factor model
- Yahoo Finance for providing historical stock data
- Kenneth French's data library for factor data
