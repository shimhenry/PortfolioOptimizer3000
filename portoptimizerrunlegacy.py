import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
from pypfopt import risk_models
from pypfopt import expected_returns
from datetime import datetime

# Plug input,output file path in here:

input_file_path = r"C:\Users\Henry Goldsmith\OneDrive\Desktop\Port Optimizer\Portfolio Optimization Template.xlsx"
input_df = pd.read_excel(input_file_path)
output_file_path = r"C:\Users\Henry Goldsmith\OneDrive\Desktop\Port Optimizer\optimized_portfolio.xlsx"

# Set gamma value, author of the pypfopt says 1 is okay for 20 assets
gamma = 0


# Ideas to improve SR
    # more recent lookback period? Less recent look back period?
    # Average RF rate ? Too rich rn
    # Asset selection, SNP plus some EM fund holdings?

# Other Improvements
    # abstaction/scope
    # Historic look back changes like interval and period
    #


def validate_tickers(tickers, end_date):
    valid_tickers = []
    failed_tickers = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, start="2000-01-01", end=end_date)['Adj Close']
            if not data.empty:
                valid_tickers.append(ticker)
            else:
                failed_tickers.append(ticker)
        except Exception as e:
            print(f"Failed to download data for {ticker}: {e}")
            failed_tickers.append(ticker)
    return valid_tickers, failed_tickers

def optimize_portfolio(tickers, risk_tolerance, principal, risk_free_rate, gamma):
# adjust end date here
    end_date = datetime.today().strftime('%Y-%m-%d')

    valid_tickers, failed_tickers = validate_tickers(tickers, end_date)

    data = yf.download(valid_tickers, start="2020-01-01", end=end_date)['Adj Close']

    # Ensure data is df
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)

    # Risk_free_asset
    risk_free_ticker = 'Risk_Free_Asset'
    mu[risk_free_ticker] = risk_free_rate
    risk_free_cov = pd.DataFrame(0, index=S.index, columns=[risk_free_ticker])
    S = pd.concat([S, risk_free_cov], axis=1)
    S.loc[risk_free_ticker] = 0

    # ChatGPT Covariance Nonsense
    S = S + S.T - np.diag(S.values.diagonal())
    S = (S + S.T) / 2  # Ensure exact symmetry
    S = np.array(S)
    min_eig = np.min(np.real(np.linalg.eigvals(S)))
    if min_eig < 0:
        S -= 10 * min_eig * np.eye(*S.shape)

    # Dynamic gamma value place holder
    # gamma = min(10, max(1, 1 + num_valid_tickers // 20))
    print(f"Using gamma value: {gamma}")

    # Risk Aware PortOp,
    ef_risk_aware = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    ef_risk_aware.add_objective(objective_functions.L2_reg, gamma=gamma)
    lambda_risk_aversion = 11 - risk_tolerance
    weights_risk_aware = ef_risk_aware.max_quadratic_utility(risk_aversion=lambda_risk_aversion)
    cleaned_weights_risk_aware = ef_risk_aware.clean_weights()

    # Max Sharpe Portfolio Optimization
    ef_max_sharpe = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    ef_max_sharpe.add_objective(objective_functions.L2_reg, gamma=gamma)
    weights_max_sharpe = ef_max_sharpe.max_sharpe()
    cleaned_weights_max_sharpe = ef_max_sharpe.clean_weights()

    # weights to pd.series
    cleaned_weights_series_risk_aware = pd.Series(cleaned_weights_risk_aware)
    cleaned_weights_series_max_sharpe = pd.Series(cleaned_weights_max_sharpe)

    # Get latest prices
    latest_prices = data.iloc[-1].copy()
    latest_prices[risk_free_ticker] = 1

    # weights -> shares and dollar amounts
    dollar_weights_risk_aware = cleaned_weights_series_risk_aware * principal
    share_weights_risk_aware = dollar_weights_risk_aware / latest_prices

    dollar_weights_max_sharpe = cleaned_weights_series_max_sharpe * principal
    share_weights_max_sharpe = dollar_weights_max_sharpe / latest_prices

    # Final Ports to dataframes
    output_risk_aware = pd.DataFrame({
        'Ticker': valid_tickers + [risk_free_ticker],
        'Weight': cleaned_weights_series_risk_aware.values.round(4),
        'Dollar Value': dollar_weights_risk_aware.values,
        'Shares': share_weights_risk_aware.values.round().astype(int)
    })

    output_max_sharpe = pd.DataFrame({
        'Ticker': valid_tickers + [risk_free_ticker],
        'Weight': cleaned_weights_series_max_sharpe.values.round(4),
        'Dollar Value': dollar_weights_max_sharpe.values,
        'Shares': share_weights_max_sharpe.values.round().astype(int)
    })

    # Failed ticker df
    failed_df = pd.DataFrame({
        'Ticker': failed_tickers,
        'Weight': 'Could not find this ticker, please correct or remove',
        'Dollar Value': None,
        'Shares': None
    })
    output_risk_aware = pd.concat([output_risk_aware, failed_df], ignore_index=True)
    output_max_sharpe = pd.concat([output_max_sharpe, failed_df], ignore_index=True)

    # # Portfolio performance metrics
    # Risk Aware metrics
    expected_return_risk_aware, annual_volatility_risk_aware, sharpe_ratio_risk_aware = ef_risk_aware.portfolio_performance(verbose=True)
    print(f"Risk Aware Sharpe Ratio: {sharpe_ratio_risk_aware}")

    # Max Sharpe performance metrics
    expected_return_max_sharpe, annual_volatility_max_sharpe, sharpe_ratio_max_sharpe = ef_max_sharpe.portfolio_performance(verbose=True)
    print(f"Max Sharpe Ratio: {sharpe_ratio_max_sharpe}")

    # Combine portfolio and performance dataframes
    # # making performance dfs
    performance_metrics_risk_aware = pd.DataFrame({
        'Metric': ['Expected Return', 'Annual Volatility', 'Sharpe Ratio'],
        'Value': [expected_return_risk_aware, annual_volatility_risk_aware, sharpe_ratio_risk_aware]
    })
    performance_metrics_max_sharpe = pd.DataFrame({
        'Metric': ['Expected Return', 'Annual Volatility', 'Sharpe Ratio'],
        'Value': [expected_return_max_sharpe, annual_volatility_max_sharpe, sharpe_ratio_max_sharpe]
    })

    with pd.ExcelWriter(output_file_path) as writer:
        output_risk_aware.to_excel(writer, sheet_name='Risk Aware Portfolio', index=False)
        performance_metrics_risk_aware.to_excel(writer, sheet_name='Risk Aware Performance', index=False)
        output_max_sharpe.to_excel(writer, sheet_name='Max Sharpe Portfolio', index=False)
        performance_metrics_max_sharpe.to_excel(writer, sheet_name='Max Sharpe Performance', index=False)

    return output_risk_aware, output_max_sharpe

tickers = input_df['Tickers'].dropna().tolist()
risk_tolerance = input_df['Risk_Tolerance'].dropna().iloc[0]
principal = input_df['Principal(USD)'].dropna().iloc[0]
risk_free_rate = input_df['RF Rate'].dropna().iloc[0]

optimized_portfolio_risk_aware, optimized_portfolio_max_sharpe = optimize_portfolio(tickers, risk_tolerance, principal, risk_free_rate, gamma)


print("Risk Aware Portfolio:")
print(optimized_portfolio_risk_aware)
print("\nMax Sharpe Portfolio:")
print(optimized_portfolio_max_sharpe)
