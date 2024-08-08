import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions, risk_models, expected_returns
from datetime import datetime
import os

def download_data(tickers, start_date, end_date):
    valid_tickers = []
    failed_tickers = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
            if not data.empty:
                valid_tickers.append(ticker)
            else:
                failed_tickers.append(ticker)
        except Exception as e:
            print(f"Failed to download data for {ticker}: {e}")
            failed_tickers.append(ticker)
    return valid_tickers, failed_tickers


def calculate_weights(mu, S, gamma, risk_tolerance):
    ef_risk_aware = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    ef_risk_aware.add_objective(objective_functions.L2_reg, gamma=gamma)
    lambda_risk_aversion = 11 - risk_tolerance
    weights_risk_aware = ef_risk_aware.max_quadratic_utility(risk_aversion=lambda_risk_aversion)
    cleaned_weights_risk_aware = ef_risk_aware.clean_weights()
    return ef_risk_aware, cleaned_weights_risk_aware



def calculate_max_sharpe_weights(mu, S, gamma):
    ef_max_sharpe = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    ef_max_sharpe.add_objective(objective_functions.L2_reg, gamma=gamma)
    weights_max_sharpe = ef_max_sharpe.max_sharpe()
    cleaned_weights_max_sharpe = ef_max_sharpe.clean_weights()
    return ef_max_sharpe, cleaned_weights_max_sharpe


def calculate_dollar_and_share_weights(cleaned_weights, principal, latest_prices):
    # Calculate dollar weights by multiplying each weight by the principal
    dollar_weights = {ticker: weight * principal for ticker, weight in cleaned_weights.items()}
    
    # Calculate share weights by dividing the dollar amount by the latest price for each ticker
    share_weights = {ticker: dollar_amount / latest_prices[ticker] for ticker, dollar_amount in dollar_weights.items()}
    
    return dollar_weights, share_weights


def create_output_df(tickers, cleaned_weights, dollar_weights, share_weights):
    # Convert dictionary values to a list and then round them
    weights_rounded = [round(weight, 4) for weight in cleaned_weights.values()]
    
    # Assuming dollar_weights and share_weights are also dictionaries,
    # convert their values to lists for DataFrame creation
    dollar_values = list(dollar_weights.values())
    shares = [round(share) for share in share_weights.values()]
    
    output = pd.DataFrame({
        'Ticker': tickers,
        'Weight': weights_rounded,
        'Dollar Value': dollar_values,
        'Shares': shares
    })
    return output


def optimize_portfolio(tickers, risk_tolerance, principal, risk_free_rate, gamma=0):
    end_date = datetime.today().strftime('%Y-%m-%d')
    valid_tickers, failed_tickers = download_data(tickers, "2010-01-01", end_date)
    data = yf.download(valid_tickers, start="2020-01-01", end=end_date)['Adj Close']
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    risk_free_ticker = 'RiskFree'
    mu[risk_free_ticker] = risk_free_rate
    risk_free_cov = pd.DataFrame(0, index=S.index, columns=[risk_free_ticker])
    S = pd.concat([S, risk_free_cov], axis=1)
    S.loc[risk_free_ticker] = 0
    S = S + S.T - np.diag(S.values.diagonal())
    S = (S + S.T) / 2
    S = np.array(S)
    min_eig = np.min(np.real(np.linalg.eigvals(S)))
    if min_eig < 0:
        S -= 10 * min_eig * np.eye(*S.shape)
    ef_risk_aware, cleaned_weights_risk_aware = calculate_weights(mu, S, gamma, risk_tolerance)
    ef_max_sharpe, cleaned_weights_max_sharpe = calculate_max_sharpe_weights(mu, S, gamma)
    latest_prices = data.iloc[-1].copy()
    latest_prices[risk_free_ticker] = 1
    dollar_weights_risk_aware, share_weights_risk_aware = calculate_dollar_and_share_weights(cleaned_weights_risk_aware, principal, latest_prices)
    dollar_weights_max_sharpe, share_weights_max_sharpe = calculate_dollar_and_share_weights(cleaned_weights_max_sharpe, principal, latest_prices)
    output_risk_aware = create_output_df(valid_tickers + [risk_free_ticker], cleaned_weights_risk_aware, dollar_weights_risk_aware, share_weights_risk_aware)
    output_max_sharpe = create_output_df(valid_tickers + [risk_free_ticker], cleaned_weights_max_sharpe, dollar_weights_max_sharpe, share_weights_max_sharpe)
    failed_df = pd.DataFrame({
        'Ticker': failed_tickers,
        'Weight': 'Could not find this ticker, please correct or remove',
        'Dollar Value': None,
        'Shares': None
    })
    output_risk_aware = pd.concat([output_risk_aware, failed_df], ignore_index=True)
    output_max_sharpe = pd.concat([output_max_sharpe, failed_df], ignore_index=True)
    expected_return_risk_aware, annual_volatility_risk_aware, sharpe_ratio_risk_aware = ef_risk_aware.portfolio_performance(verbose=True)
    expected_return_max_sharpe, annual_volatility_max_sharpe, sharpe_ratio_max_sharpe = ef_max_sharpe.portfolio_performance(verbose=True)
    performance_metrics_risk_aware = pd.DataFrame({
        'Metric': ['Expected Return', 'Annual Volatility', 'Sharpe Ratio'],
        'Value': [expected_return_risk_aware, annual_volatility_risk_aware, sharpe_ratio_risk_aware]
    })
    performance_metrics_max_sharpe = pd.DataFrame({
        'Metric': ['Expected Return', 'Annual Volatility', 'Sharpe Ratio'],
        'Value': [expected_return_max_sharpe, annual_volatility_max_sharpe, sharpe_ratio_max_sharpe]
    })
    return output_risk_aware, output_max_sharpe, performance_metrics_risk_aware, performance_metrics_max_sharpe