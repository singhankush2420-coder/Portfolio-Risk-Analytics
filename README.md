# Portfolio Risk Analytics Engine & Power BI Dashboard

An end-to-end portfolio risk analytics project built with **Python, PostgreSQL and Power BI**.

## Project objective

Build a portfolio risk engine that reconstructs historical portfolio holdings, retrieves market data, calculates performance and risk measures, stores reporting outputs in PostgreSQL, and presents the results through an interactive Power BI dashboard.

## Architecture

**Market Data → Python Risk Engine → PostgreSQL Reporting Layer → Power BI**

## Risk analytics implemented

- Time-Weighted Return (TWR)
- CAGR
- Portfolio volatility
- Beta
- Sharpe Ratio
- Sortino Ratio
- Skewness and Kurtosis
- Maximum Drawdown
- Historical VaR
- Parametric VaR
- Monte Carlo VaR
- Expected Shortfall
- Correlation analysis
- Risk Contribution / Marginal Risk Contribution
- CAPM expected returns
- Portfolio optimization
- Minimum-volatility portfolio
- Maximum-Sharpe portfolio
- Efficient-frontier analysis
- Market stress testing
- FX regression analysis
- FX stress testing

## Reporting

Power BI is used as the reporting and visualization layer. The report currently contains:

1. Portfolio Overview
2. Risk Analysis
3. Portfolio Optimization
4. Stress Testing

## Data

The current project uses Yahoo Finance market data for the development environment. The portfolio universe and historical transactions are stored in PostgreSQL.

## Security

Database credentials are not stored in the repository. The Python script reads the PostgreSQL password from the `PORTFOLIO_DB_PASSWORD` environment variable.

Create a local `.env`/environment variable before running the script. Never commit passwords, API keys, raw proprietary data, or personal credentials.

## Current status

- Risk engine: completed
- PostgreSQL reporting layer: completed
- Power BI risk report: completed
- Automated scheduled execution / Power BI refresh: planned next phase