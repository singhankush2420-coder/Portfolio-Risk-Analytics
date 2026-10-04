import psycopg2
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
## import seaborn as sns
## import datetime as dt
from scipy.stats import norm,kurtosis,skew
from scipy.optimize import minimize
import statsmodels.api as sm
import os

def get_database_connection():
    connection = psycopg2.connect(
        host="localhost",
        database="Portfolio",
        user="postgres",
        password=os.getenv("PORTFOLIO_DB_PASSWORD"),
        port="1234"
    )
    return connection


def load_transactions(connection):
    query = """
    SELECT
        transaction_id,
        trade_date,
        ticker,
        transaction_type,
        quantity,
        price
    FROM portfolio1
    ORDER BY trade_date, transaction_id;
    """

    return pd.read_sql(query, connection)


connection = get_database_connection()

transactions = load_transactions(connection)

transactions['trade_date'] = pd.to_datetime(
    transactions['trade_date']  
)

print(transactions)

connection.close()

def calculate_holdings(transactions):

    holdings = {}

    for _, row in transactions.iterrows():

        ticker = row["ticker"]
        quantity = row["quantity"]
        transaction_type = row["transaction_type"]

        if ticker not in holdings:
            holdings[ticker] = 0

        if transaction_type == "BUY":
            holdings[ticker] += quantity

        elif transaction_type == "SELL":
            holdings[ticker] -= quantity

    return holdings

holdings = calculate_holdings(transactions)

print(holdings)

tickers = transactions['ticker'].unique()
print(tickers)

st = transactions['trade_date'].min()
et = pd.Timestamp.today()

df =  pd.DataFrame()
for t in tickers:
    data = yf.download(t,start=st, end=et,auto_adjust=False)['Adj Close']
    df[t] = data

benchmark = yf.download('^NSEI',start=st,end=et,auto_adjust=False)['Adj Close']

common_dates = df.index.intersection(benchmark.index)
df_aligned = df.loc[common_dates]
benchmark_aligned = benchmark.loc[common_dates]

valid_rows_mask = df_aligned.notna().all(axis=1)
df_final = df_aligned.loc[valid_rows_mask]
benchmark_final = benchmark_aligned.loc[valid_rows_mask]

historical_holdings = pd.DataFrame(
    0,
    index=df_final.index,
    columns=tickers
)
print(historical_holdings.head())

for ticker in tickers:
    ticker_transactions = transactions[transactions['ticker']==ticker]
    print('\n', ticker)
    print(ticker_transactions)

transactions['signed_quantity'] = transactions.apply(
    lambda row: row['quantity']
    if row['transaction_type'] == 'BUY'
    else -row['quantity'],
    axis=1
)

print(
    transactions[
        ['trade_date','ticker','transaction_type','quantity','signed_quantity']
    ]
)

for ticker in tickers:
    ticker_transactions = transactions[
        transactions['ticker'] == ticker 
    ]

    for date in df_final.index:
        quantity_held = ticker_transactions[
            ticker_transactions['trade_date'] <= date 
        ]['signed_quantity'].sum()
        historical_holdings.loc[date,ticker] = quantity_held 
print(historical_holdings.head())

print(historical_holdings.loc[
    [
        "2021-01-11",
        "2021-03-15",
        "2021-06-21",
        "2022-01-10",
        "2022-08-05"
    ]
])


print(historical_holdings.index)
print(df_final.index)
print(benchmark_final.index)



position_value = pd.DataFrame(index = df_final.index, columns=tickers)

for t in tickers:
    position_value[t] = historical_holdings[t] * df_final[t]


portfolio_value = position_value.sum(axis=1)

transactions['cash_flow'] = transactions['signed_quantity'] * transactions['price']

daily_cash_flow = (transactions.groupby('trade_date')['cash_flow'].sum().reindex(portfolio_value.index, fill_value=0))

daily_twr = (portfolio_value.diff() - daily_cash_flow) / portfolio_value.shift(1)

daily_twr = daily_twr.dropna()

simple_return = (1 + daily_twr).cumprod() - 1 

benchmark_returns = benchmark_final.pct_change().dropna()

common_dates = daily_twr.index.intersection(benchmark_returns.index)

aligned_twr = daily_twr.loc[common_dates]
aligned_simple_twr = (1 + aligned_twr).cumprod() - 1
aligned_benchmark_returns = benchmark_returns.loc[common_dates].squeeze()
benchmark_simple_return = (1 + aligned_benchmark_returns).cumprod() - 1 

log_returns = np.log(1 + aligned_twr)
benchmark_log_returns = np.log(1 + aligned_benchmark_returns)


years = (aligned_twr.index[-1] - aligned_twr.index[0]).days / 365.25

portfolio_cagr = (1 + aligned_simple_twr.iloc[-1]) ** (1/years) - 1 
benchmark_cagr = (1 + benchmark_simple_return.iloc[-1]) ** (1/years) - 1 

print(f'Portfolio CAGR: {portfolio_cagr}')
print(f'Benchmark CAGR: {benchmark_cagr}')

def annualized_volatility(returns):
    return returns.std() 

portfolio_log_returns = log_returns
benchmark_returns = benchmark_log_returns

portfolio_volatility = annualized_volatility(portfolio_log_returns)
benchmark_volatility = annualized_volatility(benchmark_returns)
print(f'Portfolio Annualized Volatility: {portfolio_volatility:.4f}')
print(f'Benchmark Annualized Volatility: {benchmark_volatility}')

def beta_calculation(stock_returns,benchmark_returns):
    covariance = np.cov(stock_returns,benchmark_returns)[0][1]
    variance = np.var(benchmark_returns)
    return covariance/variance 

beta = {}
for ticker in tickers:
    stock_returns = df_final[ticker].pct_change().dropna() 
    ticker_common_dates = stock_returns.index.intersection(aligned_benchmark_returns.index)
    stock_returns_aligned = stock_returns.loc[ticker_common_dates]
    benchmark_returns_aligned = aligned_benchmark_returns.loc[ticker_common_dates].squeeze()
    beta[ticker] = beta_calculation(stock_returns_aligned,benchmark_returns_aligned)
        
print(beta)  

portfolio_beta = beta_calculation(aligned_twr,aligned_benchmark_returns)
print(portfolio_beta)

risk_free_rate = 0.0591
portfolio_return = aligned_twr.mean() * 252
portfolio_deviation = aligned_twr.std() * np.sqrt(252)

def sharpe_ratio(portfolio_return,portfolio_deviation,risk_free_rate):
    return (portfolio_return - risk_free_rate) / portfolio_deviation

sharpe_ratio = sharpe_ratio(portfolio_return,portfolio_deviation,risk_free_rate)
print(f'Sharpe Ratio: {sharpe_ratio:.4f}')

def downside_deviation(aligned_twr,risk_free_rate):
    rf = risk_free_rate / 252
    excess_return = aligned_twr - rf
    downside_return = np.minimum(0,excess_return)
    downside_deviation = np.sqrt(np.mean(downside_return ** 2)) * np.sqrt(252)
    return downside_deviation

downside_deviation = downside_deviation(aligned_twr,risk_free_rate)

def sortino_ration(portfolio_return,downside_deviation,risk_free_rate):
    return (portfolio_return - risk_free_rate) / downside_deviation

sortino_ratio = sortino_ration(portfolio_return,downside_deviation,risk_free_rate)
print(f'Sortino Ratio: {sortino_ratio:.4f}')

skewness = aligned_twr.skew()
kurt = kurtosis(aligned_twr,fisher=False)
print(f'Skewness: {skewness}')
print(f'Kurtosis: {kurt}') 

cumalative_returns = (1 + aligned_twr).cumprod()
cumalaitve_max = cumalative_returns.cummax()
def md(cumalative_returns,cumalaitve_max):
    return cumalative_returns / cumalaitve_max - 1 
maximum_drawdown = md(cumalative_returns,cumalaitve_max)
max_drawdown = maximum_drawdown.min()
print(f'Maximum Drawdown: {max_drawdown:.4f}')

plt.figure(figsize=(8,6))
maximum_drawdown.plot(label='Drrawdown', color='red')
plt.axhline(y=0, color='black',linestyle='--',linewidth=0.5)
plt.title("Portfolio Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown (%)")
plt.legend()
plt.fill_between(maximum_drawdown.index, maximum_drawdown, 0, 
                 alpha=0.3, color="red")  
plt.tight_layout()
plt.show()

## Historical VaR and Expected Shortfall Method
ci = 0.95
days = int(input('Enter the horizon days-'))
range_returns = log_returns.rolling(window = days).sum().dropna()
def histo_var(range_returns,ci,portfolio_value):
    h_var = -np.percentile(range_returns,100 * (1 - ci)) * portfolio_value.iloc[-1]
    return h_var
historical_VaR = histo_var(range_returns,ci,portfolio_value)
print(f'Historical VaR: {historical_VaR}')

def es(range_returns,ci,portfolio_value):
    var = np.percentile(range_returns, 100 * (1 - ci))
    cvar = range_returns[range_returns <= var]
    h_cvar = -cvar.mean() * portfolio_value.iloc[-1]
    return h_cvar
expected_shortfall = es(range_returns,ci,portfolio_value)
print(f'Expected_shortfall: {expected_shortfall}')

## Parametric VaR & Expected Method
mu_h = log_returns.mean()  * days
sd_h = portfolio_volatility * np.sqrt(days)
z_score = norm.ppf(1-ci)
def para(mu_h,sd_h,z_score,portfolio_value):
    par_var = -(mu_h + z_score * sd_h) * portfolio_value.iloc[-1]
    return par_var
parametric_var = para(mu_h,sd_h,z_score,portfolio_value)

def es_par(mu_h,sd_h,z_score,portfolio_value):
    alpha = 1 - ci
    p_var = mu_h + z_score * sd_h
    p_cvar = mu_h - sd_h * (norm.pdf(z_score)/ alpha)
    para_cvar = -p_cvar * portfolio_value.iloc[-1]
    return para_cvar
parametric_expected_shortfal = es_par(mu_h,sd_h,z_score,portfolio_value)

print(f'Parametric VaR: {parametric_var}')
print(f'Parametric Expected Shortfall: {parametric_expected_shortfal}')

simulations = int(input('Enter the Simulations eg:5000- '))
simulated_returns = np.random.normal(
    loc = mu_h,
    scale = sd_h,
    size = simulations
)

def monte_carlo_simulation(simulated_returns,ci,portfolio_value):
    mc_var = -np.percentile(simulated_returns, 100 * (1 - ci)) * portfolio_value.iloc[-1]
    return mc_var

monte_carlo_var = monte_carlo_simulation(simulated_returns,ci,portfolio_value)

def monte_expectedshortfall(simulated_returns,ci,portfolio_value):
    monte_var = np.percentile(simulated_returns, 100 * (1 - ci))
    monte_es = simulated_returns[simulated_returns <= monte_var]
    expected_shortfall = -monte_es.mean() * portfolio_value.iloc[-1]
    return expected_shortfall

monte_carlo_expectedshortfall = monte_expectedshortfall(simulated_returns,ci,portfolio_value)

print(f'Monte Carlo VaR: {monte_carlo_var}')
print(f'Monte Carlo ES: {monte_carlo_expectedshortfall}')

stock_log_returns = pd.DataFrame(index = df_final.index)
for tic in tickers:
    stock_log_returns[tic] = np.log(df_final[tic]/df_final[tic].shift(1)).dropna()
correlation_matrix = stock_log_returns.corr()
print(correlation_matrix)


def stock_weights(current_position_value,current_portfolio_value):
    return current_position_value / current_portfolio_value

current_portfolio_value = portfolio_value.iloc[-1]
current_position_value = position_value.iloc[-1]


weights =  stock_weights(current_position_value,current_portfolio_value) 
print(weights) 

covariance_matrix = stock_log_returns.cov()

def vol(covariance_matrix,weights):
    return  np.sqrt(weights.T @covariance_matrix @weights) 

volatility = vol(covariance_matrix,weights)
print(f'Volatility: {volatility}')

mrc = (covariance_matrix @ weights) / volatility
rc = weights * mrc
print(rc)
print(rc.sum())

rc_percent = (rc / volatility) * 100
print(rc_percent)

simple_returns= pd.DataFrame(index=df_final.index[1:], columns=tickers)
for t in tickers:
    simple_returns[t] = df_final[t].pct_change().dropna()

print(simple_returns.head())
print(simple_returns.isna().sum())

## CAPM Calculation
market_returns = np.mean(benchmark_returns) * 252

def capm_calculations(risk_free_rate,beta,market_returns):
    expected_returns = risk_free_rate + beta * (market_returns - risk_free_rate)
    return expected_returns

tcs_expected_return = capm_calculations(risk_free_rate,beta['TCS.NS'],market_returns)
reliance_expected_return = capm_calculations(risk_free_rate,beta['RELIANCE.NS'],market_returns)
infy_expected_return = capm_calculations(risk_free_rate,beta['INFY.NS'],market_returns)
hdfc_expected_return = capm_calculations(risk_free_rate,beta['HDFCBANK.NS'],market_returns)
icici_expected_return = capm_calculations(risk_free_rate,beta['ICICIBANK.NS'],market_returns) 

print(
    f'TCS Expected Return: {tcs_expected_return}\n'
    f'Reliance Expected Return: {reliance_expected_return}\n'
    f'Infy Expected Return: {infy_expected_return}\n'
    f'HDFC Expected Return: {hdfc_expected_return}\n'
    f'ICICI Expected Return: {icici_expected_return}\n'
)

capm_returns = pd.Series(
 [
    tcs_expected_return,
    reliance_expected_return,
    infy_expected_return,
    hdfc_expected_return,
    icici_expected_return
 ],
index=tickers
)

simple_returns_cov = simple_returns.cov()
print(simple_returns_cov)

random_weights = np.random.random(len(tickers))
random_weights /= np.sum(random_weights)
print(random_weights)

portfolio_weights = []
pfolio_returns = []
pfolio_volatility = []
for x in range(10000):
    random_weights = np.random.random(len(tickers))
    random_weights /= np.sum(random_weights)
    portfolio_weights.append(random_weights.copy())
    pfolio_returns.append(np.sum(random_weights * capm_returns))
    pfolio_volatility.append(np.sqrt(random_weights.T @simple_returns_cov @random_weights) * np.sqrt(252))

portfolio_weights = np.array(portfolio_weights)
weights_df = pd.DataFrame(portfolio_weights,columns = tickers)
pfolio_returns = np.array(pfolio_returns)
pfolio_volatility = np.array(pfolio_volatility)
portfolio_weights,pfolio_returns,pfolio_volatility

pfolio = pd.DataFrame({'Returns':pfolio_returns, 'Volatility':pfolio_volatility})
pfolio_data = pd.concat([weights_df,pfolio],axis=1)


min_volatility = pfolio_volatility.min()
rf = 0.0591 
pfolio_data['sharpe_ratio'] = (pfolio_returns - rf) / pfolio_volatility
maximum_sharpe_ratio = pfolio_data['sharpe_ratio'].max()

print(f'Minimum Volatility: {min_volatility:.4f}')
print(f'Maximum Sharpe Ratio: {maximum_sharpe_ratio:.4f}')

pfolio_vol = pfolio_data.loc[pfolio_data['Volatility'] == min_volatility]
pfolio_sharpe = pfolio_data.loc[pfolio_data['sharpe_ratio'] == maximum_sharpe_ratio]
print(pfolio_vol)
print(pfolio_sharpe)

### plotting the efficient frontier
plt.figure(figsize=(10,6))
plt.scatter(x=pfolio['Volatility'], y=pfolio['Returns'])
plt.scatter(x=pfolio_vol['Volatility'], y=pfolio_vol['Returns'], color='red', marker='^', s=200, label='Minimum Volatility')
plt.scatter(x=pfolio_sharpe['Volatility'], y=pfolio_sharpe['Returns'], color='green', marker='^', s=200, label='Maximum Sharpe Ratio')
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.legend()
plt.show() 

## Monte Carlo simulations 
pfolio_data = pfolio_data.sort_values(by='Volatility')
print(pfolio_data.head())

eff_pfolio = []
ef_benchmark = pfolio_data.iloc[0]
for _, row in pfolio_data.iterrows():
    if row['Returns'] >= ef_benchmark['Returns']:
        ef_benchmark = row
        eff_pfolio.append(row)

eff_pfolio = pd.DataFrame(eff_pfolio)
print(eff_pfolio.head())

plt.figure(figsize=(10,60))
plt.plot(eff_pfolio['Volatility'],eff_pfolio['Returns'], marker='*',markersize=3)
plt.scatter(x=pfolio_vol['Volatility'], y=pfolio_vol['Returns'], color='red', marker='*', s=100, label='Minimum Volatility')
plt.scatter(x=pfolio_sharpe['Volatility'], y=pfolio_sharpe['Returns'], color='green', marker='*', s=100, label='Maximum Sharpe Ratio')
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.title('Effcient frontier')
plt.tight_layout()
plt.show()

pfolio_vol = pfolio_data.loc[pfolio_data['Volatility'] == min_volatility]
pfolio_sharpe = pfolio_data.loc[pfolio_data['sharpe_ratio'] == maximum_sharpe_ratio]
print(pfolio_vol)
print(pfolio_sharpe)

print(weights)

def pfolio_variance(weights):
    return weights.T @ simple_returns_cov @ weights

x0  = weights
bounds = len(tickers) * [(0.0,1.0)]
constraints = {
        'type':'eq',
        'fun':lambda weights:np.sum(weights) - 1
}

result = minimize(
    pfolio_variance,
    x0,
    bounds = bounds,
    constraints= constraints
)
print(result)

optimized_weights = result.x

print(f'Optimized Successful:', result.success)
print('Optimized Weights:')

for Tickers, Weights in zip(tickers,optimized_weights):
    print(f'{Tickers}: {Weights:.4f}')

optimized_variance = result.fun

optimized_daliy_volatility = np.sqrt(optimized_variance)
optimized_annual_volatility = optimized_daliy_volatility * np.sqrt(252)

print(f"\nOptimized Variance: {optimized_variance:.6f}")
print(f"Optimized Annual Volatility: {optimized_annual_volatility:.4%}")

optimized_portfolio_returns = (capm_returns * optimized_weights).sum()

optimized_sharpe_ratio = (optimized_portfolio_returns - rf) / optimized_annual_volatility
print(optimized_sharpe_ratio)

def pfolio_sharpe(weights):
    portfolio_r = (capm_returns * weights).sum()
    portfolio_v = np.sqrt(weights.T @ simple_returns_cov @ weights) * np.sqrt(252)
    sharpe = (portfolio_r - rf) / portfolio_v
    return -sharpe

maximum = minimize(
    pfolio_sharpe,
    x0,
    bounds = bounds,
    constraints= constraints
)

print(maximum)

maximum_sharpe = -(maximum.fun)
new_optimized_weights = maximum.x

print(f'Optimized Successful:', maximum.success)
print(f'Optimized Weights: {new_optimized_weights}')

for ticker, weight in zip(tickers, new_optimized_weights):
    print(f'{ticker}: {weight:.4f}')

optimized_return = (capm_returns * new_optimized_weights).sum()
new_optimized_volatility = np.sqrt(new_optimized_weights.T @ simple_returns_cov @ new_optimized_weights) *np.sqrt(252)

sr = (optimized_return - rf) / new_optimized_volatility
print(sr)

comparison_table = pd.DataFrame({
    'Actual Portfolio':[
        weights.iloc[0],
        weights.iloc[1],
        weights.iloc[2],
        weights.iloc[3],
        weights.iloc[4],
        weights.sum(),
        portfolio_return,
        portfolio_volatility * np.sqrt(252),
        sharpe_ratio
    ],
    'Minimum-Volatility Portfolio':[
        optimized_weights[0],
        optimized_weights[1],
        optimized_weights[2],
        optimized_weights[3],
        optimized_weights[4],
        optimized_weights.sum(),
        optimized_portfolio_returns,
        optimized_annual_volatility,
        optimized_sharpe_ratio
    ],
    'Maximum-Sharpe Ratio':[
        new_optimized_weights[0],
        new_optimized_weights[1],
        new_optimized_weights[2],
        new_optimized_weights[3],
        new_optimized_weights[4],
        new_optimized_weights.sum(),
        optimized_return,
        new_optimized_volatility,
        maximum_sharpe
    ]
}, index = [
    'TCS',
    'Reliance',
    'Infosys',
    'HDFC',
    'ICICI',
    'Total Weight',
    'Expected Return',
    'Annual Volatility',
    'Sharpe Ratio'
])
print(comparison_table)

## Stress Testing Analysis
one_day_minimum_return = min(benchmark_returns)
five_day_rolling_return = (benchmark_returns.rolling(window=5).apply(lambda x:np.prod(1 + x) - 1))
five_day_minimum_return =  five_day_rolling_return.min()
twenty_day = (benchmark_returns.rolling(window=20).apply(lambda z:np.prod(1 + z) - 1))
twenty_day_minimum_return = twenty_day.min()
benchmark_cummlative_returns = (1 + aligned_benchmark_returns).cumprod()
benchmark_cummlative_max = benchmark_cummlative_returns.cummax()
maxdrawdown = md(benchmark_cummlative_returns,benchmark_cummlative_max)
benchmark_maximum_drawdown = maxdrawdown.min()

print(f'1 day worst return: {one_day_minimum_return}')
print(f'5 day worst return: {five_day_minimum_return}')
print(f'20 day worst return: {twenty_day_minimum_return}')
print(f'Benchmark Maximum Drardown: {benchmark_maximum_drawdown:.4f}')

stress_scenarios = pd.DataFrame({
    '1-Day':[one_day_minimum_return],
    '5-Day':[five_day_minimum_return],
    '20-Day': [twenty_day_minimum_return],
    'Max DD': [benchmark_maximum_drawdown]    
}, index= ['NIFTY'])

stress_result = pd.DataFrame(index=beta.keys(), columns = stress_scenarios.columns)

for scenarios in stress_scenarios.columns:
    nifty_shock = stress_scenarios.loc['NIFTY',scenarios]

    for ticker in beta:
        stress_result.loc[ticker,scenarios] = beta[ticker] * nifty_shock

portfolio_shock = {}
for scenarios in stress_scenarios.columns:
    nifty_shock = stress_scenarios.loc['NIFTY',scenarios]
    portfolio_shock[scenarios] = portfolio_beta * nifty_shock 
    
stress_result.loc['Portfolio'] = portfolio_shock
print(stress_result)

currency_df = yf.download('INR=X', start=st, auto_adjust=False)['Close']

currency_log_returns = np.log(currency_df / currency_df.shift(1)).dropna()

new_common_dates = portfolio_log_returns.index.intersection(currency_log_returns.index)

portfolio_new_log_returns = portfolio_log_returns.loc[new_common_dates]
benchmark_new_log_returns = benchmark_log_returns.loc[new_common_dates]
stock_new_log_returns = stock_log_returns.loc[new_common_dates]

currency_log_returns = currency_log_returns.loc[new_common_dates].squeeze()

scenarios_analysis = pd.DataFrame(
    columns=[
    'Market β',
    'Rate β',
    'Rate p-value',
    'Rate R²',
    'FX β',
    'FX p-value',
    'FX R²'
    ],
    index =[
    'TCS',
    'Reliance',
    'Infosys',
    'HDFC',
    'ICICI'
    ]
)

def fx_regression(stock_returns, benchmark_returns, currency_returns):
    regression_data = pd.concat(
        [
            stock_returns,
            benchmark_returns,
            currency_returns
        ],
        axis=1
    ).dropna()

    regression_data.columns = [
        'Stock',
        'NIFTY',
        'USDINR'
    ]

    y = regression_data['Stock']
    X = regression_data[['NIFTY','USDINR']]
    model = sm.OLS(y,X).fit()

    return {
        'Market β': model.params['NIFTY'],
        'FX β': model.params['USDINR'],
        'FX p-value': model.pvalues['USDINR'],
        'FX R²': model.rsquared
    }

fx_results = {}

for ticker in tickers:
    fx_results[ticker] = fx_regression(
        stock_new_log_returns[ticker],
        benchmark_new_log_returns,
        currency_log_returns
    )

fx_results_df = pd.DataFrame(fx_results).T
print(fx_results_df)

curr_one_day_returns = min(currency_log_returns)
curr_one_day_max_returns = max(currency_log_returns)
curr_five_day_rolling = (currency_log_returns.rolling(window=5).apply(lambda x: np.prod(1 + x) - 1))
curr_five_day_min_returns = curr_five_day_rolling.min()
curr_five_day_max_returns = curr_five_day_rolling.max()
curr_twenty_day_rolling = (currency_log_returns.rolling(window=20).apply(lambda x: np.prod(1 + x) - 1))
curr_twenty_day_min_returns = curr_twenty_day_rolling.min()
curr_twenty_day_max_returns = curr_twenty_day_rolling.max()
curr_smp_returns = currency_df.pct_change().dropna()
curr_cummalitve_return = (1 + curr_smp_returns).cumprod()
curr_cummalitve_max = curr_cummalitve_return.cummax()
curr_drawdown = md(curr_cummalitve_return,curr_cummalitve_max)
curr_maximum_drawdown = curr_drawdown.min()

fx_shocks = {
    '1D INR Appre': curr_one_day_returns,
    '1D INR Depre': curr_one_day_max_returns,
    '5D INR Appre': curr_five_day_min_returns,
    '5D INR Depre': curr_five_day_max_returns,
    '20D INR Appre': curr_twenty_day_min_returns,
    '20D INR Depre': curr_twenty_day_max_returns
}

fx_stress_table = pd.DataFrame(
  columns = [
    '1D INR Appre',
    '1D INR Depre',
    '5D INR Appre',
    '5D INR Depre',
    '20D INR Appre',
    '20D INR Depre'
  ], 
  index = [
    'TCS.NS',
    'RELIANCE.NS',
    'INFY.NS',
    'HDFCBANK.NS',
    'ICICIBANK.NS'
  ]
)

for results in fx_results_df.index:
    curr_shock = fx_results_df.loc[results, 'FX β']

    for scenarios,shocks in fx_shocks.items():
        fx_stress_table.loc[results,scenarios] = curr_shock * shocks
    

portfolio_fx_shock = {}
for scenarios,shocks in fx_stress_table.items():
    portfolio_fx_shock[scenarios] = (weights * fx_stress_table[scenarios]).sum()

fx_stress_table.loc['Portfolio'] = portfolio_fx_shock
print(fx_stress_table)

## Risk Report
print('=' * 55)
print("Portfolio Risk Report")
print("=" * 55)
print(f"Annual Return: {portfolio_return:.4f}")
print(f"Portfolio CAGR: {portfolio_cagr:.4f}")
print(f"Annual Volatility: {portfolio_deviation:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Sortino Ratio: {sortino_ratio:.4f}")
print(f"Maximum Drawdown: {max_drawdown:.4f}")
print('--' * 35)
print("VaR & ES Report")
print('--' * 35)
print(f"Historical VaR: {historical_VaR:.4f}\nHistorical Expected Shortfall: {expected_shortfall:.4f}")
print(f"Parametric VaR: {parametric_var:.4f}\nParametirc Expected Shortfall: {parametric_expected_shortfal:.4f}")
print(f"Monte Carlo VaR: {monte_carlo_var:.4f}\nMonte Carlo Expected Shortfall: {monte_carlo_expectedshortfall:.4f}")
print('--' * 35)
print("CAPM Table")
print('--' * 35)
print(capm_returns * 100)

## Power BI / SQL Reporting Output
from sqlalchemy import create_engine
engine = create_engine(
    'postgresql+psycopg2://postgres:username@localhost:password/database_name'
)
print("\nConnected to PostgreSQL for Power BI reporting")

## Report date
report_date = df_final.index.max()

## Daily Portfolio Performance
portfolio_performance = pd.DataFrame({
    'date':aligned_twr.index,
    'portfolio_return': aligned_twr.to_numpy(),
    'benchmark_return': aligned_benchmark_returns.to_numpy(),
    'portfolio_cumulative_return': aligned_simple_twr.to_numpy(),
    'benchmark_cumulative_returns': benchmark_simple_return.to_numpy()
})

portfolio_performance.to_sql(
    'portfolio_performance',
    engine,
    if_exists= 'replace',
    index=False
)
print("portfolio_performance updated.")

## PORTFOLIO SUMMARY
portfolio_summary = pd.DataFrame({
    'report_date': [report_date],
    'portfolio_cagr': [portfolio_cagr],
    'benchmark_cagr': [benchmark_cagr],
    'portfolio_volatility':[portfolio_deviation],
    'benchmark_volatility': [benchmark_volatility],
    'portfolio_beta': [portfolio_beta],
    'sharpe_ratio': [sharpe_ratio],
    'sortino_ratio': [sortino_ratio],
    'portfolio_skewness': [skewness],
    'portfolio_kurtosis': [kurt],
    'maximum_drawdown': [max_drawdown]
})

portfolio_summary.to_sql(
    'portfolio_summary',
    engine,
    if_exists = 'replace',
    index=False
)
print('portfolio_summary uploaded')

## VaR and ES 
var_es_results = pd.DataFrame({
    'report_date': [report_date],
    'historical_var': [historical_VaR],
    'historical_es': [expected_shortfall],
    'parametric_var': [parametric_var],
    'parametric_es': [parametric_expected_shortfal],
    'monte_carlo_var': [monte_carlo_var],
    'monte_carlo_es': [monte_carlo_expectedshortfall]
})

var_es_results.to_sql(
    'var_es_results',
    engine,
    if_exists = 'replace',
    index=False
)
print("var_es_results table updated.")

## Risk Contribution
risk_contribution = pd.DataFrame({
    'ticker': tickers,
    'weight': weights.to_numpy(),
    'marginal_risk_contribution': mrc.to_numpy(),
    'risk_contribution': rc.to_numpy(),
    'risk_contributions_percent': (rc_percent / 100).to_numpy()
})

risk_contribution['report_date'] = report_date
risk_contribution = risk_contribution[
    [
        'report_date',
        'ticker',
        'weight',
        'marginal_risk_contribution',
        'risk_contribution',
        'risk_contributions_percent'
    ]
]

risk_contribution.to_sql(
    'risk_contribution',
    engine,
    if_exists = 'replace',
    index=False
)
print('risk_contribution table updated')

## CAPM Results
capm_results = pd.DataFrame({
    'ticker': tickers,
    'beta': [
        beta['TCS.NS'],
        beta['RELIANCE.NS'],
        beta['INFY.NS'],
        beta['HDFCBANK.NS'],
        beta['ICICIBANK.NS']
    ],
    'capm_expected_return': capm_returns.to_numpy()
})

capm_results['report_date'] = report_date

capm_results = capm_results[
    [
        'report_date',
        'ticker',
        'beta',
        'capm_expected_return'
    ]
]

capm_results.to_sql(
    'capm_results',
    engine,
    if_exists='replace',
    index=False
)

print("capm_results table updated")

## OPTIMIZATION RESULTS

optimization_results = pd.DataFrame({
    'ticker': list(tickers) + ['Portfolio'],

    'actual_weight': list(weights) + [weights.sum()],

    'minimum_volatility_weight': list(optimized_weights)
    + [optimized_weights.sum()],

    'maximum_sharpe_weight': list(new_optimized_weights)
    + [new_optimized_weights.sum()]
})


# Add portfolio-level metrics
optimization_results['actual_expected_return'] = np.nan
optimization_results['minimum_volatility_expected_return'] = np.nan
optimization_results['maximum_sharpe_expected_return'] = np.nan

optimization_results['actual_volatility'] = np.nan
optimization_results['minimum_volatility'] = np.nan
optimization_results['maximum_sharpe_volatility'] = np.nan

optimization_results['actual_sharpe'] = np.nan
optimization_results['minimum_volatility_sharpe'] = np.nan
optimization_results['maximum_sharpe_sharpe'] = np.nan


portfolio_row = optimization_results['ticker'] == 'Portfolio'


optimization_results.loc[
    portfolio_row,
    'actual_expected_return'
] = portfolio_return

optimization_results.loc[
    portfolio_row,
    'minimum_volatility_expected_return'
] = optimized_portfolio_returns

optimization_results.loc[
    portfolio_row,
    'maximum_sharpe_expected_return'
] = optimized_return


optimization_results.loc[
    portfolio_row,
    'actual_volatility'
] = portfolio_deviation

optimization_results.loc[
    portfolio_row,
    'minimum_volatility'
] = optimized_annual_volatility

optimization_results.loc[
    portfolio_row,
    'maximum_sharpe_volatility'
] = new_optimized_volatility


optimization_results.loc[
    portfolio_row,
    'actual_sharpe'
] = sharpe_ratio

optimization_results.loc[
    portfolio_row,
    'minimum_volatility_sharpe'
] = optimized_sharpe_ratio

optimization_results.loc[
    portfolio_row,
    'maximum_sharpe_sharpe'
] = maximum_sharpe


optimization_results['report_date'] = report_date

optimization_results = optimization_results[
    [
        'report_date',
        'ticker',
        'actual_weight',
        'minimum_volatility_weight',
        'maximum_sharpe_weight',
        'actual_expected_return',
        'minimum_volatility_expected_return',
        'maximum_sharpe_expected_return',
        'actual_volatility',
        'minimum_volatility',
        'maximum_sharpe_volatility',
        'actual_sharpe',
        'minimum_volatility_sharpe',
        'maximum_sharpe_sharpe'
    ]
]

optimization_results.to_sql(
    'optimization_results',
    engine,
    if_exists='replace',
    index=False
)

print("optimization_results table updated")

## Market Stress
market_stress = stress_result.copy()

market_stress = market_stress.reset_index()

market_stress = market_stress.rename(
    columns={'index': 'ticker'}
)

market_stress['report_date'] = report_date

market_stress = market_stress[
    [
        'report_date',
        'ticker',
        '1-Day',
        '5-Day',
        '20-Day',
        'Max DD'
    ]
]

numeric_columns = [
    '1-Day',
    '5-Day',
    '20-Day',
    'Max DD'
]

market_stress[numeric_columns] = market_stress[numeric_columns].map(
    lambda x: float(x) if pd.notna(x) else None
)

market_stress.to_sql(
    'market_stress',
    engine,
    if_exists='replace',
    index=False
)

print("market_stress table updated")

## FX REGRESSION RESULTS
fx_regression_results = fx_results_df.copy()

fx_regression_results = fx_regression_results.reset_index()

fx_regression_results = fx_regression_results.rename(
    columns={'index': 'ticker'}
)

fx_regression_results['report_date'] = report_date

fx_regression_results = fx_regression_results[
    [
        'report_date',
        'ticker',
        'Market β',
        'FX β',
        'FX p-value',
        'FX R²'
    ]
]

fx_regression_results.to_sql(
    'fx_regression_results',
    engine,
    if_exists='replace',
    index=False
)

print("fx_regression_results table updated")

# TABLE 9: FX STRESS

fx_stress = fx_stress_table.copy()

fx_stress = fx_stress.reset_index()

fx_stress = fx_stress.rename(
    columns={'index': 'ticker'}
)

fx_stress['report_date'] = report_date

fx_stress = fx_stress[
    [
        'report_date',
        'ticker',
        '1D INR Appre',
        '1D INR Depre',
        '5D INR Appre',
        '5D INR Depre',
        '20D INR Appre',
        '20D INR Depre'
    ]
]

numeric_columns = [
    '1D INR Appre',
    '1D INR Depre',
    '5D INR Appre',
    '5D INR Depre',
    '20D INR Appre',
    '20D INR Depre'
]

fx_stress[numeric_columns] = fx_stress[numeric_columns].map(
    lambda x: float(x) if pd.notna(x) else None
)

fx_stress.to_sql(
    'fx_stress',
    engine,
    if_exists='replace',
    index=False
)

print("fx_stress table updated")

## CORRELATION MATRIX

correlation_matrix_sql = correlation_matrix.copy()

correlation_matrix_sql = correlation_matrix_sql.reset_index()

correlation_matrix_sql = correlation_matrix_sql.rename(
    columns={'index': 'ticker'}
)

correlation_matrix_sql['report_date'] = report_date

correlation_matrix_sql = correlation_matrix_sql[
    [
        'report_date',
        'ticker',
        'TCS.NS',
        'RELIANCE.NS',
        'INFY.NS',
        'HDFCBANK.NS',
        'ICICIBANK.NS'
    ]
]

correlation_matrix_sql.to_sql(
    'correlation_matrix',
    engine,
    if_exists='replace',
    index=False
)

print("correlation_matrix table updated")
