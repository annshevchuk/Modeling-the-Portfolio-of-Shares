import os
import pickle
import requests
import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import (
    MOEX_LIMIT, MOEX_REQUEST_TIMEOUT, MOEX_MAX_RETRIES, MOEX_RETRY_DELAY,
    TRADING_DAYS
)

def ensure_dirs(dirs):
    """Создаёт список папок, если их нет."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def get_all_tickers():
    url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/tqbr/securities.json"
    tickers = []
    start = 0
    print("Загружаем список акций с MOEX...")
    while True:
        for attempt in range(MOEX_MAX_RETRIES):
            try:
                resp = requests.get(url, params={'start': start, 'limit': MOEX_LIMIT}, timeout=MOEX_REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    break
                else:
                    print(f"  Попытка {attempt+1}: статус {resp.status_code}, повтор через {MOEX_RETRY_DELAY} сек")
                    time.sleep(MOEX_RETRY_DELAY)
            except Exception as e:
                print(f"  Попытка {attempt+1}: ошибка {e}, повтор через {MOEX_RETRY_DELAY} сек")
                time.sleep(MOEX_RETRY_DELAY)
        else:
            print(f"Не удалось загрузить страницу start={start}, прерываем")
            break
        data = resp.json()
        securities = data.get('securities', {}).get('data', [])
        if not securities:
            break
        cols = data['securities']['columns']
        secid_idx = cols.index('SECID')
        for row in securities:
            tickers.append(row[secid_idx])
        if len(securities) < MOEX_LIMIT:
            break
        start += MOEX_LIMIT
        time.sleep(0.5)
    tickers = list(set(tickers))
    print(f"✅ Всего уникальных тикеров: {len(tickers)}")
    return tickers


def get_historical_prices(ticker, start_date, end_date):
    url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{ticker}.json"
    all_data = []
    start = 0
    while True:
        for attempt in range(MOEX_MAX_RETRIES):
            try:
                params = {
                    'from': start_date.strftime('%Y-%m-%d'),
                    'till': end_date.strftime('%Y-%m-%d'),
                    'start': start,
                    'limit': MOEX_LIMIT
                }
                resp = requests.get(url, params=params, timeout=MOEX_REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    break
                else:
                    print(f"      Попытка {attempt+1}: статус {resp.status_code}, повтор")
                    time.sleep(MOEX_RETRY_DELAY)
            except Exception as e:
                print(f"      Попытка {attempt+1}: ошибка {e}, повтор")
                time.sleep(MOEX_RETRY_DELAY)
        else:
            return None
        data = resp.json()
        history = data.get('history', {}).get('data', [])
        if not history:
            break
        cols = data['history']['columns']
        df_part = pd.DataFrame(history, columns=cols)
        all_data.append(df_part)
        if len(history) < MOEX_LIMIT:
            break
        start += MOEX_LIMIT
        time.sleep(0.5)
    if not all_data:
        return None
    df = pd.concat(all_data, ignore_index=True)
    if 'TRADEDATE' not in df.columns or 'CLOSE' not in df.columns:
        return None
    df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
    df = df.sort_values('TRADEDATE')
    prices = df[['TRADEDATE', 'CLOSE']].copy()
    prices.columns = ['date', 'close']
    prices.set_index('date', inplace=True)
    prices['close'] = pd.to_numeric(prices['close'], errors='coerce')
    prices = prices[prices['close'] > 0]
    prices = prices.dropna()
    return prices['close']


def load_prices_dict(pkl_file):
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)


def save_prices_dict(pkl_file, data):
    with open(pkl_file, 'wb') as f:
        pickle.dump(data, f)

def get_dividends_sum(ticker, start_date, end_date, max_retries=3):
    url = f"https://iss.moex.com/iss/securities/{ticker}/dividends.json"
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                break
            time.sleep(2)
        except:
            time.sleep(2)
    else:
        return 0.0
    data = resp.json()
    if 'dividends' not in data:
        return 0.0
    cols = data['dividends'].get('columns', [])
    rows = data['dividends'].get('data', [])
    if not rows:
        return 0.0
    df_div = pd.DataFrame(rows, columns=cols)
    if 'lastdate' in df_div.columns and 'value' in df_div.columns:
        df_div['lastdate'] = pd.to_datetime(df_div['lastdate'])
        mask = (df_div['lastdate'] >= start_date) & (df_div['lastdate'] <= end_date)
        total = pd.to_numeric(df_div[mask]['value'], errors='coerce').sum()
        return total if not pd.isna(total) else 0.0
    return 0.0

def annual_return_from_series(prices_series, dividends, trading_days=TRADING_DAYS):
    if prices_series is None or len(prices_series) < 5:
        return None
    start_price = prices_series.iloc[0]
    end_price = prices_series.iloc[-1]
    if start_price <= 0:
        return None
    total_return = (end_price - start_price) / start_price
    if dividends > 0:
        total_return += dividends / start_price
    days = len(prices_series)
    years = days / trading_days
    if years > 0:
        annual = (1 + total_return) ** (1 / years) - 1
    else:
        annual = total_return
    return annual * 100


def compute_sharpe_with_dividends(prices_series, dividends, rf, trading_days=TRADING_DAYS):
    if prices_series is None or len(prices_series) < 10:
        return None
    returns = prices_series.pct_change().dropna()
    if len(returns) < 5:
        return None
    avg_price = prices_series.mean()
    if dividends > 0 and avg_price > 0:
        dividend_yield = dividends / avg_price
        daily_div = dividend_yield / len(returns)
        returns = returns + daily_div
    mu_daily = returns.mean()
    sigma_daily = returns.std()
    if sigma_daily == 0:
        return None
    mu_annual = (1 + mu_daily) ** trading_days - 1
    sigma_annual = sigma_daily * np.sqrt(trading_days)
    return (mu_annual - rf) / sigma_annual


def sortino_with_dividends(prices_series, dividends, rf, trading_days=TRADING_DAYS):
    if prices_series is None or len(prices_series) < 10:
        return None
    returns = prices_series.pct_change().dropna()
    if len(returns) < 5:
        return None
    avg_price = prices_series.mean()
    if dividends > 0 and avg_price > 0:
        dividend_yield = dividends / avg_price
        daily_div = dividend_yield / len(returns)
        returns = returns + daily_div
    daily_rf = (1 + rf) ** (1 / trading_days) - 1
    excess = returns - daily_rf
    mean_excess = excess.mean()
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.inf
    downside_std = np.sqrt((downside ** 2).mean())
    if downside_std == 0:
        return np.inf
    return mean_excess / downside_std * np.sqrt(trading_days)


def equal_weight_metrics(tickers, mean_ret, cov, rf):
    n = len(tickers)
    w = np.ones(n) / n
    ret_daily = np.dot(w, [mean_ret[t] for t in tickers])
    var_daily = np.dot(w, np.dot(cov.loc[tickers, tickers], w))
    risk_daily = np.sqrt(var_daily)
    annual_ret = (1 + ret_daily) ** TRADING_DAYS - 1
    annual_risk = risk_daily * np.sqrt(TRADING_DAYS)
    sharpe = (annual_ret - rf) / annual_risk if annual_risk > 0 else np.nan
    return annual_ret, annual_risk, sharpe


def portfolio_sharpe(weights, returns_df, rf, trading_days=TRADING_DAYS):
    port_returns = returns_df @ weights
    daily_rf = (1 + rf) ** (1/trading_days) - 1
    excess = port_returns - daily_rf
    mu = excess.mean()
    sigma = excess.std()
    if sigma == 0:
        return np.inf
    sharpe_daily = mu / sigma
    return sharpe_daily * np.sqrt(trading_days)


def portfolio_sortino(weights, returns_df, rf, trading_days=TRADING_DAYS):
    port_returns = returns_df @ weights
    daily_rf = (1 + rf) ** (1/trading_days) - 1
    excess = port_returns - daily_rf
    mean_excess = excess.mean()
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.inf
    downside_std = np.sqrt((downside**2).mean())
    if downside_std == 0:
        return np.inf
    sortino_daily = mean_excess / downside_std
    return sortino_daily * np.sqrt(trading_days)


def optimize_portfolio(tickers, mean_ret, cov, returns_df, rf,
                       sharpe_min=0.75, sortino_min=0.75,
                       sharpe_decline_allowed=0):
    n = len(tickers)
    w_eq = np.ones(n) / n
    ret_eq_annual, risk_eq_annual, sharpe_eq = equal_weight_metrics(tickers, mean_ret, cov, rf)

    Sigma = cov.loc[tickers, tickers].values
    returns_sub = returns_df[tickers]

    def variance(w):
        return w @ Sigma @ w

    def variance_jac(w):
        return 2 * Sigma @ w

    if n == 2:
        t1, t2 = tickers
        var1 = Sigma[0, 0]
        var2 = Sigma[1, 1]
        cov12 = Sigma[0, 1]

        denom = var1 + var2 - 2 * cov12
        if denom != 0:
            w1 = (var2 - cov12) / denom
        else:
            w1 = 0.5
        w1 = np.clip(w1, 0.0, 1.0)
        w2 = 1.0 - w1
        w_opt = np.array([w1, w2])

        sharpe_opt = portfolio_sharpe(w_opt, returns_sub, rf, TRADING_DAYS)
        sortino_opt = portfolio_sortino(w_opt, returns_sub, rf, TRADING_DAYS)

        if sharpe_opt >= sharpe_min and sortino_opt >= sortino_min:
            risk_daily = np.sqrt(variance(w_opt))
            ret_daily = np.dot(w_opt, [mean_ret[t] for t in tickers])
            annual_ret_opt = (1 + ret_daily) ** TRADING_DAYS - 1
            annual_risk_opt = risk_daily * np.sqrt(TRADING_DAYS)
            improved = (annual_risk_opt < risk_eq_annual) and (sharpe_opt >= sharpe_eq * (1 - sharpe_decline_allowed))
        else:
            w_opt = w_eq
            annual_ret_opt = ret_eq_annual
            annual_risk_opt = risk_eq_annual
            sharpe_opt = sharpe_eq
            improved = False

        return {
            'old_return': ret_eq_annual,
            'old_risk': risk_eq_annual,
            'old_sharpe': sharpe_eq,
            'new_return': annual_ret_opt,
            'new_risk': annual_risk_opt,
            'new_sharpe': sharpe_opt,
            'weights': w_opt,
            'improved': improved
        }

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: portfolio_sharpe(w, returns_sub, rf, TRADING_DAYS) - sharpe_min},
        {'type': 'ineq', 'fun': lambda w: portfolio_sortino(w, returns_sub, rf, TRADING_DAYS) - sortino_min}
    ]
    bounds = [(0, 1) for _ in range(n)]
    init = w_eq.copy()

    res = minimize(variance, init, method='SLSQP', jac=variance_jac,
                   bounds=bounds, constraints=constraints,
                   options={'ftol': 1e-9, 'disp': False})

    if not res.success:
        w_opt = w_eq
        risk_daily = np.sqrt(variance(w_eq))
        ret_daily = np.dot(w_eq, [mean_ret[t] for t in tickers])
        improved = False
    else:
        w_opt = res.x
        risk_daily = np.sqrt(res.fun)
        ret_daily = np.dot(w_opt, [mean_ret[t] for t in tickers])
        annual_risk_opt = risk_daily * np.sqrt(TRADING_DAYS)
        sharpe_opt = portfolio_sharpe(w_opt, returns_sub, rf, TRADING_DAYS)
        improved = (annual_risk_opt < risk_eq_annual) and (sharpe_opt >= sharpe_eq * (1 - sharpe_decline_allowed))

    annual_ret_opt = (1 + ret_daily) ** TRADING_DAYS - 1
    annual_risk_opt = risk_daily * np.sqrt(TRADING_DAYS)
    sharpe_opt = portfolio_sharpe(w_opt, returns_sub, rf, TRADING_DAYS)

    return {
        'old_return': ret_eq_annual,
        'old_risk': risk_eq_annual,
        'old_sharpe': sharpe_eq,
        'new_return': annual_ret_opt,
        'new_risk': annual_risk_opt,
        'new_sharpe': sharpe_opt,
        'weights': w_opt,
        'improved': improved
    }

def normalize_pair(t1, t2):
    return tuple(sorted([t1, t2]))

def normalize_triple(t1, t2, t3):
    return tuple(sorted([t1, t2, t3]))

def normalize_quad(t1, t2, t3, t4):
    return tuple(sorted([t1, t2, t3, t4]))

def cluster_by_cv(tickers, price_file, max_k=10, random_state=42):
    prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
    prices = prices[tickers]
    returns = prices.pct_change().dropna()
    mean_ret = returns.mean()
    std_ret = returns.std()
    cv = (std_ret / mean_ret.abs()).abs()
    X = cv.values.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    best_k = 2
    best_score = -1
    best_labels = None
    k_range = range(2, min(max_k, X_scaled.shape[0] - 1) + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        sizes = pd.Series(labels).value_counts()
        if (sizes >= 2).all():
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
    return pd.DataFrame({'ticker': tickers, 'cluster': best_labels}), best_k, cv