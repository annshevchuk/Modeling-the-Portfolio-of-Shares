PERIODS = {
    '2023_2024': {
        'name': '2023_2024',
        'start': '2023-12-21',
        'end': '2024-06-06',
        'rf': 0.16,
        'pkl_file': 'prices_dict_2023_2024.pkl'
    },
    '2024_2025': {
        'name': '2024_2025',
        'start': '2024-12-21',
        'end': '2025-06-06',
        'rf': 0.21,
        'pkl_file': 'prices_dict_2024_2025.pkl'
    }
}

METHODS = ['inner', 'ffill']

CORR_LOWER = -0.1
CORR_UPPER = 0.1
SHARPE_THRESHOLD = 0.75
SORTINO_THRESHOLD = 0.75

TRADING_DAYS = 252

MOEX_REQUEST_TIMEOUT = 60
MOEX_MAX_RETRIES = 5
MOEX_RETRY_DELAY = 5
MOEX_PAUSE_BETWEEN_TICKERS = 1.0
MOEX_LIMIT = 100