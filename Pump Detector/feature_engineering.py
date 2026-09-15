import numpy as np
import pandas as pd

RAW_COLS = ['Price', 'Open', 'High', 'Low', 'Volume',
           'TradeCount', 'Taker_Buy_Volume']

FEATURE_ORDER = [
    'price_change_5m', 'price_change_15m', 'price_change_60m',
    'high_low_range', 'close_open_ratio',
    'volume_spike_5m', 'volume_spike_15m', 'volume_spike_60m',
    'trade_count_spike_15m', 'trade_count_5m',
    'taker_buy_ratio',
    'volatility_15m', 'volatility_60m', 'atr_14m_pct',
    'ema_10_ratio', 'momentum_10m_pct', 'momentum_30m_pct',
    'rsi_14', 'macd_pct', 'macd_hist_pct',
    'hour_sin', 'hour_cos', 'day_of_week',
    'price_volume_impact_5m', 'price_change_per_trade_5m',
]

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_features_single(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột raw: {missing}")

    close = df['Price']
    open_ = df['Open']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    trade_count = df['TradeCount']
    taker_buy = df['Taker_Buy_Volume']

    out = pd.DataFrame(index=df.index)

    out['price_change_5m'] = close.pct_change(5)
    out['price_change_15m'] = close.pct_change(15)
    out['price_change_60m'] = close.pct_change(60)
    out['high_low_range'] = (high - low) / open_
    out['close_open_ratio'] = close / open_

    out['volume_spike_5m'] = volume / volume.rolling(5).mean()
    out['volume_spike_15m'] = volume / volume.rolling(15).mean()
    out['volume_spike_60m'] = volume / volume.rolling(60).mean()
    out['trade_count_spike_15m'] = trade_count / trade_count.rolling(15).mean()
    out['trade_count_5m'] = trade_count.rolling(5).sum()

    out['taker_buy_ratio'] = taker_buy / volume

    ret = close.pct_change()
    out['volatility_15m'] = ret.rolling(15).std()
    out['volatility_60m'] = ret.rolling(60).std()

    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_14m = true_range.rolling(14).mean()
    out['atr_14m_pct'] = atr_14m / close

    ema10 = close.ewm(span=10, adjust=False).mean()
    out['ema_10_ratio'] = close / ema10

    out['momentum_10m_pct'] = close.pct_change(10)
    out['momentum_30m_pct'] = close.pct_change(30)

    out['rsi_14'] = _rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out['macd_pct'] = macd / close
    out['macd_hist_pct'] = (macd - macd_signal) / close

    hour = df.index.hour
    out['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    out['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    out['day_of_week'] = df.index.dayofweek

    out['price_volume_impact_5m'] = out['price_change_5m'].abs() / out['volume_spike_5m']
    out['price_change_per_trade_5m'] = out['price_change_5m'] / out['trade_count_5m']

    return out[FEATURE_ORDER]

def compute_features(df: pd.DataFrame, pair_col: str = None) -> pd.DataFrame:
    if pair_col is None:
        return compute_features_single(df)

    parts = []
    for pair, g in df.groupby(pair_col, sort=False):
        feat = compute_features_single(g.sort_index())
        feat[pair_col] = pair
        parts.append(feat)
    return pd.concat(parts)

if __name__ == '__main__':
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range('2024-01-01', periods=n, freq='1min', tz='UTC')
    price = 100 + np.cumsum(rng.normal(0, 0.5, n))
    df_raw = pd.DataFrame({
        'Price': price,
        'Open': price + rng.normal(0, 0.05, n),
        'High': price + np.abs(rng.normal(0, 0.3, n)),
        'Low': price - np.abs(rng.normal(0, 0.3, n)),
        'Volume': rng.uniform(10, 100, n),
        'TradeCount': rng.integers(0, 20, n),
        'Taker_Buy_Volume': rng.uniform(0, 50, n),
    }, index=idx)

    feat = compute_features_single(df_raw)
    print(f"Shape: {feat.shape}")
    print(f"Đúng thứ tự {len(FEATURE_ORDER)} cột: {list(feat.columns) == FEATURE_ORDER}")

    mask = feat['trade_count_5m'] != 0
    lhs = feat.loc[mask, 'price_change_per_trade_5m']
    rhs = feat.loc[mask, 'price_change_5m'] / feat.loc[mask, 'trade_count_5m']
    assert np.allclose(lhs, rhs, equal_nan=True), "Công thức price_change_per_trade_5m sai!"
    print("OK: price_change_per_trade_5m = price_change_5m / trade_count_5m")

    assert np.allclose(feat['hour_sin']**2 + feat['hour_cos']**2, 1.0)
    print("OK: hour_sin^2 + hour_cos^2 = 1 (đúng chu kỳ lượng giác)")

    print("\nMẫu 5 dòng cuối (đã đủ lịch sử rolling):")
    print(feat.tail(5).to_string())
