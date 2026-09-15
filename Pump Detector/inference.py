import json

import numpy as np
import pandas as pd
import joblib

from feature_engineering import compute_features
from data_utils import make_run_ids, groupwise_ffill, windows_for_segment, contiguous_runs

DEFAULT_CONFIG_PATH = 'model_config_lgbm.json'

class PumpDetector:
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.timesteps = self.config['timesteps']
        self.feature_order = self.config['feature_order']
        self.threshold = self.config['golden_threshold']
        self.model_type = self.config.get('model_type', 'lightgbm')

        self.scaler = joblib.load(self.config['scaler_path'])

        default_model_path = ('best_pump_lgbm_model.joblib' if self.model_type == 'lightgbm'
                              else 'best_pump_lstm_model.keras')
        model_path = self.config.get('model_path', default_model_path)

        if self.model_type == 'lightgbm':
            self.model = joblib.load(model_path)
        elif self.model_type == 'lstm':
            import keras
            self.model = keras.models.load_model(model_path)
        else:
            raise ValueError(f"model_type không hỗ trợ: {self.model_type}")

    def _prepare_windows(self, raw_df: pd.DataFrame, pair_col: str = None):
        single_pair = pair_col is None
        if single_pair:
            raw_df = raw_df.copy()
            raw_df['__pair__'] = '_'
            pair_col = '__pair__'

        parts = []
        for pair, g in raw_df.groupby(pair_col, sort=False):
            feat = compute_features(g.sort_index())
            feat[pair_col] = pair
            parts.append(feat)
        feat_df = pd.concat(parts)

        run_ids = make_run_ids(feat_df[pair_col].to_numpy())
        X_raw = groupwise_ffill(feat_df[self.feature_order], run_ids)
        X_scaled = self.scaler.transform(X_raw.to_numpy(dtype=np.float64))

        pair_values = feat_df[pair_col].to_numpy()
        X_list, pairs, times = [], [], []
        for s, e in contiguous_runs(run_ids):
            Xw, _ = windows_for_segment(X_scaled[s:e], np.zeros(e - s), self.timesteps)
            if len(Xw) == 0:
                continue
            X_list.append(Xw)
            pairs.extend([pair_values[s]] * len(Xw))
            times.extend(feat_df.index[s + self.timesteps:e])

        if not X_list:
            return np.empty((0, self.timesteps, len(self.feature_order))), [], []
        return np.concatenate(X_list), pairs, times

    def _predict_proba(self, X_windows: np.ndarray) -> np.ndarray:
        if len(X_windows) == 0:
            return np.empty(0)
        if self.model_type == 'lightgbm':
            X_flat = X_windows.reshape(X_windows.shape[0], -1)
            return self.model.predict_proba(X_flat)[:, 1]
        return self.model.predict(X_windows, verbose=0).ravel()

    def predict(self, raw_df: pd.DataFrame, pair_col: str = None) -> pd.DataFrame:
        X, pairs, times = self._prepare_windows(raw_df, pair_col)
        proba = self._predict_proba(X)
        out = pd.DataFrame({'pair': pairs, 'timestamp': times, 'probability': proba})
        out['is_pump'] = out['probability'] >= self.threshold
        return out.sort_values(['pair', 'timestamp']).reset_index(drop=True)

    def predict_latest(self, raw_df: pd.DataFrame, pair_col: str = None) -> pd.DataFrame:
        result = self.predict(raw_df, pair_col)
        if result.empty:
            return result
        return (result.sort_values('timestamp')
                      .groupby('pair', as_index=False)
                      .tail(1)
                      .reset_index(drop=True))

if __name__ == '__main__':
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("Tự kiểm tra bằng dữ liệu OHLCV giả cho 3 coin...\n")
    rng = np.random.default_rng(0)

    def make_pair(n, base, vol):
        idx = pd.date_range('2024-06-01', periods=n, freq='1min', tz='UTC')
        price = base + np.cumsum(rng.normal(0, vol, n))
        price = np.maximum(price, base * 0.5)
        return pd.DataFrame({
            'Price': price, 'Open': price + rng.normal(0, vol * 0.1, n),
            'High': price + np.abs(rng.normal(0, vol * 0.6, n)),
            'Low': price - np.abs(rng.normal(0, vol * 0.6, n)),
            'Volume': rng.uniform(10, 100, n), 'TradeCount': rng.integers(0, 20, n),
            'Taker_Buy_Volume': rng.uniform(0, 50, n),
        }, index=idx)

    raw = pd.concat([
        make_pair(120, 100, 0.5).assign(pair='AAA'),
        make_pair(120, 5000, 20).assign(pair='BBB'),
        make_pair(40, 1, 0.02).assign(pair='CCC'),
    ])

    detector = PumpDetector()
    print(f"Model: {detector.config['model_name']} ({detector.model_type})")
    print(f"Ngưỡng: {detector.threshold:.4f}  Timesteps: {detector.timesteps}\n")

    hist = detector.predict(raw, pair_col='pair')
    print(f"predict() -> {len(hist)} dòng, các coin: {sorted(hist['pair'].unique())}")
    assert hist['probability'].between(0, 1).all(), "Xác suất phải trong [0,1]"
    assert (hist['is_pump'] == (hist['probability'] >= detector.threshold)).all()
    print("OK: xác suất hợp lệ, is_pump khớp đúng ngưỡng.\n")

    latest = detector.predict_latest(raw, pair_col='pair')
    print("predict_latest():")
    print(latest.to_string(index=False))
    assert len(latest) == raw['pair'].nunique(), "Phải có đúng 1 dòng / coin"
    print("\nOK: mỗi coin đúng 1 dự đoán mới nhất.")
