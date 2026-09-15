import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from typing import NamedTuple, List, Tuple

DATA_PATH = "training_dataset_extended.csv"
METADATA_COLS = ['event_id', 'open_time', 'event_time', 'currency', 'pair']
GROUP_COL = 'event_id'
LABEL_COL = 'label'
TIMESTEPS = 15

class Splits(NamedTuple):
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    feature_names: List[str]

def make_run_ids(group_ids: np.ndarray) -> np.ndarray:
    g = np.asarray(group_ids)
    if len(g) == 0:
        return np.empty(0, dtype=np.int64)
    is_new = np.empty(len(g), dtype=bool)
    is_new[0] = True
    is_new[1:] = g[1:] != g[:-1]
    return np.cumsum(is_new) - 1

def contiguous_runs(run_ids: np.ndarray) -> List[Tuple[int, int]]:
    r = np.asarray(run_ids)
    if len(r) == 0:
        return []
    change = np.flatnonzero(r[1:] != r[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(r)]))
    return list(zip(bounds[:-1].tolist(), bounds[1:].tolist()))

def groupwise_ffill(features: pd.DataFrame,
                    run_ids: np.ndarray,
                    fill_value: float = 0.0) -> pd.DataFrame:
    filled = features.groupby(run_ids, sort=False).ffill()
    return filled.fillna(fill_value)

def windows_for_segment(X_seg: np.ndarray, y_seg: np.ndarray,
                        timesteps: int = TIMESTEPS, dtype=np.float32):
    n = len(X_seg)
    if n <= timesteps:
        return (np.empty((0, timesteps, X_seg.shape[1]), dtype=dtype),
                np.empty((0,), dtype=y_seg.dtype))
    w = sliding_window_view(X_seg, window_shape=timesteps, axis=0)
    w = np.transpose(w, (0, 2, 1))[:n - timesteps].astype(dtype)
    return w, y_seg[timesteps:n]

def create_windows_by_group(X: np.ndarray,
                            y: np.ndarray,
                            run_ids: np.ndarray,
                            timesteps: int = TIMESTEPS,
                            dtype=np.float32):
    runs = contiguous_runs(run_ids)
    usable = [(s, e) for s, e in runs if (e - s) > timesteps]

    n_out = sum((e - s) - timesteps for s, e in usable)
    n_feat = X.shape[1]

    X_out = np.empty((n_out, timesteps, n_feat), dtype=dtype)
    y_out = np.empty(n_out, dtype=y.dtype)
    r_out = np.empty(n_out, dtype=np.int64)

    pos = 0
    for s, e in usable:
        k = (e - s) - timesteps
        w = sliding_window_view(X[s:e], window_shape=timesteps, axis=0)
        w = np.transpose(w, (0, 2, 1))[:k]
        X_out[pos:pos + k] = w
        y_out[pos:pos + k] = y[s + timesteps:e]
        r_out[pos:pos + k] = run_ids[s]
        pos += k

    assert pos == n_out
    return X_out, y_out, r_out

def prepare_data(path: str = DATA_PATH,
                 timesteps: int = TIMESTEPS,
                 train_frac: float = 0.70,
                 val_frac: float = 0.85,
                 split_by: str = 'row') -> Splits:
    if split_by not in ('row', 'event'):
        raise ValueError("split_by phải là 'row' hoặc 'event'")

    df = pd.read_csv(path)
    run_ids = make_run_ids(df[GROUP_COL].to_numpy())

    features_df = df.drop(columns=METADATA_COLS, errors='ignore')
    feature_names = [c for c in features_df.columns if c != LABEL_COL]

    X_df = groupwise_ffill(features_df[feature_names], run_ids)
    X_raw = X_df.to_numpy(dtype=np.float64)
    y_raw = features_df[LABEL_COL].to_numpy()

    n = len(X_raw)
    train_end = int(n * train_frac)
    val_end = int(n * val_frac)
    if split_by == 'event':
        train_end = _snap_to_run_boundary(run_ids, train_end)
        val_end = _snap_to_run_boundary(run_ids, val_end)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_raw[:train_end])
    X_val_scaled = scaler.transform(X_raw[train_end:val_end])
    X_test_scaled = scaler.transform(X_raw[val_end:])

    X_train, y_train, _ = create_windows_by_group(
        X_train_scaled, y_raw[:train_end], run_ids[:train_end], timesteps)
    X_val, y_val, _ = create_windows_by_group(
        X_val_scaled, y_raw[train_end:val_end], run_ids[train_end:val_end], timesteps)
    X_test, y_test, _ = create_windows_by_group(
        X_test_scaled, y_raw[val_end:], run_ids[val_end:], timesteps)

    return Splits(X_train, y_train, X_val, y_val, X_test, y_test,
                  scaler, feature_names)

def iter_event_folds(path: str = DATA_PATH,
                     timesteps: int = TIMESTEPS,
                     n_folds: int = 5,
                     seed: int = 42):
    from sklearn.model_selection import KFold

    df = pd.read_csv(path)
    run_ids = make_run_ids(df[GROUP_COL].to_numpy())
    runs = contiguous_runs(run_ids)

    features_df = df.drop(columns=METADATA_COLS, errors='ignore')
    feature_names = [c for c in features_df.columns if c != LABEL_COL]
    X_df = groupwise_ffill(features_df[feature_names], run_ids)
    X_raw = X_df.to_numpy(dtype=np.float64)
    y_raw = features_df[LABEL_COL].to_numpy()

    usable = [(s, e) for s, e in runs if (e - s) > timesteps]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    positions = np.arange(len(usable))

    for fold, (tr_pos, va_pos) in enumerate(kf.split(positions), start=1):
        tr_events = [usable[p] for p in tr_pos]
        va_events = [usable[p] for p in va_pos]

        train_row_mask = np.zeros(len(X_raw), dtype=bool)
        for s, e in tr_events:
            train_row_mask[s:e] = True
        scaler = StandardScaler().fit(X_raw[train_row_mask])
        X_scaled = scaler.transform(X_raw)

        def _build(events):
            Xs, ys = [], []
            for s, e in events:
                Xw, yw = windows_for_segment(X_scaled[s:e], y_raw[s:e], timesteps)
                if len(Xw):
                    Xs.append(Xw)
                    ys.append(yw)
            return np.concatenate(Xs), np.concatenate(ys)

        X_tr, y_tr = _build(tr_events)
        X_va, y_va = _build(va_events)

        yield dict(fold=fold, tr_events=tr_events, va_events=va_events,
                  X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va,
                  feature_names=feature_names)

def _snap_to_run_boundary(run_ids: np.ndarray, idx: int) -> int:
    starts = np.array([s for s, _ in contiguous_runs(run_ids)])
    return int(starts[np.argmin(np.abs(starts - idx))])

def _create_windows_old(X, y, timesteps):
    w = sliding_window_view(X, window_shape=timesteps, axis=0)
    w = np.transpose(w, (0, 2, 1))[:-1]
    return w, y[timesteps:]
