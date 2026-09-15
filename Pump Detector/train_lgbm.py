import os
import sys
import json

import numpy as np
import joblib
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, precision_score,
                             recall_score, f1_score)

from data_utils import prepare_data, TIMESTEPS
from lgbm_eval import flatten, LGB_PARAMS

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

MIN_RECALL = 0.0

OUT_PREFIX = os.environ.get('OUT_PREFIX', '')
MODEL_PATH = f'{OUT_PREFIX}best_pump_lgbm_model.joblib'
SCALER_PATH = f'{OUT_PREFIX}scaler_lgbm.joblib'
CONFIG_PATH = f'{OUT_PREFIX}model_config_lgbm.json'

def main():
    print("Đang chuẩn bị dữ liệu (ffill + cắt cửa sổ theo từng sự kiện)...")
    data = prepare_data(timesteps=TIMESTEPS)

    X_train, y_train = flatten(data.X_train), data.y_train
    X_val, y_val = flatten(data.X_val), data.y_val
    X_test, y_test = flatten(data.X_test), data.y_test

    print(f"Cấu trúc đầu vào (đã làm phẳng): {X_train.shape}")
    print(f"Nhãn 1 trong tập train: {int(y_train.sum()):,} ({y_train.mean() * 100:.2f}%)")
    print(f"Nhãn 1 trong tập val  : {int(y_val.sum()):,} ({y_val.mean() * 100:.2f}%)")
    print(f"Nhãn 1 trong tập test : {int(y_test.sum()):,} ({y_test.mean() * 100:.2f}%)\n")

    import lightgbm as lgb
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)],
    )
    print(f"\nDừng ở iteration tốt nhất: {model.best_iteration_}")

    y_val_prob = model.predict_proba(X_val)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob)
    precisions, recalls = precisions[:-1], recalls[:-1]
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)

    valid = recalls >= MIN_RECALL
    best_idx = int(np.argmax(np.where(valid, f1_scores, -1))) if valid.any() \
        else int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])

    print("\n" + "=" * 55)
    print("DÒ NGƯỠNG TRÊN TẬP VAL")
    print("=" * 55)
    print(f"Ngưỡng tốt nhất : {best_threshold:.4f}")
    print(f"  precision (val) = {precisions[best_idx]:.3f}")
    print(f"  recall    (val) = {recalls[best_idx]:.3f}")
    print(f"  f1        (val) = {f1_scores[best_idx]:.3f}")

    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_test_prob >= best_threshold).astype(int)

    print("\n" + "=" * 55)
    print("BÁO CÁO HIỆU SUẤT PHÁT HIỆN PUMP & DUMP (TẬP TEST)")
    print("=" * 55)
    print(f"Phân phối xác suất: min={y_test_prob.min():.4f} "
          f"max={y_test_prob.max():.4f} mean={y_test_prob.mean():.4f}")
    print(f"Số mẫu vượt ngưỡng {best_threshold:.4f}: {int(y_pred.sum()):,}"
          f" / {len(y_pred):,}\n")

    print(classification_report(y_test, y_pred,
                                target_names=['Normal (0)', 'Pump/Dump (1)'],
                                digits=3))
    print("MA TRẬN NHẦM LẪN:")
    print(confusion_matrix(y_test, y_pred))

    print("\nĐộ nhạy theo ngưỡng (tập test):")
    print(f"{'ngưỡng':>8} | {'precision':>9} | {'recall':>7} | {'f1':>6}")
    for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, best_threshold]:
        pred_t = (y_test_prob >= t).astype(int)
        mark = '  <-- đã chọn' if abs(t - best_threshold) < 1e-9 else ''
        print(f"{t:8.4f} | {precision_score(y_test, pred_t, zero_division=0):9.3f}"
              f" | {recall_score(y_test, pred_t, zero_division=0):7.3f}"
              f" | {f1_score(y_test, pred_t, zero_division=0):6.3f}{mark}")

    n_feat = len(data.feature_names)
    flat_names = [f"{name}_t-{TIMESTEPS - 1 - i}"
                  for i in range(TIMESTEPS) for name in data.feature_names]
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]

    print("\n" + "=" * 55)
    print("TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT")
    print("=" * 55)
    for i in top_idx:
        print(f"  {flat_names[i]:<28} {importances[i]:>8}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(data.scaler, SCALER_PATH)

    config_data = {
        "model_name": "Pump_LGBM_v1",
        "model_type": "lightgbm",
        "golden_threshold": best_threshold,
        "target_metric": "F1-Score",
        "timesteps": TIMESTEPS,
        "n_features": n_feat,
        "feature_order": data.feature_names,
        "scaler_path": SCALER_PATH,
        "best_iteration": int(model.best_iteration_),
        "val_precision": float(precisions[best_idx]),
        "val_recall": float(recalls[best_idx]),
        "val_f1": float(f1_scores[best_idx]),
    }
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)

    print(f"\nĐã lưu: {MODEL_PATH}, {SCALER_PATH}, {CONFIG_PATH}")

if __name__ == '__main__':
    main()
