import os
import sys
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import joblib
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, precision_score,
                             recall_score, f1_score)

from data_utils import prepare_data
from model_lib import build_model, compile_model

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SEED = 42
TIMESTEPS = 15

EPOCHS = int(os.environ.get('EPOCHS', 40))
BATCH_SIZE = 64
LEARNING_RATE = 0.001

MIN_RECALL = 0.0

OUT_PREFIX = os.environ.get('OUT_PREFIX', '')
MODEL_PATH = f'{OUT_PREFIX}best_pump_lstm_model.keras'
SCALER_PATH = f'{OUT_PREFIX}scaler.joblib'
CONFIG_PATH = f'{OUT_PREFIX}model_config.json'

keras.utils.set_random_seed(SEED)

print("Đang chuẩn bị dữ liệu (ffill + cắt cửa sổ theo từng sự kiện)...")
data = prepare_data(timesteps=TIMESTEPS)

X_train, y_train = data.X_train, data.y_train
X_val, y_val = data.X_val, data.y_val
X_test, y_test = data.X_test, data.y_test

N_FEATURES = X_train.shape[2]
print(f"Cấu trúc đầu vào LSTM : {X_train.shape}")
print(f"Nhãn 1 trong tập train: {int(y_train.sum()):,} ({y_train.mean() * 100:.2f}%)")
print(f"Nhãn 1 trong tập val  : {int(y_val.sum()):,} ({y_val.mean() * 100:.2f}%)")
print(f"Nhãn 1 trong tập test : {int(y_test.sum()):,} ({y_test.mean() * 100:.2f}%)\n")

model = build_model(TIMESTEPS, N_FEATURES, seed=SEED)
compile_model(model, learning_rate=LEARNING_RATE)
model.summary()

callbacks = [
    EarlyStopping(monitor='val_f1', mode='max', patience=7,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor='val_f1', mode='max',
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_f1', mode='max', factor=0.5,
                      patience=3, min_lr=1e-5, verbose=1),
]

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1,
)

y_val_prob = model.predict(X_val, batch_size=512, verbose=0).ravel()

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

y_test_prob = model.predict(X_test, batch_size=512, verbose=0).ravel()
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
for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, best_threshold]:
    pred_t = (y_test_prob >= t).astype(int)
    mark = '  <-- đã chọn' if abs(t - best_threshold) < 1e-9 else ''
    print(f"{t:8.4f} | {precision_score(y_test, pred_t, zero_division=0):9.3f}"
          f" | {recall_score(y_test, pred_t, zero_division=0):7.3f}"
          f" | {f1_score(y_test, pred_t, zero_division=0):6.3f}{mark}")

model.save(MODEL_PATH)
joblib.dump(data.scaler, SCALER_PATH)

config_data = {
    "model_name": "Pump_LSTM_v2",
    "golden_threshold": best_threshold,
    "target_metric": "F1-Score",
    "timesteps": TIMESTEPS,
    "n_features": N_FEATURES,
    "feature_order": data.feature_names,
    "scaler_path": SCALER_PATH,
    "val_precision": float(precisions[best_idx]),
    "val_recall": float(recalls[best_idx]),
    "val_f1": float(f1_scores[best_idx]),
}
with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
    json.dump(config_data, f, indent=4, ensure_ascii=False)

print(f"\nĐã lưu: {MODEL_PATH}, {SCALER_PATH}, {CONFIG_PATH}")
