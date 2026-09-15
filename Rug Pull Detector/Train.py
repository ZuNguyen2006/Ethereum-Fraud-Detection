"""Huấn luyện mô hình phát hiện ví lừa đảo.

Những thay đổi so với phiên bản trước và lý do:

  * Bỏ StandardScaler. Cây quyết định bất biến với biến đổi đơn điệu; đo thực tế
    cho AUC 0.9853 (không scale) vs 0.9855 (có scale). Scaler chỉ thêm một file
    phải giữ đồng bộ khi serving.
  * Bỏ scale_pos_weight. Nó và ngưỡng quyết định là hai nút chỉnh trùng chức
    năng; giữ xác suất không bị bóp méo rồi chỉnh ngưỡng thì dễ diễn giải hơn.
  * Optuna chấm điểm bằng StratifiedKFold thay vì một validation split duy nhất.
    Độ nhiễu split-to-split đo được là +/- 0.008 PR-AUC, lớn hơn khoảng cách
    giữa phần lớn các trial -- tune trên một split là tune theo nhiễu.
  * Tối ưu PR-AUC thay vì ROC-AUC.
  * early_stopping_rounds thật sự được bật (trước đây eval_set được truyền vào
    nhưng không có early stopping nên hoàn toàn vô tác dụng).
  * Sampler được seed -> chạy lại ra đúng kết quả cũ.
  * Ngưỡng quyết định được chọn trên out-of-fold prediction và lưu kèm model.
  * Artifact lưu ra là một bundle có kèm thứ tự feature, ngưỡng, metric, version.
"""

import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

import config as C

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Dữ liệu
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """Trả về (X, y, ngày tháng) -- ngày tháng là None nếu dataset không có."""
    if not C.CLEAN_CSV.exists():
        raise FileNotFoundError(
            f"Không thấy {C.CLEAN_CSV.name}. Chạy script chuẩn bị dữ liệu trước "
            f"(preprocessing.py cho 'eoa', prepare_rugpull1k.py cho 'rugpull1k')."
        )
    df = pd.read_csv(C.CLEAN_CSV)
    X = C.align_features(df).copy()

    # XGBoost đọc thẳng dtype category qua enable_categorical, không cần one-hot.
    # Ép trên toàn bộ X trước khi split để các fold có cùng tập nhãn category.
    for col in C.CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")

    dates = None
    if C.DATE_COLUMN and C.DATE_COLUMN in df.columns:
        dates = pd.to_datetime(df[C.DATE_COLUMN], errors="coerce")
    return X, df[C.TARGET], dates


def split_train_test(
    X: pd.DataFrame, y: pd.Series, dates: pd.Series | None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chia tập test theo SPLIT_STRATEGY."""
    if C.SPLIT_STRATEGY == "temporal":
        if dates is None:
            raise ValueError(
                f"SPLIT_STRATEGY='temporal' cần cột ngày, "
                f"dataset '{C.DATASET}' không có."
            )
        order = dates.sort_values(kind="mergesort").index
        cut = int(len(order) * (1 - C.TEST_SIZE))
        tr_idx, te_idx = order[:cut], order[cut:]
        print(f"Temporal split tại {dates.loc[tr_idx].max().date()}: "
              f"train <= mốc này, test sau mốc này")
        return X.loc[tr_idx], X.loc[te_idx], y.loc[tr_idx], y.loc[te_idx]

    if C.SPLIT_STRATEGY != "random":
        raise ValueError(f"SPLIT_STRATEGY không hợp lệ: {C.SPLIT_STRATEGY}")
    return train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.SEED, stratify=y
    )


# ---------------------------------------------------------------------------
# Đánh giá chéo
# ---------------------------------------------------------------------------
def cv_out_of_fold(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    trial: optuna.Trial | None = None,
) -> tuple[np.ndarray, list[int], list[float]]:
    """Chạy StratifiedKFold, trả về (xác suất out-of-fold, số cây, PR-AUC mỗi fold).

    Mỗi fold dùng chính phần validation của nó để early stopping. Truyền `trial`
    để Optuna cắt sớm các cấu hình kém ngay giữa chừng.
    """
    skf = StratifiedKFold(
        n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED
    )
    oof = np.zeros(len(X), dtype=float)
    best_iters: list[int] = []
    fold_scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        x_tr, x_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = XGBClassifier(**params)
        model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)

        oof[va_idx] = model.predict_proba(x_va)[:, 1]
        best_iters.append(int(model.best_iteration) + 1)
        fold_scores.append(float(average_precision_score(y_va, oof[va_idx])))

        if trial is not None:
            trial.report(float(np.mean(fold_scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return oof, best_iters, fold_scores


def base_params() -> dict:
    """Tham số cố định, không đưa vào không gian dò tìm."""
    return {
        "objective": "binary:logistic",
        # aucpr để early stopping cùng thước đo với mục tiêu tối ưu
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": C.SEED,
        "n_jobs": -1,
        "n_estimators": C.MAX_ESTIMATORS,
        "early_stopping_rounds": C.EARLY_STOPPING_ROUNDS,
        # Cho phép nhận thẳng cột dtype category (chain, consensus)
        "enable_categorical": True,
    }


def xgb_objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = {
        **base_params(),
        # 1. Dung lượng học -- n_estimators do early stopping quyết định
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        # 2. Chống nhiễu từ bot / ví cá biệt
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        # 3. Phạt độ phức tạp để ép giảm FP
        "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    _, _, fold_scores = cv_out_of_fold(params, X, y, trial=trial)
    return float(np.mean(fold_scores))


# ---------------------------------------------------------------------------
# Chọn ngưỡng quyết định
# ---------------------------------------------------------------------------
def pick_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, str]:
    """Chọn ngưỡng trên out-of-fold prediction, không bao giờ trên tập test."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision_recall_curve trả về mảng dài hơn thresholds 1 phần tử
    precision, recall = precision[:-1], recall[:-1]

    if C.THRESHOLD_POLICY == "f1":
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        best = int(np.argmax(f1))
        reason = f"F1 cao nhất = {f1[best]:.4f}"
    elif C.THRESHOLD_POLICY == "min_precision":
        ok = precision >= C.TARGET_PRECISION
        if not ok.any():
            raise ValueError(
                f"Không ngưỡng nào đạt precision >= {C.TARGET_PRECISION}. "
                f"Precision cao nhất: {precision.max():.4f}"
            )
        candidates = np.where(ok)[0]
        best = int(candidates[np.argmax(recall[candidates])])
        reason = (
            f"recall cao nhất ({recall[best]:.4f}) "
            f"với precision >= {C.TARGET_PRECISION}"
        )
    else:
        raise ValueError(f"THRESHOLD_POLICY không hợp lệ: {C.THRESHOLD_POLICY}")

    return float(thresholds[best]), reason


def threshold_table(y_true: np.ndarray, proba: np.ndarray) -> None:
    print(f"{'ngưỡng':>8} {'precision':>10} {'recall':>8} {'F1':>8} {'cảnh báo':>10}")
    for t in np.arange(0.10, 0.95, 0.05):
        pred = (proba >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        print(f"{t:8.2f} {p:10.3f} {r:8.3f} {f1:8.3f} {int(pred.sum()):10d}")


# ---------------------------------------------------------------------------
# Báo cáo
# ---------------------------------------------------------------------------
def evaluate(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    oof_ap: float,
) -> dict:
    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    m = {
        "train_roc_auc": float(roc_auc_score(y_train, train_proba)),
        "train_pr_auc": float(average_precision_score(y_train, train_proba)),
        "train_log_loss": float(log_loss(y_train, train_proba)),
        "cv_oof_pr_auc": float(oof_ap),
        "test_roc_auc": float(roc_auc_score(y_test, test_proba)),
        "test_pr_auc": float(average_precision_score(y_test, test_proba)),
        "test_log_loss": float(log_loss(y_test, test_proba)),
        "test_brier": float(brier_score_loss(y_test, test_proba)),
        "threshold": float(threshold),
    }

    print("\n=== KIỂM TRA OVERFITTING ===")
    print("ROC-AUC bão hoà quanh 0.99 nên không dùng để phát hiện overfit.")
    print("So sánh PR-AUC và log-loss giữa train / CV / test:")
    print(f"  PR-AUC   train={m['train_pr_auc']:.4f}  "
          f"CV(oof)={m['cv_oof_pr_auc']:.4f}  test={m['test_pr_auc']:.4f}")
    print(f"  log-loss train={m['train_log_loss']:.4f}  test={m['test_log_loss']:.4f}")
    gap = m["cv_oof_pr_auc"] - m["test_pr_auc"]
    print(f"  Chênh lệch CV vs test: {gap:+.4f} "
          f"({'trong khoảng nhiễu' if abs(gap) < 0.02 else 'ĐÁNG NGỜ, cần xem lại'})")
    print(f"  Brier score trên test: {m['test_brier']:.4f} "
          f"(càng gần 0 càng đáng tin khi hiển thị % rủi ro)")

    print(f"\n=== CHI TIẾT TRÊN TẬP TEST (ngưỡng = {threshold:.4f}) ===")
    test_pred = (test_proba >= threshold).astype(int)
    print(
        classification_report(
            y_test, test_pred, target_names=["Clean (0)", "Fraud (1)"], digits=4
        )
    )

    print("=== MA TRẬN NHẦM LẪN ===")
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
    print(f"  Ví sạch đoán đúng (TN)          : {tn}")
    print(f"  Ví sạch bị báo nhầm (FP)        : {fp}")
    print(f"  Vụ lừa đảo BỊ BỎ SÓT (FN)       : {fn}")
    print(f"  Vụ lừa đảo BẮT ĐÚNG (TP)        : {tp}")
    m.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

    print("\n=== ĐÁNH ĐỔI NGƯỠNG TRÊN TẬP TEST ===")
    threshold_table(y_test.to_numpy(), test_proba)
    return m


# ---------------------------------------------------------------------------
def main() -> None:
    C.setup_console()
    X, y, dates = load_data()
    print(f"Dataset '{C.DATASET}' | split '{C.SPLIT_STRATEGY}'")
    print(f"Dữ liệu sạch: {len(X)} dòng, {len(C.FEATURES)} feature "
          f"({len(C.CATEGORICAL_FEATURES)} phân loại)")
    ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
    print(f"Cân bằng lớp: {(y == 0).sum()} âm / {(y == 1).sum()} dương "
          f"({ratio:.2f}:1)")

    # Tập test được tách ra ngay và chỉ được chạm vào MỘT lần ở cuối.
    x_train, x_test, y_train, y_test = split_train_test(X, y, dates)
    print(f"Train {len(x_train)} / Test {len(x_test)} (test chỉ dùng ở bước cuối)\n")

    # --- Optuna trên CV của tập train ---
    print(f"🚀 Optuna: {C.N_TRIALS} trial x {C.N_FOLDS}-fold CV, "
          f"mục tiêu = {C.TUNING_METRIC}")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=C.SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(
        lambda t: xgb_objective(t, x_train, y_train),
        n_trials=C.N_TRIALS,
        show_progress_bar=True,
    )

    n_pruned = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    )
    print(f"\nHoàn tất ({n_pruned} trial bị cắt sớm).")
    print(f"🥇 PR-AUC (CV) cao nhất: {study.best_value:.4f}")
    print("📦 Bộ siêu tham số tốt nhất:")
    for k, v in study.best_params.items():
        print(f"   - {k}: {v if isinstance(v, int) else f'{v:.5f}'}")

    # --- Chạy lại CV với tham số tốt nhất để lấy out-of-fold prediction ---
    print("\n⚙️  Chạy lại CV với tham số tốt nhất để lấy out-of-fold prediction...")
    best = {**base_params(), **study.best_params}
    oof, best_iters, fold_scores = cv_out_of_fold(best, x_train, y_train)
    oof_ap = float(average_precision_score(y_train, oof))
    print(f"   PR-AUC từng fold: {np.round(fold_scores, 4).tolist()}")
    print(f"   PR-AUC out-of-fold tổng: {oof_ap:.4f}")
    print(f"   Số cây early stopping chọn: {best_iters}")

    # --- Chọn ngưỡng trên out-of-fold, không phải trên test ---
    threshold, reason = pick_threshold(y_train.to_numpy(), oof)
    print(f"\n🎚️  Ngưỡng chọn theo chính sách '{C.THRESHOLD_POLICY}': "
          f"{threshold:.4f} ({reason})")
    print("   Đánh đổi trên out-of-fold:")
    threshold_table(y_train.to_numpy(), oof)

    # --- Model cuối: train trên toàn bộ tập train, không early stopping ---
    # Nhân k/(k-1) vì model cuối thấy nhiều dữ liệu hơn mỗi fold CV.
    n_final = int(round(np.mean(best_iters) * C.N_FOLDS / (C.N_FOLDS - 1)))
    final_params = {
        **base_params(),
        **study.best_params,
        "n_estimators": n_final,
    }
    final_params.pop("early_stopping_rounds")
    print(f"\n⚙️  Huấn luyện model cuối với n_estimators = {n_final}...")
    model = XGBClassifier(**final_params)
    model.fit(x_train, y_train, verbose=False)

    metrics = evaluate(model, x_train, y_train, x_test, y_test, threshold, oof_ap)

    # --- Lưu bundle: model + hợp đồng dữ liệu + metric + version ---
    cleaning = {}
    if C.CLEAN_REPORT_JSON.exists():
        cleaning = json.loads(C.CLEAN_REPORT_JSON.read_text(encoding="utf-8"))

    bundle = {
        "schema_version": 2,
        "dataset": C.DATASET,
        "split_strategy": C.SPLIT_STRATEGY,
        "categorical_features": C.CATEGORICAL_FEATURES,
        "model": model,
        # Thứ tự feature là một phần của hợp đồng: XGBoost nhận array theo vị trí,
        # truyền sai thứ tự sẽ cho kết quả sai mà không báo lỗi.
        "feature_names": C.FEATURES,
        "target": C.TARGET,
        "threshold": threshold,
        "threshold_policy": C.THRESHOLD_POLICY,
        "metrics": metrics,
        "cv_fold_pr_auc": fold_scores,
        "best_params": study.best_params,
        "final_n_estimators": n_final,
        "seed": C.SEED,
        "cleaning_report": cleaning,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "xgboost": xgboost.__version__,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "optuna": optuna.__version__,
        },
    }
    joblib.dump(bundle, C.MODEL_BUNDLE)
    print(f"\n💾 Đã lưu bundle vào {C.MODEL_BUNDLE.name}")
    print("   Không còn scaler: XGBoost không cần chuẩn hoá đầu vào.")
    print("   Khi dự đoán, dùng config.align_features(df) để đảm bảo đúng thứ tự cột.")


if __name__ == "__main__":
    main()
