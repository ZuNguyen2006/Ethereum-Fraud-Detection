"""Cấu hình dùng chung cho toàn bộ pipeline.

Trước đây danh sách feature được khai báo lặp ở cả preprocessing.py lẫn Train.py
và hai bên có thể lệch nhau âm thầm. Mọi hằng số dùng chung giờ nằm ở đây.
"""

import sys
from pathlib import Path

import pandas as pd


def setup_console() -> None:
    """Ép stdout sang UTF-8.

    Console Windows mặc định là cp1252 và sẽ ném UnicodeEncodeError khi gặp
    tiếng Việt có dấu hoặc emoji trong log.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Đường dẫn
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"

# Dataset 1: ví EOA lừa đảo (Ethereum Fraud Detection)
RAW_CSV = ROOT / "transaction_dataset.csv"
EOA_CSV = ROOT / "dataset_final.csv"
EOA_REPORT = ROOT / "cleaning_report.json"

# Dataset 2: RugPull1K -- cấp token/project, tải từ HuggingFace
RUGPULL1K_XLSX = DATA_DIR / "dataset.xlsx"
RUGPULL1K_CSV = DATA_DIR / "rugpull1k_clean.csv"
RUGPULL1K_REPORT = DATA_DIR / "rugpull1k_report.json"

# ---------------------------------------------------------------------------
# Tái lập kết quả
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Feature
# ---------------------------------------------------------------------------
# Các cột lấy thẳng từ dataset gốc (đã strip + lower tên cột).
CORE_FEATURES = [
    "avg min between sent tnx",
    "avg min between received tnx",
    "sent tnx",
    "received tnx",
    "unique received from addresses",
    "unique sent to addresses",
    "avg val received",
    "max value received",
    "total erc20 tnxs",
    "erc20 uniq sent addr",
    "erc20 uniq rec addr",
]

# Các cột phái sinh do preprocessing.py tạo ra.
#
# Lưu ý đo đạc: 5-fold CV cho thấy 3 feature này đóng góp +0.001 PR-AUC, nằm
# gọn trong nhiễu (+/- 0.005). Giữ lại vì chúng không gây hại và dễ diễn giải,
# nhưng đừng kỳ vọng chúng nâng điểm.
ENGINEERED_FEATURES = [
    "tnx_per_unique_receiver",
    "val_received_skewness",
    "erc20_ratio",
]

EOA_FEATURES = CORE_FEATURES + ENGINEERED_FEATURES
TARGET = "flag"

# Mẫu số của các tỷ lệ phái sinh. Code cũ dùng 1e-5, biến 537 dòng có
# sent+received == 0 thành erc20_ratio = 300_000 -- một outlier hoàn toàn nhân
# tạo. eps = 1.0 cho cùng chất lượng model mà không tạo giá trị bệnh hoạn.
RATIO_EPS = 1.0

# ---------------------------------------------------------------------------
# Chính sách làm sạch dữ liệu (P0)
# ---------------------------------------------------------------------------
# 829 dòng trong dataset gốc bị thiếu (NaN) toàn bộ 3 cột ERC20, và 100% trong
# số đó có flag = 1. Đây là artifact của quá trình thu thập dữ liệu, không phải
# hành vi on-chain. Ba cách xử lý và hệ quả đo được (5-fold CV, PR-AUC):
#
#   "drop"      -> loại hẳn 829 dòng.            PR-AUC 0.965  <-- mặc định
#   "fill_zero" -> hành vi cũ, fillna(0).        PR-AUC 0.944
#   (giữ NaN)   -> KHÔNG cài đặt. Cho PR-AUC 0.983 nhưng đó là điểm giả:
#                  model chỉ học "API không trả dữ liệu ERC20" và sẽ sập khi
#                  gặp dữ liệu thật.
#
# "fill_zero" gộp 829 dòng fraud vào chung 4399 dòng có số 0 thật (0% fraud),
# tức là phá huỷ thông tin chứ không bổ khuyết. Chỉ dùng để tái lập số cũ.
ERC20_MISSING_POLICY = "drop"

ERC20_COLUMNS = [
    "total erc20 tnxs",
    "erc20 uniq sent addr",
    "erc20 uniq rec addr",
]

# Bỏ các dòng mà toàn bộ feature đều bằng 0 (282 dòng) -- không mang thông tin.
DROP_ALL_ZERO_ROWS = True

# Bỏ các nhóm dòng trùng feature nhưng mâu thuẫn nhãn (18 nhóm). Không có cách
# nào biết nhãn nào đúng nên giữ lại chỉ làm nhiễu tín hiệu học.
DROP_CONFLICTING_LABELS = True

# Khử trùng lặp TRƯỚC khi split. Dataset gốc có 1.113 dòng trùng feature khiến
# 12.8% tập test là bản sao của dòng trong tập train.
DROP_DUPLICATES = True

# 25 địa chỉ ví xuất hiện nhiều lần -- cùng một ví thì không được nằm ở cả hai
# phía của split.
DROP_DUPLICATE_ADDRESSES = True

# ---------------------------------------------------------------------------
# RugPull1K -- feature cấp token/project
# ---------------------------------------------------------------------------
# Bộ NGHIÊM NGẶT: chỉ những gì biết được TRƯỚC khi thanh khoản bị rút.
# Đã loại end_date và G1/W1/X1 bản toàn kỳ trong prepare_rugpull1k.py.
RUGPULL1K_FEATURES_STRICT = [
    "chain",
    "consensus",
    "n_transactions",
    "token_concentration_ratio",
    "variance_total",
    "variance_holders_gt1pct",
    "token_supply",
    "n_initial_deposits",
    "G1_half",   # Google hits tại điểm giữa vòng đời
    "W1_half",   # tweet tại điểm giữa
    "X1_half",   # hoạt động X tại điểm giữa
]

# has_website / has_twitter CỐ TÌNH bị loại: 998/999 và 997/999 dòng đều bằng 1
# -- kể cả scam cũng có đủ website lẫn Twitter. Độ quan trọng đo được đúng bằng
# 0.0000. Chỉ KHỐI LƯỢNG tín hiệu OSINT (G1_half, X1_half) mới phân biệt được,
# còn sự TỒN TẠI của chúng thì không.

# Bộ MỞ RỘNG: thêm 4 chỉ báo giá theo quý. Paper không nói rõ Q1-Q4 đo ở mốc
# nào; nếu chúng trải hết vòng đời project thì đã bao gồm cả cú sập giá, tức là
# rò rỉ. Chạy cả hai bộ rồi so chênh lệch để biết.
RUGPULL1K_FEATURES_EXTENDED = RUGPULL1K_FEATURES_STRICT + ["Q1", "Q2", "Q3", "Q4"]

# XGBoost xử lý trực tiếp qua enable_categorical, không cần one-hot.
RUGPULL1K_CATEGORICAL = ["chain", "consensus"]

# start_year CỐ TÌNH không nằm trong feature: scam dồn vào 2021-2022 nên model
# sẽ học "năm 2022 = scam" và vô dụng với token mới. Cột vẫn có trong CSV để
# dùng cho temporal split.

# ---------------------------------------------------------------------------
# Chọn dataset đang huấn luyện
# ---------------------------------------------------------------------------
#   "eoa"        -> ví lừa đảo Ethereum (dataset_final.csv)
#   "rugpull1k"  -> token/project (data/rugpull1k_clean.csv)
DATASET = "rugpull1k"

# Chỉ áp dụng cho rugpull1k: "strict" (mặc định) hoặc "extended"
RUGPULL1K_FEATURE_SET = "strict"

# Cách chia tập test:
#   "random"   -> stratified ngẫu nhiên, so sánh được với kết quả công bố
#   "temporal" -> train trên project cũ, test trên project mới. Sát thực tế hơn:
#                 model thật luôn phải dự đoán token chưa từng thấy.
#
# Đo thực tế (compare_rugpull1k.py): temporal cho PR-AUC test CAO HƠN random
# (+0.019 strict, +0.014 extended). Với chỉ ~200 dòng test thì chênh lệch cỡ này
# nằm trong nhiễu -- kết luận đúng là "không khác nhau đáng kể", chứ không phải
# temporal tốt hơn. Mặc định vẫn để temporal vì nó là phép đo trung thực cho
# tình huống production.
SPLIT_STRATEGY = "temporal"

if DATASET == "eoa":
    CLEAN_CSV = EOA_CSV
    CLEAN_REPORT_JSON = EOA_REPORT
    FEATURES = EOA_FEATURES
    CATEGORICAL_FEATURES: list[str] = []
    DATE_COLUMN: str | None = None
    MODEL_BUNDLE = ROOT / "rug_pull_model.joblib"
elif DATASET == "rugpull1k":
    CLEAN_CSV = RUGPULL1K_CSV
    CLEAN_REPORT_JSON = RUGPULL1K_REPORT
    FEATURES = (
        RUGPULL1K_FEATURES_STRICT
        if RUGPULL1K_FEATURE_SET == "strict"
        else RUGPULL1K_FEATURES_EXTENDED
    )
    CATEGORICAL_FEATURES = RUGPULL1K_CATEGORICAL
    DATE_COLUMN = "start_date"
    MODEL_BUNDLE = ROOT / f"rugpull1k_model_{RUGPULL1K_FEATURE_SET}.joblib"
else:
    raise ValueError(f"DATASET không hợp lệ: {DATASET}")

# ---------------------------------------------------------------------------
# Huấn luyện (P1)
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
N_FOLDS = 5
N_TRIALS = 40

# Optuna tối ưu theo PR-AUC (average_precision), không phải ROC-AUC. Với bài
# toán mất cân bằng và mối quan tâm nghiệp vụ là FP/FN thì ROC-AUC bão hoà và
# không phân biệt được các cấu hình tốt/xấu.
TUNING_METRIC = "average_precision"

# Trần số cây; số cây thực tế do early stopping quyết định.
MAX_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 50

# Cách chọn ngưỡng quyết định trên out-of-fold prediction:
#   "f1"            -> ngưỡng cho F1 cao nhất (mặc định trung tính)
#   "min_precision" -> recall cao nhất với ràng buộc precision >= TARGET_PRECISION
THRESHOLD_POLICY = "f1"
TARGET_PRECISION = 0.90


def align_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sắp cột theo đúng thứ tự model mong đợi, báo lỗi nếu thiếu.

    XGBoost nhận numpy array theo vị trí. Nếu backend truyền cột sai thứ tự thì
    dự đoán sẽ sai một cách âm thầm, không có exception nào. Luôn đi qua hàm này
    trước khi gọi predict.
    """
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Thiếu {len(missing)} feature: {missing}")
    return df[FEATURES]
