"""Chuẩn hoá RugPull1K (data/dataset.xlsx) thành CSV sạch, không rò rỉ thời gian.

Ba việc chính:

  1. Dịch 27 tên cột tiếng Ba Tư sang tiếng Anh. Phải map THEO VỊ TRÍ chứ không
     theo tên: tên gốc chứa ký tự RTL/non-breaking space ẩn nên so khớp chuỗi
     thất bại âm thầm.

  2. Ép kiểu các cột số đang bị lưu dạng text (dấu phẩy nghìn, \\xa0, xuống dòng).

  3. Loại cột rò rỉ thời gian. Đây là phần quan trọng nhất:

       end_date  -> ngày project SỤP ĐỔ. Thời lượng của scam có median 291 ngày
                    so với 1.104 ngày của project bình thường. Tại thời điểm cần
                    dự đoán ta không biết ngày này.
       G1/W1/X1 -> tín hiệu OSINT đo trên TOÀN vòng đời, tức tính cả giai đoạn
                    sau khi sụp đổ. Bản `_half` đo tại điểm giữa mới dùng được.

     Cùng loại bẫy với 829 dòng ERC20-NaN của dataset cũ: feature tương quan
     hoàn hảo với nhãn vì nó được đo sau khi sự kiện đã xảy ra.
"""

import json

import numpy as np
import pandas as pd

import config as C

# Map theo vị trí cột (xem docstring). Thứ tự phải khớp đúng file gốc.
RAW_NAMES = [
    "project_name",              # عنوان
    "Q1", "Q2", "Q3", "Q4",      # chỉ báo giá theo quý
    "chain",                     # نوع شبکه
    "n_transactions",            # تعداد تراکنش ها
    "token_concentration_ratio", # نسبت تمرکز توکن ها به تعداد افراد
    "variance_total",            # واریانس کل
    "variance_holders_gt1pct",   # واریانس هولدرهای بالای یک درصد
    "token_supply",              # موجودی توکن ها
    "ticker",                    # نوع ارز دیجیتالی درگیر
    "n_initial_deposits",        # تعداد واریزهای اولیه
    "consensus",                 # نوع شبکه بلاکچین (pos-pow)
    "contract_online",           # قرارداد هوشمند (آنلاین)
    "contract_offline",          # قرارداد هوشمند (آفلاین)
    "website",
    "x",
    "label",                     # کلاس: scam / normal
    "start_date",                # شروع پروژه
    "end_date",                  # پایان پروژه  <-- RÒ RỈ
    "G1", "G1_half",             # Google hits: toàn kỳ / tại điểm giữa
    "W1", "W1_half",             # Tweet: toàn kỳ / tại điểm giữa
    "X1", "X1_half",             # X activity: toàn kỳ / tại điểm giữa
]

NUMERIC_COLUMNS = [
    "Q1", "Q2", "Q3", "Q4",
    "n_transactions", "token_concentration_ratio",
    "variance_total", "variance_holders_gt1pct",
    "token_supply", "n_initial_deposits",
    "G1", "G1_half", "W1", "W1_half", "X1", "X1_half",
]

# Bỏ hẳn: định danh, hằng số, và feature đo sau khi sự kiện xảy ra.
LEAKY_COLUMNS = ["end_date", "G1", "W1", "X1"]
CONSTANT_COLUMNS = ["contract_online", "contract_offline"]
ID_COLUMNS = ["project_name", "ticker", "website", "x"]

# Ethereum ra đời 2015; mốc trước đó chắc chắn là lỗi nhập liệu.
MIN_PLAUSIBLE_YEAR = 2013

# FANTOM và FTM là cùng một chain bị nhập thành hai giá trị.
CHAIN_ALIASES = {"FTM": "FANTOM"}

# Các cột đuôi nặng cần ép về thang log.
#
# variance_holders_gt1pct chạm 3.19e117 và token_supply chạm 1.16e59 (số dư
# tính theo đơn vị nhỏ nhất của token, bình phương lên thì bùng nổ). XGBoost
# chuyển dữ liệu sang float32 ở tầng dưới, trần float32 là 3.4e38 -- mọi giá
# trị vượt ngưỡng biến thành inf và fit ném lỗi.
#
# Cây quyết định bất biến với biến đổi đơn điệu nên log KHÔNG làm đổi chất
# lượng model; nó chỉ giữ số nằm trong dải biểu diễn được. Dùng dạng có dấu
# để an toàn cả với giá trị âm.
LOG_TRANSFORM_COLUMNS = [
    "n_transactions", "token_concentration_ratio",
    "variance_total", "variance_holders_gt1pct",
    "token_supply", "n_initial_deposits",
    "G1_half", "W1_half", "X1_half",
    "Q1", "Q2", "Q3", "Q4",
]


def signed_log1p(s: pd.Series) -> pd.Series:
    """log1p giữ dấu: đơn điệu trên toàn trục số, an toàn với giá trị âm."""
    return np.sign(s) * np.log1p(s.abs())


def to_numeric(s: pd.Series) -> pd.Series:
    """Ép về số, dọn non-breaking space, dấu phẩy nghìn, xuống dòng, ký tự '-'."""
    if s.dtype.kind in "if":
        return s
    cleaned = (
        s.astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("\n", "", regex=False)
        .str.strip()
        .replace({"-": None, "": None, "nan": None, "NaT": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def main() -> None:
    C.setup_console()
    df = pd.read_excel(C.RUGPULL1K_XLSX)
    report: dict = {"source": C.RUGPULL1K_XLSX.name, "n_raw": len(df)}

    if df.shape[1] != len(RAW_NAMES):
        raise ValueError(
            f"Kỳ vọng {len(RAW_NAMES)} cột, file có {df.shape[1]}. "
            "File nguồn có thể đã đổi -- kiểm tra lại map theo vị trí."
        )
    df.columns = RAW_NAMES
    print(f"Đọc {len(df)} dòng x {df.shape[1]} cột từ {C.RUGPULL1K_XLSX.name}")

    # --- Nhãn ---
    df["flag"] = (
        df["label"].astype(str).str.strip().str.lower().map({"scam": 1, "normal": 0})
    )
    if df["flag"].isna().any():
        bad = df.loc[df["flag"].isna(), "label"].unique()
        raise ValueError(f"Giá trị nhãn lạ: {bad}")

    # --- Ép kiểu số ---
    print("\n=== ÉP KIỂU CỘT SỐ ===")
    coercion: dict[str, int] = {}
    for col in NUMERIC_COLUMNS:
        num = to_numeric(df[col])
        failed = int(num.isna().sum() - df[col].isna().sum())
        coercion[col] = max(failed, 0)
        if failed > 0:
            print(f"  {col:26s} {failed:3d} ô không ép được -> NaN")
        df[col] = num
    report["coercion_failures"] = coercion
    print(f"  tổng cộng {sum(coercion.values())} ô -> NaN "
          f"(để nguyên NaN, XGBoost xử lý được)")

    # --- Feature phái sinh ---
    df["has_website"] = df["website"].notna().astype(int)
    df["has_twitter"] = df["x"].notna().astype(int)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["start_year"] = df["start_date"].dt.year
    df["start_month"] = df["start_date"].dt.month
    df["chain"] = df["chain"].astype(str).str.strip().str.upper().replace(CHAIN_ALIASES)
    df["consensus"] = df["consensus"].astype(str).str.strip()

    # --- Kiểm tra cột hằng số trước khi bỏ ---
    print("\n=== CỘT HẰNG SỐ (bỏ) ===")
    for col in CONSTANT_COLUMNS:
        vals = df[col].astype(str).str.strip().str.lower().nunique()
        print(f"  {col:20s} {df[col].nunique()} giá trị thô, "
              f"{vals} sau khi chuẩn hoá chữ hoa/thường -> "
              f"{'hằng số' if vals == 1 else 'CẢNH BÁO: không phải hằng số'}")

    print("\n=== CỘT RÒ RỈ THỜI GIAN (bỏ) ===")
    end = pd.to_datetime(df["end_date"], errors="coerce")
    dur = (end - df["start_date"]).dt.days
    med = dur.groupby(df["flag"]).median()
    print(f"  end_date : thời lượng median normal={med.get(0, float('nan')):.0f} ngày "
          f"vs scam={med.get(1, float('nan')):.0f} ngày -> phân biệt hoàn hảo, "
          f"nhưng không biết trước được")
    for full, half in [("G1", "G1_half"), ("W1", "W1_half"), ("X1", "X1_half")]:
        print(f"  {full:2s} -> giữ {half} (đo tại điểm giữa vòng đời)")

    # --- Loại dòng lỗi ---
    print("\n=== LOẠI DÒNG LỖI ===")
    before = len(df)
    impossible = df["start_year"] < MIN_PLAUSIBLE_YEAR
    if impossible.any():
        years = sorted(df.loc[impossible, "start_year"].dropna().unique().tolist())
        print(f"  start_year < {MIN_PLAUSIBLE_YEAR}: -{int(impossible.sum())} dòng {years}")
    df = df[~impossible]

    n_bad_range = int((end.loc[df.index] < df["start_date"]).sum())
    if n_bad_range:
        print(f"  end_date < start_date: {n_bad_range} dòng (giữ lại -- "
              f"end_date không dùng làm feature)")

    # --- Ép thang log cho các cột đuôi nặng (xem LOG_TRANSFORM_COLUMNS) ---
    print("\n=== ÉP THANG LOG (tràn float32) ===")
    F32_MAX = np.finfo(np.float32).max
    logged = []
    for col in LOG_TRANSFORM_COLUMNS:
        before_max = df[col].abs().max()
        df[col] = signed_log1p(df[col])
        flag = "  <-- vượt trần float32" if before_max > F32_MAX else ""
        if before_max > 1e6 or flag:
            print(f"  {col:26s} max {before_max:.3e} -> {df[col].abs().max():7.2f}{flag}")
        logged.append(col)
    report["log_transformed"] = logged

    # --- Chọn cột giữ lại ---
    drop = set(LEAKY_COLUMNS + CONSTANT_COLUMNS + ID_COLUMNS + ["label"])
    keep = [c for c in df.columns if c not in drop]
    out = df[keep].copy()

    # --- Khử trùng lặp trên feature ---
    feat_cols = [c for c in out.columns if c not in ("flag", "start_date")]
    before = len(out)
    out = out.drop_duplicates(subset=feat_cols, keep="first")
    print(f"  trùng feature: -{before - len(out)} dòng")

    out = out.sort_values("start_date").reset_index(drop=True)

    report["n_clean"] = len(out)
    report["class_balance"] = {str(k): int(v) for k, v in out["flag"].value_counts().items()}
    report["dropped_columns"] = {
        "leaky": LEAKY_COLUMNS, "constant": CONSTANT_COLUMNS, "id": ID_COLUMNS,
    }
    report["columns"] = list(out.columns)

    out.to_csv(C.RUGPULL1K_CSV, index=False)
    C.RUGPULL1K_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    pos, neg = report["class_balance"].get("1", 0), report["class_balance"].get("0", 0)
    print(f"\nKết quả: {report['n_raw']} -> {len(out)} dòng, "
          f"{len(out.columns) - 2} feature + flag + start_date")
    print(f"Cân bằng lớp: {neg} normal / {pos} scam ({pos / len(out):.1%} scam)")
    print(f"Khoảng thời gian: {out['start_date'].min().date()} -> {out['start_date'].max().date()}")
    print(f"Đã ghi {C.RUGPULL1K_CSV.name} và {C.RUGPULL1K_REPORT.name}")


if __name__ == "__main__":
    main()
