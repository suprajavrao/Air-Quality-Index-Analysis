"""Build a small, memory-friendly modelling extract from the raw KSPCB CSVs.

Pure standard library (no pandas/numpy) so it can run even when the numeric
stack is under memory pressure. Writes data/processed/model_sample.csv with
canonical column names used by app.py.
"""

import csv
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw")
OUT = os.path.join(HERE, "..", "data", "processed", "model_sample.csv")

# 0-based positions in the KSPCB header:
# Station ID, State, City, Station Name, Timestamp, PM2.5, PM10, NO, NO2, NOx,
# NH3, SO2, CO, Ozone, Benzene, Toluene, Xylene, O Xylene, Eth-Benzene,
# MP-Xylene, AT, RH, WS, WD, RF, TOT-RF, SR, BP, VWS
COLS = {
    "PM2.5": 5,
    "PM10": 6,
    "NO": 7,
    "NO2": 8,
    "NH3": 10,
    "SO2": 11,
    "CO": 12,
    "Ozone": 13,
    "Temperature": 20,
    "RelativeHumidity": 21,
    "WindSpeed": 22,
}
FIELD_ORDER = ["PM2.5", "PM10", "NO", "NO2", "NH3", "SO2", "CO", "Ozone",
               "Temperature", "RelativeHumidity", "WindSpeed"]

TARGET_TOTAL = 6000


def to_float(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    return value


def main():
    files = sorted(glob.glob(os.path.join(RAW, "blr-*.csv")))
    if not files:
        raise SystemExit("No blr-*.csv files found in data/raw")

    per_file = max(1, TARGET_TOTAL // len(files))
    rows = []

    for path in files:
        taken = 0
        seen = 0
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                continue
            if len(header) < 23 or "Timestamp" not in header[4]:
                print(f"skip (unexpected header): {os.path.basename(path)}")
                continue
            for row in reader:
                if len(row) < 23:
                    continue
                seen += 1
                if seen % 150 != 0:
                    continue
                target = to_float(row[COLS["PM2.5"]])
                if target is None:
                    continue
                record = [target]
                for name in FIELD_ORDER[1:]:
                    record.append(to_float(row[COLS[name]]))
                rows.append(record)
                taken += 1
                if taken >= per_file:
                    break
        print(f"{os.path.basename(path)}: {taken} rows sampled from {seen}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(FIELD_ORDER)
        # keep only rows with enough information to model
        kept = 0
        for record in rows:
            non_null = sum(1 for v in record if v is not None)
            if non_null >= len(FIELD_ORDER) // 2:
                writer.writerow(["" if v is None else round(v, 3) for v in record])
                kept += 1

    size_kb = os.path.getsize(OUT) / 1024
    print(f"wrote {OUT}: {kept} rows, {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
