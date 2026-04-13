from pathlib import Path
import re

import pandas as pd


TARGET_BRIDGE = "잠실대교"
TARGET_HOURS = [f"{hour:02d}시" for hour in range(6, 24)]


def normalize_text(value):
    return re.sub(r"\s+", "", str(value)).strip()


def resolve_base_dir(script_dir: Path) -> Path:
    candidates = [
        script_dir / "Data" / "Trafficdata",
        script_dir / "Trafficdata",
        script_dir / "TrafficData",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Trafficdata folder not found.")


def find_month_file(base_dir: Path, month: int) -> Path:
    patterns = [
        f"{month:02d}*2024*.xlsx",
        f"{month:02d}*.xlsx",
    ]
    for pattern in patterns:
        matches = sorted(base_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Monthly file not found for month={month:02d}")


def find_hour_columns(df: pd.DataFrame) -> dict[str, str]:
    normalized_cols = {normalize_text(col): col for col in df.columns}
    hour_map = {}
    for hour in range(6, 24):
        canonical = f"{hour:02d}시"
        candidates = [canonical, f"{hour}시"]
        for candidate in candidates:
            if candidate in normalized_cols:
                hour_map[canonical] = normalized_cols[candidate]
                break
    return hour_map


def extract_month_row(file_path: Path, month: int) -> dict:
    excel_file = pd.ExcelFile(file_path)

    for sheet_name in excel_file.sheet_names:
        for header_row in range(6):
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
            except Exception:
                continue

            df.columns = [str(col).strip() for col in df.columns]
            hour_map = find_hour_columns(df)
            if len(hour_map) != len(TARGET_HOURS):
                continue

            location_col = None
            bridge_mask = None

            for col in df.columns:
                normalized_values = df[col].map(normalize_text)
                mask = normalized_values.eq(TARGET_BRIDGE)
                if mask.any():
                    location_col = col
                    bridge_mask = mask
                    break

            if location_col is None or bridge_mask is None or not bridge_mask.any():
                continue

            selected = df.loc[bridge_mask, [hour_map[hour] for hour in TARGET_HOURS]].copy()
            selected = selected.apply(pd.to_numeric, errors="coerce").fillna(0)
            month_sum = selected.sum(axis=0)

            return {
                "month": month,
                **{
                    canonical_hour: month_sum[hour_map[canonical_hour]]
                    for canonical_hour in TARGET_HOURS
                },
            }

    raise ValueError(f"Could not extract {TARGET_BRIDGE} from {file_path.name}")


def main():
    script_dir = Path(__file__).resolve().parent
    base_dir = resolve_base_dir(script_dir)
    rows = [extract_month_row(find_month_file(base_dir, month), month) for month in range(1, 13)]
    result_df = pd.DataFrame(rows)
    ordered_columns = ["month"] + TARGET_HOURS
    result_df = result_df[ordered_columns].sort_values("month").reset_index(drop=True)

    output_path = script_dir / "Data" / "Trafficdata" / "2024_jamsil_mooneojim.csv"
    result_df.to_csv(output_path, index=False)

    print(result_df.head())
    print(result_df.shape)


if __name__ == "__main__":
    main()
