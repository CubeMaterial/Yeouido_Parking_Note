from __future__ import annotations

import json
import os
import ssl
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

BASE_DIR = Path(__file__).resolve().parent
MPL_CACHE_DIR = BASE_DIR / ".cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "outputs"
WEATHER_CACHE_PATH = OUTPUT_DIR / "yeouido_weather_daily.csv"
MERGED_OUTPUT_PATH = OUTPUT_DIR / "yeouido_parking_weather_merged.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "yeouido_parking_weather_yearly_summary.csv"
CHART_OUTPUT_PATH = OUTPUT_DIR / "yeouido_parking_usage_by_weather_year.png"

WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"
YEOUIDO_LATITUDE = 37.525
YEOUIDO_LONGITUDE = 126.925
TIMEZONE = "Asia/Seoul"
PARKING_NAME_FILTER = "여의도"
PARKING_NAME_FILTER_LABEL = "Yeouido"

DATE_COLUMN_CANDIDATES = [
    "날짜",
    "일자",
    "기준일자",
    "date",
    "Date",
    "일시",
    "datetime",
]
USAGE_COLUMN_CANDIDATES = [
    "주차대수",
    "이용건수",
    "주차건수",
    "입차대수",
    "usage",
    "count",
]
TIME_COLUMN_CANDIDATES = [
    "이용시간",
    "주차시간",
    "usage_time",
]
PARKING_LOT_COLUMN_CANDIDATES = [
    "주차장명",
    "주차장",
    "parking_lot_name",
]
WEATHER_CATEGORY_LABELS = {
    "clear": "Clear",
    "cloudy": "Cloudy",
    "precipitation": "Precipitation",
}
WEATHER_CATEGORY_LABELS_KO = {
    "clear": "맑음",
    "cloudy": "흐림",
    "precipitation": "비/눈",
}
WEATHER_CATEGORY_ORDER = ["clear", "cloudy", "precipitation"]
WEATHER_COLORS = {
    "clear": "#4C9BE8",
    "cloudy": "#8A94A6",
    "precipitation": "#2E5B9A",
}
PRECIPITATION_CODES = {
    51,
    53,
    55,
    56,
    57,
    61,
    63,
    65,
    66,
    67,
    71,
    73,
    75,
    77,
    80,
    81,
    82,
    85,
    86,
    95,
    96,
    99,
}
CLEAR_CODES = {0, 1}
CAFILE_CANDIDATES = [
    Path("/etc/ssl/cert.pem"),
    Path("/private/etc/ssl/cert.pem"),
]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    read_errors: list[str] = []
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            read_errors.append(f"{encoding}: {exc}")

    error_text = "\n".join(read_errors)
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Unable to decode {path.name}:\n{error_text}")


def pick_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {column.strip(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for column in columns:
        lowered = column.lower()
        if any(candidate.lower() in lowered for candidate in candidates):
            return column
    return None


def normalize_date_column(series: pd.Series) -> pd.Series:
    extracted = (
        series.astype(str)
        .str.extract(r"(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})", expand=False)
        .fillna(series.astype(str))
        .str.replace(".", "-", regex=False)
        .str.replace("/", "-", regex=False)
    )
    return pd.to_datetime(extracted, errors="coerce").dt.normalize()


def normalize_numeric_column(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^\d.\-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_parking_usage(data_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    usage_frames: list[pd.DataFrame] = []
    loaded_files: list[Path] = []

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".xls", ".xlsx"}:
            continue

        frame = read_table(path)
        if frame.empty:
            continue

        frame.columns = [str(column).strip() for column in frame.columns]
        date_column = pick_column(frame.columns.tolist(), DATE_COLUMN_CANDIDATES)
        usage_column = pick_column(frame.columns.tolist(), USAGE_COLUMN_CANDIDATES)
        time_column = pick_column(frame.columns.tolist(), TIME_COLUMN_CANDIDATES)
        parking_lot_column = pick_column(frame.columns.tolist(), PARKING_LOT_COLUMN_CANDIDATES)

        if date_column is None or usage_column is None or parking_lot_column is None:
            continue

        normalized = pd.DataFrame(
            {
                "date": normalize_date_column(frame[date_column]),
                "parking_usage": normalize_numeric_column(frame[usage_column]),
                "parking_lot_name": frame[parking_lot_column].astype(str).str.strip(),
                "source_file": path.name,
            }
        )

        if time_column is not None:
            normalized["usage_time"] = normalize_numeric_column(frame[time_column])
        else:
            normalized["usage_time"] = pd.NA

        normalized = normalized.dropna(subset=["date", "parking_usage"])
        if normalized.empty:
            continue

        usage_frames.append(normalized)
        loaded_files.append(path)

    if not usage_frames:
        raise FileNotFoundError("No parking usage files with date and usage columns were found in the Data directory.")

    usage_data = pd.concat(usage_frames, ignore_index=True)
    usage_data = usage_data[usage_data["parking_lot_name"].str.contains(PARKING_NAME_FILTER, na=False)].copy()
    if usage_data.empty:
        raise ValueError(f'No parking rows matched the filter "{PARKING_NAME_FILTER}".')

    daily_usage = (
        usage_data.groupby("date", as_index=False)
        .agg(
            parking_usage=("parking_usage", "sum"),
            usage_time=("usage_time", "sum"),
            source_rows=("parking_usage", "size"),
            parking_lot_count=("parking_lot_name", "nunique"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return daily_usage, loaded_files


def classify_weather(weather_code: int, precipitation_sum: float, snowfall_sum: float) -> str:
    if precipitation_sum > 0 or snowfall_sum > 0 or weather_code in PRECIPITATION_CODES:
        return "precipitation"
    if weather_code in CLEAR_CODES:
        return "clear"
    return "cloudy"


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": YEOUIDO_LATITUDE,
        "longitude": YEOUIDO_LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "weather_code,precipitation_sum,snowfall_sum",
        "timezone": TIMEZONE,
    }
    url = f"{WEATHER_API_URL}?{urlencode(params)}"
    ssl_context = build_ssl_context()
    with urlopen(url, timeout=60, context=ssl_context) as response:
        payload = json.load(response)

    daily = payload.get("daily")
    if daily is None:
        raise ValueError("Weather API response did not include daily data.")

    weather = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]).normalize(),
            "weather_code": pd.to_numeric(daily["weather_code"]),
            "precipitation_sum": pd.to_numeric(daily["precipitation_sum"]),
            "snowfall_sum": pd.to_numeric(daily["snowfall_sum"]),
        }
    )
    weather["weather_category"] = weather.apply(
        lambda row: classify_weather(
            int(row["weather_code"]),
            float(row["precipitation_sum"]),
            float(row["snowfall_sum"]),
        ),
        axis=1,
    )
    weather["weather_label"] = weather["weather_category"].map(WEATHER_CATEGORY_LABELS)
    weather["weather_label_ko"] = weather["weather_category"].map(WEATHER_CATEGORY_LABELS_KO)
    return weather


def build_ssl_context() -> ssl.SSLContext:
    for cafile in CAFILE_CANDIDATES:
        if cafile.exists():
            return ssl.create_default_context(cafile=str(cafile))
    return ssl.create_default_context()


def load_or_fetch_weather(required_dates: pd.Series) -> pd.DataFrame:
    required_date_keys = set(required_dates.dt.strftime("%Y-%m-%d"))

    if WEATHER_CACHE_PATH.exists():
        cached = pd.read_csv(WEATHER_CACHE_PATH, parse_dates=["date"])
        cached["date"] = cached["date"].dt.normalize()
        cached_keys = set(cached["date"].dt.strftime("%Y-%m-%d"))
        if required_date_keys.issubset(cached_keys):
            return cached[cached["date"].dt.strftime("%Y-%m-%d").isin(required_date_keys)].copy()

    weather = fetch_weather(
        required_dates.min().strftime("%Y-%m-%d"),
        required_dates.max().strftime("%Y-%m-%d"),
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weather.to_csv(WEATHER_CACHE_PATH, index=False, encoding="utf-8-sig")
    return weather[weather["date"].dt.strftime("%Y-%m-%d").isin(required_date_keys)].copy()


def build_summary(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()
    merged["year"] = merged["date"].dt.year
    summary = (
        merged.groupby(["year", "weather_category"], as_index=False)
        .agg(
            days=("parking_usage", "size"),
            average_usage=("parking_usage", "mean"),
            peak_usage=("parking_usage", "max"),
        )
    )
    summary["weather_category"] = pd.Categorical(
        summary["weather_category"],
        categories=WEATHER_CATEGORY_ORDER,
        ordered=True,
    )
    summary = summary.sort_values(["year", "weather_category"]).reset_index(drop=True)
    summary["weather_label"] = summary["weather_category"].map(WEATHER_CATEGORY_LABELS)
    summary["weather_label_ko"] = summary["weather_category"].map(WEATHER_CATEGORY_LABELS_KO)
    return summary


def create_chart(summary: pd.DataFrame, chart_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))

    years = sorted(summary["year"].unique().tolist())
    year_positions = list(range(len(years)))
    bar_width = 0.22
    category_offsets = {
        "clear": -bar_width,
        "cloudy": 0.0,
        "precipitation": bar_width,
    }

    ymax = float(summary["peak_usage"].max())
    label_padding = ymax * 0.015

    for category in WEATHER_CATEGORY_ORDER:
        category_summary = (
            summary[summary["weather_category"].astype(str) == category]
            .set_index("year")
            .reindex(years)
            .reset_index()
        )
        x_positions = [position + category_offsets[category] for position in year_positions]
        bars = ax.bar(
            x_positions,
            category_summary["average_usage"],
            color=WEATHER_COLORS[category],
            width=bar_width,
            label=WEATHER_CATEGORY_LABELS[category],
        )
        ax.scatter(
            x_positions,
            category_summary["peak_usage"],
            color="#D62728",
            marker="D",
            s=34,
            zorder=3,
        )

        for index, bar in enumerate(bars):
            peak_value = category_summary.iloc[index]["peak_usage"]
            if pd.isna(peak_value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                float(peak_value) + label_padding,
                f"{float(peak_value):,.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.scatter([], [], color="#D62728", marker="D", s=34, label="Peak daily usage")
    ax.set_xticks(year_positions, [str(year) for year in years])
    ax.set_xlabel("Year")
    ax.set_ylabel("Average daily parking usage (vehicles)")
    ax.set_title(
        f'Yeouido Parking Usage by Weather Category and Year\n'
        f'(only parking lots containing "{PARKING_NAME_FILTER_LABEL}")'
    )
    ax.set_ylim(0, ymax * 1.14)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    daily_usage, loaded_files = load_parking_usage(DATA_DIR)
    weather = load_or_fetch_weather(daily_usage["date"])

    merged = daily_usage.merge(weather, on="date", how="left", validate="1:1")
    if merged["weather_category"].isna().any():
        missing_dates = merged.loc[merged["weather_category"].isna(), "date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Weather data is missing for {len(missing_dates)} dates: {missing_dates[:10]}")

    summary = build_summary(merged)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_to_save = merged.copy()
    merged_to_save["date"] = merged_to_save["date"].dt.strftime("%Y-%m-%d")
    merged_to_save["year"] = pd.to_datetime(merged_to_save["date"]).dt.year
    merged_to_save.to_csv(MERGED_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    create_chart(summary, CHART_OUTPUT_PATH)

    print(f"Loaded {len(loaded_files)} parking usage file(s):")
    for path in loaded_files:
        print(f"- {path.name}")
    print()
    print("Weather category mapping:")
    for category in WEATHER_CATEGORY_ORDER:
        print(f"- {WEATHER_CATEGORY_LABELS[category]} = {WEATHER_CATEGORY_LABELS_KO[category]}")
    print()
    print(f'Parking lot filter: "{PARKING_NAME_FILTER}"')
    print()
    print(summary.to_string(index=False, formatters={"average_usage": "{:,.2f}".format, "peak_usage": "{:,.0f}".format}))
    print()
    print(f"Merged data saved to: {MERGED_OUTPUT_PATH}")
    print(f"Summary saved to: {SUMMARY_OUTPUT_PATH}")
    print(f"Chart saved to: {CHART_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
