
import re

import numpy as np
import pandas as pd

HEADER_TRANSLATIONS = {
    "תאריך": "date",
    "שעה": "time",
    "תחזית ביקוש יום מראש": "day-ahead-forecast",
    "תחזית ביקוש עדכנית": "updated-demand-forecast",
    "ביקוש בפועל": "actual-demand",
}

IMS_HEADER_TRANSLATIONS = {
    "תחנה": "station",
    "תאריך ושעה (שעון חורף)": "datetime",
    "טמפרטורה (C°)": "temp",
    "טמפרטורה לחה (C°)": "wet_bulb",
    "טמפרטורת נקודת הטל (C°)": "dew_point",
    "לחות יחסית (%)": "relative_humidity_percent",
    "כיוון הרוח (מעלות)": "wind_dir_deg",
    "מהירות רוח (מטר לשניה)": "wind_speed_m_s",
}

IMS_STATION_TRANSLATIONS = {
    'חיפה אוניברסיטה 10/2001-12/2025': 'Haifa',
    'תל-אביב חוף 04/2005-12/2025': 'TelAviv',
    'ירושלים מרכז 01/1997-12/2025': 'Jerusalem'
}

IMS_STATION_RANGE_PATTERN = re.compile(r"\d{2}/\d{4}-\d{2}/\d{4}$")

PRICE_COLUMNS = [
    "date",
    "time",
    "day_ahead_constrained",
    "day_ahead_unconstrained",
    "real_time_constrained",
    "real_time_unconstrained",
]

# 30-minute settlement periods.
PERIOD_MIN = 30


def noga_csv() -> None:
    xlsx = pd.read_excel(
        "data/noga.xlsx",
        skiprows=1)

    xlsx.rename(columns=HEADER_TRANSLATIONS, inplace=True)
    xlsx.to_csv("data/noga.csv", index=False)


def noga_fixed_csv():
    '''
    noga.csv:
        - date: 2024-01-01
        - time: 00:05:00
        - day-ahead-forecast: 7415
        - actual-demand: 7378.6

    1. Reads noga.csv
    2. Interpolate missing values (day-ahead-forecast, actual-demand).
    3. Saves to noga.fixed.csv
    '''
    noga = pd.read_csv("data/noga.csv")

    noga['date'] = pd.to_datetime(noga['date'])
    noga['time'] = pd.to_timedelta(noga['time']).dt.total_seconds() / 60
    noga['time'] = noga['time'].astype(int)

    noga['forecast'] = pd.to_numeric(
        noga['day-ahead-forecast'],
        errors='coerce') \
        .replace(0, np.nan) \
        .interpolate()

    noga['actual'] = pd.to_numeric(
        noga['actual-demand'],
        errors='coerce') \
        .replace(0, np.nan) \
        .interpolate()

    noga = noga.drop(columns=[
        'day-ahead-forecast',
        'actual-demand',
        'updated-demand-forecast'
    ])

    noga.to_csv("data/noga.fixed.csv", index=False)


def ims_csv() -> None:
    ims = pd.read_csv("data/ims.he.csv", encoding="utf-8")
    ims.rename(columns=IMS_HEADER_TRANSLATIONS, inplace=True)
    ims["station"] = ims["station"].astype(str).map(
        lambda s: IMS_STATION_TRANSLATIONS.get(s, s))  # type: ignore

    ims['datetime'] = pd.to_datetime(
        ims['datetime'],
        format="%d-%m-%Y %H:%M",
    )

    ims = ims.pivot(
        index=["datetime"],
        columns="station")

    ims.columns = [f'{c}_{s}' for c, s in ims.columns]
    ims = ims.reset_index()
    dt = ims['datetime'].dt
    ims.insert(1, "date", dt.strftime("%Y-%m-%d"))
    ims.insert(2, "time", dt.hour * 60 + dt.minute)
    ims.drop(columns=["datetime"], inplace=True)

    ims.to_csv("data/ims.csv", index=False)

    row = ims.iloc[0]
    for col in ims.columns:
        print(f"- {col}: {row[col]}")


def price_csv():
    '''
    Noga SMP export (₪/MWh, 30-minute periods), Hebrew headers in two rows:
    date, time, day-ahead (constrained, unconstrained),
    real-time (constrained, unconstrained).
    '''
    xlsx = pd.read_excel(
        "data/Smp_01_04_2023-01_01_2026_he-IL.xlsx",
        skiprows=2,
        header=None,
        names=PRICE_COLUMNS)

    xlsx["date"] = pd.to_datetime(xlsx["date"]).dt.strftime("%Y-%m-%d")
    for col in PRICE_COLUMNS[2:]:
        xlsx[col] = pd.to_numeric(xlsx[col], errors="coerce")

    xlsx.to_csv("data/price.csv", index=False)


def data_csv():
    noga = pd.read_csv("data/noga.fixed.csv")
    ims = pd.read_csv("data/ims.csv")
    price = pd.read_csv("data/price.csv")

    data = pd.merge(noga, ims, on=["date", "time"], how="left")
    int_cols = [col for col in ims.columns if col not in ["date", "time"]]
    data[int_cols] = data[int_cols].interpolate()

    # Each 5-minute slot takes the day-ahead price of its settlement period.
    price["slot"] = pd.to_timedelta(price["time"]).dt.total_seconds() // 60
    price = price.rename(columns={"day_ahead_constrained": "p_da"})
    data["slot"] = data["time"] // PERIOD_MIN * PERIOD_MIN
    data = pd.merge(
        data,
        price[["date", "slot", "p_da"]],
        on=["date", "slot"],
        how="left")

    dt = pd.to_datetime(data['date']).dt
    data = data.dropna()

    data['year'] = dt.year
    data['month'] = dt.month - 1
    data['day'] = (dt.dayofweek + 1) % 7
    # Settlement period index: key for the synthetic imbalance prices.
    days = (pd.to_datetime(data['date']) - pd.Timestamp("2023-01-01")).dt.days
    data['period'] = days * (24 * 60 // PERIOD_MIN) + data['slot'] // PERIOD_MIN
    data['date_ord'] = days

    data.drop(columns=["date", "slot"], inplace=True)
    data.columns = [c.replace("Tel Aviv", "TelAviv") for c in data.columns]
    data.to_csv("data/data.csv", index=False)


def sample_csv():
    data = pd.read_csv("data/data.csv")
    sample = data[[
        "year", "month", "day", "time",
        "forecast", "actual", 'temp_Haifa']]

    print(sample)


if __name__ == "__main__":
    import sys

    # e.g. `uv run -m noga.data price_csv` (as used by Taskfile.yml).
    if len(sys.argv) > 1:
        globals()[sys.argv[1]]()
    else:
        sample_csv()
