
from __future__ import annotations

import re

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
    "תאריך ושעה (שעון חורף)": "datetime_winter_time",
    "טמפרטורה (C°)": "temperature_c",
    "טמפרטורה לחה (C°)": "wet_bulb_temperature_c",
    "טמפרטורת נקודת הטל (C°)": "dew_point_temperature_c",
    "לחות יחסית (%)": "relative_humidity_percent",
    "כיוון הרוח (מעלות)": "wind_direction_degrees",
    "מהירות רוח (מטר לשניה)": "wind_speed_m_s",
}

IMS_STATION_TRANSLATIONS = {
    'חיפה אוניברסיטה 10/2001-12/2025': 'Haifa',
    'תל-אביב חוף 04/2005-12/2025': 'Tel Aviv',
    'ירושלים מרכז 01/1997-12/2025': 'Jerusalem'
}

IMS_STATION_RANGE_PATTERN = re.compile(r"\d{2}/\d{4}-\d{2}/\d{4}$")


def noga_csv() -> None:
    xlsx = pd.read_excel(
        "data/noga.xlsx",
        skiprows=1)

    xlsx.rename(columns=HEADER_TRANSLATIONS, inplace=True)
    xlsx.to_csv("data/noga.csv", index=False)


def ims_csv() -> None:
    ims = pd.read_csv("data/ims.he.csv", encoding="utf-8")
    ims.rename(columns=IMS_HEADER_TRANSLATIONS, inplace=True)
    ims["station"] = ims["station"].astype(str).map(
        lambda s: IMS_STATION_TRANSLATIONS.get(s, s))

    values = [
        col for col in ims.columns
        if col not in {"datetime_winter_time", "station"}]

    ims = ims.pivot(
        index=["datetime_winter_time"],
        columns="station",
        values=values)

    ims.columns = ['_'.join(col).strip() for col in ims.columns.values]

    ims.reset_index(inplace=True)

    dt = pd.to_datetime(
        ims["datetime_winter_time"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )
    ims.insert(1, "date", dt.dt.strftime("%d-%m-%Y"))
    ims.insert(2, "time", dt.dt.strftime("%H:%M"))
    ims.drop(columns=["datetime_winter_time"], inplace=True)

    ims.to_csv("data/ims.csv", index=False)


def data_csv() -> None:
    noga = pd.read_csv("data/noga.csv")
    ims = pd.read_csv("data/ims.csv")

    noga["date"] = pd.to_datetime(
        noga["date"],
        format="%Y-%m-%d").dt.strftime("%d-%m-%Y")

    noga["time"] = pd.to_datetime(
        noga["time"],
        format="%H:%M:%S").dt.strftime("%H:%M")

    ims["date"] = pd.to_datetime(
        ims["date"],
        format="%d-%m-%Y").dt.strftime("%d-%m-%Y")

    ims["time"] = pd.to_datetime(
        ims["time"],
        format="%H:%M").dt.strftime("%H:%M")

    data = pd.merge(noga, ims, on=["date", "time"], how="inner")

    # TO NUMERIC
    data['day-ahead-forecast'] = pd.to_numeric(
        data['day-ahead-forecast'],
        errors='coerce')

    data['actual-demand'] = pd.to_numeric(
        data['actual-demand'],
        errors='coerce')

    data = data.dropna()

    # TIME FEATURES
    date = pd.to_datetime(
        data['date'],
        format="%d-%m-%Y")

    data['year'] = date.dt.year
    data['month'] = date.dt.month
    data['day'] = (date.dt.dayofweek + 1) % 7

    data['hour'] = pd.to_datetime(
        data['time'],
        format="%H:%M").dt.hour

    cols = [
        # TIME
        # 'date',
        # 'time',
        'year',
        'month',
        'day',
        'hour',
        # TEMPERATURE
        'temperature_c_Haifa',
        'temperature_c_Jerusalem',
        'temperature_c_Tel Aviv',
        'wet_bulb_temperature_c_Haifa',
        'wet_bulb_temperature_c_Jerusalem',
        'wet_bulb_temperature_c_Tel Aviv',
        'dew_point_temperature_c_Haifa',
        'dew_point_temperature_c_Jerusalem',
        'dew_point_temperature_c_Tel Aviv',
        # HUMIDITY
        'relative_humidity_percent_Haifa',
        'relative_humidity_percent_Jerusalem',
        'relative_humidity_percent_Tel Aviv',
        # WIND
        'wind_direction_degrees_Haifa',
        'wind_direction_degrees_Jerusalem',
        'wind_direction_degrees_Tel Aviv',
        'wind_speed_m_s_Haifa',
        'wind_speed_m_s_Jerusalem',
        'wind_speed_m_s_Tel Aviv',
        # DEMAND
        'day-ahead-forecast',
        'updated-demand-forecast',
        'actual-demand']

    data = data[cols]
    data.to_csv("data/data.csv", index=False)


if __name__ == "__main__":
    data_csv()
