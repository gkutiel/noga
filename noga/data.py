
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
        errors='coerce').interpolate()

    noga['actual'] = pd.to_numeric(
        noga['actual-demand'],
        errors='coerce').interpolate()

    noga = noga.drop(columns=[
        'day-ahead-forecast',
        'actual-demand',
        'updated-demand-forecast'
    ])

    noga.to_csv("data/noga.fixed.csv", index=False)


def ims_fixed_csv():
    ims = pd.read_csv("data/ims.csv")

    ims['date'] = pd.to_datetime(ims['date'], format='%m-%d-%Y')
    ims['time'] = pd \
        .to_timedelta(ims['time'] + ':00') \
        .dt.total_seconds() / 60

    ims.to_csv("data/ims.fixed.csv", index=False)


def data_csv():
    noga = pd.read_csv("data/noga.fixed.csv")
    ims = pd.read_csv("data/ims.fixed.csv")


if __name__ == "__main__":
    print('-' * 10)
    ims_fixed_csv()
