
from __future__ import annotations

import pandas as pd

HEADER_TRANSLATIONS = {
    "תאריך": "date",
    "שעה": "time",
    "תחזית ביקוש יום מראש": "day-ahead-forecast",
    "תחזית ביקוש עדכנית": "updated-demand-forecast",
    "ביקוש בפועל": "actual-demand",
}


def noga_csv() -> None:
    xlsx = pd.read_excel(
        "data/noga.xlsx",
        skiprows=1)

    xlsx.rename(columns=HEADER_TRANSLATIONS, inplace=True)
    xlsx.to_csv("data/noga.csv", index=False)


if __name__ == "__main__":
    noga_csv()
