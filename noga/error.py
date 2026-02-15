import pandas as pd


def error_csv():
    columns = [
        'year',
        'month',
        'day',
        'hour',
        'day-ahead-forecast',
        'actual-demand']

    data = pd.read_csv(
        "data/data.csv",
        usecols=columns)

    data = data[data['year'] == 2023]

    data['error'] = data['day-ahead-forecast'] - data['actual-demand']
    data['abs-error'] = data['error'].abs()
    data['error-percentage'] = data['abs-error'] / data['actual-demand'] * 100

    data.to_csv("data/error.csv", index=False)


def error_stat():
    data = pd.read_csv("data/error.csv")
    by_month = data.groupby("month")
    err_by_month = by_month[["abs-error", "error-percentage"]].mean()
    err_by_month.to_csv("data/error_by_month.csv")
    # TODO make a bar chart of error by day of week
    # save it to plots.


if __name__ == "__main__":
    error_csv()
    error_stat()
