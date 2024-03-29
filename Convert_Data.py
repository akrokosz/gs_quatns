import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def strip_date(df):
    df['Month'] = df['Date'].dt.month
    df = df.drop(columns=['Date'])
    return df


def split_by_attr(attr: str, data_train: pd.DataFrame):
    X_data_train = data_train[[col for col in data_train.columns if col != attr]]
    y_data_train = data_train[attr]
    return X_data_train, y_data_train


def numerize(data: pd.Series):
    unique_value = data.unique()
    string_to_int = {value: i for i, value in enumerate(unique_value)}
    data = data.map(string_to_int)
    return data


def get_train_data(path: str) -> pd.DataFrame:
    # WCZYTANIE Z PLIKU
    data_train = pd.read_csv(path, sep=',')
    # USTAWIENIE TYPU DATY
    data_train["Date"] = pd.to_datetime(data_train["Date"], format='%d/%m/%Y')
    # USTAWIENIE TYPU STR NA SECTOR I RATING
    data_train[["Sector", "Rating"]] = data_train[["Sector", "Rating"]].astype(str)
    return data_train


def drop_all_except(df_expanded, *exceptions):
    return df_expanded.drop([col for col in df_expanded.columns if col not in exceptions], axis=1)


def get_data_income_tax_from(param):
    data_income_tax: pd.DataFrame = pd.read_csv(param)
    data_income_tax.rename(columns={"DATE": "Date"}, inplace=True)
    data_income_tax["Date"] = pd.to_datetime(data_income_tax["Date"], format='%Y-%m-%d')
    data_income_tax = data_income_tax[data_income_tax["Date"].dt.year >= 2005]
    new_rows = []
    for _, row in data_income_tax.iterrows():
        year = row['Date'].year
        day = row['Date'].day
        original_month = row['Date'].month
        for month in range(1, 13):
            new_row = row.copy()
            new_row['Date'] = pd.Timestamp(year=year, month=month, day=day)
            new_rows.append(new_row)
    df_expanded = pd.DataFrame(new_rows)
    df_expanded.sort_values(by='Date', inplace=True)
    df_expanded["Year-Month"] = df_expanded["Date"].dt.to_period('M')
    df_expanded = drop_all_except(df_expanded, "Year-Month", "Date", "IITTRHB")
    return df_expanded


def get_data_cfnfci(param):
    data_cfnfci = pd.read_csv(param)
    data_cfnfci.rename(columns={"Friday_of_Week": "Date"}, inplace=True)
    data_cfnfci["Date"] = pd.to_datetime(data_cfnfci["Date"], format='%m/%d/%Y')
    data_cfnfci = data_cfnfci[data_cfnfci["Date"].dt.year >= 2005]
    data_cfnfci["Year-Month"] = data_cfnfci["Date"].dt.to_period('M')
    data_cfnfci = data_cfnfci.groupby(data_cfnfci['Year-Month']).mean()
    data_cfnfci = drop_all_except(data_cfnfci, "NFCI", "Date", "Year-Month")
    return data_cfnfci


def get_data_cfnai_series(param):
    data_cfnai_series = pd.read_csv(param)
    data_cfnai_series["Date"] = pd.to_datetime(data_cfnai_series["Date"], format='%Y/%m')
    data_cfnai_series = data_cfnai_series[data_cfnai_series["Date"].dt.year >= 2005]
    data_cfnai_series["Year-Month"] = data_cfnai_series["Date"].dt.to_period('M')
    data_cfnai_series = drop_all_except(data_cfnai_series, "CFNAI", "Date", "Year-Month")
    return data_cfnai_series


def predict_missing_values_except(data: pd.DataFrame, excpetions: list[str]):
    imputer = IterativeImputer(random_state=0)
    imputed = imputer.fit_transform(data[[col for col in data.columns if col not in excpetions]])
    data[[col for col in data.columns if col not in excpetions]] = pd.DataFrame(imputed, columns=[
        [col for col in data.columns if col not in excpetions]])
    return data


def convert_train_data(data_train: pd.DataFrame) -> tuple[DataFrame, Series]:
    # X_train, y_train = split_by_attr("Rating", data_train)
    data_train["Rating"] = numerize(data_train["Rating"])

    data_train["Year-Month"] = data_train["Date"].dt.to_period('M')

    data_train = strip_date(data_train)

    data_train = pd.get_dummies(data_train, columns=["Month"], prefix=["Month"])
    data_train = pd.get_dummies(data_train, columns=['Sector'], prefix='Sector')

    data_income_tax = get_data_income_tax_from("Dataset/income_tax.csv")

    data_train = pd.merge(data_train, data_income_tax, on="Year-Month", how='inner')

    data_cfnfci = get_data_cfnfci("Dataset/cfnfci.csv")
    data_train = pd.merge(data_train, data_cfnfci, on='Year-Month', how='inner')

    data_cfnai_series = get_data_cfnai_series("Dataset/CFNAI.csv")
    data_train = pd.merge(data_train, data_cfnai_series.drop("Date", axis=1), on='Year-Month', how='inner')

    data_train = data_train.drop(["Date_x", "Date_y"], axis=1)
    # data_train = predict_missing_values_except(data_train, ['Date', 'Year-Month'])

    data_train = data_train.drop(["Year-Month"], axis=1)

    X_train, y_train = split_by_attr("Rating", data_train)

    return X_train, y_train


def get_test_data(path):
    data_test = pd.read_csv(path, sep=',')
    data_test["Date"] = pd.to_datetime(data_test["Date"], format='%d/%m/%Y')
    data_test["Sector"] = data_test["Sector"].astype(str)
    return data_test

    # X_data_test = get_test_data("Dataset/test_fixed.csv")


def convert_test_data(X_data_test: pd.DataFrame) -> pd.DataFrame:
    X_data_test["Year-Month"] = X_data_test["Date"].dt.to_period('M')

    X_data_test = strip_date(X_data_test)

    X_data_test = pd.get_dummies(X_data_test, columns=["Month"], prefix=["Month"])
    X_data_test = pd.get_dummies(X_data_test, columns=['Sector'], prefix='Sector')

    data_income_tax = get_data_income_tax_from("Dataset/income_tax.csv")

    X_data_test = pd.merge(X_data_test, data_income_tax, on="Year-Month", how='inner')

    data_cfnfci = get_data_cfnfci("Dataset/cfnfci.csv")
    X_data_test = pd.merge(X_data_test, data_cfnfci, on='Year-Month', how='inner')

    data_cfnai_series = get_data_cfnai_series("Dataset/CFNAI.csv")
    X_data_test = pd.merge(X_data_test, data_cfnai_series.drop("Date", axis=1), on='Year-Month', how='inner')

    X_data_test = X_data_test.drop(["Date_x", "Date_y"], axis=1)
    # data_train = predict_missing_values_except(data_train, ['Date', 'Year-Month'])

    X_data_test = X_data_test.drop(["Year-Month"], axis=1)

    return X_data_test
