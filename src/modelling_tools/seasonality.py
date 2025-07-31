import pandas as pd
import numpy as np


def add_seasonality(
    df: pd.DataFrame,
    date_col: str,
    period: float,
    fourier_order: int,
    seasonality_name: str = None,
) -> pd.DataFrame:
    """Add seasonality features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing the date column shape (N, n_cols)
        period (float): Period of the seasonality (in days)
        fourier_order (int): Order of the Fourier series to use
        date_col (str): Name of the date column in the DataFrame
        seasonality_name (str, optional): Name for the seasonality column. Defaults to None.
    """
    if seasonality_name is None:
        seasonality_name = f"seasonality_period_{period}"

    # convert each date col to nanoseconds since epoch
    dates = df[date_col].to_numpy(dtype=np.int64)
    # convert to days since epoch
    day_to_nanosec = 3600 * 24 * int(1e9)
    dates = dates / day_to_nanosec
    df["t_seasonality"] = dates

    # shape (fourier_order)
    n_vals = np.arange(1, fourier_order + 1)
    # shape (fourier order, N)
    t = np.outer(n_vals, df["t_seasonality"] * 2 * np.pi / period)

    # the actual sin and cos features
    sin_col_names = [f"{seasonality_name}_sin_{i}" for i in n_vals]
    cos_col_names = [f"{seasonality_name}_cos_{i}" for i in n_vals]
    sin_df = pd.DataFrame(np.sin(t.T), columns=sin_col_names, index=df.index)
    cos_df = pd.DataFrame(np.cos(t.T), columns=cos_col_names, index=df.index)

    out_df = pd.concat([df, sin_df, cos_df], axis=1)
    return out_df


def add_daily_seasonality(
    df: pd.DataFrame,
    date_col: str,
    fourier_order: int = 4,
    seasonality_name: str = "seasonality_daily",
):
    return add_seasonality(
        df,
        period=1,
        fourier_order=fourier_order,
        date_col=date_col,
        seasonality_name=seasonality_name,
    )


def add_weekly_seasonality(
    df: pd.DataFrame,
    date_col: str,
    fourier_order: int = 3,
    seasonality_name: str = "seasonality_weekly",
):
    return add_seasonality(
        df,
        period=7,
        fourier_order=fourier_order,
        date_col=date_col,
        seasonality_name=seasonality_name,
    )


def add_yearly_seasonality(
    df: pd.DataFrame,
    date_col: str,
    fourier_order: int = 10,
    seasonality_name: str = "seasonality_yearly",
):
    return add_seasonality(
        df,
        period=365.25,
        fourier_order=fourier_order,
        date_col=date_col,
        seasonality_name=seasonality_name,
    )
