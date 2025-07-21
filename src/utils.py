import numpy as np
import pandas as pd
from weka.core.dataset import Instances, Instance
from weka.core.dataset import Attribute
from functools import wraps
from sklearn.utils import check_array
from sklearn.utils.validation import check_consistent_length
from sklearn.metrics import root_mean_squared_error, mean_absolute_error


# ========== Preprocessing Functions ==========
# Time-Series splitter
def split_time_series(data: pd.DataFrame, train_size: float = 0.8, date_col: str = 'Date') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits time series data into training and testing sets based on unique years.

    Parameters:
        data (pd.DataFrame): Input DataFrame with a datetime column.
        train_size (float): Proportion of years to include in the training set (default is 0.8).
        date_col (str): Name of the datetime column (default is 'Date').

    Returns:
        pd.DataFrame: Training set containing the first N years.
        pd.DataFrame: Testing set containing the remaining years.
    """
    years = data[date_col].dt.year.unique()
    total_years = len(years)
    train_years_count = int(train_size * total_years)

    train_years = years[:train_years_count]
    test_years = years[train_years_count:]

    train = data[data[date_col].dt.year.isin(train_years)].copy()
    test = data[data[date_col].dt.year.isin(test_years)].copy()

    return train, test


# Pandas DataFrame to Weka Instances (last column = target)
def pandas_to_weka_instances(df: pd.DataFrame, relation_name="data") -> Instances:
    attributes = [Attribute.create_numeric(col) for col in df.columns]
    dataset = Instances.create_instances(relation_name, attributes, 0)
    for row in df.itertuples(index=False):
        inst = Instance.create_instance(list(row))
        dataset.add_instance(inst)
    dataset.class_index = df.shape[1] - 1  # target is last column
    return dataset


## ========== Feature Engineering ==========
# Add base flow separation
def add_base_flow(data, *, BFI_max, a, flow_col='Qt', date_col='Date'):
    """
    Estimates and adds base flow ('BFt') to the DataFrame using a recursive digital filter method.
    
    Parameters:
        data (pd.DataFrame): Input DataFrame containing streamflow and date columns.
        BFI_max (float): Maximum base flow index (BFImax), typically site-specific.
        a (float): Filter parameter controlling the recession rate (default is 0.98).
        flow_col (str): Name of the streamflow column (default is 'Qt').
        date_col (str): Name of the datetime column (default is 'Date').
    
    Returns:
        pd.DataFrame: A copy of the input DataFrame with an added 'BFt' column representing base flow.
    """
    data = data.copy()
    
    def baseflow_recursive_filter(Qt):
        bft = [0]
        for k in range(1, len(Qt)):
            yk = Qt.iloc[k]
            bk_1 = bft[k - 1]
            bk = ((1 - BFI_max) * a * bk_1 + (1 - a) * BFI_max * yk) / (1 - a * BFI_max)
            bft.append(min(yk, bk))
        return pd.Series(bft, index=Qt.index)
    
    data['BFt'] = data.groupby(data[date_col].dt.year)[flow_col].transform(baseflow_recursive_filter).round(1).values
    return data


# Add moving averages
def add_moving_averages(data, *, lags=[3, 5, 7], flow_col='Qt', date_col='Date'):
    """
    Adds shifted rolling mean columns (moving averages) of streamflow to the DataFrame.
    
    For each lag in the list, computes a backward-looking moving average of the streamflow
    (shifted by 1 time step), grouped by year. The resulting columns are named 'QMOV{n}'.
    
    Parameters:
        data (pd.DataFrame): Input DataFrame containing streamflow and datetime columns.
        lags (list of int): List of lag window sizes (in time steps) to compute moving averages for.
        flow_col (str): Name of the streamflow column (default is 'Qt').
        date_col (str): Name of the datetime column (default is 'Date').

    Returns:
        pd.DataFrame: A copy of the input DataFrame with added 'QMOV{n}' columns.
    """
    data = data.copy()

    for n in lags:
        data[f'QMOV{n}'] = (
            data.groupby(data[date_col].dt.year)[flow_col]
              .apply(lambda x: x.shift(-n//2+1).rolling(n).mean())
              .bfill().ffill()
              .values.round(1)
        )
    return data


# Add lagged variables
def add_lagged_variables(data, *, vars=['Qt', 'Rt', 'BFt'], lags=[3, 3, 3], date_col='Date'):
    """
    Adds lagged versions of selected variables to the DataFrame, grouped by year.
    
    For each variable in `vars`, creates columns with values lagged by 1 to `n` time steps,
    where `n` is specified in the corresponding `lags` list. Grouping is done by year.

    Parameters:
        data (pd.DataFrame): Input DataFrame with time series and a datetime column.
        vars (list of str): List of variable names to lag (default is ['Qt', 'Rt', 'BFt']).
        lags (list of int): List of lag counts for each variable (must match length of `vars`).
        date_col (str): Name of the datetime column (default is 'Date').

    Returns:
        pd.DataFrame: A copy of the input DataFrame with added lagged variable columns.
    """
    if len(vars) != len(lags):
        raise ValueError("Length of 'vars' must match length of 'lags'.")

    data = data.copy()

    for var, max_lag in zip(vars, lags):
        for i in range(1, max_lag + 1):
            data[f'{var}-{i}'] = (
                data.groupby(data[date_col].dt.year)[var]
                  .apply(lambda x: x.shift(i))
                  .bfill()
                  .values
            )
    return data



# ========== Metrics ==========
def validate_inputs(func):
    @wraps(func)
    def wrapper(y_true, y_pred, *args, **kwargs):
        y_true = check_array(y_true, ensure_2d=False, dtype='float64')
        y_pred = check_array(y_pred, ensure_2d=False, dtype='float64')
        check_consistent_length(y_true, y_pred)
        return func(y_true, y_pred, *args, **kwargs)
    return wrapper


# Fractional Standard Error, FSE
@validate_inputs
def fractional_standard_error(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    mean_qobs = np.mean(y_true)
    if mean_qobs == 0:
        raise ValueError("Mean of observed values is zero; FSE is undefined.")
    return rmse / mean_qobs

# Nash–Sutcliffe Efficiency, NSE
@validate_inputs
def nash_sutcliffe_efficiency(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator == 0:
        raise ValueError("The variance of y_true is zero; NSE is undefined.")
    return 1 - (numerator / denominator)

# Multiplicative Bias, MB
@validate_inputs
def multiplicative_bias(y_true, y_pred):
    sum_sim = np.sum(y_pred)
    sum_obs = np.sum(y_true)
    if sum_obs == 0:
        raise ValueError("Sum of observed values is zero; MB is undefined.")
    return sum_sim / sum_obs

# Probability of Detection, POD
@validate_inputs
def probability_of_detection(y_true, y_pred, percentile=90):
    q90 = np.percentile(y_true, percentile)
    hits = (y_true >= q90) & (y_pred >= q90)
    actual_events = y_true >= q90
    if np.sum(actual_events) == 0:
        raise ValueError("No observed events exceed the Q90 threshold; POD is undefined.")
    return np.sum(hits) / np.sum(actual_events)

# False Alarm Rate, FA
@validate_inputs
def false_alarm_rate(y_true, y_pred, percentile=90):
    q90 = np.percentile(y_true, percentile)
    false_alarms = (y_true < q90) & (y_pred >= q90)
    non_events = y_true < q90
    if np.sum(non_events) == 0:
        raise ValueError("All observations exceed Q90; FA is undefined.")
    return np.sum(false_alarms) / np.sum(non_events)

# RMSE to standard deviation ratio, RSR
@validate_inputs
def rmse_to_std_ratio(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    std_obs = np.std(y_true, ddof=1)
    if std_obs == 0:
        raise ValueError("Standard deviation of observed values is zero; RSR is undefined.")
    return rmse / std_obs

# MAE to mean ratio, MMR
@validate_inputs
def mae_to_mean_ratio(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mean_obs = np.mean(y_true)
    if mean_obs == 0:
        raise ValueError("Mean of observed values is zero; MMR is undefined.")
    return mae / mean_obs

# Mean absolute relative error, MARE
@validate_inputs
def mean_absolute_relative_error(y_true, y_pred):
    q99 = np.percentile(y_true, 99)
    mask = y_true > q99
    if not np.any(mask):
        raise ValueError("No observed values greater than the 99th percentile; MARE is undefined.")
    y_true_peak = y_true[mask]
    y_pred_peak = y_pred[mask]
    
    if np.any(y_true_peak == 0):
        raise ValueError("Zero values found among peak observed values; MARE is undefined due to division by zero.")
    relative_errors = np.abs(y_pred_peak - y_true_peak) / y_true_peak
    return np.mean(relative_errors)