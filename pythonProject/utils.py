import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean
from mlforecast.lgb_cv import LightGBMCV
from mlforecast.target_transforms import Differences, LocalStandardScaler

from utilsforecast.losses import mape, rmse
from utilsforecast.plotting import plot_series


def preprocess_data(df, dfbu):
    df['unique_id'] = df.but_num_business_unit.astype(str) + '_' + df.dpt_num_department.astype(str)
    # make columns names for mlforecast
    df['ds'] = pd.to_datetime(df['day_id'])
    if 'turnover' in df.columns:
        df['y'] = df.turnover

    # add business unit features
    df = pd.merge(df, dfbu, on='but_num_business_unit', how='left')

    if 'turnover' in df.columns:
        # remove outliers and negative turnover values
        y_mean = df.y.mean()
        y_std = df.y.std()
        df.y = df.y.apply(lambda y: y_mean if y > y_mean + 10 * y_std else y)
        df.y = df.y.apply(lambda y: y if y > 0 else 0)
        df = df.drop(['turnover'], axis=1)
    df = df.drop(['day_id'], axis=1)
    return df


def read_and_preprocess_data():
    """
    Read data from parquet files and preprocess it
    :return: df_train, df_test, df_train_full - train and test dataframes;
     df_train_full - train data with all series including the ones unavailable for validation
    """
    df_train_raw = pd.read_parquet('data-ds-test/data/train.parquet')
    df_test_raw = pd.read_parquet('data-ds-test/data/test.parquet')
    dfbu = pd.read_parquet('data-ds-test/data/bu_feat.parquet')
    df_train_full = preprocess_data(df_train_raw, dfbu)
    df_test = preprocess_data(df_test_raw, dfbu)
    # make pivot tables to examine individual time series
    train_pivot = df_train_full.sort_values('unique_id').pivot(index='ds', columns='unique_id', values=['y'])
    train_pivot.columns = train_pivot.columns.to_series().str.join('_')
    train_pivot = train_pivot.dropna(axis=1)

    # remove time series with no data in validation period
    good_ids = [i.replace('y_', '') for i in train_pivot[train_pivot.index > '2016-10-01'].dropna(axis=1).columns]
    df_train = df_train_full[df_train_full.apply(lambda row: row['unique_id'] in good_ids, 1)]
    return df_train, df_test, df_train_full


def wmape(df_, models, id_col="unique_id", target_col="y"):
    series_abs_sums = df_.groupby(id_col)[target_col].apply(lambda c: c.abs().sum())
    series_abs_sums.name = 'series_abs_sum'

    df = df_.join(series_abs_sums, on=id_col, how='left')
    df[models] = df[models].sub(df[target_col], axis=0).div(df['series_abs_sum'], axis=0)
    res = df.groupby(id_col)[models].apply(lambda c: c.abs().sum())
    return res


def compute_cross_val_score(cv_df, models, metric=rmse):
    res = []
    for cutoff in cv_df.cutoff.unique():
        res.append(metric(cv_df[cv_df.cutoff == cutoff], models)[models].mean(axis=0))
    return pd.concat(res, axis=1).mean(axis=1)


def make_res_df(cv_df, metric=rmse, model_name='Naive'):
    cv_df = cv_df.reset_index()
    deps = [73, 88, 117, 127]
    res = []
    overall_res = metric(cv_df, models=[model_name], id_col='cutoff')[model_name].mean()
    res.append({'dep name': 'global', model_name + ' ' + metric.__name__: overall_res})
    for dep in deps:
        dep_val = metric(cv_df[cv_df['unique_id'].str.endswith(f'_{dep}')],
                         models=[model_name], id_col='cutoff')[model_name].mean()
        res.append({'dep name': dep, model_name + ' ' + metric.__name__: dep_val})
    return pd.DataFrame(res)


def plot_cv(df, df_cv, uid):
    cutoffs = df_cv.query('unique_id == @uid')['cutoff'].unique()
    fig = make_subplots(rows=len(cutoffs), cols=1, shared_xaxes=True,
                        subplot_titles=[f"Cutoff: {cutoff}" for cutoff in cutoffs])

    actual_color = 'blue'
    predicted_color = 'red'

    for i, cutoff in enumerate(cutoffs):
        df_filtered = df.query('unique_id == @uid').set_index('ds')
        df_cv_filtered = df_cv.query('unique_id == @uid & cutoff == @cutoff').set_index('ds')

        trace1 = go.Scatter(x=df_filtered.index, y=df_filtered['y'], mode='lines', name='Actual',
                            line=dict(color=actual_color))
        trace2 = go.Scatter(x=df_cv_filtered.index, y=df_cv_filtered['LGBMRegressor'], mode='lines', name='Predicted',
                            line=dict(color=predicted_color))

        fig.add_trace(trace1, row=i + 1, col=1)
        fig.add_trace(trace2, row=i + 1, col=1)

    fig.update_layout(height=250 * len(cutoffs), width=900, title_text=uid, showlegend=True)
    fig.show()


def plot_test_predictions(df_train_full, df_test_, uuids):
    fig = make_subplots(rows=len(uuids), cols=1, shared_xaxes=True,
                        subplot_titles=uuids)

    actual_color = 'blue'
    predicted_color = 'red'
    for i, uuid in enumerate(uuids):
        df_filtered = df_train_full.query('unique_id == @uuid').set_index('ds')
        df_cv_filtered = df_test_.query('unique_id == @uuid').set_index('ds')

        trace1 = go.Scatter(x=df_filtered.index, y=df_filtered['y'], mode='lines', name='Actual',
                            line=dict(color=actual_color))
        trace2 = go.Scatter(x=df_cv_filtered.index, y=df_cv_filtered['LGBMRegressor'], mode='lines', name='Predicted',
                            line=dict(color=predicted_color))

        fig.add_trace(trace1, row=1 + i, col=1)
        fig.add_trace(trace2, row=1 + i, col=1)
    fig.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True, xaxis3_showticklabels=True)
    fig.show()
