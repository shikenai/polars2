import math

import matplotlib
import polars as pl
import matplotlib.pyplot as plt


def calculate_growth_rate(_df):
    # 成長率zの計算式
    print(_df)
    x = _df['Close'].iloc[0]
    y = _df['Close'].iloc[-1]
    print(x)
    print(y)
    n = _df.shape[0]
    z = math.pow(y / x, 1 / n)
    return z


def normalize_growth_rate(_df: pl.DataFrame, growth_rate):
    _df = _df.with_columns((pl.lit(growth_rate)).alias('test'))
    _df = _df.with_columns(_df.select('Close').with_row_count('num'))
    _df = _df.with_columns(pl.col('test') ** pl.col('num'))
    _df = _df.with_columns((pl.col('Close') / pl.col('test')).round(1).alias('unti'))
    x = _df['Date'].to_list()
    y = _df['unti'].to_list()
    matplotlib.use('TkAgg')
    plt.plot(x, y, marker='o', linestyle='-')
    plt.show()
    return _df
