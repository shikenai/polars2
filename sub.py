import polars as pl
import matplotlib.pyplot as plt


def set_yesterday_entity(df: pl.DataFrame):
    df = df.with_columns(
        (((pl.col('Close') - pl.col('Open')) * 100) / pl.col('Close')).round(2).shift(1).alias('entity_rate'))
    # print(df[['1days_profit', 'entity_rate']])

    return df
