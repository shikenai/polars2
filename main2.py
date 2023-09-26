import matplotlib
import polars as pl
import pandas as pd
import os
import datetime as dt
import time
import matplotlib.pyplot as plt
import math
import itertools
import sub2

def processing(df):
    df_date = df['Attributes']
    new_columns = df.iloc[0].tolist()
    df.columns = new_columns
    # for i in range(1, ((len(new_columns) - 1) // 5) + 1):
    for i in range(1, 2):
        brand_code = df.columns[i]
        extracted_df = pd.concat([df_date, df[brand_code]], axis=1)
        extracted_df = extracted_df.drop([0, 1])
        extracted_df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        extracted_df = extracted_df.astype(
            {'Close': float, 'High': float, 'Open': float, 'Low': float, 'Volume': float})
        extracted_df['Date'] = pd.to_datetime(extracted_df['Date']).dt.date
        extracted_df = extracted_df.drop(["Volume"], axis=1)
        extracted_df = extracted_df.dropna()
        extracted_df = extracted_df.reset_index(drop=True)

        growth_rate = sub2.calculate_growth_rate(extracted_df)
        extracted_pl = pl.DataFrame(extracted_df)
        extracted_pl = sub2.normalize_growth_rate(extracted_pl, growth_rate)
        print(extracted_pl)
        print(growth_rate)

        return extracted_pl


pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(50)
pl.Config.set_tbl_width_chars(200)
t1 = time.time()

base_data_path = os.path.join(os.getcwd(), "data", "base")
_file_name = os.listdir(base_data_path)
file_name = [s for s in _file_name if "topix" in s]
file_path = os.path.join(base_data_path, file_name[0])
_df = pd.read_csv(file_path)

df = processing(_df)
# print(df)

print(time.time() - t1)
