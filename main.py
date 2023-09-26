import matplotlib
import polars as pl
import pandas as pd
import os
import datetime as dt
import time
import matplotlib.pyplot as plt
import math
import itertools
import sub


def processing(df):
    df_date = df['Attributes']
    new_columns = df.iloc[0].tolist()
    df.columns = new_columns
    schema = [('count', pl.Float64),
              ('mean', pl.Float64),
              ('std', pl.Float64),
              ('min', pl.Float64),
              ('25%', pl.Float64),
              ('50%', pl.Float64),
              ('75%', pl.Float64),
              ('max', pl.Float64),
              ('brand', pl.Utf8),
              ('signX', pl.Utf8)]
    # last_df = pl.DataFrame({}, ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'brand', 'signX'])
    last_df = pl.DataFrame({}, schema=schema)
    for i in range(1, ((len(new_columns) - 1) // 5) + 1):
    # for i in range(1, 3):
        ignore_list = ['5830.jp', '5831.jp', "5832.jp", "9147.jp"]
        brand_code = df.columns[i]
        if brand_code in ignore_list:
            continue
        extracted_df = pd.concat([df_date, df[brand_code]], axis=1)
        extracted_df = extracted_df.drop([0, 1])
        extracted_df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        extracted_df = extracted_df.astype(
            {'Close': float, 'High': float, 'Open': float, 'Low': float, 'Volume': float})
        extracted_df['Date'] = pd.to_datetime(extracted_df['Date']).dt.date
        extracted_df = extracted_df.reset_index(drop=True)
        extracted_df = extracted_df.drop(["Volume"], axis=1)
        extracted_df = add_macd(extracted_df)
        # extracted_df = add_rsi(extracted_df)
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.max_columns', 500)
        pd.options.display.float_format = '{:.2f}'.format
        extracted_pl = pl.DataFrame(extracted_df)
        extracted_pl = add_expected_profit(extracted_pl)
        extracted_pl = check_close_value_transition(extracted_pl)

        extracted_pl = delete_outlier(extracted_pl, "Close_diff", 2)
        extracted_pl = sub.set_yesterday_entity(extracted_pl)
        days_list = [5, 10, 25, 60]
        extracted_pl = add_ma(extracted_pl, days_list, "Close")
        extracted_pl = set_macd_sign(extracted_pl)

        extracted_pl = extracted_pl.drop_nulls()
        # extracted_pl = extracted_pl.filter((extracted_pl['signX'] == 2) & (extracted_pl['entity_rate'] > 0))
        # extracted_pl = extracted_pl.filter(extracted_pl['signX'] == -2)
        temp_pl = extracted_pl['Close_diff'].describe()

        sign_list = [2, 1, 0, -1, -2]
        for sign in sign_list:
            filtered_pl = extracted_pl.filter(extracted_pl['signX'] == sign)
            print(brand_code)
            # print(filtered_pl['Close_diff'].describe()['value'])
            # print(filtered_pl['Close_diff'].describe()['value'].alias(str(sign)))
            # print(filtered_pl['Close_diff'])
            described_pl = filtered_pl['Close_diff'].describe()['value'].alias(str(sign))
            temp_pl = temp_pl.hstack([described_pl])
        temp_pl = temp_pl.drop('statistic').transpose()

        temp_pl = temp_pl.with_columns(pl.lit(brand_code).alias('brand'))
        temp_pl = temp_pl.rename(
            {"column_0": "count", "column_2": "mean", "column_3": "std", "column_4": "min", "column_5": "25%",
             "column_6": "50%", "column_7": "75%", "column_8": "max"}).drop('column_1')
        sign_lists = pl.Series(["all", "2", "1", "0", "-1", "-2"])

        temp_pl = temp_pl.with_columns((sign_lists).alias('signX'))

        last_df = pl.concat([last_df, temp_pl])
    print(last_df)
    last_df.write_csv('test.csv')
    print('done')


def set_macd_sign(_df):
    # print('sign')
    target_columns = 'macd_hist_3ma_diff'
    positive_columns = [c for c in _df.columns if c.endswith('MA_positive')]
    for n in range(4, 0, -1):
        # print(n)
        m = n - 1
        _df = _df.with_columns(pl.col(target_columns).shift(m).alias(f'sign{m}'))
    # _df = _df.with_columns((((pl.col('sign0') - pl.col('sign3'))/3).round(2) + pl.col('sign0')).alias('sign_next'))

    _df = _df.with_columns(
        pl.when((_df[positive_columns[0]] == 1) & (_df[positive_columns[1]] == 1) & (_df[positive_columns[2]] == 1) & (
                _df[positive_columns[3]] == 1) & (_df['sign0'] > _df['sign1']) & (_df['sign1'] > _df['sign2']) & (
                        _df['sign2'] > _df['sign3']) & (
                        _df['sign2'] < 0) & (_df['sign1'] > 0)).then(1)
            .when(
            (_df[positive_columns[0]] == 1) & (_df[positive_columns[1]] == 1) & (_df[positive_columns[2]] == 1) & (
                    _df[positive_columns[3]] == 1) & (_df['sign0'] > _df['sign1']) & (
                    _df['sign1'] > _df['sign2']) & (_df['sign2'] > _df['sign3']) & (
                    _df['sign1'] < 0) & (_df['sign0'] > 0)).then(2)
            .when(
            (_df[positive_columns[0]] == -1) & (_df[positive_columns[1]] == -1) & (_df[positive_columns[2]] == -1) & (
                    _df[positive_columns[3]] == -1) & (_df['sign0'] < _df['sign1']) & (
                    _df['sign1'] < _df['sign2']) & (_df['sign2'] < _df['sign3']) & (
                    _df['sign2'] > 0) & (_df['sign1'] < 0)).then(-1)
            .when(
            (_df[positive_columns[0]] == -1) & (_df[positive_columns[1]] == -1) & (_df[positive_columns[2]] == -1) & (
                    _df[positive_columns[3]] == -1) & (_df['sign0'] < _df['sign1']) & (
                    _df['sign1'] < _df['sign2']) & (_df['sign2'] < _df['sign3']) & (
                    _df['sign1'] > 0) & (_df['sign0'] < 0)).then(-2)
            .otherwise(0).alias('signX')
    )
    # print(_df[['Close_diff',
    #           'sign3', 'sign2', 'sign1', 'sign0', 'signX', 'Close_5MA_positive', 'Close_10MA_positive', 'Close_25MA_positive', 'Close_60MA_positive', '1days_profit', '2days_profit', '3days_profit']])
    # print('this')
    # print(_df.columns)
    return _df


def test(_df):
    _list = [1, 2, -1, -2, 0]
    _days = [1, 2, 3]
    # matplotlib.use('inline')
    # ['GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo',
    #  'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']

    matplotlib.use('TkAgg')
    for n in _list:
        filtered_df = _df.filter(_df['signX'] == n)
        # print('sign:' + str(n))
        # print(filtered_df[['1days_profit', '2days_profit', '3days_profit']].describe())
        # plt.title('sign:' + str(n))
        # plt.hist(filtered_df['1days_profit'], alpha=0.3)
        # plt.hist(filtered_df['2days_profit'], alpha=0.3)
        # plt.hist(filtered_df['3days_profit'], alpha=0.3)
        # plt.legend()
        # plt.show()

        # print(filtered_df[f'{d}days_profit'].describe().mean())
        # print(_df[f'{d}days_profit'].describe().mean())
    # print('all')
    # print(_df[['1days_profit', '2days_profit', '3days_profit']].describe())
    # # print(_df[['1days_profit','2days_profit','3days_profit']])
    # print(_df)


def filtering(_df):
    all_columns = _df.columns
    # print(all_columns)
    # targets = [c for c in all_columns if ">" in c]
    # print(targets)
    # targets = [c for c in targets if "sign" not in c]
    # print(targets)
    target_col = 'signX'
    signals = [2, 1, 0, -1, -2]
    for signal in signals:
        filtered_df = _df.filter(_df[target_col] == signal)
        # print(filtered_df.describe())


def add_expected_profit(_df, sign_col='macd_sign'):
    """
    趣旨：ある列(sign_col)をサイン（１：購入、ー１：売却）として参照し、
    　　　そのサインが観測された翌日に成り行きで購入・売却したとき、
    　　　n日後の終値がいくらになっているかを列で取得する。
    　　　シンプルな取引（成り行き売買）でどの程度の利益が見込まれそうかを判断したい。
    :param sign_col: 売買購入シグナルとする列
    :param _df:
    :return: _df
    【追加する列の解説】
    趣旨記載の「n日後」として、１～３日とする。
    """
    term = [1, 2, 3]
    for n in term:
        # _df = _df.with_columns(
        #     ((pl.col('Close')).shift(n * -1) - pl.col('Open').shift(-1)).alias(
        #         f'{n}days_profit'))
        _df = _df.with_columns(
            (((pl.col('Close').shift(n * -1) - pl.col('Open').shift(-1)) * 100 / pl.col('Open').shift(-1)).round(
                2)).alias(
                f'{n}days_profit'))
    return _df


def add_rsi(_df):
    period = 14
    delta = _df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rsi = 100 * (avg_gain / (avg_gain + avg_loss))
    _df['RSI'] = rsi
    return _df


def add_macd(_df):
    """
    :param _df:
    :return:_df
    【追加する列の解説】
    macd_hist ：いわゆるmacdのヒストグラム。誤解を恐れずに言えば、負数であれば減少傾向で、正数であれば上昇傾向にあるといえる
    macd_hist_3ma_diff :macdのヒストグラムの差分をとり、その差分の3日移動平均を取ったもの。より短期間の「傾向」を判断するために取っている指標
    macd_hist_positive :macd_hist_3ma_positiveが正数であるか負数であるかを判断するための列
    macd_hist_positive_continual :macd_hist_positiveが、正数・負数ともに、どの程度継続しているのかを判断するための列
    macd_sign :売買シグナルにしようとしているもの。長期的な傾向と、逆の方向の短期的傾向を示した場合に、数字が入力される
    　　　　　　　１の場合、購入サインで、-1の場合、売却サインといえる
    """
    es = 12
    el = 26
    sg = 9
    # print(_df)

    ema12 = round(_df['Close'].ewm(span=es, adjust=False).mean() * 100 / _df['Close'], 2)
    ema26 = round(_df['Close'].ewm(span=el, adjust=False).mean() * 100 / _df["Close"], 2)
    macd = ema12 - ema26
    sig = round(macd.ewm(span=sg, adjust=False).mean() * 100 / _df["Close"], 3)
    hist = macd - sig
    # macd_hist いわゆるMACDのヒストグラム。端数処理しているので若干違うかもしれんけども。
    _df['macd_hist'] = hist
    # _df['macd_hist_diff1'] = _df['macd_hist'].diff()
    # _df['macd_hist_diff2'] = _df['macd_hist'].diff().rolling(window=3).mean()
    _df['macd_hist_3ma_diff'] = _df['macd_hist'].diff().rolling(window=3).mean().round(3) * 10

    # macd_positive macd_histが、正の方向であれば1を、負の方向であれば-1を持つ。
    _df['macd_hist_positive'] = _df['macd_hist_3ma_diff'].apply(lambda x: 1 if x > 0 else -1)
    # _df = _df.drop(['macd_hist_diff'], axis=1)
    val = 0
    result = []
    for sign in _df['macd_hist_positive']:
        if sign == 1:
            if val < 0:
                val = 1
            else:
                val += 1
        elif sign == -1:
            if val > 0:
                val = -1
            else:
                val -= 1
        result.append(val)
    # macd_continual は、macd_positiveが、どれほど正（あるいは負）の状態を続けているのかを示すもの。
    _df['macd_hist_positive_continual'] = result
    new_df = _df.iloc[1:]
    mask = (new_df["macd_hist_positive_continual"] == 1) | (new_df["macd_hist_positive_continual"] == -1)
    first_index = mask.idxmax()
    # print(first_index)
    _df = _df[first_index:].reset_index(drop=True)
    _df['macd_trend_conversion'] = 0
    _df.loc[(_df["macd_hist"] < 0) & (_df['macd_hist_positive_continual'] > 0), 'macd_trend_conversion'] = 1
    _df.loc[(_df["macd_hist"] > 0) & (_df['macd_hist_positive_continual'] < 0), 'macd_trend_conversion'] = -1

    # print(_df)
    return _df


def check_close_value_transition(_df):
    _df = _df.with_columns((pl.col('Close').diff(n=1) * 100 / pl.col('Close')).round(2).alias('Close_diff'))
    # _df = _df.with_columns(pl.col('Close').diff(n=1).alias('Close_diff'))
    _df = _df.with_columns(pl.col('Close_diff').diff(n=1).alias('Close_dd'))
    # print(_df[['Close', 'Close_diff', 'Close_dd']])
    # matplotlib.use('TkAgg')
    # plt.figure(figsize=(10, 10))
    # # plt.plot(_df['Close'], label='Close')
    # # plt.plot(_df['Close_diff'], label='Close_diff')
    # # plt.plot(_df['Close_dd'], label='Close_dd')
    # plt.scatter(_df['Close_dd'], _df['Close_diff'])
    # plt.hist(_df['Close_diff'])
    # plt.legend()
    # plt.show()
    # statistics = _df['Close_diff'].agg([
    #     pl.sum('sum'),
    #     pl.mean('mean'),
    #     pl.median('median'),
    #     pl.min('min'),
    #     pl.max('max'),
    #     pl.std('std')
    # ])
    # print(statistics)

    return _df


def add_ma(_df, days_list, col_name):
    ma_col_name_list = []
    for days in days_list:
        ma_col_name = f"{col_name}_{str(days)}MA"
        ma_col_name_list.append(ma_col_name)
        _df = _df.with_columns(
            [(pl.col(col_name).rolling_mean(window_size=days, min_periods=days).alias(
                ma_col_name).round(1))])
        diff_name = ma_col_name + "_diff"
        positive_col_name = ma_col_name + "_positive"
        _df = _df.with_columns(
            (pl.col(ma_col_name).diff(n=1) * 100 / pl.col(ma_col_name)).round(3).alias(diff_name))

        _df = _df.with_columns(
            [pl.when(_df[diff_name] > 0).then(1).otherwise(-1).alias(positive_col_name)])

    return _df


def check_ma_gxdx(_df):
    ma_list = [c for c in _df.columns if c.endswith('MA')]
    targets = list(itertools.combinations(ma_list, 2))
    for t in targets:
        gxdx_name = t[0].replace('Close_', "") + ">" + t[1].replace('Close_', "")
        _df = _df.with_columns(pl.when(_df[t[0]] > _df[t[1]]).then(1).otherwise(-1).alias(
            gxdx_name))
        shifted_name = 'shifted_' + gxdx_name
        _df = _df.with_columns(pl.col(gxdx_name).shift().alias(shifted_name))
        sign_name = 'sign_' + gxdx_name
        _df = _df.with_columns(pl.when((pl.col(gxdx_name) == 1) & (pl.col(shifted_name) == 1)).then(2)
                               .when((pl.col(gxdx_name) == 1) & (pl.col(shifted_name) == -1)).then(1)
                               .when((pl.col(gxdx_name) == -1) & (pl.col(shifted_name) == -1)).then(-2)
                               .when((pl.col(gxdx_name) == -1) & (pl.col(shifted_name) == 1)).then(-1).alias(sign_name))
        _df = _df.drop(shifted_name)

    return _df


def delete_outlier(_df, target_col, sigma):
    mean = _df[target_col].mean()
    std = _df[target_col].std()

    upper_standard = mean + sigma * std
    downer_standard = mean - sigma * std

    _df = _df.filter((_df[target_col] < upper_standard) & (_df[target_col] > downer_standard))

    return _df


def set_debugger(target_df):
    target_df = target_df.with_columns((pl.col('Close').diff(n=1) * 100 / pl.col("Close")).round(2).alias('Close_diff'))
    target_df = target_df.drop_nulls()
    a = target_df.get_column("Close_diff")
    # print(a.hist(bin_count=4))
    # target_df = target_df.with_columns(pl.col('Close_diff').diff(n=1).diff(n=1).alias('debugger'))
    return target_df


def evaluate(target_df):
    targets = [1, -1]
    lists = [c for c in target_df.columns if c.endswith('MA_positive')]
    for t in targets:
        extracted_df = target_df.filter(pl.col('macd_sign') == t)
        extracted_df_positive = extracted_df.filter(pl.col('Close_3MA_positive') == 1)
        plt.hist(extracted_df_positive['Close-Open'], bins=20)
        plt.title(f'macd_sign: ' + str(t) + "3ma positive")
        plt.show()

        extracted_df_negative = extracted_df.filter(pl.col('Close_3MA_positive') == -1)
        plt.hist(extracted_df_negative['Close-Open'], bins=20)
        plt.title(f'macd_sign: ' + str(t) + "3ma negative")
        plt.show()


def calculate_growth_rate(_df):
    # 成長率zの計算式
    x = _df.select('Close')[0]
    y = _df.select('Close')[-1]
    n = _df.shape[0]
    z = math.pow(y / x, 1 / n)
    return z


def calculate_average(a, b, c, d):
    diff1 = a - b
    diff2 = b - c
    diff3 = c - d
    average = round((diff1 + diff2 + diff3) / 3, 2)
    return average


pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(50)
pl.Config.set_tbl_width_chars(200)

base_data_path = os.path.join(os.getcwd(), "data", "base")
_file_name = os.listdir(base_data_path)
file_name = [s for s in _file_name if "topix" in s]
file_path = os.path.join(base_data_path, file_name[0])
_df = pd.read_csv(file_path)

t1 = time.time()
processing(_df)

print(time.time() - t1)
