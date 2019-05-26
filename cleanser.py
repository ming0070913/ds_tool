import sys
import re
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
from IPython.core.display import display, HTML

def clean_data(_df, settings, logging=True):
    def log(msg):
        if logging:
            print(msg)
    
    df = _df.copy()
    
    try:
        for col in settings:
            col_type, parse_format, fillna = settings[col]
            
            if col not in df.columns:
                log('[Warning] Missing {}'.format(col))
                continue

            if col_type == 'int':
                fillna = fillna if fillna != '' else '0'
                if fillna == '0':
                    fill_val = 0
                elif fillna == 'mean':
                    fill_val = int(df[col].mean())
                elif fillna == 'median':
                    fill_val = int(df[col].median())
                count_nan = df[col].isnull().sum()
                df[col] = df[col].fillna(fill_val).astype(np.float).astype(np.int32)
                log('Set {} as Interger, Repalced {} NaN to {}'.format(col, count_nan, fill_val))

            elif col_type == 'float':
                df[col] = df[col].astype(np.float)
                fillna = fillna if fillna != '' else 'mean'
                if fillna == '0.0':
                    fill_val = 0.0
                elif fillna == 'mean':
                     fill_val = df[col].mean()
                elif fillna == 'median':
                    fill_val = df[col].median()
                count_nan = df[col].isnull().sum()
                df[col] = df[col].fillna(fill_val)
                log('Set {} as Float, Repalced {} NaN to {}'.format(col, count_nan, fill_val))

            elif col_type == 'binary':
                if df[col].dtype == 'object':
                    count_nan = df[col].isnull().sum()
                    df[col] = df[col].str.strip().astype(str)
                    df.loc[df[col] == 'Y', col] = '1'
                    df.loc[df[col] == 'N', col] = '0'
                    fill_val = fillna if fillna != '' else '0'
                    df.loc[((df[col] != '1') & (df[col] != '0')), col] = fill_val
                    df[col] = df[col].astype(np.int32)
                    log('Set {} as Float, Repalced {} NaN to {}'.format(col, count_nan, fill_val))
                else:
                    log('[Warning] Binary Column {} is of type {}'.format(col, df[col].dtype))

            elif col_type == 'class':
                fill_val = fillna if fillna != '' else 'Missing'
                df[col] = df[col].fillna(fill_val).astype(np.str)
                df[col] = df[col].str.strip()
                log('Set {} as Str Class'.format(col))

            elif col_type == 'date' or col_type == 'datetime':
                parse_format = parse_format if parse_format != '' else '%Y/%m/%d'
                df[col] = pd.to_datetime(df[col], format=parse_format, errors='coerce')
                log('Set {} as Date'.format(col))

            elif col_type == 'isnull':
                df[col + '_NOT_NULL'] = 1 - (df[col].isnull()).astype(np.int32)
                df.drop(col, 1, inplace=True)
                log('Dropped {}'.format(col))

            elif col_type == 'iszero':
                df[col + '_NOT_ZERO'] = 1 - (df[col].isnull() | df[col] == 0).astype(np.int32)
                df.drop(col, 1, inplace=True)
                log('Dropped {}'.format(col))

            elif col_type == 'drop':
                df.drop(col, 1, inplace=True)
                log('Dropped {}'.format(col))

            else:
                raise 'Unknown column type'
                
    except:
        import traceback
        print('Error on: {}'.format(col))
        print(sys.exc_info())
        print(traceback.print_exc())
        return df
    
    return df

def drop_timestamp(_df, settings, logging=True):
    def log(msg):
        if logging:
            print(msg)
            
    df = _df.copy()
    
    try:
        for col in settings:
            col_type, parse_format, fillna = settings[col]
            if col_type == 'date' or col_type == 'datetime':
                if col in df.columns:
                    df.drop(col, 1, inplace=True)
                    log('Dropped {}'.format(col))
    except:
        import traceback
        print('Error on: {}'.format(col))
        print(sys.exc_info())
        print(traceback.print_exc())
        return df
    
    return df

def one_hot(_df, settings, logging=True):
    def log(msg):
        if logging:
            print(msg)
    
    df = _df.copy()
    
    try:
        for col in settings:
            col_type, parse_format, fillna = settings[col]
            
            if col_type == 'class':
                if col not in df.columns:
                    log('[Warning] Missing {}'.format(col))
                    continue

                df = pd.concat([df, pd.get_dummies(df[col], prefix=col, drop_first=True)], axis=1)
                df.drop(col, 1, inplace=True)
    
    except:
        import traceback
        print('Error on: {}'.format(col))
        print(sys.exc_info())
        print(traceback.print_exc())
        return df
    
    return df

def info(df):
    if not display:
        return
    display(df.head())
    display(df.describe(exclude=[np.number]))
    display(df.describe())

def analyze(df):
    count_records = df.shape[0]
    count_nan_col = df.isnull().sum(axis = 0)
    
    for col in df:
#         if col != 'new_cell': continue

        count_null = count_nan_col[col]
        null_perc = count_null / count_records
        
        unique_values = len(df[col].value_counts())
        class_perc = unique_values / count_records

        if null_perc >= 1.0:
            print("'{}':\t['drop', '', ''],\t# {} % null XXX".format(col, round(null_perc * 100, 1)))
            continue

        if unique_values == 1:
            print("'{}':\t['drop', '', ''],\t# 100.0 % same XXX".format(col))
            continue

        test_recrods = int(count_records * 0.2)
        is_nan = 0
        is_binary = 0
        is_int = 0
        is_float = 0
        is_date = 0
        is_datetime = 0
        is_str = 0
        
        for i in range(test_recrods):
            v_o = str(df.iloc[i][col]).strip()
            if df.iloc[i][col] == np.nan or v_o == 'nan':
                is_nan += 1
                continue
            v = v_o
            if v in ['1', '0', 1, 0]:
                is_binary += 1
                is_int += 1
            elif v in ['Y', 'N']:
                is_binary += 1
                is_str += 1
            elif re.match('^\-?[0-9]{1,10}$', v): is_int += 1
            elif re.match('^\-?[0-9]{0,10}\.[Ee\+\-0-9]*$', v): is_float += 1
            elif re.match('^[0-9]{1,4}(\/|-)[0-9]{1,2}(\/|-)[0-9]{1,4}$', v): is_date += 1
            elif re.match('^[0-9]{1,4}(\/|-)[A-Za-z]{3}(\/|-)[0-9]{1,4}$', v): is_date += 1
            elif re.match('^[0-9]{1,4}(\/|-)[0-9]{1,2}(\/|-)[0-9]{1,4}(T| )[0-9]{1,2}:[0-9]{1,2}(:[0-9]{1,2})?Z?([\+0-9:]+)*$', v) \
                or re.match('^[0-9]{1,4}(\/|-)[A-Za-z]{3}(\/|-)[0-9]{1,4}(T| )[0-9]{1,2}:[0-9]{1,2}(:[0-9]{1,2})?Z?([\+0-9:]+)*$', v):
                is_datetime += 1
            else: is_str += 1

        total = test_recrods - is_nan

        nan_remark = '# TODO: HAS NAN {}% ???'.format(round(null_perc*100, 1)) if is_nan > 0 else ''

        if total == 0:
            print("'{}':\t['', '', ''],\t# Too many NaN !!!!!!!!! {}".format(col, nan_remark))
            continue
        if is_binary / total >= 0.9:
            print("'{}':\t['binary', '', ''],\t{}".format(col, nan_remark))
            continue
        if is_int / total >= 0.9:
            if class_perc > 0.9:
                print("'{}':\t['drop', '', ''],\t# {} % int unique XXXXXXXXXXXX".format(col, round(class_perc * 100, 2)))
                continue
            if class_perc < 0.01 and unique_values < 30:
                print("'{}':\t['class', '', ''],\t# {} % int unique ---------------".format(col, round(class_perc * 100, 2)))
                continue
            print("'{}':\t['int', '', ''],\t".format(col, nan_remark))
            continue
        if (is_int + is_float) / total >= 0.9:
            print("'{}':\t['float', '', ''],\t".format(col, nan_remark))
            continue
        if is_date / total >= 0.9:
            formats = ["%d-%b-%y", "%Y/%m/%d", "%Y-%m-%d", "%y-%m-%d"]
            for parse_format in formats:
                format_correct = True
                try: datetime.datetime.strptime(v, parse_format)
                except: format_correct = False
                if format_correct: break
            if format_correct:
                print("'{}':\t['datetime', '{}', ''],\t".format(col, parse_format, nan_remark))
            else:
                print("'{}':\t['drop', '', ''],\t# TODO: Unknow Date Format, example {} ---------------  {}"
                      .format(col, v, nan_remark))
            continue
        if is_datetime / total >= 0.9:
            formats = ["%Y-%m-%dT%H:%M:%SZ", "%d-%b-%y %H:%M:%S", "%d-%b-%y %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", 
                       "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%y-%m-%d %H:%M:%S", "%y-%m-%d %H:%M"]
            for parse_format in formats:
                format_correct = True
                try: datetime.datetime.strptime(v, parse_format)
                except: format_correct = False
                if format_correct: break
            if format_correct:
                print("'{}':\t['datetime', '{}', ''],\t".format(col, parse_format, nan_remark))
            else:
                print("'{}':\t['drop', '', ''],\t# TODO: Unknow Datetime Format, example {} ---------------  {}"
                      .format(col, v, nan_remark))
            continue
        if is_str / total >= 0.9:
            if class_perc > 0.5 or unique_values > 30:
                print("'{}':\t['drop', '', ''],\t# {} % str unique XXXXXXXXXXXXX"
                      .format(col, round(class_perc * 100, 2)))
            else:
                print("'{}':\t['class', '', ''],\t".format(col, nan_remark))
            continue
        else:
            pass # todo
            print('# TOFIX: {} -----------------------------'.format(col))
            print('# Stats:', is_nan, is_binary, is_int, is_float, is_date, is_datetime, is_str, class_perc, unique_values, 'total', total)
            # print(df[col][0])
            # print(df[col][1])

def compare(df1, df2, full=True):
    def b(t, bold = False):
        if type(t) == np.float64:
            t = "{:0.1f}".format(t)
        elif type(t) == np.int:
            t = "{:0d}".format(t)
        elif t == None:
            t = '.'
        else:
            t = str(t)
            if len(t)>5:
                t = t[:5] + '..' 

        if bold:
            return "<strong>" + t + "</strong>"
        else:
            return t

    def col_stats(s):
        p = 100.0 / len(s) # percent

        r = {
            "na": sum(s.isnull()) * p,
            "min": None,
            "mean": None,
            "max": None,
            "kurt": None,
            "skew": None,
            "unique": None,
            "mode": None
        }

        try:
            r["mode"] = s.mode()[0]
            r["min"] = s.min()
            r["mean"] = s.mean()
            r["max"] = s.max()
            r["kurt"] = s.kurt()
            r["skew"] = s.skew()
        except:
            pass

        unqiue = s.nunique()
        if unqiue * p < 20:
            r["unique"] = unqiue
            r["mode"] = s.mode()[0]

        return r
    
    print("Comparing DFs")
    print("Shape of DF1: {}".format(df1.shape))
    print("Shape of DF2: {}".format(df2.shape))
    print("")
    
    columns = set(list(df1.columns) + list(df2.columns))
    
    col_missing1 = []
    col_missing2 = []
    
    rows = []
    
    for col in columns:
        row = []
        stats1 = {
            "na": None,
            "min": None,
            "mean": None,
            "max": None,
            "kurt": None,
            "skew": None,
            "unique": None,
            "mode": None
        }
        stats2 = {
            "na": None,
            "min": None,
            "mean": None,
            "max": None,
            "kurt": None,
            "skew": None,
            "unique": None,
            "mode": None
        }
        
        if col in df1:
            stats1 = col_stats(df1[col])
        else:
            col_missing1.append(col)
            
        if col in df2:
            stats2 = col_stats(df2[col])
        else:
            col_missing2.append(col)
        
        # Display
        row.append(col)
        
        c = stats1['na'] != None and stats2['na'] != None and abs(stats1['na'] - stats2['na']) > 5
        row.append(b(stats1['na'], c))
        row.append("/")
        row.append(b(stats2['na'], c))
        
        c = stats1['unique'] != None and stats2['unique'] != None and abs(stats1['unique'] - stats2['unique']) > 1
        row.append(b(stats1['unique'], c))
        row.append("/")
        row.append(b(stats2['unique'], c))
        
        c = (stats1['mode'] != stats2['mode'])
        row.append(b(stats1['mode'], c))
        row.append("/")
        row.append(b(stats2['mode'], c))
        
        row.append(b(stats1['min'], 0))
        row.append("/")
        row.append(b(stats2['min'], 0))
        
        row.append(b(stats1['mean'], 0))
        row.append("/")
        row.append(b(stats2['mean'], 0))
        
        row.append(b(stats1['max'], 0))
        row.append("/")
        row.append(b(stats2['max'], 0))
        
        row.append(b(stats1['kurt'], 0))
        row.append("/")
        row.append(b(stats2['kurt'], 0))
        
        row.append(b(stats1['skew'], 0))
        row.append("/")
        row.append(b(stats2['skew'], 0))
        
        rows.append(row)
    
    print("DF1 does not has columns {}".format(str(col_missing1)))
    print("DF2 does not has columns {}".format(str(col_missing2)))
    print("")
    
    html = "<table><tr>"
    html += "<th>Columns</th>"
    html += "<th colspan='3' style='text-align:center'>Null</th>"
    html += "<th colspan='3' style='text-align:center'>Unique</th>"
    html += "<th colspan='3' style='text-align:center'>Mode</th>"
    html += "<th colspan='3' style='text-align:center'>Min</th>"
    html += "<th colspan='3' style='text-align:center'>Mean</th>"
    html += "<th colspan='3' style='text-align:center'>Max</th>"
    html += "<th colspan='3' style='text-align:center'>Kurtosis</th>"
    html += "<th colspan='3' style='text-align:center'>Skewness</th>"
    html += "</tr>"
    for r in rows:
        html += "<tr>"
        for c in r:
            html += "<td>" + str(c) + "</td>"
        html += "</tr>"
    html += "</table>"
    
    display(HTML(html))
    
    return