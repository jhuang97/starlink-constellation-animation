import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unicodedata import normalize
from dateutil.parser import parse
import re


def clean_normalize_whitespace(x):
    """ Normalize unicode characters and strip trailing spaces
    """
    if isinstance(x, str):
        return normalize('NFKC', x).strip()
    else:
        return x


def de_2tuple(x):
    if type(x) == tuple and len(x) == 2:
        if x[0] == x[1]:
            return x[0]
        else:
            return x[1]
    else:
        return x


def de_ref(x):
    return re.sub(r'\s?\[\d+\]\s?', '', x)


table_sl = pd.read_html('https://en.wikipedia.org/wiki/Starlink', match='Starlink launches')
print(f'Total tables: {len(table_sl)}')

df = table_sl[-1]

df = df.applymap(clean_normalize_whitespace)
print(type(df.columns[0]) == tuple)
df.columns = df.columns.to_series().apply(clean_normalize_whitespace)
df.rename(columns=de_2tuple, inplace=True)
df.rename(columns=de_ref, inplace=True)
df.rename(columns={'Deorbited [87]': 'Deorbited'}, inplace=True)
df.replace(to_replace=r'\[\d+\]$', value='', regex=True, inplace=True)
df.replace(to_replace=r'\[\d+\]$', value='', regex=True, inplace=True)
df.replace(to_replace='Tintin[91]v0.1', value='Tintin v0.1', inplace=True)
df.replace(to_replace='°', value='', regex=True, inplace=True)
df = df.applymap(clean_normalize_whitespace)

print(df.head())
print(df.info())

df['match'] = df.Mission.eq(df.Mission.shift())
# print(df['match'])

pd.set_option("display.max_rows", None, "display.max_columns", None)
# print(df[df['match'] == False].iloc[:, [0, 1, 3]])
mask = ~df['match'] & df['Outcome'].str.contains('Success')


def parse_table_dt(dt):
    d = dt.split(',')[0]
    return parse(d).strftime('%Y-%m-%d')


df1 = df[mask].copy()
df1['Inclination'] = df1['Inclination'].apply(pd.to_numeric)
df1['Deployed'] = df1['Deployed'].apply(pd.to_numeric)
df1['Working'] = df1['Working'].apply(pd.to_numeric)
df1['launch_date'] = df1['Date and time, UTC'].apply(parse_table_dt)
df2 = df1.iloc[:, [0, 1, 7, 8, 9, 12]].copy()
print(df2)
df2.to_pickle('../sat_info/starlink_launches.pkl')