import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Load The Data
df_minute = pd.read_csv('EURUSD_minute.csv', parse_dates={'Datetime': ['Date', 'Time']})
df_minute.set_index('Datetime', inplace=True)
print("Top rows (Minute Data):\n", df_minute.head())
print("\nData types:\n", df_minute.dtypes)
print("\nMissing values:\n", df_minute.isnull().sum())
print("\nBasic stats:\n", df_minute.describe())

df_hour = pd.read_csv('EURUSD_hour.csv', parse_dates={'Datetime': ['Date', 'Time']})
df_hour.set_index('Datetime', inplace=True)
print("\nHourly Data Shape:", df_hour.shape)

df_news = pd.read_csv('eurusd_news.csv')
df_news['Date'] = pd.to_datetime(df_news['Article'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0], errors='coerce')
df_news.dropna(subset=['Date'], inplace=True)
print("\nNews Data Head:\n", df_news.head())

# Data Preprocessing
df_minute.fillna(method='ffill', inplace=True)
df_hour.fillna(method='ffill', inplace=True)

df_minute.drop_duplicates(inplace=True)
df_hour.drop_duplicates(inplace=True)

df_news.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

print("\nAfter cleaning - Minute shape:", df_minute.shape)
print("News unique dates:", df_news['Date'].nunique())

# Exploratory Data Analysis (EDA)
df_minute['BidSpread'] = df_minute['BH'] - df_minute['BL']  # Bid range
df_minute['AskSpread'] = df_minute['AH'] - df_minute['AL']  # Ask range
df_minute['BidReturn'] = df_minute['BC'].pct_change()
df_minute['Volatility'] = df_minute['BidReturn'].rolling(window=60).std()  # 1-hour rolling