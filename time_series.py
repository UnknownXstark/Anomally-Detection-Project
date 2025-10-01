import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import Isolation_forest
from sklearn.metrics import precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

df_minute = pd.read_csv('EURUSD_minute.csv', parse_dates={'Datetime': ['Date', 'Time']})
df_minute.set_index('Datetime', inplace=True)
print("Top rows (Minute Data):\n", df_minute.head())
print("\nData types:\n", df_minute.dtypes)
print("\nMissing values:\n", df_minute.isnull().sum())
print("\nBasic stats:\n", df_minute.describe())