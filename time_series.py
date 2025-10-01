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

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.histplot(df_minute['BidSpread'], bins=50, kde=True, ax=axes[0,0])
axes[0,0].set_title('Bid Spread Distribution')
sns.boxplot(y=df_minute['BidReturn'], ax=axes[0,1])
axes[0,1].set_title('Bid Return Boxplot')
sns.heatmap(df_minute[['BO', 'BH', 'BL', 'BC', 'BCh', 'AO', 'AH', 'AL', 'AC', 'ACh', 'BidSpread', 'BidReturn']].corr(), annot=True, cmap='coolwarm', ax=axes[1,0])
axes[1,0].set_title('Correlation Heatmap')
df_minute['Volatility'].plot(ax=axes[1,1], title='Rolling Volatility Over Time')
plt.tight_layout()
plt.show()

Q1 = df_minute['BidReturn'].quantile(0.25)
Q3 = df_minute['BidReturn'].quantile(0.75)
IQR = Q3 - Q1
outliers = df_minute[(df_minute['BidReturn'] < (Q1 - 1.5 * IQR)) | (df_minute['BidReturn'] > (Q3 + 1.5 * IQR))]
print(f"Potential outliers (extreme returns): {len(outliers)} / {len(df_minute)} ({len(outliers)/len(df_minute)*100:.2f}%)")

# Feature Engineering and Wrangling
df_minute['LogBidSpread'] = np.log1p(df_minute['BidSpread'])
df_minute['Hour'] = df_minute.index.hour
df_minute['DayOfWeek'] = df_minute.index.dayofweek
df_minute['TimeBucket'] = pd.cut(df_minute['Hour'], bins=[0,6,12,18,24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
df_minute['PriceRatio'] = df_minute['AC'] / df_minute['BC']  # Ask/Bid close ratio

df_minute = pd.get_dummies(df_minute, columns=['TimeBucket'], drop_first=True)

num_features = ['BO', 'BH', 'BL', 'BC', 'BCh', 'AO', 'AH', 'AL', 'AC', 'ACh', 'BidSpread', 'AskSpread', 'BidReturn', 'Volatility', 'LogBidSpread', 'PriceRatio', 'Hour', 'DayOfWeek']
scaler = StandardScaler()
df_minute[num_features] = scaler.fit_transform(df_minute[num_features].fillna(0))  # Fill NaN from pct_change

print("After engineering - New columns:", [col for col in df_minute.columns if 'TimeBucket' in col])
print(df_minute[num_features + ['Hour']].head())

# Baseline Anomaly Detection Model
X = df_minute[num_features].dropna()

# Chronological split (train 2002-2015, test 2016-2019 for evolving normals)
train_end = '2016-01-01'
X_train = X[X.index < train_end]
X_test = X[X.index >= train_end]

model = IsolationForest(contamination=0.01, random_state=42)  # 1% anomalies
model.fit(X_train)
anomaly_scores = model.decision_function(X_test)
anomaly_labels = model.predict(X_test)  # -1 = anomaly

df_test = X_test.copy()
df_test['Anomaly'] = anomaly_labels
df_test['Score'] = anomaly_scores
top_anomalies = df_test[df_test['Anomaly'] == -1].sort_values('Score')

print("Detected anomalies:", (df_test['Anomaly'] == -1).sum())
print("Top 5 anomaly dates:\n", top_anomalies.head().index)

df_test_reset = df_test.reset_index()
df_test_reset = df_test_reset.merge(df_news[['Date', 'Article']], left_on='Datetime', right_on='Date', how='left')
# Simulate "high impact" anomalies (parse impact from Article if possible)
# For now, assume top news dates as true positives
news_dates = df_news['Date'].dropna().unique()
true_anomalies = df_test_reset[df_test_reset['Datetime'].isin(news_dates)]
true_pos = len(df_test_reset[(df_test_reset['Anomaly'] == -1) & (df_test_reset['Datetime'].isin(news_dates))])
print(f"True positives (news-correlated): {true_pos}")