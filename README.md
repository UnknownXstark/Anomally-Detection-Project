### ✅ Notes:
- ```bash
  pip install -r requirements.txt

# Anomaly Detection in EUR/USD Forex Data

This project explores **anomaly detection** in the **EUR/USD foreign exchange market** using the **Isolation Forest** algorithm.  
The goal is to detect unusual market movements or "shocks" that may correspond to real-world news events or other external factors.

---

##  Project Overview

###  What is Anomaly Detection?
Anomaly detection refers to identifying rare items, events, or observations that deviate significantly from the majority of the data.  
In finance, anomalies often represent unusual volatility, sudden price spikes, or unexpected shifts in trading behavior.

###  Why the EUR/USD Forex Pair?
EUR/USD is one of the most traded currency pairs globally and is highly sensitive to:
- Economic announcements  
- Central bank policies  
- Political and market news  

Detecting anomalies here helps traders and analysts understand hidden patterns and potential trading signals.

---

##  Workflow Summary

### 1. **Data Loading**
- **EURUSD_minute.csv** and **EURUSD_hour.csv** contain historical bid/ask price data.  
- **eurusd_news.csv** contains news articles/dates for event correlation.

### 2. **Preprocessing**
- Filled missing values and removed duplicates.  
- Merged datasets and cleaned news data.  
- Created additional features like spreads, returns, volatility, and time-based features (hour, day, session).

### 3. **Exploratory Data Analysis (EDA)**
- Visualized distributions, correlations, and rolling volatility.
- Detected potential outliers based on return values.

### 4. **Feature Engineering**
- Log-transformed spreads, standardized numerical features, and one-hot encoded time buckets.

### 5. **Model Training**
Used **Isolation Forest** (from `sklearn.ensemble`) for unsupervised anomaly detection.
```python
model = IsolationForest(contamination=0.005, random_state=42)
model.fit(X_train)