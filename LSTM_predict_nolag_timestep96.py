import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# df_train = pd.read_csv("training_context_features1_nolag_allcopy_timestep96.csv")
df_train = pd.read_csv("training_context_features1_nolag_allcopy_timestep96_12864.csv")
df_predict = pd.read_csv("forecastedApril2025_1.csv")

df_train.columns = df_train.columns.str.strip()
df_predict.columns = df_predict.columns.str.strip()

# print(df_train.info())
# print(df_predict.info())
# print(df_predict.columns)


# Convert 'time' to datetime
df_predict['Time'] = pd.to_datetime(df_predict['Time'], format='%d-%m-%y %H:%M', errors='coerce')
df_train['Time'] = pd.to_datetime(df_train['Time'], errors='coerce')

print(df_predict.head())
print(df_train.head())

df_predict = df_predict.sort_values("Time").reset_index(drop=True)
df_predict.columns = df_predict.columns.str.replace('[\[\]<>]', '', regex=True)
df_predict = df_predict.drop('Texttime', axis=1, errors='ignore')


# Convert all other object columns to float
for col in df_predict.select_dtypes(include='object').columns:
    if col not in ['Time']:
        df_predict[col] = df_predict[col].str.replace(',', '')  # Remove thousands separators
        df_predict[col] = df_predict[col].str.replace(' ', '')  # (Optional) Remove extra spaces
        df_predict[col] = df_predict[col].astype(float)

        
# context_rows = df_train.tail(500)
context_rows = df_train.copy()
df_forecast = pd.concat([context_rows, df_predict], ignore_index=True)

print(df_predict.head())
# print(df_forecast[495:510])
print(df_forecast.head())

# print(df_forecast.columns)
# print(df_forecast.head())
# print(df_forecast[495:510])

# Add cyclical and datetime features=
df_forecast['hour'] = df_forecast['Time'].dt.hour
df_forecast['month'] = df_forecast['Time'].dt.month
df_forecast['dow'] = df_forecast['Time'].dt.dayofweek

df_forecast['hour_sin'] = np.sin(2 * np.pi * df_forecast['hour'] / 24)
df_forecast['hour_cos'] = np.cos(2 * np.pi * df_forecast['hour'] / 24)
df_forecast['month_sin'] = np.sin(2 * np.pi * df_forecast['month'] / 12)
df_forecast['month_cos'] = np.cos(2 * np.pi * df_forecast['month'] / 12)
df_forecast['dow_sin'] = np.sin(2 * np.pi * df_forecast['dow'] / 7)
df_forecast['dow_cos'] = np.cos(2 * np.pi * df_forecast['dow'] / 7)

# print(df_forecast.info())
# print(df_forecast.columns)
# print(df_forecast.head())
# print(df_forecast[['Time','lag_1', 'lag_96', 'rolling_mean_96', 'rolling_std_96']][50:100])
# print(len(df_forecast))
# print(len(df_predict))

# first_nan_index = df_forecast['rolling_mean_96'].isna().idxmax()
# print("First NaN in 'rolling_mean_96' is at row index:", first_nan_index)



# print(df_train.columns.tolist())
# print("Columns in df_forecast:", df_forecast.columns.tolist())

# print(row_500)
# print(df_forecast['Germany DA EUR/MWh'].iloc[405:500].isnull().sum())
# print(df_forecast[features].iloc[405:500].isnull().sum())

# print(df_forecast[['Time','lag_1', 'lag_96', 'rolling_mean_96', 'rolling_std_96', 'Germany DA EUR/MWh']][495:510])

# print("Before dropna:", df_forecast.shape[0])
# df_forecast = df_forecast.dropna().reset_index(drop=True)
# print("After dropna:", df_forecast.shape[0])

# print(df_forecast[495:510].isna().sum())
# print(df_forecast[495:510].isna().any(axis=1))


def create_lag_features(df, col, lags):
    for lag in lags:
        df[f'lag_{lag}'] = df[col].shift(lag)
    return df

def create_rolling_features(df, col, windows):
    for w in windows:
        # df[f'rolling_mean_{w}'] = df[col].rolling(window=w).mean()
        df[f'rolling_std_{w}'] = df[col].rolling(window=w).std()
    return df

target_col = 'Germany DA EUR/MWh'

df_forecast = create_lag_features(df_forecast, target_col, lags=[1, 4])
df_forecast = create_rolling_features(df_forecast, target_col, windows=[96])


features = ['Wind offshore MWh', 'Wind onshore MWh', 'Photovoltaics MWh',
            'Actual grid load consumption MWh', 'Residual load MWh',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
            'lag_1', 'lag_4','rolling_std_96']




# print("Before dropna:", df_forecast.shape[0])
# df_forecast = df_forecast.dropna().reset_index(drop=True)
# print("After dropna:", df_forecast.shape[0])


# print("X_forecast shape before scaling:", df_forecast.shape)
# print("Any NaNs?", df_forecast.isnull().any().any())
# print("Total NaNs per column:\n", df_forecast.isnull().sum())

# print("len(df_forecast) before dropna:", len(df_forecast))

# df_forecast = df_forecast.dropna().reset_index(drop=True)


timesteps = 96  # Using last 6 hours of data (15-min intervals)
n_steps = len(df_predict)
preds = []

model = tf.keras.models.load_model('final_lstm_model_trained_until_2025_04_01_96_12864.keras', compile=False)
# model.compile(loss='mse', optimizer='adam')

# Create a new column for predicted prices
df_forecast['predicted_price'] = np.nan

# Get starting point: index where prediction begins
start_idx = len(df_forecast) - len(df_predict)
# start_idx = 500

print("len(df_forecast):", len(df_forecast))
print("len(df_predict):", len(df_predict))
print("start_idx:", start_idx)

X_forecast = df_forecast.iloc[:start_idx][features]
scaler = StandardScaler()

# scaler = MinMaxScaler(feature_range=(0, 1))  # Scale features to range [0, 1]

scaler.fit(df_forecast[features].iloc[:start_idx])
X_forecast_scaled = scaler.transform(df_forecast[features].iloc[:start_idx])
# X_forecast_scaled = scaler.fit_transform(X_forecast)

# print(len(df_forecast))
# print(len(df_predict))

# X_forecast_scaled = list(X_forecast_scaled)
# print(f"Starting recursive prediction from index {start_idx} to {start_idx + n_steps}")

# Check shape and NaN before filtering
# print("Before filtering:")
# print("Shape:", X_forecast_scaled.shape)
# print("Any NaNs?", np.isnan(X_forecast_scaled).any())
# print("Rows with NaNs:", np.isnan(X_forecast_scaled).any(axis=1).sum())

# loops over each item (x) in your existing list X_forecast_scaled
# For each x (which is usually a NumPy array representing a single feature row), this checks whether any value in x is NaN.
# only include x in the new list if it does not contain any NaNs.

X_forecast_scaled = [x for x in X_forecast_scaled if not np.isnan(x).any()]

# After filtering
# print("\nAfter filtering:")
# print("Length of X_forecast_scaled list:", len(X_forecast_scaled))

# nan_summary = pd.DataFrame(X_forecast, columns=features).isna().sum()
# print("NaN count per feature:\n", nan_summary)
# print("Clean rows before scaling:", X_forecast.dropna().shape[0])

train_start_time = df_forecast.iloc[0]['Time']
train_end_time = df_forecast.iloc[start_idx - 1]['Time']
print(f"Training used for forecasting from {train_start_time} to {train_end_time}")


for i in range(n_steps):
    input_seq = np.array(X_forecast_scaled[-timesteps:])    # shape: (timesteps, features)
    input_seq = np.expand_dims(input_seq, axis=0)   # shape: (1, timesteps, features)

    pred = model.predict(input_seq, verbose=0)[0, 0]
    preds.append(pred)

    current_idx = start_idx + i
    # df_forecast.at[current_idx, 'predicted_price'] = pred
    df_forecast.at[current_idx, 'Germany DA EUR/MWh'] = pred

    # Update lag_1 for next row
    if current_idx + 1 < len(df_forecast):
        df_forecast.at[current_idx + 1, 'lag_1'] = pred

    # Update lag_4 for next row
    if current_idx + 4 < len(df_forecast):
        df_forecast.at[current_idx + 4, 'lag_4'] = pred

    # Update lag_96
    # if current_idx + 96 < len(df_forecast):
    #     df_forecast.at[current_idx + 96, 'lag_96'] = pred


    df_forecast.at[current_idx, 'predicted_price'] = pred

    print(f"Step {i}: Predicting for index {current_idx}, predicted value = {pred:.3f}")


    # Recalculate rolling mean/std
    recent_prices = df_forecast['Germany DA EUR/MWh'].iloc[current_idx - 95: current_idx + 1]
    # rolling_mean = recent_prices.mean()
    rolling_std = recent_prices.std()

    if current_idx + 1 < len(df_forecast):
        # df_forecast.at[current_idx + 1, 'rolling_mean_96'] = rolling_mean
        df_forecast.at[current_idx + 1, 'rolling_std_96'] = rolling_std

        try:
            new_features = df_forecast.loc[[current_idx + 1], features]
            try:
                 new_scaled = scaler.transform(new_features)
            except Exception as e:
                 print(f"Scaler error at step {i}, index {current_idx + 1}: {e}")
                 break
            X_forecast_scaled.append(new_scaled.flatten())

        except Exception as e:
            print(f"Error updating input sequence at index {current_idx + 1}: {e}")
            break


# Extract only the predicted values and their timestamps
predicted_df = df_forecast.loc[start_idx-5:, ['Time', 'predicted_price', 'Price comparison DA', 'Germany DA EUR/MWh', 'lag_1', 'lag_4']]
print(predicted_df.head(20))


filtered_df = predicted_df.dropna(subset=['Price comparison DA', 'predicted_price'])
rmse = np.sqrt(mean_squared_error(filtered_df['Price comparison DA'], filtered_df['predicted_price']))
r2 = r2_score(filtered_df['Price comparison DA'], filtered_df['predicted_price'])

print(f"Forecast RMSE: {rmse:.2f}, R²: {r2:.3f}")

df_train[target_col].hist(alpha=0.6, label='Train')
df_predict['Price comparison DA'].hist(alpha=0.6, label='April 2025')
plt.legend()
plt.title('Target Variable Distribution')


plt.figure(figsize=(14, 6))
plt.plot(df_forecast['Time'], df_forecast['Price comparison DA'], label='Actual Price', linewidth=2)
plt.plot(df_forecast['Time'], df_forecast['predicted_price'], label='Predicted Price', linestyle='--', linewidth=2)
plt.xlabel("Time")
plt.ylabel("Price [EUR/MWh]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


predicted_df.to_csv("lstm_price_forecast_april2025_timestep96.csv", index=False)






# def create_sequences_for_forecast(X, timesteps):
#     Xs = []
#     for i in range(timesteps, len(X)):
#         Xs.append(X[i-timesteps:i])
#     return np.array(Xs)


# X_real_lstm = create_sequences_for_forecast(X_forecast_scaled, timesteps)
# print("Shape of X_real_lstm:", X_real_lstm.shape)


# model = tf.keras.models.load_model('trained_lstm_model.h5', compile=False)
# y_pred_real = model.predict(X_real_lstm)


# df_forecast['Predicted Price'] = np.nan
# df_forecast.iloc[timesteps:, df_forecast.columns.get_loc('Predicted Price')] = y_pred_real.flatten()



# plt.figure(figsize=(14, 6))
# plt.plot(df_forecast['Time'], df_forecast['Price comparison DA'], label='Actual Price', linewidth=2)
# plt.plot(df_forecast['Time'][timesteps:], y_pred_real, label='Predicted Price', linestyle='--', linewidth=2)
# plt.title("Actual vs. Predicted Prices")
# plt.xlabel("Time")
# plt.ylabel("Price [EUR/MWh]")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()