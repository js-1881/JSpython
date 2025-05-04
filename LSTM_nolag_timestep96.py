import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Load and preprocess the data
file_name = "historicaldata20232025.csv"

try:
    df = pd.read_csv(file_name)
    print("Data loaded successfully.\n")
    print(df.head())
except FileNotFoundError:
    print("File not found. Please check the file path.")

# Step 2: Clean and format the dataframe
df['Time'] = pd.to_datetime(df['Time'], format='%d-%m-%y %H:%M')
df = df.drop(['Texttime', 'Unnamed: 9'], axis=1, errors='ignore')
df = df.sort_values("Time").reset_index(drop=True)
df.columns = df.columns.str.replace('[\[\]<>]', '', regex=True)

print(df.head())

# Convert all object columns to float, clean strings if needed
for col in df.select_dtypes(include='object').columns:
    if col != 'Time':
        df[col] = df[col].str.replace(',', '')
        df[col] = df[col].str.replace(' ', '')
        df[col] = df[col].astype(float)

# Step 3: Add datetime & cyclical features
df['hour'] = df['Time'].dt.hour
df['month'] = df['Time'].dt.month
df['dow'] = df['Time'].dt.dayofweek

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)

# Step 4: Create lag and rolling features
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
df = create_lag_features(df, target_col, lags=[1, 4])
df = create_rolling_features(df, target_col, windows=[96])
df.dropna(inplace=True)

# Step 5: Define features and target
features = ['Wind offshore MWh', 'Wind onshore MWh', 'Photovoltaics MWh',
            'Actual grid load consumption MWh', 'Residual load MWh',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
            'lag_1', 'lag_4','rolling_std_96'
            ]

X = df[features]
y = df[target_col]

# Step 6: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Create sequences for LSTM
# LSTMs don't just look at individual data points — 
# they learn from a sequence of past data. 
# So we need to convert input into overlapping sequences

# Use timesteps > 1 to let LSTM capture past patterns
def create_sequences(X, y, timesteps):
    Xs, ys = [], []
    for i in range(timesteps, len(X)):      # range(start, stop, step)
        Xs.append(X[i-timesteps:i])         # last N timesteps
        ys.append(y.iloc[i])                # target to predict  
    return np.array(Xs), np.array(ys)

timesteps = 96  # Using last 24 hours of data (15-min intervals)
X_lstm, y_lstm = create_sequences(X_scaled, y, timesteps)

print("X_lstm shape:", X_lstm.shape)    # (n_samples, timesteps, n_features)
print("y_lstm shape:", y_lstm.shape)

# Step 8: TimeSeriesSplit Cross-Validation
tscv = TimeSeriesSplit(n_splits=2)
rmses, r2_scores = [], []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_lstm), 1):
    X_train, X_test = X_lstm[train_idx], X_lstm[test_idx]
    y_train, y_test = y_lstm[train_idx], y_lstm[test_idx]

    print(f"\nFold {fold}")
    print("Train indices:", train_idx)
    print("Test indices:", test_idx)
    print("X_train shape:", X_lstm[train_idx].shape)
    print("X_test shape:", X_lstm[test_idx].shape)

    # Step 9: Build LSTM Model
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(128, return_sequences=True),
        Dropout(0.4),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Step 10: Train with early stopping
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0, callbacks=[early_stop])

    # Step 11: Predict and Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    rmses.append(rmse)
    r2_scores.append(r2)

    print(f"\nFold {fold} RMSE: {rmse:.2f}")
    print(f"R2: {r2:.3f}")
    print(f"Train: {df.iloc[train_idx + timesteps]['Time'].min()} to {df.iloc[train_idx + timesteps]['Time'].max()}")
    print(f"Test: {df.iloc[test_idx + timesteps]['Time'].min()} to {df.iloc[test_idx + timesteps]['Time'].max()}")

# Step 12: Report results
print(f"\n✅ Average RMSE: {np.mean(rmses):.2f}")
print(f"✅ Average R2: {np.mean(r2_scores):.3f}")

# Step 13: Plot actual vs. predicted (last fold)
plt.figure(figsize=(14, 6))
plt.plot(df.iloc[test_idx + timesteps]['Time'], y_test, label='Actual', linewidth=2)
plt.plot(df.iloc[test_idx + timesteps]['Time'], y_pred, label='Predicted', linestyle='--', linewidth=2)
plt.title("✅ Actual vs. Predicted Prices (Last Fold)")
plt.xlabel("Time")
plt.ylabel("Price [EUR/MWh]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save the trained model
model.save('trained_lstm_model_nolag_timestep96_12864.keras')

cols_to_save = ['Time', 'Germany DA EUR/MWh', 'Wind offshore MWh', 'Wind onshore MWh', 'Photovoltaics MWh', 'Actual grid load consumption MWh', 
                'Residual load MWh', 'lag_1', 'lag_4', 'rolling_std_96', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos']
df_train_with_features = df[cols_to_save].copy()
# df_train_with_features.tail(500).to_csv('training_context_features1_nolag_tiemstep96_tail500_1.csv', index=False)
# df_train_with_features.to_csv('training_context_features1_nolag_allcopy_timestep96_.csv', index=False)

df_train_with_features.tail(500).to_csv('training_context_features1_nolag_tiemstep96_tail500_12864.csv', index=False)
df_train_with_features.to_csv('training_context_features1_nolag_allcopy_timestep96_12864.csv', index=False)

# .\.venv\Scripts\activate


# importances = model.feature_importances_
# plt.figure(figsize=(8, 6))
# plt.barh(X_train.columns, importances)
# plt.title("Feature Importances")
# plt.xlabel("Importance")
# plt.tight_layout()
# plt.show()


# Step 14: Retrain using full historical data before forecast
print("\n🔁 Retraining model on full historical data before forecasting...")

# Full training set ends on the last date that want to include
cutoff_time = pd.Timestamp("2025-04-01 23:45:00")
full_df = df[df['Time'] <= cutoff_time].copy()

# Recreate full scaled X and y
full_X = full_df[features]
full_y = full_df[target_col]
full_X_scaled = scaler.transform(full_X)

# Recreate sequences
X_full_lstm, y_full_lstm = create_sequences(full_X_scaled, full_y, timesteps)

# Retrain model
model_full = Sequential([
    Input(shape=(X_full_lstm.shape[1], X_full_lstm.shape[2])),
    LSTM(128, return_sequences=True),
    Dropout(0.4),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(1)
])

model_full.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

model_full.fit(X_full_lstm, y_full_lstm, epochs=20, batch_size=32, verbose=1, callbacks=[early_stop])

# Save full retrained model
model_full.save("final_lstm_model_trained_until_2025_04_01_96_12864.keras")
print("✅ Full model retrained and saved.")


# df_forecast = pd.read_csv("forecastedApril2025.csv")

# # Convert 'time' to datetime
# df_forecast['Time'] = pd.to_datetime(df_forecast['Time'], dayfirst=True, errors='coerce')

# df_forecast = df_forecast.sort_values("Time").reset_index(drop=True)
# df_forecast.columns = df_forecast.columns.str.replace('[\[\]<>]', '', regex=True)
# df_forecast = df_forecast.drop('Texttime', axis=1, errors='ignore')


# # Convert all other object columns to float
# for col in df_forecast.select_dtypes(include='object').columns:
#     if col not in ['Time']:
#         df_forecast[col] = df_forecast[col].str.replace(',', '')  # Remove thousands separators
#         df_forecast[col] = df_forecast[col].str.replace(' ', '')  # (Optional) Remove extra spaces
#         df_forecast[col] = df_forecast[col].astype(float)


# # Rename columns to match model input
# df_forecast.rename(columns={
#     "F Wind offshore MWh": "Wind offshore MWh",
#     "F Wind onshore MWh": "Wind onshore MWh",
#     "F Photovoltaics MWh": "Photovoltaics MWh",
#     "F grid load consumption MWh MWh": "Actual grid load consumption MWh",
#     "F Residual load MWh": "Residual load MWh"
# }, inplace=True)


# # Add cyclical and datetime features=
# df_forecast['hour'] = df_forecast['Time'].dt.hour
# df_forecast['month'] = df_forecast['Time'].dt.month
# df_forecast['dow'] = df_forecast['Time'].dt.dayofweek

# df_forecast['hour_sin'] = np.sin(2 * np.pi * df_forecast['hour'] / 24)
# df_forecast['hour_cos'] = np.cos(2 * np.pi * df_forecast['hour'] / 24)
# df_forecast['month_sin'] = np.sin(2 * np.pi * df_forecast['month'] / 12)
# df_forecast['month_cos'] = np.cos(2 * np.pi * df_forecast['month'] / 12)
# df_forecast['dow_sin'] = np.sin(2 * np.pi * df_forecast['dow'] / 7)
# df_forecast['dow_cos'] = np.cos(2 * np.pi * df_forecast['dow'] / 7)

# df_forecast = create_lag_features(df_forecast, 'Germany DA EUR/MWh', lags=[1, 96])
# df_forecast = create_rolling_features(df_forecast, 'Germany DA EUR/MWh', windows=[96])

# features = ['Wind offshore MWh', 'Wind onshore MWh', 'Photovoltaics MWh',
#             'Actual grid load consumption MWh', 'Residual load MWh',
#             'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
#             'lag_96', 'lag_1', 'rolling_mean_96', 'rolling_std_96']

# X_forecast = df_forecast[features]


# scaler = StandardScaler()
# X_forecast_scaled = scaler.fit_transform(X_forecast)

# timesteps = 24  # Using last 6 hours of data (15-min intervals)

# def create_sequences_for_forecast(X, timesteps):
#     Xs = []
#     for i in range(timesteps, len(X)):
#         Xs.append(X[i-timesteps:i])
#     return np.array(Xs)

# X_real_lstm = create_sequences_for_forecast(X_forecast _scaled, timesteps)
