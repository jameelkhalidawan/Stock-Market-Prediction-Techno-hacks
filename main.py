import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot


# Load and preprocess the data, excluding the 'Date' column
def load_and_preprocess_data(filename):
    df = pd.read_csv(filename)

    # Exclude the 'Date' column
    df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

    # Handle missing or zero values
    df = df.replace(0, np.nan)
    df = df.dropna()

    # Normalize the data
    scaler = MinMaxScaler()
    df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = scaler.fit_transform(
        df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])

    return df, scaler  # Return the scaler as well


# Load and preprocess data for a specific stock
data, scaler = load_and_preprocess_data('AMZN.csv')

# Define the number of previous days to use for prediction
look_back = 10


# Create sequences of data for LSTM
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data.iloc[i:i + look_back].values)
        y.append(data.iloc[i + look_back]['Close'])
    return np.array(X), np.array(y)


X, y = create_sequences(data, look_back)

# Split the data into training and test sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test, y_train, y_test = X[:split_index], X[split_index:], y[:split_index], y[split_index:]

# Convert data to float32 AFTER dropping the 'Date' column
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# Create and train the LSTM model with dropout layers
model = Sequential()
model.add(LSTM(100, input_shape=(look_back, 6), return_sequences=True))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Use early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

# Make predictions on the test set
y_pred = model.predict(X_test)

# Manually transform the predicted values back to the original scale for 'Close'
y_test_actual = y_test * (scaler.data_max_[-3] - scaler.data_min_[-3]) + scaler.data_min_[-3]
y_pred_actual = y_pred * (scaler.data_max_[-3] - scaler.data_min_[-3]) + scaler.data_min_[-3]

# Get the 'Date' column for plotting X-axis labels
date_column = data.iloc[split_index + look_back:].index

# Calculate Mean Squared Error
mse = mean_squared_error(y_test_actual, y_pred_actual)
print(f"Mean Squared Error: {mse}")

# Plot actual vs. predicted prices with correct date labels
plt.figure(figsize=(12, 6))
plt.plot(date_column, y_test_actual, label='Actual')
plt.plot(date_column, y_pred_actual, label='Predicted')
plt.xlabel('Data')
plt.ylabel('Stock Price (Close)')
plt.xticks(rotation=45)  # Rotate X-axis labels for clarity
plt.legend()
plt.title('Stock Price Prediction')
plt.show()

# Plot the training and validation loss over epochs
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Calculate residuals
residuals = y_test_actual - y_pred_actual

# Plot the residuals
plt.figure(figsize=(12, 6))
plt.plot(date_column, residuals, label='Residuals')
plt.axhline(0, color='red', linestyle='--', label='Zero Residual')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.xticks(rotation=45)
plt.legend()
plt.title('Residual Plot')
plt.show()

# Plot a histogram of residuals without specifying the color
plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=30, alpha=0.5, label='Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.legend()
plt.title('Histogram of Residuals')
plt.show()


# Create an autocorrelation plot of residuals
plt.figure(figsize=(12, 6))
autocorrelation_plot(residuals)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of Residuals')
plt.show()
