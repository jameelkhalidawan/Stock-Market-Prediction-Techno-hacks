# Stock-Market-Prediction-Techno-hacks

This repository contains code for predicting stock prices using Long Short-Term Memory (LSTM) neural networks. The code loads historical stock price data from a CSV file, preprocesses the data, trains an LSTM model, and evaluates its performance. It also includes various visualizations to help understand the model's behavior.

**Prerequisites**

Before running the code, make sure you have the following dependencies installed:

pandas
numpy
scikit-learn
TensorFlow
matplotlib
You can install these packages using pip:

pip install pandas numpy scikit-learn tensorflow matplotlib

**Usage**
Clone this repository to your local machine:

git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction

Place your historical stock price data in a CSV file (e.g., AMZN.csv) in the same directory.

Modify the filename variable in the code to specify the name of your CSV file:

filename = 'AMZN.csv'  # Change this to your CSV file name
Run the code:
python stock_price_prediction.py

**Code Structure**

stock_price_prediction.py: The main Python script containing the stock price prediction code.
AMZN.csv: Example historical stock price data for Amazon (Replace with your own data).
README.md: This README file.

**Data Preprocessing**

The code performs the following data preprocessing steps:

Loads the data from the CSV file and selects relevant columns.
Handles missing or zero values by replacing them with NaN and dropping rows with NaN values.
Normalizes the data using Min-Max scaling.

**Model Architecture**

The LSTM model architecture used for stock price prediction consists of two LSTM layers followed by a Dense layer. Dropout layers are used for regularization to prevent overfitting.

model = Sequential()
model.add(LSTM(100, input_shape=(look_back, 6), return_sequences=True))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

**Training**

The model is trained on the preprocessed data with early stopping to prevent overfitting. Training and validation loss are monitored during training.

**Evaluation**

The code evaluates the model by calculating Mean Squared Error (MSE) between actual and predicted stock prices. It also generates several visualizations:

Actual vs. Predicted Stock Prices
Training and Validation Loss
Residual Plot
Histogram of Residuals
Autocorrelation of Residuals

**License**

This code is provided under the MIT License. Feel free to use, modify, and distribute it as needed.

