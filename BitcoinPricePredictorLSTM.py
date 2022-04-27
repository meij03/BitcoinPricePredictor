import math # Mathematical functions
import numpy as np # Fundamental package for scientific computing with Python
import pandas as pd # Additional functions for analysing and manipulating data
from datetime import date, timedelta, datetime # Date Functions
from pandas.plotting import register_matplotlib_converters # This function adds plotting functions for calender dates
import matplotlib.pyplot as plt # Important package for visualization - we use this to plot the market data
import matplotlib.dates as mdates # Formatting dates
from sklearn.metrics import mean_absolute_error, mean_squared_error # Packages for measuring model performance / errors
from keras.models import Sequential # Deep learning library, used for neural networks
from keras.layers import LSTM, Dense, Dropout # Deep learning classes for recurrent and regular densely-connected layers
from keras.callbacks import EarlyStopping # EarlyStopping during model training
from sklearn.preprocessing import RobustScaler, MinMaxScaler # This Scaler removes the median and scales the data according to the quantile range to normalize the price data
import seaborn as sns


# df = pd.read_excel("cake_results_edit_sent_analysis.xlsx")
df = pd.read_csv("merge_bitcoin_n_fear.csv")

# Indexing Batches
# 02-01-2018
# df['Date'] = pd.to_datetime(df.Date, format = '%Y-%m-%d %H:%M:%S')
df['Date'] = pd.to_datetime(df.Date, format = '%m-%d-%Y')

# df['Date'] = pd.to_datetime(df.Date)

# print(df['Date'])
df = df.dropna()
train_df = df.sort_values(by=['Date']).copy()
date_index = train_df.index
train_df = train_df.reset_index(drop=True).copy()
# print(train_df.head(5))
# print(train_df.tail(5))

# List of considered Features
# FEATURES = ['Open','High','Low','Close','Volume_(BTC)','Volume_(Currency)','Weighted_Price','fng_value','fng_classification']
FEATURES = ['Open','High','Low','Close','Volume_(BTC)','Volume_(Currency)','Weighted_Price','fng_value']


# Create the dataset with features and filter the data to the list of FEATURES
data = pd.DataFrame(train_df)
data_filtered = data[FEATURES]



# We add a prediction column and set dummy values to prepare the data for scaling
data_filtered_ext = data_filtered.copy()
data_filtered_ext['Prediction'] = data_filtered_ext['Weighted_Price']

# Print the tail of the dataframe
# print(data_filtered_ext.tail())

# Get the number of rows in the data
nrows = data_filtered.shape[0]

# Convert the data to numpy values
np_data_unscaled = np.array(data_filtered)
np_data = np.reshape(np_data_unscaled, (nrows, -1))
print(np_data.shape)

# Transform the data by scaling each feature to a range between 0 and 1
scaler = MinMaxScaler()
np_data_scaled = scaler.fit_transform(np_data_unscaled)

# Creating a separate scaler that works on a single column for scaling predictions
scaler_pred = MinMaxScaler()
df_Close = pd.DataFrame(data_filtered_ext['Weighted_Price'])
np_Close_scaled = scaler_pred.fit_transform(df_Close)

# Set the sequence length - this is the timeframe used to make a single prediction
sequence_length = 50

# Prediction Index
index_Close = data.columns.get_loc("Weighted_Price")

# Split the training data into train and train data sets
# As a first step, we get the number of rows to train the model on 80% of the data
train_data_len = math.ceil(np_data_scaled.shape[0] * 0.8)

# Create the training and test data
train_data = np_data_scaled[0:train_data_len, :]
test_data = np_data_scaled[train_data_len - sequence_length:, :]

# The RNN needs data with the format of [samples, time steps, features]
# Here, we create N samples, sequence_length time steps per sample, and 6 features
def partition_dataset(sequence_length, data):
    x, y = [], []
    data_len = data.shape[0]
    for i in range(sequence_length, data_len):
        x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columsn
        y.append(data[i, index_Close]) #contains the prediction values for validation,  for single-step prediction
    
    # Convert the x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x, y

# Generate training data and test data
x_train, y_train = partition_dataset(sequence_length, train_data)
x_test, y_test = partition_dataset(sequence_length, test_data)

# Print the shapes: the result is: (rows, training_sequence, features) (prediction value, )
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Validate that the prediction value and the input match up
# The last close price of the second input sample should equal the first prediction value
print(x_train[1][sequence_length-1][index_Close])
print(y_train[0])

# Configure the neural network model
model = Sequential()

# Model with n_neurons = inputshape Timestamps, each with x_train.shape[2] variables
n_neurons = x_train.shape[1] * x_train.shape[2]
print(n_neurons, x_train.shape[1], x_train.shape[2])
model.add(LSTM(n_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(LSTM(n_neurons, return_sequences=False))
model.add(Dense(5))
model.add(Dense(1))
#hidden layer or bidirectional LSTM

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Training the model
# epochs = 50
epochs = 50
batch_size = 16
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                   callbacks=[early_stop])

# Plot training & validation loss values
fig, ax = plt.subplots(figsize=(20, 10), sharex=True)
plt.plot(history.history["loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
ax.xaxis.set_major_locator(plt.MaxNLocator(epochs))
plt.legend(["Train", "Test"], loc="upper left")
plt.grid()
plt.savefig("Model_Loss.png")
plt.show()

# Get the predicted values
y_pred_scaled = model.predict(x_test)

# Unscale the predicted values
y_pred = scaler_pred.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

# Mean Absolute Error (MAE)
MAE = mean_absolute_error(y_test_unscaled, y_pred)
print(f'Median Absolute Error (MAE): {np.round(MAE, 2)}')

# Mean Absolute Percentage Error (MAPE)
MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100
print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

# Median Absolute Percentage Error (MDAPE)
MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled)) ) * 100
print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')

# # # The date from which on the date is displayed
# # display_start_date = pd.Timestamp('today') - timedelta(days=500)

# # Add the date column
# print(data_filtered)
data_filtered_sub = data_filtered.copy()
data_filtered_sub['Date'] = date_index

# # Add the difference between the valid and predicted prices
train = data_filtered_sub[:train_data_len + 1]
valid = data_filtered_sub[train_data_len:]
valid.insert(1, "Prediction", y_pred.ravel(), True)
print(valid)
print(valid["Weighted_Price"])
valid.insert(1, "Difference", valid["Prediction"] - valid["Weighted_Price"], True)

# # # Zoom in to a closer timeframe
# # valid = valid[valid['Date'] > display_start_date]
# # train = train[train['Date'] > display_start_date]

# Visualize the data
fig, ax1 = plt.subplots(figsize=(22, 10), sharex=True)
xt = train['Date']; yt = train[["Weighted_Price"]]
xv = valid['Date']; yv = valid[["Weighted_Price", "Prediction"]]
plt.title("Predictions vs Actual Values", fontsize=20)
plt.ylabel("price in usd", fontsize=18)
plt.plot(xt, yt, color="#039dfc", linewidth=2.0)
plt.plot(xv, yv["Prediction"], color="#E91D9E", linewidth=2.0)
plt.plot(xv, yv["Weighted_Price"], color="black", linewidth=2.0)
plt.legend(["Train", "Test Predictions", "Actual Values"], loc="upper left")
plt.savefig("Predictions_vs_Actual.png")
plt.show()

# # Create the bar plot with the differences
x = valid['Date']
y = valid["Difference"]

# Create custom color range for positive and negative differences
valid.loc[y >= 0, 'diff_color'] = "#2BC97A"
valid.loc[y < 0, 'diff_color'] = "#C92B2B"

plt.bar(x, y, width=0.8, color=valid['diff_color'])
plt.grid()
plt.savefig("Predictions_vs_Actual.png")
plt.show()

# df_temp = df[-sequence_length:]
# new_df = df_temp.filter(FEATURES)

# N = sequence_length

# # Get the last N day closing price values and scale the data to be values between 0 and 1
# last_N_days = new_df[-sequence_length:].values
# last_N_days_scaled = scaler.transform(last_N_days)

# # Create an empty list and Append past N days
# X_test_new = []
# X_test_new.append(last_N_days_scaled)

# # Convert the X_test data set to a numpy array and reshape the data
# pred_price_scaled = model.predict(np.array(X_test_new))
# pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled.reshape(-1, 1))

# # Print last price and predicted price for the next day
# predicted_price = np.round(pred_price_unscaled.ravel()[0], 2)
# print(new_df['Weighted_Price'])
# price_today = np.round(new_df['Weighted_Price'][-1], 2)
# change_percent = np.round(100 - (price_today * 100)/predicted_price, 2)

# plus = '+'; minus = ''
# print(f'The close price for Cake today was {price_today}')
# print(f'The predicted close price is {predicted_price} ({plus if change_percent > 0 else minus}{change_percent}%)')

# Tutorial: https://www.relataly.com/stock-market-prediction-using-multivariate-time-series-in-python/1815/#h-implementing-a-multivariate-time-series-prediction-model-in-python


#TO DO NEXT TIME
#one hot encode above value
#experiment w layers
#experiment vs the other dataset 
