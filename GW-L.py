
import os
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.decomposition import PCA
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')


df = pd.read_csv('timeseries.csv')
season = {'Winter':1,'Spring':2,'Summer':3,'Autumn':4}

df = df[:96431:]
df.index = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
df['G(i)'] = pd.to_numeric(df['G(i)'], errors='coerce')

day  = 60 * 60 * 24
year =  365.2425 * day
month = 30 * day

df["Seconds"] = df.index.map(pd.Timestamp.timestamp)
df ["day sin"] = np.sin(df['Seconds'] * ( 2 * np.pi /day))
df ["day cos"] = np.cos(df['Seconds'] * ( 2 * np.pi /day))
# df ["year sin"] = np.sin(df['Seconds'] * ( 2 * np.pi /year))
# df ["year cos"] = np.cos(df['Seconds'] * ( 2 * np.pi /year))

# 1971
df = df.drop(["Seconds"], axis = 1)

# Check for any NaN values after conversion
# print("NaN values in G(i):", df['G(i)'].isna().sum())
df= df[df['G(i)'] > 10]

def get_season(month):
    if month in [12, 1, 2]:
        return season['Winter']
    elif month in [3, 4, 5]:
        return season['Spring']
    elif month in [6, 7, 8]:
        return season['Summer']
    elif month in [9, 10, 11]:
        return season['Autumn']

# Apply the function to create the season column
df['Season'] = df.index.month.map(get_season)

# Load data into DataFrame
data = df.copy()
data.drop(['Int'], axis = 1,inplace=True)

# Ensure 'time' column is of string type
data['time'] = data['time'].astype(str)

# Validate 'time' column format and filter rows with valid datetime-like strings
valid_time_format = r'^\d{8}:\d{4}$'
data = data[data['time'].str.match(valid_time_format, na=False)]

# Convert 'time' column to datetime
data['time'] = pd.to_datetime(data['time'], format='%Y%m%d:%H%M')

# Set 'time' as the index
data.set_index('time', inplace=True)

# Convert all columns to numeric, coercing errors to NaN
data = data.apply(pd.to_numeric, errors='coerce')

# Handle missing values by filling with the mean of each column
data.fillna(data.mean(), inplace=True)

# print(data.describe())

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

# Fit and transform the data
scaled_data = scaler.fit_transform(data)

# Create a new DataFrame with scaled data, preserving column names and index
scaled_df = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

# Display the first few rows and basic statistics to confirm
# print(scaled_df.head())


print(scaled_df.head(15))
data=scaled_df


# Box Plots
data_box=data.copy()
data_box.drop(['day sin','day cos'], axis=1,inplace=True)
plt.figure(figsize=(12, 6))
sns.boxplot(data=data_box)
plt.title('Box Plot of Features')
plt.show()

# TODO Merhawi
# this for hourly basis
# Now resample to hourly intervals
# Verify index is DatetimeIndex
data_hourly=data.copy()


data_2019_M1_D1= data_hourly[(data_hourly.index.year == 2019) & (data_hourly.index.month == 1) &
                        (data_hourly.index.day == 1)]



data_2019= data_hourly[(data_hourly.index.year == 2019) & (data_hourly.index.month == 12)]

plt.figure(figsize=(20, 12))
plt.subplot(1,2,1)
plt.plot(data_2019_M1_D1.index, data_2019_M1_D1['G(i)'], marker='o', label='G(i) per Hour')
plt.xlabel('Time')
plt.ylabel('G(i)')
plt.title('Hourly Plot of G(i) (as recorded) ')
plt.legend()

plt.subplot(1,2,2)
plt.plot(data_2019_M1_D1.index, data_2019_M1_D1['H_sun'], marker='v',color='orange' ,label='G(i) per Hour')
plt.xlabel('Time')
plt.ylabel('H(i)')
plt.title(' Plot of H_sun (as recorded)')
plt.legend()

plt.show()
  
  

plt.figure(figsize=(10, 5))
plt.plot(data_2019_M1_D1.index, data_2019_M1_D1['G(i)'], color='orange', label='G(i) - 2019')
plt.plot(data_2019_M1_D1.index, data_2019_M1_D1['H_sun'], marker='v', label='G(i) per Hour')
plt.title('month Data for G(i) vs H_sun in 2019')
plt.xlabel('Time')
plt.ylabel('H_sun and G(i)')
plt.legend()
plt.show()


data_2019_M1=data.copy()


data_2019_M1.drop(['Season','day cos', 'day sin'], axis=1, inplace=True)
data_2019_M1 = data_2019_M1[(data.index.year == 2019) & (data.index.month == 1)]
# Additional KDE Plot for each feature to observe distribution
plt.figure(figsize=(10, 10))
for column in data_2019_M1.columns:
    sns.kdeplot(data_2019_M1[column] , label=column)
plt.title('KDE Plot of Features')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()


 ## Correlation Analysis
# Correlation Heatmap

data.drop(['day sin','day cos','Season'],axis=1,inplace=True)
plt.figure(figsize=(6, 5))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# Pair Plot for Visualizing Relationships Between Features
sns.pairplot(data, palette='bright')
plt.suptitle('Pair Plot of Features', y=1.02, fontsize=16)
plt.show()


df = df.dropna(subset=['time']) 
df['time'].tail(20)


df = df[:96431:]  # Truncate the DataFrame to include only the first 96,431 rows.

df.drop(['Int', 'time'], axis = 1,inplace=True)
df_multi = df.iloc[:,:]  # Select columns from index 1 to 4 (inclusive), typically removing the time column assuming it's at index 0.

df.tail(30)  # Display the last 30 rows of the DataFrame to check the end of the dataset.

print(df_multi.head(10))  # Print the first 10 rows of the subset DataFrame to verify the correct columns and data.

print(df_multi.dtypes)  # Print the data types of the columns in the subset DataFrame to ensure they are appropriate for further analysis.


# ToDo Merhawi Check the result by using MinMaxScaler() 
# Initialize MinMaxScaler
scaler = MinMaxScaler() 

# Fit and transform the data
scaled_data = scaler.fit_transform(df_multi)

# Create a new DataFrame with scaled data, preserving column names and index
scaled_df = pd.DataFrame(scaled_data, columns=df_multi.columns, index=df_multi.index)

# Display the first few rows and basic statistics to confirm
print(scaled_df.head())


def df_to_X_y(df_as_np, window_size=5):
#   df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [a for a in df_as_np[i:i+window_size,:]]
    X.append(row)
    label = df_as_np[i+window_size,0]
    y.append(label)
  return np.array(X), np.array(y)



WINDOW_SIZE = 5
df_scaled = scaled_df.to_numpy()
X1, y1 = df_to_X_y(df_scaled, WINDOW_SIZE)
print(X1[0])
print("#############################")
print(y1[0])
X1.shape, y1.shape

X_train1, y_train1 = X1[:30684], y1[:30684] 
X_val1, y_val1 = X1[30685:35064], y1[30685:35064] # 1 year for validation [61,367 , 70127]
X_test1, y_test1 = X1[35065:], y1[35065:] #  1 year testing [70,128 , 78887]
X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape




# ---------------------------------------------
# Model Definition
# ---------------------------------------------

# Replace 'seq_length' and 'num_features' with actual values
seq_length = 5       # For example, using past 24 hours
num_features = 7      # Number of features in your dataset

def build_optimized_model_LSTM():
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(units=128, return_sequences=True, input_shape=(seq_length, num_features)))
    model.add(Dropout(rate=0.1))
    
    # Second LSTM layer
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(rate=0.1))
    
    # Third LSTM layer
    model.add(LSTM(units=32))
    model.add(Dropout(rate=0.1))
    
    # Fully connected layer
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(rate=0.1))
    
    # Output layer
    model.add(Dense(1))  # Output layer for regression
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=Huber(delta=1.0),
        metrics=['mae']
    )
    
    return model

# Instantiate the model
model = build_optimized_model_LSTM()

# ---------------------------------------------
# Callbacks
# ---------------------------------------------

early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=20,
    verbose=1,
    mode='min'
)

model_checkpoint = ModelCheckpoint(
    'model_optimized_best_1.keras',
    save_best_only=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=5,
    verbose=1
)


history_LSTM = model.fit(
    X_train1, y_train1,
    epochs=50,
    batch_size=64,  # Adjust based on your system's memory capacity
    validation_data=(X_val1, y_val1),
    callbacks=[early_stopping, model_checkpoint, lr_scheduler],
    verbose=1,
    shuffle=False
)


model_LSTM = tf.keras.models.load_model('model_optimized_best.keras')
print(model_LSTM.summary())


plt.plot(history_LSTM.history['loss'])
plt.plot(history_LSTM.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



# Access the data_min_ and data_max_ for the first feature
data_min = scaler.data_min_[0]
data_max = scaler.data_max_[0]

Train_prediction = model_LSTM.predict(X_test1).flatten()

# Manually inverse transform the predictions and actuals
Train_Predictions_transformed = Train_prediction * (data_max - data_min) + data_min
y_train1_transformed = y_test1 * (data_max - data_min) + data_min

# Prepare the DataFrame with actual and predicted values
train_results = pd.DataFrame({
    'Test Predictions': Train_Predictions_transformed,
    'Actuals': y_train1_transformed
})

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(train_results['Test Predictions'][0:60], label="Predicted G(i)")
plt.plot(train_results['Actuals'][0:60], label="Actual G(i) ")
plt.ylabel('G(i) in W/m²')
plt.xlabel("time step (hourly)")
plt.title('Test Data Predictions vs Actuals LSTM')
plt.legend()
plt.show()


# Define metrics using NumPy
def mae_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mba_np(y_true, y_pred):
    return np.abs(np.mean(y_pred - y_true))

def rmse_np(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

# Actual and predicted values

Train_prediction_ERR = model_LSTM.predict(X_test1).flatten()

# Manually inverse transform the predictions and actuals
# Prepare the DataFrame with actual and predicted values
train_results = pd.DataFrame({
    'Test Predictions': Train_prediction_ERR,
    'Actuals': y_test1
})
y_true = train_results['Test Predictions'][0:60].values
y_pred = train_results['Actuals'][0:60]

# Calculate metrics
mae_value = mae_np(y_true, y_pred)
mba_value = mba_np(y_true, y_pred)
rmse_value = rmse_np(y_true, y_pred)

print("MAE:", mae_value)
print("MBA:", mba_value)
print("RMSE:", rmse_value)

def mae_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mba_np(y_true, y_pred):
    return np.mean(y_pred - y_true)

def rmse_np(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))




def build_optimized_gru_model():
    model = Sequential()
    seq_length=5
    num_features=7
    
    # Input layer
    model.add(InputLayer(input_shape=(seq_length, num_features)))
    
    # First GRU layer
    model.add(GRU(units=128, return_sequences=True))
    model.add(Dropout(rate=0.1))
    
    # Second GRU layer
    model.add(GRU(units=128, return_sequences=True))
    model.add(Dropout(rate=0.1))
    
    # Third GRU layer
    model.add(GRU(units=64))
    model.add(Dropout(rate=0.1))
    
    # Fully connected layer
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(rate=0.1))
    
    # Output layer
    model.add(Dense(1))  # Linear activation for regression
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Instantiate the model
best_model_gru=build_optimized_gru_model()

early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=20,
    verbose=1,
    mode='min'
)

model_checkpoint = ModelCheckpoint(
    'model_optimized_GRU_Best.keras',
    save_best_only=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=5,
    verbose=1
)




# print(best_model.summary())
history_GRU = best_model_gru.fit(
    X_train1, y_train1,
    epochs=200,
    validation_data=(X_val1,y_val1),
    callbacks=[early_stopping, model_checkpoint, lr_scheduler],
    batch_size=64,
    verbose=1,
    shuffle=False
)




import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_GRU.history['loss'])
plt.plot(history_GRU.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plot training & validation MAE values
plt.subplot(1, 2, 2)
plt.plot(history_GRU.history['mae'])
plt.plot(history_GRU.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.show()


# Access the data_min_ and data_max_ for the first feature
data_min = scaler.data_min_[0]
data_max = scaler.data_max_[0]
best_model_gru = tf.keras.models.load_model('model_optimized_GRU_Best.keras')
print(best_model_gru.summary())

Train_prediction = best_model_gru.predict(X_test1).flatten()

# Manually inverse transform the predictions and actuals
Train_Predictions_transformed = Train_prediction * (data_max - data_min) + data_min
y_train1_transformed = y_test1 * (data_max - data_min) + data_min

# Prepare the DataFrame with actual and predicted values
train_results = pd.DataFrame({
    'Test Predictions': Train_Predictions_transformed,
    'Actuals': y_train1_transformed
})

# Plotting 
plt.figure(figsize=(12, 6))
plt.plot(train_results['Test Predictions'][0:60], label="Predicted G(i)")
plt.plot(train_results['Actuals'][0:60], label="Actual G(i) ")
plt.ylabel('G(i) in W/m²')
plt.xlabel("time step (hourly)")
plt.title('Test Data Predictions vs Actuals GRU')
plt.legend()
plt.show()


# Define metrics using NumPy
def mae_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mba_np(y_true, y_pred):
    return np.abs(np.mean(y_pred - y_true))

def rmse_np(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

# Actual and predicted values

Train_prediction_Testing_GRU = best_model_gru.predict(X_test1).flatten()

# Manually inverse transform the predictions and actuals
# Prepare the DataFrame with actual and predicted values
train_results = pd.DataFrame({
    'Test Predictions': Train_prediction_Testing_GRU,
    'Actuals': y_test1
})
y_true = train_results['Test Predictions'][0:60].values
y_pred = train_results['Actuals'][0:60]

# Calculate metrics
mae_value = mae_np(y_true, y_pred)
mba_value = mba_np(y_true, y_pred)
rmse_value = rmse_np(y_true, y_pred)

print("MAE:", mae_value)
print("MBA:", mba_value)
print("RMSE:", rmse_value)

def mae_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mba_np(y_true, y_pred):
    return np.abs(np.mean(y_pred - y_true))

def rmse_np(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))





# ---------------------------------------------
# Model Definition
# ---------------------------------------------

# Replace 'seq_length' and 'num_features' with actual values
seq_length = 5       # For example, using past 24 hours
num_features = 7      # Number of features in your dataset

def build_optimized_rnn_model():
    model = Sequential()
    
    # First RNN layer
    model.add(InputLayer(input_shape=(seq_length, num_features)))
    model.add(SimpleRNN(units=64, return_sequences=True))
    model.add(Dropout(rate=0.2))
    
    # Second RNN layer
    model.add(SimpleRNN(units=64, return_sequences=True))
    model.add(Dropout(rate=0.2))
    
    # Third RNN layer
    model.add(SimpleRNN(units=32))
    model.add(Dropout(rate=0.2))
    
    # Fully connected layer
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(rate=0.2))
    
    # Output layer
    model.add(Dense(1))  # Output layer for regression
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=Huber(),
        metrics=['mae']
    )
    
    return model

# Instantiate the model
model_RNN = build_optimized_rnn_model()

# ---------------------------------------------
# Callbacks
# ---------------------------------------------

early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=20,
    verbose=1,
    mode='min'
)

model_checkpoint = ModelCheckpoint(
    'model_optimized_rnn_best.keras',
    save_best_only=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=5,
    verbose=1
)

# ---------------------------------------------
# Model Training
# ---------------------------------------------

history_RNN = model_RNN.fit(
    X_train1, y_train1,
    epochs=50,
    validation_data=(X_val1, y_val1),
    callbacks=[early_stopping, model_checkpoint, lr_scheduler],
    batch_size=64
)

model_RNN=build_optimized_rnn_model()


plt.plot(history_RNN.history['loss'])
plt.plot(history_RNN.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Define metrics using NumPy
def mae_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mba_np(y_true, y_pred):
    return np.abs(np.mean(y_pred - y_true))

def rmse_np(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

# Actual and predicted values
model_RNN = tf.keras.models.load_model('model_optimized_rnn_best.keras')
print(model_RNN.summary())

Train_prediction_Testing_RNN = model_RNN.predict(X_test1).flatten()

# Manually inverse transform the predictions and actuals
# Prepare the DataFrame with actual and predicted values
train_results = pd.DataFrame({
    'Test Predictions': Train_prediction_Testing_RNN,
    'Actuals': y_test1
})
y_true = train_results['Test Predictions'][0:60].values
y_pred = train_results['Actuals'][0:60]

# Calculate metrics
mae_value = mae_np(y_true, y_pred)
mba_value = mba_np(y_true, y_pred)
rmse_value = rmse_np(y_true, y_pred)

print("MAE:", mae_value)
print("MBA:", mba_value)
print("RMSE:", rmse_value)

def mae_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mba_np(y_true, y_pred):
    return np.mean(y_pred - y_true)

def rmse_np(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


# Access the data_min_ and data_max_ for the first feature
data_min = scaler.data_min_[0]
data_max = scaler.data_max_[0]

Train_prediction_RNN = model_RNN.predict(X_test1).flatten()
Train_prediction_GRU = best_model_gru(X_test1)
Train_prediction_LSTM = model_LSTM.predict(X_test1).flatten()

# Manually inverse transform the predictions and actuals
Train_Predictions_transformed = Train_prediction_RNN * (data_max - data_min) + data_min
y_train1_transformed = y_test1 * (data_max - data_min) + data_min

# Prepare the DataFrame with actual and predicted values
train_results = pd.DataFrame({
    'Test Predictions': Train_Predictions_transformed,
    'Actuals': y_train1_transformed
})

# Plotting 
plt.figure(figsize=(12, 6))
plt.plot(train_results['Test Predictions'][0:60], label="Predicted G(i)")
plt.plot(train_results['Actuals'][0:60], label="Actual G(i) ")
plt.ylabel('G(i) in W/m²')
plt.xlabel("time step (hourly)")
plt.title('Test Data Predictions vs Actuals RNN')
plt.legend()
plt.show()


# Access the data_min_ and data_max_ for the first feature
data_min = scaler.data_min_[0]
data_max = scaler.data_max_[0]

# Get predictions from all models
Train_prediction_RNN = model_RNN.predict(X_test1).flatten()
Train_prediction_GRU = best_model_gru.predict(X_test1).flatten()
Train_prediction_LSTM = model_LSTM.predict(X_test1).flatten()

# Manually inverse transform the predictions and actuals
Train_Predictions_RNN_transformed = Train_prediction_RNN * (data_max - data_min) + data_min
Train_Predictions_GRU_transformed = Train_prediction_GRU * (data_max - data_min) + data_min
Train_Predictions_LSTM_transformed = Train_prediction_LSTM * (data_max - data_min) + data_min
y_test1_transformed = y_test1 * (data_max - data_min) + data_min

# Prepare the DataFrame with actual and predicted values
train_results = pd.DataFrame({
    'Actuals': y_test1_transformed,
    'RNN Predictions': Train_Predictions_RNN_transformed,
    'GRU Predictions': Train_Predictions_GRU_transformed,
    'LSTM Predictions': Train_Predictions_LSTM_transformed
})
x_h=0
y_h=24
# Plotting
plt.figure(figsize=(12, 6))
plt.plot(train_results['Actuals'][x_h:y_h], label="Actual G(i)",marker='o', linestyle='-')
plt.plot(train_results['RNN Predictions'][x_h:y_h], label="RNN Predicted G(i)")
plt.plot(train_results['GRU Predictions'][x_h:y_h], label="GRU Predicted G(i)")
plt.plot(train_results['LSTM Predictions'][x_h:y_h], label="LSTM Predicted G(i)")
plt.ylabel('G(i) in W/m²')
plt.xlabel("Time Step (Hourly)")
plt.title('Test Data Predictions vs Actuals for RNN, GRU, and LSTM Models')
plt.legend()
plt.show()



error_RNN = Train_Predictions_RNN_transformed - y_test1_transformed  # Flatten if necessary to make 1D
# Plotting the error distribution with a histogram
plt.figure(figsize=(10, 6))
sns.histplot(error_RNN, kde=True, color="RED", bins=30,label="RNN model")
plt.title("Error Distribution (Predicted - Actual)")
plt.xlabel("Bias Error")
plt.ylabel("Frequency")
plt.legend()
plt.show()


error_GRU = Train_Predictions_GRU_transformed - y_test1_transformed  # Flatten if necessary to make 1D
# Plotting the error distribution with a histogram
plt.figure(figsize=(10, 6))
sns.histplot(error_GRU, kde=True, color="ORANGE", bins=30,label="GRU model")
plt.title("Error Distribution (Predicted - Actual)")
plt.xlabel("Bias Error")
plt.ylabel("Frequency")
plt.legend()
plt.show()


error_LSTM = Train_Predictions_LSTM_transformed - y_test1_transformed  # Flatten if necessary to make 1D
# Plotting the error distribution with a histogram
plt.figure(figsize=(10, 6))
sns.histplot(error_LSTM, kde=True, color="blue", bins=30,label="LSTM model")
plt.title("Error Distribution (Predicted - Actual)")
plt.xlabel("Bias Error")
plt.ylabel("Frequency")
plt.legend()
plt.show()




