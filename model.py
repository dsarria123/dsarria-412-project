import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


# 1. Prepare the data
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
        labels.append(data[i+sequence_length][0])  
    return np.array(sequences), np.array(labels)

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['WVHT', 'SwH', 'SwP', 'WWH', 'WWP', 'APD', 'WDIR', 'WSPD', 'PRES']])

# Create sequences with a window size of 48 (for example, past 48 timesteps)
sequence_length = 48
X, y = create_sequences(scaled_data, sequence_length)

# Split data into train and test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 2. Build the model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50),
    Dense(1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# 3. Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# 4. Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# 5. Make predictions (inverse transform to get original scale)
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Rescale predictions
