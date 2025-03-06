import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Assume X_train is already defined and has shape (num_samples, num_features)
num_features = x_train.shape[1]

# Create a simple neural network model for regression
model = Sequential([
    Dense(64, activation='relu', input_shape=(num_features,)),  # num_features is created up above so it can be used here
    Dense(64, activation='relu'),
    Dense(1, activation='linear')  # Output layer for regression
])


#Looked into good values to use here and Adam does best around 0.001 training rate, I chose 0.002 just to speed it up a little bit
model.compile(optimizer=Adam(learning_rate=0.002), loss='mse')

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)