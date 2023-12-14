import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the cleaned data
file_path = 'd:/GitHub/Extrusion-Parameter-Optimization/data_clean.csv'
data = pd.read_csv(file_path)

# Assuming 'Layer Height' and 'Layer Width' are features
# and 'B (Robot Speed)' and 'C (Extrusion Speed)' are targets
features = data[['Layer Height', 'Layer Width']]
targets = data[['B (Robot Speed)', 'C (Extrusion Speed)']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2))  # Two output neurons for two target values

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')

# Train the model and save the history
history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1, validation_split=0.2)

# Plotting the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Optionally, you can save your model
# model.save('path/to/your/model.h5')
