import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Generate synthetic data for binary classification
np.random.seed(42)
data = np.random.rand(1000, 20)  # 1000 samples, 20 features
labels = np.random.randint(0, 2, 1000)  # Binary labels (0 or 1)

# Split data into training and testing sets
trainData, testData, trainLabels, testLabels = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# One-hot encode the labels for the neural network
trainLabels_onehot = to_categorical(trainLabels, num_classes=2)
testLabels_onehot = to_categorical(testLabels, num_classes=2)

# Define a simple neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')  # 2 output classes for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(trainData, trainLabels_onehot, epochs=10, batch_size=32, verbose=1)

# Make predictions on the test data
predictions = model.predict(testData)
predicted_labels = np.argmax(predictions, axis=1)  # Convert probabilities to binary labels

# Compute Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(testLabels, predicted_labels)
print(f"Matthews Correlation Coefficient (MCC): {mcc}")
