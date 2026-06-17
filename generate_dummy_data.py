import numpy as np

# Define dataset parameters
num_samples_train = 1000  # Number of training samples
num_samples_test = 200  # Number of testing samples
seq_length = 120  # Length of time-series sequence
num_features = 7  # Number of features (channels)

# Generate random data for features (X)
X_train = np.random.rand(num_samples_train, seq_length, num_features).astype(np.float32)
X_test = np.random.rand(num_samples_test, seq_length, num_features).astype(np.float32)

# Generate random labels for classification (y)
y_train = np.random.randint(0, 2, size=(num_samples_train,))  # Binary classification (0 or 1)
y_test = np.random.randint(0, 2, size=(num_samples_test,))

# Save datasets as .npy files
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Mock dataset generated successfully!")
