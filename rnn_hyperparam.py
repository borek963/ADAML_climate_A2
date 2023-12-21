from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Flatten
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


class RnnModel:
    def __init__(self,
                 data_frame,
                 unit_type,
                 target_column='humidity',
                 seq_len=20,
                 split_ratio=0.8,
                 lag=1,
                 n_units=80,
                 lr=0.001,
                 stacked_count=1):

        self.unit_type = unit_type
        self.data = data_frame.values
        self.target_column = int(data_frame.columns.get_loc(target_column))

        # Normalize the data using Min-Max scaling
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data_normalized = self.scaler.fit_transform(self.data)

        self.seq_len = seq_len
        self.lag = lag
        self.n_units = n_units
        self.stacked_count = stacked_count

        # Prepare sequences for input (X) and output (Y)
        X, Y = [], []

        for i in range(len(self.data_normalized) - self.seq_len - lag):
            x_sequence = self.data_normalized[i:i + self.seq_len]
            y_sequence = self.data_normalized[i + self.seq_len + lag]
            X.append(x_sequence)
            Y.append(y_sequence)

        # Convert the lists to NumPy arrays
        self.X = np.array(X)
        self.Y = np.array(Y)  # [:, self.target_column]

        # Split the data into training and testing sets (80% training, 20% testing)
        self.split_ratio = split_ratio
        self.split_index = int(split_ratio * len(self.X))

        # self.X_train, self.X_test = X[:self.split_index], X[self.split_index:]
        # self.Y_train, self.Y_test = Y[:self.split_index], Y[self.split_index:]
        
        # Split the data into training and validation sets (2 years training, 1 validation)
        self.n = 365

        self.X_train, self.X_test = X[:2*self.n], X[2*self.n + 1:3*self.n]
        self.Y_train, self.Y_test = Y[:2*self.n], Y[2*self.n + 1:3*self.n]

        # Create the LSTM model
        if self.unit_type == "base":
            self.model = Sequential()
            self.model.add(SimpleRNN(units=self.n_units, input_shape=(self.seq_len, 4)))
            self.model.add(Dense(units=4))

        elif self.unit_type == "lstm":
            self.model = Sequential()
            for _ in range(self.stacked_count):
                self.model.add(LSTM(units=self.n_units, input_shape=(self.seq_len, 4)), return_sequences=True)
            self.model.add(LSTM(units=self.n_units, input_shape=(self.seq_len, 4)))
            self.model.add(Dense(units=4))

        else:
            raise Exception("Unknown RNN unit...")

        # Compile the model
        custom_optimizer = Adam(learning_rate=lr)
        self.model.compile(optimizer=custom_optimizer, loss='mean_squared_error')

        self.validation_predictions = None
        self.validation_target = None

        self.mse = np.inf
        self.r2 = np.inf

    def train(self, epochs=100, batch_size=20, num_channels=4):
        self.model.fit(np.array(self.X_train),
                       np.array(self.Y_train),
                       epochs=epochs, batch_size=batch_size)

        # Inference using predicted values as input for the next prediction
        validation_target = self.Y_test
        validation_predictions = []

        last_x = self.X_test[0]
        while len(validation_predictions) < len(validation_target):
            p = self.model.predict(np.expand_dims(np.array(last_x), 0))
            validation_predictions.append(p)
            last_x = np.roll(last_x, -1, axis=0)
            last_x[-1] = p

        # Denormalize the predictions for better interpretation
        self.validation_predictions = self.scaler.inverse_transform(
            np.array(validation_predictions).reshape(-1, num_channels))
        self.validation_target = self.scaler.inverse_transform(
            np.array(self.Y_test).reshape(-1, num_channels))

        # Evaluate the model
        self.mse = mean_squared_error(np.array(self.validation_target)[:, self.target_column],
                                      np.array(self.validation_predictions)[:, self.target_column])
        self.r2 = r2_score(np.array(self.validation_target)[:, self.target_column],
                           np.array(self.validation_predictions)[:, self.target_column])

        print(f'Mean Squared Error: {self.mse}')
        print(f'R-squared: {self.r2}')

    def plot_validation_pred_target(self):
        plt.plot(self.validation_target[:, self.target_column], label="targets")
        plt.plot(self.validation_predictions[:, self.target_column], label="predictions")
        if self.unit_type == "base":
            plt.title("RNN Predictions")
        elif self.unit_type == "lstm":
            plt.title("LSTM Predictions")
        plt.legend()
        plt.show()
