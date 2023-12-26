from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Flatten
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time

class RnnModel:
    def __init__(self,
                 data_frame,
                 unit_type,
                 target_column='humidity',
                 seq_len=20,
                 split_ratio=0.8,
                 lag=0,
                 n_units=80,
                 lr=0.001,
                 mode='valid'):

        self.unit_type = unit_type
        self.data = data_frame.values
        self.target_column = int(data_frame.columns.get_loc(target_column))

        # Normalization
        data_mean = self.data[:int(split_ratio*len(data_frame))].mean(axis=0)
        data_std = self.data[:int(split_ratio*len(data_frame))].std(axis=0)
        self.data_normalized = (self.data - data_mean)/data_std
        self.stats = np.array([data_mean, data_std])
        
        self.seq_len = seq_len
        self.lag = lag
        self.n_units = n_units

        # Prepare sequences for input (X) and output (Y)
        X, Y = [], []

        for i in range(len(self.data_normalized) - self.seq_len - lag):
            x_sequence = self.data_normalized[i:i + self.seq_len]
            y_sequence = self.data_normalized[i + self.seq_len + lag]
            X.append(x_sequence)
            Y.append([y_sequence[self.target_column]])

        # Convert the lists to NumPy arrays
        self.X = np.array(X)
        self.Y = np.array(Y)

        split_index1 = int(split_ratio * len(self.X))
        split_index2 = int((split_ratio + (1-split_ratio)/2) * len(self.X))
        if mode == 'valid':
            # Split the data into training and validation sets (split_ratio training, (1-split_ratio)/2 validation)
            self.X_train, self.X_test = X[:split_index1], X[split_index1:split_index2]
            self.Y_train, self.Y_test = Y[:split_index1], Y[split_index1:split_index2]
        elif mode == 'test':
            # Split the data into training and test sets (split_ratio + (1-split_ratio)/2 training, (1-split_ratio)/2 testing)
            self.X_train, self.X_test = X[:split_index2], X[split_index2:]
            self.Y_train, self.Y_test = Y[:split_index2], Y[split_index2:]

        # Create the RNN model
        if self.unit_type == "base":
            self.model = Sequential()
            self.model.add(SimpleRNN(units=self.n_units, input_shape=(self.seq_len, 4)))
            self.model.add(Dense(units=4, activation="tanh"))
            self.model.add(Dense(units=1))
        # Create the LSTM model
        elif self.unit_type == "lstm":
            self.model = Sequential()
            self.model.add(LSTM(units=self.n_units, input_shape=(self.seq_len, 4)))
            self.model.add(Dense(units=4, activation="tanh"))
            self.model.add(Dense(units=1))

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
        start = time.time()
        self.model.fit(np.array(self.X_train),
                       np.array(self.Y_train),
                       epochs=epochs, batch_size=batch_size)
        end = time.time()
        print("Training time: " + str(end - start))

        # Inference using predicted values as input for the next prediction
        validation_target = self.Y_test
        validation_predictions = []

        last_x = self.X_test[0]
        count = 1
        while len(validation_predictions) < len(validation_target):
            p = self.model.predict(np.expand_dims(np.array(last_x), 0))
            # Add the new predicted value for one day to the prediction list
            validation_predictions.append(p)
            # Roll to predict new day in next iteration
            last_x = np.roll(last_x, -1, axis=0)
            try:
              next_x = self.X_test[count]
            except:
              break
            
            # actual values of other variables + predicted value
            new_values = next_x[-1,:]
            new_values[self.target_column] = p
            last_x[-1,:] = new_values
            count += 1

        # Denormalize the predictions for better interpretation
        self.validation_predictions = np.array(validation_predictions, dtype=object).reshape(-1, 1)*self.stats[1,1] + self.stats[0,1]
        self.validation_target = np.array(validation_target).reshape(-1, 1)*self.stats[1,1] + self.stats[0,1]

        # Evaluate the model
        self.mse = mean_squared_error(self.validation_target, self.validation_predictions)
        self.r2 = r2_score(self.validation_target, self.validation_predictions)

        print(f'Mean Squared Error: {self.mse}')
        print(f'R-squared: {self.r2}')

    def plot_validation_pred_target(self):
        plt.plot(self.validation_target, label="targets")
        plt.plot(self.validation_predictions, label="predictions")
        if self.unit_type == "base":
            plt.title("RNN Predictions")
        elif self.unit_type == "lstm":
            plt.title("LSTM Predictions")
        plt.legend()
        plt.show()
