import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import SGD, Adam
import matplotlib.pyplot as plt


class ArModel:
    def __init__(self, data_frame, steps_ahead=10, target_column='humidity'):
        self.T = steps_ahead
        self.X = []
        self.Y = []

        self.target_column = target_column
        self.series = data_frame[self.target_column]

        for t in range(len(self.series) - self.T):
            x = self.series[t:t + self.T]
            self.X.append(x)
            y = self.series[t + self.T]
            self.Y.append(y)

        self.X = np.array(self.X).reshape(-1, self.T)
        self.Y = np.array(self.Y)
        self.N = len(self.X)

        self.mse = np.inf
        self.r2 = np.inf

        self.model = None
        self.history_obj = None

        self.validation_target = None
        self.validation_predictions = None

    def train(self):
        i = Input(shape=(self.T,))
        x = Dense(1)(i)
        # Create a linear regression model
        self.model = Model(i, x)
        self.model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=0.1)
        )

        # Train the model on the training data
        self.history_obj = self.model.fit(self.X[:-self.N // 2], self.Y[:-self.N // 2],
                                          epochs=80,
                                          validation_data=(self.X[-self.N // 2:], self.Y[-self.N // 2:])
                                          )

        self.validation_target = self.Y[-self.N // 2:]
        self.validation_predictions = []
        last_x = self.X[-self.N // 2]

        while len(self.validation_predictions) < len(self.validation_target):
            p = self.model.predict(last_x.reshape(1, -1), verbose=0)[0, 0]
            self.validation_predictions.append(p)
            last_x = np.roll(last_x, -1)
            last_x[-1] = p

        # Evaluate the model
        self.mse = mean_squared_error(self.validation_target, self.validation_predictions)
        self.r2 = r2_score(self.validation_target, self.validation_predictions)

        print(f'Mean Squared Error: {self.mse}')
        print(f'R-squared: {self.r2}')

    def plot_train_progress(self):
        plt.plot(self.history_obj.history["loss"], label="Training Loss")
        plt.plot(self.history_obj.history["val_loss"], label="Validation loss")
        plt.legend()

    def plot_validation_pred_target(self):
        # Plot the true values
        plt.plot(self.validation_target, label="forecast_target")
        plt.plot(self.validation_predictions, label="forecast_prediction")
        plt.legend()
