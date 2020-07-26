import tensorflow as tf
import numpy as np


class LSTMNetworks1:

    lstm_model1 = tf.keras.models.Sequential()
    lstm_model2 = tf.keras.models.Model()
    history = []

    def __init__(self, timesteps, features, batch_size):

        # Sequence to Sequence Model.
        encoder_inputs = tf.keras.layers.Input(shape=(timesteps, features), batch_size=batch_size)
        encoder = [tf.keras.layers.LSTM(300, return_state=True,  return_sequences=True, stateful=True,),
                   tf.keras.layers.LSTM(300, return_state=True,  return_sequences=True,  stateful=True),
                   tf.keras.layers.LSTM(300, return_state=True, stateful=True)]

        encoder_layer_1, _, _ = encoder[0](encoder_inputs)
        encoder_layer_2, _, _ = encoder[1](encoder_layer_1)
        _, state_h, state_c = encoder[2](encoder_layer_2)

        encoder_states = [state_h, state_c]

        decoder_inputs = tf.keras.layers.Input(shape=(None, 4), batch_size=batch_size)
        decoder = [tf.keras.layers.LSTM(300, return_state=True, return_sequences=True, stateful=True),
                   tf.keras.layers.LSTM(300, return_state=True, return_sequences=True,  stateful=True),
                   tf.keras.layers.LSTM(300, return_state=True, return_sequences=True,  stateful=True, dropout=0.1)]

        decoder_layer_1, _, _ = decoder[0](decoder_inputs, initial_state=encoder_states)
        decoder_layer_2, _, _ = decoder[1](decoder_layer_1)
        decoder_output, state_h, state_c = decoder[2](decoder_layer_2)
        decoder_dense = tf.keras.layers.Dense(4  )
        decoder_dense_output = decoder_dense(decoder_output)

        self.lstm_model2 = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_dense_output)

    def fit(self, x_train, y_train, y_final):
        self.lstm_model2.compile(optimizer='adam', loss='mse')

        epochs = 15
        slice_per_step = x_train.shape[0]
        slice_index = slice_per_step
        while slice_index <= y_train.shape[0]:
            y_train_slice = y_train[slice_index - slice_per_step: slice_index]
            y_final_slice = y_final[slice_index - slice_per_step: slice_index]
            slice_index = slice_index + slice_per_step
            for epoch in range(epochs):
                print("Epoch:", epoch + 1, "/", epochs)
                self.history.append(self.lstm_model2.fit([x_train, y_train_slice], y_final_slice, epochs=1, shuffle=False))
                self.lstm_model2.reset_states()

    def predict(self, x_train, y_train):
        pred = []
        slice_per_step = x_train.shape[0]
        print(y_train.shape)
        print(x_train.shape)
        slice_index = slice_per_step
        while slice_index <= y_train.shape[0]:
            y_train_slice = y_train[slice_index - slice_per_step: slice_index]
            slice_index = slice_index + slice_per_step

            if slice_index == y_train.shape[0]:
                predicted_vals = self.lstm_model2.predict([x_train, y_train_slice])
                pred.append(predicted_vals)
        pred_np = np.array(pred)
        return pred_np[-1][-1]

    def get_losses(self):
        loss = []
        for i in range(0, len(self.history)):
            loss.append(self.history[i].history['loss'])
        return loss

    def get_val_losses(self):
        val_loss = []
        for i in range(0, len(self.history)):
            val_loss.append(self.history[i].history['val_loss'])
        return val_loss

    def save_trained_model(self, name):
        path = '../Trained Models/' + name
        self.lstm_model2.save(path)

    def load_model(self, name):
        path = '../Trained Models/' + name
        self.lstm_model2 = tf.keras.models.load_model(path)


