from DataHandling import DataHandling
from TIndicators import TIndicators
from LSTMNetworks import LSTMNetworks1

import pandas as pd
import numpy as np
# import sklearn.metrics as skm

import matplotlib.pyplot as plt
# import plotly.graph_objects as go

# ------------------------------------Basic Setup-------------------------------------------------- #

# Read the whole dataset
df_intraday_full = pd.read_csv('../Datasets/intraday_1min_AXISBANK.csv', converters={'Time': lambda x: str(x)})

# Set up global variables
batch_size = 30
timesteps_e = 60
timesteps_d = 15
features = 16
from_index = 50
to_index = 54125
test_train_split_index = 54000

# Set up global objects for custom classes
lstm1 = LSTMNetworks1(timesteps=timesteps_e, features=features, batch_size=batch_size)
dh = DataHandling(timesteps_e, timesteps_d, features)
ti = TIndicators()

# Slice the dataframe to required length
df_intraday = df_intraday_full[:to_index]

# Store stock price related information
time = df_intraday['Time']
open_p = df_intraday['Open']
high_p = df_intraday['High']
low_p = df_intraday['Low']
close_p = df_intraday['Close']
volume = df_intraday['Volume']

# ------------------------------------Setting up the data--------------------------------------------- #
# Stock Price Information
time_req = dh.slice_from(time, from_index, 1)
open_p_req = dh.slice_from(open_p, from_index - 1, 1)
high_p_req = dh.slice_from(high_p, from_index - 1, 1)
low_p_req = dh.slice_from(low_p, from_index - 1, 1)
close_p_req = dh.slice_from(close_p, from_index - 1, 1)
volume_req = dh.slice_from(volume, from_index, 1)
#
# Computing Log-Returns on the stock prices data.
open_p_req_lr = dh.slice_from(dh.log_returns(open_p_req), 1, 1)
high_p_req_lr = dh.slice_from(dh.log_returns(high_p_req), 1, 1)
low_p_req_lr = dh.slice_from(dh.log_returns(low_p_req), 1, 1)
close_p_req_lr = dh.slice_from(dh.log_returns(close_p_req), 1, 1)

# Momentum Indicators
so, ss = ti.stochastic_oscillator(high=high_p, low=low_p,
                                  close=close_p, periods=15, ma_periods=3)
so_req = dh.slice_from(so, from_index, 1)
roc_req = dh.slice_from(ti.roc(close_p, 15), from_index, 1)
rsi_req = dh.slice_from(ti.rsi(close_p, periods=15), from_index, 1)
kama_req = dh.slice_from(ti.kama(close_p, periods=15), from_index, 1)

# Trend Indicators
macd_req = dh.slice_from(ti.macd(close_p), from_index, 1)
psar_req = dh.slice_from(ti.psar(high_p, low_p, close_p), from_index, 1)
vortex_req = dh.slice_from(ti.psar(high_p, low_p, close_p), from_index, 1)
cci_req = dh.slice_from(ti.cci(high_p, low_p, close_p), from_index, 1)
adx_req = dh.slice_from(ti.adx(high_p, low_p, close_p, 15), from_index, 1)

# Volume Indicators
acci_req = dh.slice_from(ti.acc_dist_index(high_p, low_p, close_p, volume), from_index, 1)
cmf_req = dh.slice_from(ti.chaikin_money_flow(high_p, low_p, close_p, volume), from_index, 1)
vpt_req = dh.slice_from(ti.volume_price_trend(close_p, volume), from_index, 1)
fi_req = dh.slice_from(ti.force_index(close_p, volume, 15), from_index, 1)

# Volatility Indicators
atr_req = dh.slice_from(ti.atr(high_p, low_p, close_p, 15), from_index, 1)
bbs_req = dh.slice_from(ti.bollinger_bands(close_p), from_index, 6)

# Creating single numpy array for each category
stock_data = np.concatenate([open_p_req_lr, high_p_req_lr, low_p_req_lr, close_p_req_lr], axis=1)
momentum_indicators = np.concatenate([rsi_req, so_req, roc_req, kama_req], axis=1)
trend_indicators = np.concatenate([macd_req, psar_req, vortex_req, cci_req, adx_req], axis=1)
volume_indicators = np.concatenate([acci_req, cmf_req, vpt_req, fi_req], axis=1)
volatility_indicators = np.concatenate([atr_req, bbs_req], axis=1)

# Performing Min-Max Normalization.
stock_data_n = dh.time_normalize(stock_data, for_layer='e')
volume_n = dh.time_normalize(volume_req, for_layer='e')
momentum_indicators_n = dh.time_normalize(momentum_indicators, for_layer='e')
trend_indicators_n = dh.time_normalize(trend_indicators, for_layer='e')
volume_indicators_n = dh.time_normalize(volume_indicators, for_layer='e')
volatility_indicators_n = dh.time_normalize(volatility_indicators, for_layer='e')


# Apply dimensionality reduction via PCA
stock_data_pca = dh.pca(stock_data_n, 3)
momentum_indicators_pca = dh.pca(momentum_indicators_n, 3)
trend_indicators_pca = dh.pca(trend_indicators_n, 4)
volume_indicators_pca = dh.pca(volume_indicators_n, 3)
volatility_indicators_pca = dh.pca(volatility_indicators_n, 5)

# Forming our final dataset
data_req = np.concatenate([volume_n, momentum_indicators_pca, trend_indicators_pca,
                           volume_indicators_pca, volatility_indicators_pca], axis=1)

data_req_df = pd.DataFrame(data_req)

stock_data_req = np.concatenate([open_p_req_lr, high_p_req_lr, low_p_req_lr, close_p_req_lr], axis=1)
stock_data_n_dec = stock_data_req
stock_data_n_dec_pd = pd.DataFrame(stock_data_n_dec)


# --------------------------- Setting up Training Data -------------------------------------#
# Set up training X
x_train_raw = data_req[:test_train_split_index]
x_train = dh.create_x(x_train_raw)

# Set up training Y
y_raw = stock_data_n_dec
y_raw_n = y_raw[:test_train_split_index]
y_dec_train = dh.create_y(y_raw_n, dim=4)

y_final_n = y_raw[timesteps_d: test_train_split_index + timesteps_d]
y_final = dh.create_y(y_final_n, dim=4)

# --------------------------- Setting up Testing Data -------------------------------------#
# Set up Test Data X
look_back_e = (timesteps_e * batch_size) - timesteps_e
x_test_raw = data_req[(test_train_split_index - look_back_e): -timesteps_d]
# print(x_test_raw.shape)
x_test = dh.create_x(x_test_raw)

look_back_d = (timesteps_e * batch_size) - timesteps_e
y_dec_test_raw = stock_data_n_dec[(test_train_split_index - look_back_d): -timesteps_d]
y_dec_test = dh.create_y(y_dec_test_raw, dim=4)

# # Set up True results for Test X (next 15 window).
# df_test = df_intraday_full.iloc[to_index: to_index+timesteps_e]
# time_test_y = df_test['Time']
# open_p_test_y = df_test['Open']
# high_p_test_y = df_test['High']
# low_p_test_y = df_test['Low']
# close_p_test_y = df_test['Close']
# real_oc = pd.concat([open_p_test_y, high_p_test_y, low_p_test_y, close_p_test_y], axis=1)

# --------------------------------------------LSTM Network Model--------------------------------------- #

# Train the model
lstm1.fit(x_train, y_dec_train, y_final)

# Save the trained model
lstm1.save_trained_model('test-x.h5')

# Call the saved model.
# lstm1.load_model('test-2 .h5')


# Predict with the model
pred = lstm1.predict(x_test, y_dec_test)


# Reformat the predicted data values from 3D array to 2D.
open_pred = dh.slice_from(pred[:, 0], 0, 1)
high_pred = dh.slice_from(pred[:, 1], 0, 1)
low_pred = dh.slice_from(pred[:, 2], 0, 1)
close_pred = dh.slice_from(pred[:, 3], 0, 1)
pred_oc_lr = np.concatenate([open_pred, high_pred, low_pred, close_pred], axis=1)

dh.set_seeds(open_p_req[-timesteps_d:], high_p_req[-timesteps_d:], low_p_req[-timesteps_d:], close_p_req[-timesteps_d:])
pred_final_df = pd.DataFrame(dh.inverse_log_returns(pred_oc_lr))
pred_final_df.columns = ['Open', 'High', 'Low', 'Close']
print(pred_final_df[:5])

real_vals = np.concatenate([open_p_req[-timesteps_d:], high_p_req[-timesteps_d:], low_p_req[-timesteps_d:],
                            close_p_req[-timesteps_d:]], axis=1)
real_final_df = pd.DataFrame(real_vals)
real_final_df.columns = ['Open', 'High', 'Low', 'Close']


# # --------------------------------------------Check the model--------------------------------------- #
print(real_final_df[:5])

time_slice = time[-timesteps_d:][:5]
# print(time_slice)
plt.clf()
plt.plot(time_slice, pred_final_df['Close'][:5], label="Predicted Closing Value")
plt.plot(time_slice, real_final_df['Close'][:5], label="Real Closing Value")
plt.xlabel('Time')
plt.ylabel('Closing Prices')
plt.title('Comparison between predicted and real prices')
plt.legend()
plt.show()

plt.clf()
length = range(close_p_req_lr.shape[0])
plt.plot(length, close_p_req_lr, label='Log Returned Close Values')
plt.title('Closing Prices - Log Returned')
plt.xlabel('No. of observations')
plt.ylabel('Close Prices - LR')
plt.show()



