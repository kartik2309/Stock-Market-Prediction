# Stock-Market-Prediction

TIndicators.py file contains the class TIndicators that contains the code for computing various technical indicators as required by the project.

DataHandling.py file contains the class DataHandling that contains the code for various data transformations as required by the project.

LSTMNetworks.py contains the class LSTMNetworks1 which contains the definition of LSTM Network (encoder-decoder architecture-based stateful LSTM Networks) and various supporting functions for this LSTM Network.

main.py file is where we transform the OHLC stock price data and train the LSTM networks and make predictions. (Note that large amount intraday 1 minute data is required for the LSTM Networks to be trained well and make prediction better and caputure the trend in regular scenarios)

Listed below are few parameters for main.py
<ul>
  <li>df_intraday_full (The full dataset that is to read into this variable)</li>
  <li>batch_size (Batch size for Stateful LSTM)</li>
  <li>timesteps_e (Timestep set for encoder layer of LSTM)</li>
  <li>timesteps_d (Timestep set for encoder layer of LSTM)</li>
  <li>features (Number of features in the data)</li>
  <li>from_index (The starting point from where you want the OHLC stock price data to be considered)</li>
  <li>to_index (The ending point from where you want the OHLC stock price data to be considered)</li>
  <li>test_train_split_index (The point from where we want to split our data into training and testing)</li>
</ul>
