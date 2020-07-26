import ta.momentum as tam
import ta.trend as tat
import ta.volume as tvl
import ta.volatility as tvt
from sklearn.decomposition import PCA
import numpy as np


class TIndicators:

    # Momentum Indicators
    def roc(self, close, periods):
        roc_ = tam.ROCIndicator(close=close, n=periods)
        roc = roc_.roc().to_numpy()
        return roc

    def rsi(self, close, periods):
        rsi_ = tam.RSIIndicator(close=close, n=periods)
        rsi = rsi_.rsi().to_numpy()
        return rsi

    def stochastic_oscillator(self, high, low, close, periods, ma_periods):
        so_ = tam.StochasticOscillator(high=high, low=low, close=close, n=periods, d_n=ma_periods)
        so = so_.stoch().to_numpy()
        ss = so_.stoch_signal()

        return so, ss

    def kama(self, close, periods):
        kama_ = tam.KAMAIndicator(close=close, n=periods)
        kama_val = kama_.kama().to_numpy()
        return kama_val

    # Trend Indicators
    def macd(self, close):
        macd_ = tat.MACD(close)
        macd = macd_.macd().to_numpy()
        return macd

    def psar(self, high, low, close):
        psar_ = tat.PSARIndicator(high=high, low=low, close=close, step=0.01)
        psar = psar_.psar().to_numpy()
        return psar

    def vortex(self, high, low, close, periods):
        vortex_ = tat.VortexIndicator(high=high, low=low, close=close, n=periods)
        vortex_diff = vortex_.vortex_indicator_diff().to_numpy()
        return vortex_diff

    def cci(self, high, low, close):
        cci = tat.cci(high=high, low=low, close=close).to_numpy()
        return cci

    def adx(self, high, low, close, periods,):
        adx_ = tat.ADXIndicator(high=high, low=low, close=close, n=periods)
        adx = adx_.adx().to_numpy()
        return adx

    # Volume Indicators
    def acc_dist_index(self, high, low, close, volume):
        acci_ = tvl.AccDistIndexIndicator(high=high, low=low, close=close, volume=volume)
        acci = acci_.acc_dist_index().to_numpy()
        return acci

    def chaikin_money_flow(self, high, low, close, volume):
        cmf_ = tvl.ChaikinMoneyFlowIndicator(high=high, low=low, close=close, volume=volume)
        cmf = cmf_.chaikin_money_flow().to_numpy()
        return cmf

    def volume_price_trend(self, close, volume):
        vpt_ = tvl.VolumePriceTrendIndicator(close=close, volume=volume)
        vpt = vpt_.volume_price_trend().to_numpy()
        return vpt

    def force_index(self, close, volume, period):
        fi_ = tvl.ForceIndexIndicator(close=close, volume=volume, n=period)
        fi = fi_.force_index().to_numpy()
        return fi

    # Volatility Indicators
    def atr(self, high, low, close, periods):
        atr_ = tvt.AverageTrueRange(high=high, low=low, close=close, n=periods)
        atr = atr_.average_true_range().to_numpy()
        return atr

    def bollinger_bands(self, close):
        bbs_ = tvt.BollingerBands(close=close)
        rows = close.shape[0]

        bhd = bbs_.bollinger_hband().to_numpy().reshape(rows, 1)
        bchb = bbs_.bollinger_hband_indicator().to_numpy().reshape(rows, 1)
        blb = bbs_.bollinger_lband().to_numpy().reshape(rows, 1)
        bclb = bbs_.bollinger_lband_indicator().to_numpy().reshape(rows, 1)
        bmv = bbs_.bollinger_mavg().to_numpy().reshape(rows, 1)
        bwb = bbs_.bollinger_wband().to_numpy().reshape(rows, 1)

        bbs = np.concatenate([bhd, bchb, blb, bclb, bmv, bwb], axis=1)
        return bbs



