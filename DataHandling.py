from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
from pandas import DataFrame
from keras_preprocessing.sequence import TimeseriesGenerator


class DataHandling:

    timesteps_e = 0
    features = 0
    timesteps_d = 0
    min_l = []
    max_l = []
    normalizer = MinMaxScaler()
    seed = []

    def __init__(self, timesteps_e, timesteps_d, features):
        self.timesteps_e = timesteps_e
        self.timesteps_d = timesteps_d
        self.features = features

    def time_normalize(self, data_req,  for_layer, mode='ft'):
        if for_layer == 'e':
            if mode == 'ft':
                i = self.timesteps_e
                while i <= data_req.shape[0]:
                    slice_req = data_req[i - self.timesteps_e: i]
                    data_req[i - self.timesteps_e: i] = self.normalizer.fit_transform(slice_req)
                    i = i + self.timesteps_e
            elif mode == 't':
                data_req = self.normalizer.transform(data_req)
        else:
            if mode == 'ft':
                i = self.timesteps_d
                while i <= data_req.shape[0]:
                    slice_req = data_req[i - self.timesteps_d: i]
                    data_req[i - self.timesteps_d: i] = self.normalizer.fit_transform(slice_req)
                    i = i + self.timesteps_d
            elif mode == 't':
                data_req = self.normalizer.transform(data_req)
        return data_req

    def inverse_time_normalizer(self, data_req):
        y_in = self.normalizer.inverse_transform(data_req)
        return y_in

    def get_min_max(self, data_req):
        obtained_min = data_req.min()
        obtained_max = data_req.max()
        self.min_l.append(obtained_min)
        self.max_l.append(obtained_max)

    def obtain_real_values(self, data_req):
        iter_ = 0
        real_val_list = []
        while iter_ != len(self.min_l):
            single_row = data_req[:, iter_]
            real_val = single_row * (self.max_l[iter_] - self.min_l[iter_]) + self.min_l[iter_]
            real_val_list.append(real_val)
            iter_ = iter_ + 1
        real_val_np = np.array(real_val_list).reshape(data_req.shape[0], data_req.shape[1])
        return real_val_np

    def pca(self, data_req, components):
        pca_ = PCA()
        i = self.timesteps_e
        while i <= data_req.shape[0]:
            slice_req = data_req[i - self.timesteps_e: i]
            data_req[i - self.timesteps_e: i] = pca_.fit_transform(slice_req)
            i = i + self.timesteps_e
        pca_val = data_req[:, :components]
        return pca_val

    def slice_from(self, data_req, from_index, columns, format_req='np'):
        data_req_pd = DataFrame(data_req)
        rows = data_req_pd.shape[0] - from_index
        data_req_from = data_req_pd[from_index:]
        if format_req == 'pd':
            return data_req_from
        else:
            data_req_np = data_req_from.to_numpy()
            data_req_res = data_req_np.reshape(rows, columns)
            return data_req_res

    def create_x(self, data_req):
        samples = int(data_req.shape[0] / self.timesteps_e)
        x = data_req.reshape(samples, self.timesteps_e, self.features)
        return x

    def create_y(self, data_req, dim):
        samples = int(data_req.shape[0] / self.timesteps_d)
        y = data_req.reshape(samples, self.timesteps_d, dim)
        return y

    def log_transform(self, data_req):
        data_req_log = np.log(data_req)
        return data_req_log

    def log_returns(self, data_req):
        data_req_df = DataFrame(data_req)
        data_req_pct_change = data_req_df.pct_change(periods=1)
        data_req_lr = self.log_transform(1+data_req_pct_change)
        return data_req_lr

    def inverse_log_returns(self, data_req):
        data_req_o = np.exp(data_req) - 1
        for data_index in range(data_req.shape[0]):
            for seed_index in range(len(self.seed)):
                data_req_o[data_index, seed_index] = data_req_o[data_index, seed_index] * self.seed[seed_index] + \
                                                     self.seed[seed_index]
                self.seed[seed_index] = data_req_o[data_index, seed_index]
        return data_req_o

    def set_seeds(self, open_p, high_p, low_p, close_p):
        self.seed.append(open_p[open_p.shape[0] - 1])
        self.seed.append(high_p[high_p.shape[0] - 1])
        self.seed.append(low_p[low_p.shape[0] - 1])
        self.seed.append(close_p[close_p.shape[0] - 1])



