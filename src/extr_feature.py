#!/usr/bin/env python3
# %%
# from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from sklearn import preprocessing
from datetime import datetime
from dateutil import tz
import pytz
import os, sys

import csv
import matplotlib.pyplot as plt
import pywt
from influxdb import InfluxDBClient
import operator
import scipy.signal as sg
import scipy as sp
import tsfel

# import algo_dsp as dsp
import warnings
warnings.filterwarnings("ignore")

from utils import *
from pdb import set_trace as st


def robust_mean(input_arr):
    input_arr = np.array(input_arr)
    # st()
    too_slow = np.where(input_arr>200)[0]
    if len(too_slow) != 0:
        input_arr = np.delete(input_arr, too_slow, 0)

    too_quick = np.where(input_arr<30)[0]
    if len(too_quick) != 0:
        input_arr = np.delete(input_arr, too_quick, 0)

    if len(input_arr) == 0:
        # st()
        return -1
    # st()
    # sort_arr = np.sort(input_arr)
    max_arr = max(input_arr)
    min_arr = min(input_arr)
    range_ = np.arange(min_arr, max_arr)
    threshold = 10
    in_range_list = []
    for i in range_:
        candidates = np.where(abs(input_arr - i)<=threshold)[0]
        in_range_list.append([i, len(candidates)])
    in_range_arr = np.array(in_range_list)
    # st()
    # if in_range_arr

    robust_mean_dist = np.mean(in_range_arr[np.where(in_range_arr[:,1]==max(in_range_arr[:,1]))][:,0])

    # st()
    return robust_mean_dist

# plt.rcParams['figure.figsize'] = [20, 8]  # Bigger images
def save_feature_file(data_file, all_features_labels, num_labels):
    row_size = all_features_labels.shape[1] - num_labels
    out_name = data_file
    out_name = os.path.splitext(out_name)[0]
    # print(np.round(label_data_set[0, :]))
    # out_name = data_file
    # out_name = os.path.splitext(out_name)[0]
    # # while(True):
    # #     new_name = os.path.splitext(out_name)[0]
    # #     if out_name == new_name:
    # #         break
    # #     out_name = new_name

    # print(data_file, out_name)
    # exit()
    feature_npy_file = f'{out_name}.{row_size}_{num_labels}.npy' #len(feature_label_range)
    feature_csv_file = f'{out_name}.{row_size}_{num_labels}.csv'
    np.save(feature_npy_file, all_features_labels)
    print("Saved features and labels into: ", feature_npy_file)
    all_features_labels.to_csv(feature_csv_file, index=False)
    print("Saved features and labels into: ", feature_csv_file)

def extract_features_set(feature_name, data_file, num_data_labels):
    full_data_set = load_data_file(data_file)
    # full_data_set = full_data_set[0:100,]
    full_signal_set = full_data_set[:, :-num_data_labels]
    full_label_set = full_data_set[:, -num_data_labels:]
    num_rows = full_signal_set.shape[0]
    num_labels = num_data_labels

    all_features_labels = []
    start = 0
    slide = 1000 # save every 100 rows
    while start < num_rows:
        end = min(start + slide, num_rows)
        if start == end:
            break
        signal_set = full_signal_set[start:end, :]
        label_set = full_label_set[start:end, :]

        if feature_name == 'tsfel':
            features_set = tsfel_feature_extraction(signal_set, fs=100)
        elif feature_name == 'tsfel_win':
            features_set = tsfel_win_feature_extraction(signal_set, fs=100)
        elif feature_name == 'spectral':
            features_set = spectral_features_extractor(signal_set, fs=100)
        else:
            print("Wrong feature name input!")
            exit()

        row_size = features_set.shape[1]

        # label_set = pd.DataFrame(label_set, columns=label_names)
        label_set = pd.DataFrame(label_set)
        # label_set = label_set.loc[saved_index,:].reset_index(drop=True) ### select the index of data and reset the index
        features_labels = pd.concat( (features_set, label_set ), axis=1) ## NaN problem occurs here. As features_set and label_set does not have same number of data.

        if start == 0:
            all_features_labels = features_labels
        else:
            all_features_labels = pd.concat( (all_features_labels, features_labels), axis=0)



        all_features_labels = all_features_labels.reset_index(drop=True)

        save_feature_file(data_file, all_features_labels, num_labels)
        start = end

    # st()

    return all_features_labels


from sklearn.neighbors import KNeighborsClassifier
from lime_timeseries import LimeTimeSeriesExplainer
from sklearn.metrics import mean_absolute_error as mae

def lime_selector(feature_set, num_labels, target_label_name='S'):
    feature_set = feature_set.to_numpy()
    x_train = feature_set[:, :-num_labels]
    y_train = np.round(feature_set[:, label_index(target_label_name)])
    print(x_train.shape, y_train.shape) # y_train)

    knn = KNeighborsClassifier(n_jobs=-1)
    knn.fit(x_train, y_train)

    # print(f'MAE KNeighborsClassifier: %f' % (mae(y_test, knn.predict(x_test))))

    # idx = 15 # explained instance
    idx = np.where(y_train==120)
    num_features = 5 # how many feature contained in explanation
    num_slices = round(x_train.shape[1]/20) # split time series
    series = np.mean(x_train[idx, :].squeeze(),0)
    # import pdb; pdb.set_trace()

    explainer = LimeTimeSeriesExplainer()
    exp = explainer.explain_instance(series, knn.predict_proba, num_features=num_features, num_samples=5000, num_slices=num_slices,
                                        replacement_method='total_mean')


    values_per_slice = math.ceil(len(series) / num_slices)

    index_weight = []
    for i in range(num_features):
        feature, weight = exp.as_list()[i]
        start = feature * values_per_slice
        end = start + values_per_slice
        # index_weight =
        index_weight += list (np.arange(start, end))

        # color = 'red' if weight < 0 else 'green'
        # plt.axvspan(start , end, color=color, alpha=abs(weight*20))

    selected_feature_set = pd.DataFrame(np.concat( (x_train[:,index_weight], feature_set[:, -num_labels:]), axis=1))

    return selected_feature_set

def spectral_features_extractor(signal_set,fs = 100):
    print("spectral_features_extractor:", signal_set.shape)
    N = signal_set.shape[1]

    feature_set = []
    for ind, signal in enumerate(signal_set[:,]):
        _, cur_x = dsp.signal_fft(signal, N, fs)

        feature_set.append(cur_x)
    extracted_features = pd.DataFrame(feature_set)
    return extracted_features

# # Feature Extraction
# **TSFEL** is a python package for extracting features from a sequential/time series data. It extracts **statistical**, **spectral**, and **temporal** features from the raw signal. While extracing, some of the features have missing values in terms of positive or negative infinite. I replace those with median of the features. Since, we are collecting features based on window, we compute summary statistics (mean, median, min, max, sd) for each of the feaures. In this process, the resulting features produce better results than directly using window level features. This is completely result based. No theoritical analysis is performed.

def tsfel_win_feature_extraction(signal_set, fs=100, window_size=200):
    print("tsfel_win_feature_extraction:", signal_set.shape)

    cfg = tsfel.get_features_by_domain()
    window_mean = []
    window_median = []
    window_max = []
    window_min = []
    window_std = []
    for i in range(signal_set.shape[0]):
        features = tsfel.time_series_features_extractor(cfg, signal_set[i], window_size = window_size, fs=fs)
        mean = features.mean()
        med = features.median()
        mini = features.min()
        maxi = features.max()
        sd = features.std()
        window_mean.append(mean)
        window_median.append(med)
        window_max.append(maxi)
        window_min.append(mini)
        window_std.append(sd)
    extracted_features = pd.concat([pd.DataFrame(window_median), pd.DataFrame(window_mean), pd.DataFrame(window_min), pd.DataFrame(window_max), pd.DataFrame(window_std) ], axis= 1)
    return extracted_features

def tsfel_feature_extraction(signal_set,fs=100):
    print("tsfel_feature_extraction:", signal_set.shape)

    cfg = tsfel.get_features_by_domain()
    extracted_features = tsfel.time_series_features_extractor(cfg, signal_set,fs=fs)
    return extracted_features

if __name__ == "__main__":

    if(len(sys.argv) > 2):
        data_file = sys.argv[1]
        num_data_labels = int(sys.argv[2])
        feature_name = sys.argv[3]
    else:
        print(f"Usage: {sys.argv[0]} data_file num_labels tsfel/tsfel_win/spectral")
        print(f"Example: {sys.argv[0]} ../data/synthetic_data.npy 1 tsfel")
        exit()

    features_labels = extract_features_set(feature_name, data_file,
        num_data_labels = num_data_labels,
)

    save_feature_file(data_file, features_labels, num_data_labels)

# %%
