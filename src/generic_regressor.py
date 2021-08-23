#!/usr/bin/env python3
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

import numpy as np
import time
import sys
import re
import pytz
import time
import math
import subprocess
import sys
import random
import webbrowser
import numpy as np
from datetime import datetime
from dateutil import tz
import random
import copy
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
# from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
kernel = DotProduct() + WhiteKernel()
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPRegressor
import sklearn.metrics
from pdb import set_trace as st
from sklearn.metrics import mean_absolute_error, \
                            mean_squared_error, \
                            explained_variance_score, \
                            max_error,\
                            mean_squared_log_error, \
                            median_absolute_error, \
                            r2_score, \
                            mean_poisson_deviance, \
                            mean_gamma_deviance \
                            # mean_absolute_percentage_error\

from sklearn.svm import SVR
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
# from algo_dsp import HelenaDSP
# from algo_nn import NeuralNetworks
from joblib import dump, load
import torch
from tqdm import tqdm

sns.set()
# get_ipython().run_line_magic('matplotlib', 'inline')
import sys, os
# from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

import tsfel
# import algo_dsp as dsp

from utils import *
# from check_distribution import *
from collections import Counter
from eval_result_stats import eval_result_stats


# # X_data can be one row or multiple rows of sensor signals
# def process_data(X_data):
#     # %% [markdown]
#     # # Noise removal
#     # We performed noise removal using FFT transformation
#     X_result = X_data
#     result = pd.DataFrame([dsp.denoising_using_fft(signal= X_result[i], threshold= 1 ) for i in range(X_result.shape[0])]).values
#     # signal_test = pd.DataFrame([dsp.butter_lowpass_filter(signal_test[i], 16, 100, 5) for i in range(signal_test.shape[0])]).values
#     return X_result

# %% [markdown]
# # Standardization and Outlier removing
# First, we standardized the data using z-score standardization and then the outliers are detected if the data is outside of 3-SD in the positive side. Then the outliers are replaced using KNN techniqe. One interesting point is that- if we use 3-SD in both pos and neg side to detect the outlier then the reuslt is not as good as using 3-SD only in positive side. Might be the lower points in the signal convey more information and are not outliers.
# In the pre-processing: we always use parameters from training data to standardize and impute for both training and test data to avoid data leakage.
def scale_data(X_train, X_test):
    signal_train = X_train
    scaler = preprocessing.StandardScaler()
    imputer  = KNNImputer(n_neighbors= 3)
    train_scaled = scaler.fit_transform(signal_train)
    train_scaled = np.where(train_scaled > 3, np.nan, train_scaled)
    train_scaled = imputer.fit_transform(train_scaled)

    signal_test = X_test
    test_scaled = scaler.transform(signal_test)
    test_scaled= np.where(test_scaled > 3, np.nan, test_scaled)
    test_scaled = imputer.transform(test_scaled)
    return train_scaled, test_scaled

# %% [markdown]


# %% [markdown]
# # Feature selection
# After the feature extraction, I removed features if they are highly correlated and contains low variance using variance threshold technique. After dropping the unnecessary variables, the data normalized where training parameters were used to normalize both train and test data
def select_features(train_features, test_features, train_labels):
    X_train = train_features
    X_test = test_features
    Y_train = train_labels

    print("before VarianceThreshold feature selection", X_train.shape, X_test.shape)
    selector = VarianceThreshold(threshold=0)
    X_train = selector.fit_transform(X_train)
    X_test = selector.transform(X_test)
    print("after VarianceThreshold feature selection", X_train.shape, X_test.shape)

    # print("before correlated feature drop", X_train.shape, X_test.shape)
    # X_train = pd.DataFrame(X_train)
    # X_test = pd.DataFrame(X_test)
    # X_train.drop("Unnamed: 0", axis= 1)
    # X_test.drop("Unnamed: 0", axis= 1)
    # corr_features = tsfel.correlated_features(X_train, threshold= .99) # original .99
    # X_train.drop(corr_features, axis=1, inplace=True)
    # X_test.drop(corr_features, axis=1, inplace=True)
    # print("after correlated feature drop selection", X_train.shape, X_test.shape)

    # print("before sklearn feature selection", X_train.shape, X_test.shape)
    # clf = ExtraTreesClassifier(n_estimators=50)
    # clf = clf.fit(X_train, Y_train[:, 0])
    # # print(clf.feature_importances_)
    # # array([ 0.04...,  0.05...,  0.4...,  0.4...])
    # model = SelectFromModel(clf, prefit=True)
    # X_train = model.transform(X_train)
    # X_test = model.transform(X_test)
    # print("after sklearn feature selection", X_train.shape, X_test.shape)

    scaler = preprocessing.StandardScaler()
    ## scaler = preprocessing.MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


class AnomalousPredictionFix:
    def __init__(self):
        self.min_val = None
        self.max_val = None
        self.mean= 0
    def fit(self,x):
        self.min_val = x.min()
        self.max_val=x.max()
        self.mean=x.mean()

    def transform(self,x):
        x = np.nan_to_num(x,nan=self.mean,posinf=self.mean,neginf=self.mean)
        x[x>self.max_val+2] = self.mean
        x[x<self.min_val-2] = self.mean
        return x

def combine_abbreviation(list):
    str_re = ''
    for each_l in list:
        str_re += each_l
    return str_re

def get_name(param_alg):
    for ind, each_alg in enumerate(param_alg):
        if ind == 0:
            model_name = combine_abbreviation( re.findall('([A-Z])', each_alg[0]) ) + '_' + each_alg[2][0]
        else:
            model_name = model_name + '_' + combine_abbreviation( re.findall('([A-Z])', each_alg[0]) ) + '_' + each_alg[2][0]
    return model_name+'.joblib'

class GenericRegressor():
    """docstring for GenericRegressor"""
    def __init__(self, param_alg, param_transformers, feature_selector, param_model_path='../models', load_model=False):
        super(GenericRegressor, self).__init__()
        self.param_alg = param_alg
        self.param_transformers = param_transformers
        self.feature_selector = feature_selector
        self.param_model_path = param_model_path
        self.load_model = load_model

        self.model_name = get_name(self.param_alg)


    def fit(self, X_train, Y_train):

        self.transformers = []
        if len(self.param_transformers) != 0:
            for jnd, each_trans in enumerate(self.param_transformers):
                cur_trans = copy.deepcopy(each_trans[1])
                X_train = cur_trans.fit_transform(X_train)
                self.transformers.append(cur_trans)
        self.feature_selector_index = []
        if len(self.feature_selector) != 0:        
            for jnd, each_selector in enumerate(self.feature_selector):
                selector_index = each_selector[1]
                self.feature_selector_index += list(selector_index)
                X_train = X_train[:,selector_index]

            ## remove repeat features
            self.feature_selector_index = list(set(self.feature_selector_index))

        # import pdb; pdb.set_trace()

        # self.selector_X = VarianceThreshold(threshold=0)
        # X_train = self.selector_X.fit_transform(X_train)

        # self.scaler_X = preprocessing.StandardScaler()
        # X_train = self.scaler_X.fit_transform(X_train)


        if Y_train.ndim == 1:
            self.num_of_target = 1
            Y_train = Y_train[:,np.newaxis]
        else:
            self.num_of_target = Y_train.shape[1]
        self.transformers.append(self.num_of_target)
        # import pdb; pdb.set_trace()

        self.regressors = []
        for ind, each_alg in enumerate(self.param_alg):

            if each_alg[2] == 'integrated':
                vars()[each_alg[0]] = each_alg[1].fit(X_train, Y_train)
                self.regressors.append([each_alg[0], vars()[each_alg[0]]])
            if each_alg[2] == 'separated':
                for jnd in range(self.num_of_target):
                    cur_reg = copy.deepcopy(each_alg[1])
                    # import pdb; pdb.set_trace()
                    vars()[each_alg[0]+f'_{jnd}'] = cur_reg.fit(X_train, Y_train[:,jnd][:,np.newaxis])
                    self.regressors.append([each_alg[0], vars()[each_alg[0]+f'_{jnd}']])


        ### save model
        dump([self.regressors, self.transformers, self.feature_selector_index],
                os.path.join(self.param_model_path, f'{self.model_name}.joblib'))
        print(f'Model {self.model_name}.joblib is saved!')



    def predict(self, input):
        if self.load_model == False:
            print('No Trained Model is loaded, please set load_model as True and set loading path as well if you wish to use trained model for prediction.')
        else:
            [self.regressors, self.transformers, self.feature_selector_index] = load(os.path.join(self.param_model_path, f'{self.model_name}.joblib'))
            print('Trained Model is Loaded!')

        ### applying selectors and scalers
        if len(self.transformers) != 0:
            for each_trans in self.transformers[:-1]:
                input = each_trans.transform(input)

        if len(self.feature_selector_index) != 0:
            input = input[:,self.feature_selector_index]

        ### get number of targets
        self.num_of_target = self.transformers[-1]



        ### predicting and bagging
        results = []
        
        param_alg_arr = np.array(self.param_alg)
        for [each_name, each_reg] in self.regressors:
            
            if (each_name == 'FCN') or (each_name == 'CNN') or (each_name == 'LSTM'):
                regressor = param_alg_arr[np.where(param_alg_arr[:,0] == each_name)[0][0],1]
                regressor.load_model('../models')
                cur_pred = regressor.predict(input)
                # st()
            else:    
                cur_pred = each_reg.predict(input)
            
            if cur_pred.ndim == 2:
                for i in range(self.num_of_target):
                    results.append(cur_pred[:,i])

            else:
                results.append(cur_pred)

        if np.array(results).shape[1] == 1:
            re_arr = np.mean(np.array(results).T.reshape((-1,self.num_of_target)),0)[np.newaxis, :]
        else:
            re_arr = np.mean(np.array(results).T.reshape((np.array(results).shape[1],-1,self.num_of_target)),1)

        return re_arr

    def score(self,X,Y,criteria='ieee'):
        input = X
        if self.load_model == False:
            print('No Trained Model is loaded, please set load_model as True and set loading path as well if you wish to use trained model for prediction.')
        else:
            [self.regressors, self.transformers, self.feature_selector_index] = load(os.path.join(self.param_model_path, f'{self.model_name}.joblib'))

            print('Trained Model is Loaded!')

        ### applying selectors and scalers
        if len(self.transformers) != 0:
            for each_trans in self.transformers[:-1]:
                input = each_trans.transform(input)

        if len(self.feature_selector_index) != 0:
            input = input[:,self.feature_selector_index]


        ### get number of targets
        self.num_of_target = self.transformers[-1]



        ### predicting and bagging
        results = []
        for each_reg in self.regressors:
            st()
            cur_pred = each_reg.predict(input)
            if cur_pred.ndim == 2:
                for i in range(self.num_of_target):
                    results.append(cur_pred[:,i])

            else:
                results.append(cur_pred)

        if np.array(results).shape[1] == 1:
            re_arr = np.mean(np.array(results).T.reshape((-1,self.num_of_target)),0)[np.newaxis, :]
        else:
            re_arr = np.mean(np.array(results).T.reshape((np.array(results).shape[1],-1,self.num_of_target)),1)

        eval_re = []
        if criteria == 'ieee':
            for i in range(self.num_of_target):
                if Y.ndim == 2:
                    eval_re.append(eval_result_stats(Y[:,i], re_arr[:,i]))
                else:
                    eval_re.append(eval_result_stats(Y, re_arr))

        return np.array(eval_re)

    def get_param(self):
        pass

if __name__ == '__main__':
    kn = 1
    load_feature = True # False: recalculate features; True: load features
    if(len(sys.argv) >=2):
      kn = int(sys.argv[1])
    if(len(sys.argv) >=3):
      load_feature = str2bool(sys.argv[2])


    ### prepare data

    file_list = [
      ("../../data/b8_27_eb_5b_35_37_80_200_float.npy", "../../data/windoow_all_feat_with_summary_vali1.csv"),
      # ("../../data/b8_27_eb_6c_6e_22_80_200_float.npy", "../../data/windoow_all_feat_with_summary_vali2.csv"),
      # ("../../data/b8_27_eb_80_1c_cf_80_200_float.npy", "../../data/windoow_all_feat_with_summary_vali3.csv"),
      # ("../../data/b8_27_eb_23_4b_2b_80_200_float.npy", "../../data/windoow_all_feat_with_summary_vali4.csv"),
    ]
    vital_indexes = [('H', -4), ('R', -3), ('S', -2), ('D', -1)]
    # vital_indexes = [('S', -2)]
    # vital_indexes = [('H', -4)]
    # vital_indexes = [('R', -3)]
    # vitals = [('HRSD', 0, 4)] # some regressor does not support multiple laabels and cannot use this
    data_set = load_separated_features_data(file_list=file_list)

    X_train, Y_train, X_test, Y_test = prepare_train_test_data(data_set=data_set,
                                                                split_ratio = 0.7,
                                                                time_sorted=False,
                                                                target_indexes=vital_indexes,
                                                                target_distribution="none",
                                                                target_range=np.arange(120,141,2))

    X_train, X_test = select_features(X_train, X_test, Y_train)

    import pdb; pdb.set_trace()

    # param_alg =  [('KNeighborsRegressor', KNeighborsRegressor(n_neighbors=3, n_jobs=-1, weights='distance'), 'integrated'),
    #                 ('BayesianRidge', linear_model.BayesianRidge(),  'separated')]

    # ( Algo_Name, Algo_Model, Integrated_or_Seperated_MultiVar_Regression_Training)
    regressor_algorithms = [('Ridge', linear_model.Ridge(alpha=.5), 'integrated'),
                            ('LassoLars',linear_model.LassoLars(alpha=.1), 'separated'),
                            ('BayesianRidge', linear_model.BayesianRidge(),  'separated'),
                            ('KNeighborsRegressor', KNeighborsRegressor(n_neighbors=3, n_jobs=-1, weights='distance'), 'integrated'),
                            ]

    transformers = [('VarianceThreshold',VarianceThreshold(threshold=0)),
                    ('StandardScaler',preprocessing.StandardScaler())]

    '''
    Train Model
    '''
    generic_reg = GenericRegressor(regressor_algorithms, transformers, param_model_path='../models', load_model=False)
    generic_reg.fit(X_train, Y_train)


    '''
    Predict
    '''
    generic_reg = GenericRegressor(regressor_algorithms, transformers, param_model_path='../models', load_model=True)

    ## input and outout size
    pred_re_dict = generic_reg.predict(X_test[0:1,:])
    print(f'input size {X_test[0:1,:].shape}, output size {pred_re_dict.shape}')

    pred_re_dict = generic_reg.predict(X_test[0:2,:])
    print(f'input size {X_test[0:2,:].shape}, output size {pred_re_dict.shape}')



    import pdb; pdb.set_trace()
