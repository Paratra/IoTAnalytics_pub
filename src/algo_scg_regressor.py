#!/usr/bin/env python3
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

import numpy as np
import json
import sys

import numpy as np

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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from pdb import set_trace as st
# from algo_dsp import HelenaDSP
# from algo_nn import NeuralNetworks

# import torch
from tqdm import tqdm

sns.set()
# get_ipython().run_line_magic('matplotlib', 'inline')
import sys, os
import joblib as jb #import dump, load
import warnings
warnings.filterwarnings('ignore')

import tsfel

from utils import *
# from check_distribution import *
from collections import Counter

from generic_regressor import GenericRegressor


if __name__ == '__main__':
    if(len(sys.argv) < 4):
        print(f"Usage: {sys.argv[0]} data_file config_file number_of_labels fit/predict/score")
        print(f"Example: {sys.argv[0]} ../data/synthetic_data.385_1.npy ../data/config.json 1 fit")
        print(f"Example: {sys.argv[0]} ../data/synthetic_data.385_1.npy ../data/config.json 1 predict")
        print(f"Example: {sys.argv[0]} ../data/synthetic_data.385_1.npy ../data/config.json 1 score")
        exit()

    ### get arguments
    data_file = sys.argv[1]
    config_file = sys.argv[2]
    num_labels = int(sys.argv[3])
    run_mode = sys.argv[4]

    ### load data and config
    data_set = np.load(data_file)
    config = json.load(open(config_file, 'r'))

    st()


    X = data_set[:,:-num_labels]
    Y = data_set[:,-num_labels:]



    regressor_algorithms = [
                          #('Ridge', linear_model.Ridge(alpha=.5), 'integrated'),
                          # ('LassoLars',linear_model.LassoLars(alpha=.1), 'separated'),
                          ('BayesianRidge', linear_model.BayesianRidge(),  'separated'),
                          ('KNeighborsRegressor', KNeighborsRegressor(n_neighbors=3, n_jobs=-1, weights='distance'), 'integrated'),
                          ('GradientBoostingRegressor', GradientBoostingRegressor(random_state=1, n_estimators=10), 'separated'),
                        #   ('RandomForestRegressor', RandomForestRegressor(max_depth=2, random_state=0), 'separated')
                          ]

    # transformers = [('VarianceThreshold',VarianceThreshold(threshold=0)),
    #               ('StandardScaler',preprocessing.StandardScaler())]

    transformers = []

    feature_selector = []

    if run_mode == 'fit':
        generic_reg = GenericRegressor(regressor_algorithms, transformers, feature_selector, param_model_path='../models', load_model=False)
        generic_reg.fit(X, Y)

        # import pdb; pdb.set_trace()

    elif run_mode == 'predict':
        generic_reg = GenericRegressor(regressor_algorithms, transformers, feature_selector, param_model_path='../models', load_model=True)

        ## get prediction
        pred_re_dict = generic_reg.predict(X)
        st()

        ## input and outout size
        pred_re_dict = generic_reg.predict(X[0:1,:])
        print(f'input size {X[0:1,:].shape}, output size {pred_re_dict.shape}')
        print(f'input {X[0:1,:]}, output {pred_re_dict}')

        pred_re_dict = generic_reg.predict(X[0:2,:])
        print(f'input size {X[0:2,:].shape}, output size {pred_re_dict.shape}')
        print(f'input {X[0:2,:]}, output {pred_re_dict}')


    elif run_mode == 'score':
        generic_reg = GenericRegressor(regressor_algorithms, transformers, feature_selector, param_model_path='../models', load_model=True)

        ## evaluation
        eval_re = generic_reg.score(X, Y, criteria='ieee')
        for ind in range(eval_re.shape[0]):
            print('------------------------')
            print(f'Target {ind+1}: \n MD: {eval_re[ind,0]} \n MAD: {eval_re[ind, 1]} \n MAPE: {eval_re[ind, 2]} \n CP5: {eval_re[ind, 3]} \n CP10: {eval_re[ind, 4]} \n CP15: {eval_re[ind, 5]}')

        # print()

        # import pdb; pdb.set_trace()
