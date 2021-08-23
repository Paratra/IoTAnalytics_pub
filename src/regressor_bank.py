'''
author: ming
ming.song.cn@outlook.com
copyright@2020
'''


import os
from pdb import Pdb
import sys
import numpy as np
import torch
from torch.optim import *
from torch import nn, optim, cuda
from torch.utils.data import Dataset, DataLoader
# from torch.utils.data import
from sklearn import preprocessing
import copy
from random import sample
from math import isnan
import datetime
import pickle
from scgkit2.signal.signal_distort import signal_distort
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from pdb import set_trace as st
import warnings
warnings.filterwarnings("ignore")

# batch_size = 1000
# test_only = False
# VISUAL_FLAG = False
# # test_only = bool(sys.argv[1])
# lr = 0.001
# dim_feature = 100

def get_size(input_shape,k_size,max_pool_k_size, layers):
    for i in range(layers):
        if i == 0:
            size = int((input_shape - k_size + 1)/max_pool_k_size)
        elif i == layers-1:
            size = int((size - k_size + 1))
        else:
            size = int((size - k_size + 1)/max_pool_k_size)
    return size

class Initial_Dataset(Dataset):
    """docstring for ."""

    def __init__(self, X, Y):  # before this length is data, after is label

        self.array_Tx = X
        self.array_Ty = Y

    def __getitem__(self, index):
        data_ = self.array_Tx[index, :]
        gt_ = self.array_Ty[index, :] #

        return data_, gt_

    def __len__(self):
        return self.array_Tx.shape[0]


#
# class CNN_LSTM_Net(nn.Module):
#     """docstring for CNN_LSTM_Net."""
#
#     def __init__(self, LOG=False):
#         super(CNN_LSTM_Net, self).__init__()
#         #### define layers
#
#         ## CNN part
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3)
#         self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7)
#         self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5)
#         self.batch_norm1d_1 = nn.BatchNorm1d(128)
#         self.batch_norm1d_2 = nn.BatchNorm1d(256)
#         self.batch_norm1d_3 = nn.BatchNorm1d(64)
#
#         self.max_pool1d = nn.MaxPool1d(kernel_size=2)
#         self.prelu = nn.PReLU()
#         self.dropout = nn.Dropout(p=0.5)
#
#         ## LSTM part
#         self.lstm = nn.LSTM(input_size=746, hidden_size=128, batch_first=True, num_layers=1)
#         self.decoding_layer = nn.Linear(128, 4)
#
#
#
#     def forward(self, x):
#         # import pdb; pdb.set_trace()
#         conv1 = self.conv1(x)
#         conv1 = self.batch_norm1d_1(conv1)
#         conv1 = self.prelu(conv1)
#         conv1 = self.dropout(conv1)
#         conv1 = self.max_pool1d(conv1)
#
#         conv2 = self.conv2(conv1)
#         conv2 = self.batch_norm1d_2(conv2)
#         conv2 = self.prelu(conv2)
#         conv2 = self.dropout(conv2)
#         conv2 = self.max_pool1d(conv2)
#
#         out, (hid, c) = self.lstm(conv2)
#         pred = self.decoding_layer(hid[0])
#
#         return pred

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3000, hidden_size=128, batch_first=True, num_layers=3)
        self.decoding_layer = nn.Linear(128, 4)
#
    def forward(self, input_seq):
        out, (hid, c) = self.lstm(input_seq)
        pred = self.decoding_layer(hid[0])
        return pred

class LstmAttentionNet(nn.Module):
    def __init__(self, num_layers, hidden_size, output_features):
        super(LstmAttentionNet, self).__init__()
        # hidden_size = 100
        attention_size = hidden_size
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.w_omega = nn.Parameter(torch.randn(hidden_size,attention_size))
        self.b_omega = nn.Parameter(torch.randn(attention_size))
        self.u_omega = nn.Parameter(torch.randn(attention_size,1))
        self.decoding_layer = nn.Linear(hidden_size, output_features)


    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = x.unsqueeze(2)
        out, (h, c) = self.lstm(x)
        v = torch.matmul(out,self.w_omega)+self.b_omega
        vu = torch.matmul(v, self.u_omega)
        weight= nn.functional.softmax(vu,dim=1)
        out_weighted = torch.sum(out*weight,1)
        y_pred = self.decoding_layer(out_weighted)

        return y_pred#, weight


class CNN_Net(nn.Module):
    """docstring for CNN_Net."""

    def __init__(self, input_shape, layers, output_features, out_channels, kernel_size):
        super(CNN_Net, self).__init__()
        #### define layers
        assert len(out_channels) == layers
        self.layers = layers
        self.out_channels = out_channels
        # ## CNN part
        self.net = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        for i in range(layers):
            if i == 0:
                self.net.append( nn.Conv1d(in_channels=1, out_channels=out_channels[i], kernel_size=kernel_size) ) 
            else:
                self.net.append( nn.Conv1d(in_channels=out_channels[i-1], out_channels=out_channels[i], kernel_size=kernel_size) )
            
            self.batch_norm.append( nn.BatchNorm1d(out_channels[i]) )

        # , nn.BatchNorm1d(out_channels[i])
       
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5)
        # self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5)
        # self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5)
        # self.batch_norm1d_1 = nn.BatchNorm1d(128)
        # self.batch_norm1d_2 = nn.BatchNorm1d(256)
        # self.batch_norm1d_3 = nn.BatchNorm1d(64)

        self.max_pool1d = nn.MaxPool1d(kernel_size=3)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(p=0.5)

        ## LSTM part
        # self.lstm = nn.LSTM(input_size=3000, hidden_size=64, batch_first=True, num_layers=1)

        # self.decoding_layer1 = nn.Linear(self.flatten_size, 128)
        # st()
        # flatten_size = 
        
        flatten_size = get_size(input_shape=input_shape,k_size=kernel_size,max_pool_k_size=3, layers=layers )
        self.decoding_layer1 = nn.Linear(flatten_size*out_channels[-1], 128)
        self.decoding_layer2 = nn.Linear(128, output_features)
        self.flatten = nn.Flatten()



    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = torch.unsqueeze(x, 1)
        for i in range(self.layers):
            if i == self.layers - 1:
                x = self.net[i](x)


            else:
                
                # # st()
                # self.net[i]()
                
                x = self.net[i](x)
                # st()
                x = self.batch_norm[i](x)
                x = torch.relu(x)
                x = self.dropout(x)
                x = self.max_pool1d(x)
                

        # flatten_size = x.shape[1] * x.shape[2]
        # flatten = self.flatten(x)
        
        # self.decoding_layer1 = nn.Linear(flatten_size, 128)

        # st()
        flatten = self.flatten(x)
        decode1 = self.decoding_layer1(flatten)
        pred = self.decoding_layer2(decode1)
        # st()
        return pred



class AE_Net(nn.Module):
    """docstring for AE_Net."""

    def __init__(self, input_shape):
        super(AE_Net, self).__init__()

        self.encoder_hidden_layer = nn.Linear(
            in_features=input_shape, out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=64
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=64, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=input_shape
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        state_logit = self.encoder_output_layer(activation)
        # import pdb; pdb.set_trace()
        code = torch.relu(state_logit)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = activation
        # reconstructed = torch.relu(activation)
        # import pdb; pdb.set_trace()
        return reconstructed, state_logit


class FCN_Net(nn.Module):
    """docstring for FCN_Net."""

    def __init__(self, input_features, output_features, layers, neurons):
        super(FCN_Net, self).__init__()
        #### define layers

        # self.net = []
        self.net = nn.ModuleList()
        for i in range(layers):
            if i == 0:
                self.net.append( nn.Linear(in_features=input_features, out_features=neurons) )
            if i == layers-1:
                self.net.append( nn.Linear(in_features=neurons, out_features=output_features) )
            else:
                self.net.append( nn.Linear(in_features=neurons, out_features=neurons) )



        # self.dropout = nn.Dropout(p=0.5)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        # import pdb; pdb.set_trace()
        
        for ind, each_layer in enumerate(self.net):
            if ind == len(self.net)-1:
                pred = each_layer(x)
            else:
                
                x = each_layer(x)
                x = torch.relu(x)

        return pred


class FCN_Model():
    """docstring for FCN_Model."""

    def __init__(self, input_features=1000, output_features=1, layers=6, neurons=20, learning_rate=0.001, batch_size=32, epoch_number=500):
        super(FCN_Model, self).__init__()
        ####
        self.device = torch.device('cuda' if cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch_number = epoch_number

        # self.ae_Net = AE_Net(input_shape=input_shape)
        self.reg_Net = FCN_Net(input_features=input_features, output_features=output_features, layers=layers, neurons=neurons)
        # self.reg_Net = LstmAttentionNet()

        # self.ae_Net = self.ae_Net.to(device = self.device)
        self.reg_Net = self.reg_Net.to(device = self.device)

        print(f"Using device:{self.device}")




    # def fit(self, all_data, window_len, devide_factor, learning_rate=0.001, batch_size=32, epoch_number=500, CONTINUE_TRAINING = False):
    def fit(self, X, Y):
        # self.data = all_data
        # self.window_len = X.shape[1]

        self.h_norm = 90
        self.r_norm = 20
        self.s_norm = 200
        self.d_norm = 100
        # data_train, data_test = self.normalize_and_devide(all_data, window_len, devide_factor)

        train_dataset = Initial_Dataset(X, Y)
        # self.scaler_x, self.scaler_y = train_dataset.get_scalers()
        # test_dataset = Initial_Dataset(X, Y)

        # import pdb; pdb.set_trace()

        train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=4)
        # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

        ### training component
        loss_fn = torch.nn.MSELoss()
        optimizer_reg = optim.Adam(self.reg_Net.parameters(), lr=self.learning_rate)

        scheduler_reg = lr_scheduler.StepLR(optimizer_reg,step_size=5, gamma = 0.95)

        self.last_error = 1e5
        for e in range(self.epoch_number):

            for train_tensor_x, train_tensor_y in train_loader:
                optimizer_reg.zero_grad()

                # train_tensor_x_distorted = self.batch_scg_distorted(train_tensor_x, noise=0.3, sampling_rate=100, noise_frequency=[5, 10, 100])


                train_tensor_x = torch.tensor(train_tensor_x,dtype=torch.float32,device=self.device)
                train_tensor_y = torch.tensor(train_tensor_y,dtype=torch.float32,device=self.device)


                train_y_pred_reg = self.reg_Net(train_tensor_x)

                train_loss_tensor_reg = loss_fn(train_tensor_y, train_y_pred_reg)
                train_loss_reg = train_loss_tensor_reg.item()

                train_loss_tensor = train_loss_tensor_reg
                train_loss = train_loss_reg

                reg_pred_arr = train_y_pred_reg.cpu().detach().numpy().squeeze()
                reg_gt_arr = train_tensor_y.cpu().detach().numpy().squeeze()
                train_mae = mean_absolute_error(reg_gt_arr, reg_pred_arr)
                # st()
                train_loss_tensor.backward()

                optimizer_reg.step()
            
            print(f'Epoch {e} train MSE: {train_loss} ')
            print(f'        train REG MAE: {train_mae}')

            self.error = train_mae
            if self.error < self.last_error:
                self.save_model(model_path='../models')
                self.last_error = self.error

            # st()


            # if e % 5 == 0 or e == self.epoch_number-1:
            #     loss_test = []
            #     pred_list = []
            #     gt_list = []
            #     for test_tensor_x, test_tensor_y in test_loader:

            #         test_tensor_x = torch.tensor(test_tensor_x,dtype=torch.float32,device=self.device)
            #         test_tensor_y = torch.tensor(test_tensor_y,dtype=torch.float32,device=self.device)

            #         test_y_pred_reg = self.reg_Net(test_tensor_x)

            #         test_loss_tensor_reg = loss_fn(test_tensor_y,test_y_pred_reg)

            #         test_loss_tensor = test_loss_tensor_reg
            #         reg_pred_arr = test_y_pred_reg.cpu().detach().numpy().squeeze()
            #         reg_gt_arr = test_tensor_y.cpu().detach().numpy().squeeze()
            #         gt_list.append(reg_gt_arr)
            #         pred_list.append(reg_pred_arr)

            #         test_loss = test_loss_tensor.item()
            #         loss_test.append(test_loss)
            #     print(f'Epoch {e} test MSE: {np.mean(loss_test)} ')
            #     print(f'        test REG MAE: {mean_absolute_error(gt_list, pred_list)*self.s_norm} ')

                # self.error = np.mean(loss_test)
                # if self.error < self.last_error:
                #     self.save_model(model_path='../models')
                #     self.last_error = self.error



            # learning rate decay
            scheduler_reg.step()
            print('--------------------------------------------------------------')
                # import pdb; pdb.set_trace()


        # import pdb; pdb.set_trace()

    def save_model(self, model_path='../models'):

        print('save model...')
        # with open(os.path.join(model_path,"scaler_param.pk"),"wb+") as f:
        #     pickle.dump([self.scaler_x,self.scaler_y,self.window_len],f)

        # torch.save(self.ae_Net.state_dict(), os.path.join(model_path,"AE_model_param.pk"))
        torch.save(self.reg_Net.state_dict(), os.path.join(model_path,"FCN_model_param.pk"))
        # with open(os.path.join(model_path,"error.pk"),"wb+") as f:
        #     pickle.dump(self.error,f)
        print('save done!')
        # test_error_0 = self.error


    def load_model(self, model_path):
        # if os.path.exists(os.path.join(model_path,"scaler_param.pk")):
        #     with open(os.path.join(model_path,"scaler_param.pk"),"rb+") as f:
        #         [self.scaler_x,self.scaler_y] = pickle.load(f)
        # else:
        #     print(f'scaler_param.pk not exist!')
        #     quit()

        if os.path.exists(os.path.join(model_path,"FCN_model_param.pk")):
            # self.ae_Net.load_state_dict(torch.load(os.path.join(model_path,"AE_model_param.pk"),map_location=torch.device(self.device)))
            self.reg_Net.load_state_dict(torch.load(os.path.join(model_path,"FCN_model_param.pk"),map_location=torch.device(self.device)))

        else:
            print(f'model_param.pk not exist!')
            quit()

        print('Model parameters loaded!')

        # if os.path.exists(os.path.join(model_path,"error.pk")):
        #     with open(os.path.join(model_path,"error.pk"),"rb+") as f:
        #         self.error = pickle.load(f)
        # else:
        #     print(f'error.pk not exist!')
        #     quit()

    def predict(self, pred_x):
        pred_result = []
        for each_input in pred_x:
            train_tensor_x = torch.tensor(each_input,dtype=torch.float32,device=self.device)
            train_y_pred_reg_tensor = self.reg_Net(train_tensor_x)
            train_y_pred_reg_array = train_y_pred_reg_tensor.cpu().detach().numpy().squeeze()
            pred_result.append(train_y_pred_reg_array)

        return np.array(pred_result)

        # return np.round(self.train_y_pred)[0]


    def evaluate(self, X,Y):
        # self.data = data
        test_dataset = Initial_Dataset(X, Y)


        test_loader = DataLoader(test_dataset, 1, shuffle=True, num_workers=4)


        gt_list = []
        pred_list = []
        for test_tensor_x, test_tensor_y in test_loader:

            # test_tensor_x_distorted = self.batch_scg_distorted(test_tensor_x, noise=0.3, sampling_rate=100, noise_frequency=[5, 10, 100])
            # test_arr_x_distorted = test_tensor_x_distorted.cpu().detach().numpy().squeeze()

            test_tensor_x = torch.tensor(test_tensor_x,dtype=torch.float32,device=self.device)
            test_tensor_y = torch.tensor(test_tensor_y,dtype=torch.float32,device=self.device)

            # test_y_pred_ae, test_state_logit = self.ae_Net(test_tensor_x_distorted)
            test_y_pred_reg_tensor = self.reg_Net(test_tensor_x)

            test_y_pred_reg_arr = test_y_pred_reg_tensor.cpu().detach().numpy().squeeze()
            test_y_arr = test_tensor_y.cpu().detach().numpy().squeeze()

            gt_list.append(test_y_arr)
            pred_list.append(test_y_pred_reg_arr)
        # st()
        gt_arr = np.array(gt_list)
        pred_arr = np.array(pred_list)

        for i in range(gt_arr.shape[1]):
            mae = mean_absolute_error(gt_arr[:,i], pred_arr[:,i])
            var = np.var(abs(gt_arr[:,i] - pred_arr[:,i] ))
            print(f'Target {i+1}: MAE: {mae}, VAR: {var}')








class CNN_Model():
    """docstring for CNN_Model."""

    def __init__(self, input_shape, out_channels, kernel_size, output_features=1, layers=6, learning_rate=0.001, batch_size=32, epoch_number=500):
        super(CNN_Model, self).__init__()
        ####
        self.device = torch.device('cuda' if cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch_number = epoch_number

        # self.ae_Net = AE_Net(input_shape=input_shape)
        self.reg_Net = CNN_Net(input_shape=input_shape, layers=layers, output_features=output_features, out_channels=out_channels, kernel_size=kernel_size)
        # self.reg_Net = LstmAttentionNet()

        # self.ae_Net = self.ae_Net.to(device = self.device)
        self.reg_Net = self.reg_Net.to(device = self.device)

        print(f"Using device:{self.device}")


    # def fit(self, all_data, window_len, devide_factor, learning_rate=0.001, batch_size=32, epoch_number=500, CONTINUE_TRAINING = False):
    def fit(self, X, Y):

        train_dataset = Initial_Dataset(X, Y)


        train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=4)
        # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

        ### training component
        loss_fn = torch.nn.MSELoss()
        optimizer_reg = optim.Adam(self.reg_Net.parameters(), lr=self.learning_rate)

        scheduler_reg = lr_scheduler.StepLR(optimizer_reg,step_size=5, gamma = 0.95)

        self.last_error = 1e5
        for e in range(self.epoch_number):

            for train_tensor_x, train_tensor_y in train_loader:
                optimizer_reg.zero_grad()

                train_tensor_x = torch.tensor(train_tensor_x,dtype=torch.float32,device=self.device)
                train_tensor_y = torch.tensor(train_tensor_y,dtype=torch.float32,device=self.device)


                train_y_pred_reg = self.reg_Net(train_tensor_x)
                

                train_loss_tensor_reg = loss_fn(train_tensor_y, train_y_pred_reg)
                train_loss_reg = train_loss_tensor_reg.item()

                train_loss_tensor = train_loss_tensor_reg
                train_loss = train_loss_reg

                reg_pred_arr = train_y_pred_reg.cpu().detach().numpy().squeeze()
                reg_gt_arr = train_tensor_y.cpu().detach().numpy().squeeze()
                train_mae = mean_absolute_error(reg_gt_arr, reg_pred_arr)
                # st()
                train_loss_tensor.backward()

                optimizer_reg.step()
            
            print(f'Epoch {e} train MSE: {train_loss} ')
            print(f'        train REG MAE: {train_mae}')

            self.error = train_mae
            if self.error < self.last_error:
                self.save_model(model_path='../models')
                self.last_error = self.error

            # learning rate decay
            scheduler_reg.step()
            print('--------------------------------------------------------------')
                # import pdb; pdb.set_trace()


        # import pdb; pdb.set_trace()

    def save_model(self, model_path='../models'):

        print('saving model...')

        torch.save(self.reg_Net.state_dict(), os.path.join(model_path,"CNN_model_param.pk"))

        print('save done!')



    def load_model(self, model_path):

        if os.path.exists(os.path.join(model_path,"CNN_model_param.pk")):
            self.reg_Net.load_state_dict(torch.load(os.path.join(model_path,"CNN_model_param.pk"),map_location=torch.device(self.device)))

        else:
            print(f'model_param.pk not exist!')
            quit()

        print('Model parameters loaded!')


    def predict(self, pred_x):
        pred_result = []
        for each_input in pred_x:
            train_tensor_x = torch.tensor(each_input,dtype=torch.float32,device=self.device)
            train_y_pred_reg_tensor = self.reg_Net(train_tensor_x)
            train_y_pred_reg_array = train_y_pred_reg_tensor.cpu().detach().numpy().squeeze()
            pred_result.append(train_y_pred_reg_array)

        return np.array(pred_result)


        # return np.round(self.train_y_pred)[0]


    def evaluate(self, X,Y):
        # self.data = data
        test_dataset = Initial_Dataset(X, Y)

        test_loader = DataLoader(test_dataset, 1, shuffle=True, num_workers=4)


        gt_list = []
        pred_list = []
        for test_tensor_x, test_tensor_y in test_loader:

            # test_tensor_x_distorted = self.batch_scg_distorted(test_tensor_x, noise=0.3, sampling_rate=100, noise_frequency=[5, 10, 100])
            # test_arr_x_distorted = test_tensor_x_distorted.cpu().detach().numpy().squeeze()

            test_tensor_x = torch.tensor(test_tensor_x,dtype=torch.float32,device=self.device)
            test_tensor_y = torch.tensor(test_tensor_y,dtype=torch.float32,device=self.device)

            # test_y_pred_ae, test_state_logit = self.ae_Net(test_tensor_x_distorted)
            test_y_pred_reg_tensor = self.reg_Net(test_tensor_x)

            test_y_pred_reg_arr = test_y_pred_reg_tensor.cpu().detach().numpy().squeeze()
            test_y_arr = test_tensor_y.cpu().detach().numpy().squeeze()

            gt_list.append(test_y_arr)
            pred_list.append(test_y_pred_reg_arr)
        # st()
        gt_arr = np.array(gt_list)
        pred_arr = np.array(pred_list)

        for i in range(gt_arr.shape[1]):
            mae = mean_absolute_error(gt_arr[:,i], pred_arr[:,i])
            var = np.var(abs(gt_arr[:,i] - pred_arr[:,i] ))
            print(f'Target {i+1}: MAE: {mae}, VAR: {var}')




class LSTM_Model():
    """docstring for LSTM_Model."""

    def __init__(self, num_layers=5, hidden_size=100, output_features=4, learning_rate=0.001, batch_size=32, epoch_number=500):
        super(LSTM_Model, self).__init__()
        ####
        self.device = torch.device('cuda' if cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch_number = epoch_number

        self.reg_Net = LstmAttentionNet(num_layers=num_layers, hidden_size=hidden_size, output_features=output_features)

        self.reg_Net = self.reg_Net.to(device = self.device)

        print(f"Using device:{self.device}")


    def fit(self, X, Y):

        train_dataset = Initial_Dataset(X, Y)


        train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=4)
        # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

        ### training component
        loss_fn = torch.nn.MSELoss()
        optimizer_reg = optim.Adam(self.reg_Net.parameters(), lr=self.learning_rate)

        scheduler_reg = lr_scheduler.StepLR(optimizer_reg,step_size=5, gamma = 0.95)

        self.last_error = 1e5
        for e in range(self.epoch_number):

            for train_tensor_x, train_tensor_y in train_loader:
                optimizer_reg.zero_grad()

                train_tensor_x = torch.tensor(train_tensor_x,dtype=torch.float32,device=self.device)
                train_tensor_y = torch.tensor(train_tensor_y,dtype=torch.float32,device=self.device)


                train_y_pred_reg = self.reg_Net(train_tensor_x)
                # st()

                train_loss_tensor_reg = loss_fn(train_tensor_y, train_y_pred_reg)
                train_loss_reg = train_loss_tensor_reg.item()

                train_loss_tensor = train_loss_tensor_reg
                train_loss = train_loss_reg

                reg_pred_arr = train_y_pred_reg.cpu().detach().numpy().squeeze()
                reg_gt_arr = train_tensor_y.cpu().detach().numpy().squeeze()
                train_mae = mean_absolute_error(reg_gt_arr, reg_pred_arr)
                # st()
                train_loss_tensor.backward()

                optimizer_reg.step()
            
            print(f'Epoch {e} train MSE: {train_loss} ')
            print(f'        train REG MAE: {train_mae}')

            self.error = train_mae
            if self.error < self.last_error:
                self.save_model(model_path='../models')
                self.last_error = self.error

            # learning rate decay
            scheduler_reg.step()
            print('--------------------------------------------------------------')
                # import pdb; pdb.set_trace()


        # import pdb; pdb.set_trace()

    def save_model(self, model_path='../models'):

        print('saving model...')

        torch.save(self.reg_Net.state_dict(), os.path.join(model_path,"LSTM_model_param.pk"))

        print('save done!')



    def load_model(self, model_path='../models'):

        if os.path.exists(os.path.join(model_path,"LSTM_model_param.pk")):
            self.reg_Net.load_state_dict(torch.load(os.path.join(model_path,"LSTM_model_param.pk"),map_location=torch.device(self.device)))

        else:
            print(f'model_param.pk not exist!')
            quit()

        print('Model parameters loaded!')


    def predict(self, pred_x):
        pred_result = []
        for each_input in pred_x:
            train_tensor_x = torch.tensor(each_input,dtype=torch.float32,device=self.device)
            train_y_pred_reg_tensor = self.reg_Net(train_tensor_x)
            train_y_pred_reg_array = train_y_pred_reg_tensor.cpu().detach().numpy().squeeze()
            pred_result.append(train_y_pred_reg_array)

        return np.array(pred_result)


    def evaluate(self, X,Y):
        # self.data = data
        test_dataset = Initial_Dataset(X, Y)

        test_loader = DataLoader(test_dataset, 1, shuffle=True, num_workers=4)


        gt_list = []
        pred_list = []
        for test_tensor_x, test_tensor_y in test_loader:

            # test_tensor_x_distorted = self.batch_scg_distorted(test_tensor_x, noise=0.3, sampling_rate=100, noise_frequency=[5, 10, 100])
            # test_arr_x_distorted = test_tensor_x_distorted.cpu().detach().numpy().squeeze()

            test_tensor_x = torch.tensor(test_tensor_x,dtype=torch.float32,device=self.device)
            test_tensor_y = torch.tensor(test_tensor_y,dtype=torch.float32,device=self.device)

            # test_y_pred_ae, test_state_logit = self.ae_Net(test_tensor_x_distorted)
            test_y_pred_reg_tensor = self.reg_Net(test_tensor_x)

            test_y_pred_reg_arr = test_y_pred_reg_tensor.cpu().detach().numpy().squeeze()
            test_y_arr = test_tensor_y.cpu().detach().numpy().squeeze()

            gt_list.append(test_y_arr)
            pred_list.append(test_y_pred_reg_arr)
        # st()
        gt_arr = np.array(gt_list)
        pred_arr = np.array(pred_list)

        for i in range(gt_arr.shape[1]):
            mae = mean_absolute_error(gt_arr[:,i], pred_arr[:,i])
            var = np.var(abs(gt_arr[:,i] - pred_arr[:,i] ))
            print(f'Target {i+1}: MAE: {mae}, VAR: {var}')








def main():
    scaler = preprocessing.StandardScaler()



    # dataset = np.load('../../data/real_data/data_label_train.1000_6.6_6.npy')
    dataset = np.load('../../data/real_data/data_label_train.1000_6.npy')[:10,:]

    X = dataset[:,:-6]
    Y = dataset[:,-4:-2]

    # dataset_test = np.load('../../data/real_data/data_label_test.1000_6.6_6.npy')
    # X_test = dataset_test[:,:-6]
    # Y_test = dataset_test[:,-4:-2]

    # X = scaler.fit_transform(X)
    # X_test = scaler.transform(X_test)
    # st()

    # dataset_time_sort = dataset[np.argsort( (dataset[:, -5]) )]
    # np.random.shuffle(dataset)

    # auto_encoder = FCN_Model(input_features=6, output_features=2, layers=30, neurons=128, learning_rate=0.0001, batch_size=32, epoch_number=500)
    # auto_encoder = CNN_Model(out_channels=[64,64,32], kernel_size=5, output_features=2, layers=3, learning_rate=0.001, batch_size=32, epoch_number=500)
    auto_encoder = LSTM_Model(num_layers=1, hidden_size=100, output_features=2, learning_rate=0.001, batch_size=32, epoch_number=500)

    
    auto_encoder.fit(X, Y)

    auto_encoder.load_model('../models')
    auto_encoder.evaluate(X_test, Y_test)

if __name__ == '__main__':
    main()
