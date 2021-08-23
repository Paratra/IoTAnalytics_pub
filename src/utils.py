#!/usr/bin/env python3

import time
import math
import subprocess
import sys
import random
import webbrowser
import numpy as np
from datetime import datetime
from dateutil import tz
import pytz
import os, sys

from influxdb import InfluxDBClient
import operator
import copy
from collections import Counter

from scipy.stats import norm
from scipy.special import softmax
import pandas as pd

import matplotlib.pyplot as plt
import re

import tsfel

# get_ipython().system(' pip install tsfel # installing TSFEL for feature extraction')
def str2bool(v):
  return v.lower() in ("true", "1", "https", "load")

def mac_to_int(mac):
    res = re.match('^((?:(?:[0-9a-f]{2}):){5}[0-9a-f]{2})$', mac.lower())
    if res is None:
        raise ValueError('invalid mac address')
    return int(res.group(0).replace(':', ''), 16)

def int_to_mac(macint):
    # if type(macint) != int:
    #     raise ValueError('invalid integer')
    newint = int(macint)
    return ':'.join(['{}{}'.format(a, b)
                     for a, b
                     in zip(*[iter('{:012x}'.format(newint))]*2)])

# This function converts the time string to epoch time xxx.xxx (second.ms).
# Example: time = "2020-08-13T02:03:00.200", zone = "UTC" or "America/New_York"
# If time = "2020-08-13T02:03:00.200Z" in UTC time, then call timestamp = local_time_epoch(time[:-1], "UTC"), which removes 'Z' in the string end
def local_time_epoch(time, zone):
    local_tz = pytz.timezone(zone)
    localTime = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f")
    local_dt = local_tz.localize(localTime, is_dst=None)
    # utc_dt = local_dt.astimezone(pytz.utc)
    epoch = local_dt.timestamp()
    # print("epoch time:", epoch) # this is the epoch time in seconds, times 1000 will become epoch time in milliseconds
    # print(type(epoch)) # float
    return epoch

# This function converts the epoch time xxx.xxx (second.ms) to time string.
# Example: time = "2020-08-13T02:03:00.200", zone = "UTC" or "America/New_York"
# If time = "2020-08-13T02:03:00.200Z" in UTC time, then call timestamp = local_time_epoch(time[:-1], "UTC"), which removes 'Z' in the string end
def epoch_time_local(epoch, zone):
    local_tz = pytz.timezone(zone)
    time = datetime.fromtimestamp(epoch).astimezone(local_tz).strftime("%Y-%m-%dT%H:%M:%S.%f")
    return time

# This function converts the grafana URL time to epoch time. For exmaple, given below URL
# https://sensorweb.us:3000/grafana/d/OSjxFKvGk/caretaker-vital-signs?orgId=1&var-mac=b8:27:eb:6c:6e:22&from=1612293741993&to=1612294445244
# 1612293741993 means epoch time 1612293741.993; 1612294445244 means epoch time 1612294445.244
def grafana_time_epoch(time):
    return time/1000

def influx_time_epoch(time):
    return time/10e8

def generate_waves(fs, fd, seconds):
    # fs = 10e3
    # N = 1e5
    N = fs * seconds
    amp = 2*np.sqrt(2)
    # fd = fs/20
    noise_power = 0.001 * fs / 2
    time = np.arange(N) / fs
    signal = amp*np.sin(2*np.pi*fd*time)
    # signal += amp*np.sin(2*np.pi*(fd*1.5)*time)
    signal += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
    return signal, time

def select_with_gaussian_distribution(data_name, all_data, label_indexes = [('S', -2)]):
    '''
    label_indexes is one below:
        [('H', -4)], [('R', -3)], [('S', -2)], [('D', -1)]
    '''
    index = label_indexes[0][1]

    data = all_data[:,index]
    values = list(set(data))
    num_sample = int(0.8*len(data))

    ### Gaussian Distibution
    mu = np.mean(data)
    sigma = np.std(data)
    n, bins = np.histogram(data, bins=len(values)-1, density=1)
    y = norm.pdf(bins, mu, sigma)

    ### calculate propability
    p_table = np.concatenate((np.array(values)[np.newaxis,:],y[np.newaxis,:]),0)
    p_list = []
    for each_ in data:
        index = np.where(p_table[0,:]==each_)[0][0]
        p_list.append(p_table[1,index])

    # import pdb; pdb.set_trace()
    p_arr = softmax(p_list)
    sample_index = np.random.choice(np.arange(data.shape[0]), size=num_sample, p=p_arr)
    result_data = all_data[sample_index]

    print(f"\nPerformed data selection with gaussian distribution on {data_name}\n")
    return result_data


def select_with_uniform_distribution(data_name, all_data,label_indexes = [('S', -2)]):
    '''
    label_indexes is one below:
        [('H', -4)], [('R', -3)], [('S', -2)], [('D', -1)]
    '''
    index = label_indexes[0][1]
    data = all_data[:,index]
    minimum_num_sample = int(len(data)/len(list(set(data))))

    counter_result = Counter(data)
    dict_counter = dict(counter_result)
    result_dict = dict_counter.copy()

    # we keep them even if they have less than minimum
    # for item in dict_counter.items():
    #     if item[1] < minimum_num_sample:
    #         del result_dict[item[0]]

    keys = list(result_dict.keys())

    result_index_list = []
    for each_key in keys:
        result_index_list += list(np.random.choice(np.where(data == each_key)[0], size=minimum_num_sample))

    result_data = all_data[result_index_list]
    print(f"\nPerformed data selection with uniform distribution on {data_name}\n")

    return result_data

def label_index(label_name, labels_list = ['ID', 'Time', 'H', 'R', 'S', 'D']):
    # print(labels_list.index(label_name))
    return labels_list.index(label_name)-len(labels_list)  

def eval_data_stats(data_set_name, data_set, labels_list = ['ID', 'Time', 'H', 'R', 'S', 'D'], show=False ):

    num_labels = len(labels_list)
    time_index = label_index ('Time', labels_list)
    time_min = epoch_time_local(min(data_set[:, time_index]), "America/New_York")
    time_max = epoch_time_local(max(data_set[:, time_index]), "America/New_York")

    print(f"\n{data_set_name} statistics:")
    print(f"time_min: {time_min}; time_max: {time_max}")
    print(f"{num_labels} labels are: {labels_list}")

    for label_name in labels_list[2:]:
        index = label_index (label_name, labels_list)

        target_ = data_set[:,index]
        target_pd = pd.DataFrame(target_,columns=[label_name])
        print(target_pd.describe())
        if show:
            plt.figure(f"{data_set_name} - distribution of {label_name}")
            plt.hist(target_,bins=30)
        # plt.show()

# def eval_data_stats(data_name, data, label_indexes  = [('S', -2)], time_index=-5, show=True ):
#     '''
#     label_indexes is one below:
#         [('H', -4)], [('R', -3)], [('S', -2)], [('D', -1)]
#     '''

#     label_name = label_indexes[0][0]
#     index = label_indexes[0][1]


#     # print('---------------------Original----------------------')
#     target_ = data[:,index]

#     target_pd = pd.DataFrame(target_,columns=[label_name])
#     print(f"\n{data_name} - statistics of ", target_pd.describe())

#     time_min = epoch_time_local(min(data[:, time_index]), "America/New_York")
#     time_max = epoch_time_local(max(data[:, time_index]), "America/New_York")
#     print(f"time_min: {time_min}; time_max: {time_max}\n")

#     if show:
#         plt.figure(f"{data_name} - distribution of {label_name}")
#         plt.hist(target_,bins=30)
#         # plt.show()

# This function write an array of data to influxdb. It assumes the sample interval is 1/fs.
# influx - the InfluxDB info including ip, db, user, pass. Example influx = {'ip': 'https://sensorweb.us', 'db': 'algtest', 'user':'test', 'passw':'sensorweb'}
# dataname - the dataname such as temperature, heartrate, etc
# timestamp - the epoch time (in second) of the first element in the data array, such as datetime.now().timestamp()
# fs - the sampling interval of readings in data
# unit - the unit location name tag
def write_influx(influx, unit, table_name, data_name, data, start_timestamp, fs):
    # print("epoch time:", timestamp)
    max_size = 100
    count = 0
    total = len(data)
    prefix_post  = "curl -s -POST \'"+ influx['ip']+":8086/write?db="+influx['db']+"\' -u "+ influx['user']+":"+ influx['passw']+" --data-binary \' "
    http_post = prefix_post
    for value in data:
        count += 1
        http_post += "\n" + table_name +",location=" + unit + " "
        http_post += data_name + "=" + str(value) + " " + str(int(start_timestamp*10e8))
        start_timestamp +=  1/fs
        if(count >= max_size):
            http_post += "\'  &"
            # print(http_post)
            print("Write to influx: ", table_name, data_name, count)
            subprocess.call(http_post, shell=True)
            total = total - count
            count = 0
            http_post = prefix_post
    if count != 0:
        http_post += "\'  &"
        # print(http_post)
        print("Write to influx: ", table_name, data_name, count, data)
        subprocess.call(http_post, shell=True)

# This function read an array of data from influxdb.
# influx - the InfluxDB info including ip, db, user, pass. Example influx = {'ip': 'https://sensorweb.us', 'db': 'testdb', 'user':'test', 'passw':'sensorweb'}
# dataname - the dataname such as temperature, heartrate, etc
# start_timestamp, end_timestamp - the epoch time (in second) of the first element in the data array, such as datetime.now().timestamp()
# unit - the unit location name tag
def read_influx(influx, unit, table_name, data_name, start_timestamp, end_timestamp):
    if influx['ip'] == '127.0.0.1' or influx['ip'] == 'localhost':
        client = InfluxDBClient(influx['ip'], '8086', influx['user'], influx['passw'], influx['db'],  ssl=influx['ssl'])
    else:
        client = InfluxDBClient(influx['ip'].split('//')[1], '8086', influx['user'], influx['passw'], influx['db'],  ssl=influx['ssl'])

    # client = InfluxDBClient(influx['ip'].split('//')[1], '8086', influx['user'], influx['passw'], influx['db'],  ssl=True)
    query = 'SELECT "' + data_name + '" FROM "' + table_name + '" WHERE "location" = \''+unit+'\' AND time >= '+ str(int(start_timestamp*10e8))+' AND time < '+str(int(end_timestamp*10e8))
    # query = 'SELECT last("H") FROM "labelled" WHERE ("location" = \''+unit+'\')'

    # print(query)
    result = client.query(query)
    # print(result)

    points = list(result.get_points())
    values =  list(map(operator.itemgetter(data_name), points))
    times  =  list(map(operator.itemgetter('time'),  points))
    # print(times)
    # times = [local_time_epoch(item[:-1], "UTC") for item in times] # convert string time to epoch time
    # print(times)

    data = values #np.array(values)
    # print(data, times)
    return data, times



### example
# some_data = np.load('path/to/data.npy')
# result_ = extract_specific_bp(some_data, num_extraction=2, num_row=-2) ### -2 is SBP,  -1 is DBP in that data order.

def extract_specific_data(data, range_set, num_extraction, index_col):
    '''
    imports:
        import numpy as np
        import random
    args:
        data --- array like data
        num_extraction --- how many to extract
        index_col --- the index of target label located in array
    return:
        result --- array like data including targets
    '''
    if range_set == []:
        range_set = set(data[:,index_col])    ## get all target values
    print(len(range_set), range_set)

    # result = []
    i = 0
    for each_target in range_set:
        ### if there is not enough data for requesting, skip
        if len(np.where(data[:,index_col]==each_target)[0]) < num_extraction:
            print(f'For target {each_target}, there is only {len(np.where(data[:,index_col]==each_target)[0])} data, however, you request {num_extraction}. Skip!')
            continue

        ### randomly get indexs of request target
        # selection_index = random.sample(list(np.where(data[:,index_col]==each_target)[0]), num_extraction)
        # print("selection_index:", selection_index)
        ### append requested target to result
        # result.append(data[index,:])
        if i ==0:
            ### randomly get indexs of request target
            selection_index = random.sample(list(np.where(data[:,index_col]==each_target)[0]), num_extraction)
            # print("selection_index:", selection_index)

        else:
            selection_index = np.append(selection_index, random.sample(list(np.where(data[:,index_col]==each_target)[0]), num_extraction), axis=0)
        i += 1

    # print("final selection_index:", selection_index)
    result = data[selection_index,:]
    return np.array(result), selection_index

def load_data_files(file_list):
  for ind, data_file in enumerate(file_list):
    data_set = load_data_file(data_file)
    if ind == 0:
        total_set = data_set
    else:
        total_set = np.concatenate( (total_set, data_set), 0)
  return total_set

def load_data_file(data_file):
    if data_file.endswith('.csv'):
        data_set = pd.read_csv(data_file).to_numpy()
    elif data_file.endswith('.npy'):
        data_set = np.load(data_file)
    return data_set

# # This function only applied to old separated data and feature set
# def load_data_feature_files(file_list, num_labels=6, load_feature = True):

#     for ind, (data_file, feature_file) in enumerate(file_list):
#         data_set = np.load(data_file)    #bed H2
#         if load_feature:
#             features_set = load_features_data(feature_file)
#         else:
#             features_set = extract_features_data(data_set, data_file, with_windowed = False)
#         # features_set = extract_or_load_features_data(data_set, feature_file, load_feature) #.to_numpy()
#         print(data_set.shape, features_set.shape)
#         # features_set = extract_or_load_features_data(data_set, feature_file, load_feature).drop("Unnamed: 0", axis= 1).to_numpy()
#         features_labels = np.concatenate( (features_set, data_set[:, -num_labels:]), 1)
#         if ind == 0:
#             all_features_labels = features_labels
#         else:
#             all_features_labels = np.concatenate( (all_features_labels, features_labels), 0)
        
#         data_file_noext = os.path.splitext(data_file)[0]
#         npy_name = data_file_noext+"_feature_label.npy"
#         np.save(npy_name, all_features_labels)

#     # print_stats_distribution("Whole data set", data_set, [('H', -4), ('R', -3), ('S', -2), ('D', -1)])

#     return all_features_labels

def prepare_train_test_data(data_set, split_ratio, time_sorted, 
    target_index, target_distribution, target_range, 
    num_labels=6, time_index=-5, id_index=-6, show = False):
    '''
    label_indexes is one below:
        [('H', -4)], [('R', -3)], [('S', -2)], [('D', -1)]
    '''
    # eval_data_stats("all original set", data_set, target_index, time_index )

    # target_index = target_indexes[2][1]
    # time_index = -5

    data_size = data_set.shape[0]
    # minimum_num_sample = int(data_size/len(target_range)/4)

    if time_sorted:
        sorted_indexes = np.argsort(data_set[:, time_index])
        data_set = data_set[sorted_indexes]
    else: # random shuffle
        np.random.shuffle(data_set) 

    print(data_set.shape)

    train_size = int(data_size*split_ratio)
    test_size = data_size - train_size

    train_set = data_set[:train_size, :]
    test_set = data_set[train_size:, :]

    if target_distribution == "uniform":
        train_set = select_with_uniform_distribution(train_set, target_index)
        test_set = select_with_uniform_distribution(test_set, target_index)
    elif target_distribution == "gaussian":
        train_set = select_with_gaussian_distribution(train_set, target_index)
        test_set = select_with_gaussian_distribution(test_set, target_index)

    data_set = np.concatenate((train_set, test_set), 0)
    # eval_data_stats("all data set", data_set, vital_indexes)
    if show:
        eval_data_stats(f"whole set with {target_distribution}:", data_set)
        eval_data_stats(f"train set with {target_distribution}:", train_set)
        eval_data_stats(f"test set with {target_distribution}:", test_set)

    X_train = train_set[:,:-num_labels]
    X_test = test_set[:,:-num_labels]

    Y_train = train_set[:, -num_labels-2:]  # 2 for ID and time  
    Y_test = test_set[:, -num_labels-2:]

    return X_train, Y_train, X_test, Y_test



# def extract_features_data(data_set, data_file, with_windowed = True):

#     filename_noext = os.path.splitext(data_file)[0]
    
#     if with_windowed:
#         features_ready = windowed_feature_extraction(data_set, window_size = 200)
#         wstr = "_wd"
#     else:
#         features_ready = unwindowed_feature_extraction(data_set)
#         wstr = "_uw"

#     csv_name = filename_noext+ wstr+ "_feature.csv"
#     features_ready.to_csv(csv_name)
#     print("Save features into: ", csv_name)

#     features_ready = features_ready.to_numpy()
#     npy_name = filename_noext+ wstr+ "_feature.npy"
#     np.save(npy_name, features_ready)
#     print("Save features into: ", npy_name)

#     # if filename.endswith('.csv'):
#     #   features_ready.to_csv(filename)
#     # elif filename.endswith('.npy'):
#     #    np.save(filename, features_ready)

#     return features_ready

# def load_features_data(feature_file):
#     if feature_file.endswith('.csv'):
#         features_ready = pd.read_csv(feature_file).to_numpy()
#     elif feature_file.endswith('.npy'):
#         features_ready = np.load(feature_file)
#     return features_ready
# if __name__ == "__main__":
