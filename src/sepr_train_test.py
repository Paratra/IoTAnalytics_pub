#!/usr/bin/env python3
# %% Split a data set to train and test data set
from utils import *


if __name__ == "__main__":

    if(len(sys.argv) > 3):
        data_file = sys.argv[1]
        num_labels = int(sys.argv[2])
        split_ratio = float(sys.argv[3])
    else:
        print(f"Usage: {sys.argv[0]} data_file num_labels split_ratio")
        print(f"Example: {sys.argv[0]} ../data/synthetic_data.385_1.npy 1 0.7")   
        exit()     

    out_name = data_file
    [out_name, third] = os.path.splitext(out_name)
    [out_name, second] = os.path.splitext(out_name)
    [zero, first] = os.path.splitext(out_name)
    print("Read file:", zero, first, second, third)
    train_file = zero+"_train"+first+second+third
    test_file = zero+"_test"+first+second+third
    # print(train_file, test_file)

    # if num_labels == 16:
    #     time_index = -15
    #     id_index = -16
    
    # if num_labels == 6:
    #     time_index = -5
    #     id_index = -6
        
    data_set = load_data_file(data_file)
    
    data_size = data_set.shape[0]

    split_ratio = 0.7
    # time_sorted = True
    # if time_sorted:
    #     sorted_indexes = np.argsort(data_set[:, time_index])
    #     data_set = data_set[sorted_indexes]
    # else: 
    #     np.random.shuffle(data_set) 

    np.random.shuffle(data_set) 


    train_size = int(data_size*split_ratio)
    test_size = data_size - train_size

    train_set = data_set[:train_size, :]
    test_set = data_set[train_size:, :]

    # print(data_set.shape, train_set.shape, test_set.shape)

    np.save(train_file, train_set)
    np.save(test_file, test_set)

    print(f"Separated {data_file} of {data_set.shape} into {train_file} of {train_set.shape} and {test_file} of {test_set.shape}")