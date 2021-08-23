__author__ = "Ming"
__email__ = "ming.song@uga.edu"

import numpy as np
import random
import matplotlib.pyplot as plt
from pdb import set_trace as st
import sys

def signal_xHz(fi, sample):
    return np.sin(np.linspace(0, fi * 2 * np.pi , sample))

def gen_frequency_list(num_comp):
    return np.sort(np.random.randint(0,1000,num_comp))

def gen_data(fre_component, num_sample):
    
    X_ = []
    y_ = []
    for i in range(num_sample):
        for fre_ in fre_component:
            X_.append(signal_xHz(fre_, sample=500))
            y_.append(fre_)

    X_arr = np.array(X_)
    y_arr = np.array(y_)[:,np.newaxis]

    data = np.concatenate((X_arr, y_arr), 1)

    return data

def main():

    if len(sys.argv) < 3:
        print('Usage:')

        print('First parameter (int) should be number of components:')
        print("     e.g. 4 ")
        print('Second parameter (int) should be each component data length:')
        print('     e.g. 1000')

        print('Example:')
        print('python3 gen_synthetic_data.py 4 1000')

        exit()

    fre_list = gen_frequency_list(int(sys.argv[1]))
    each_sample_num = int(sys.argv[2])
    data = gen_data(fre_component=fre_list, num_sample=each_sample_num)

    np.random.shuffle(data)

    np.save('../data/synthetic_data',data)

    print("Data saved to ../data/synthetic_data.npy")
    print(f'Frequency components are {fre_list}')
    print(f'Data size is {each_sample_num*len(fre_list)}')
    print(f'Size of each row is 501, first 500 is X, last dimention is y')



if __name__ == '__main__':
    main()