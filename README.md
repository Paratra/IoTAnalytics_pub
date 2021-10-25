# IoTAnalytics

1. gen_synthetic_data.py
    Usage:
        First parameter (int) should be number of components:
            e.g. 4 
        Second parameter (int) should be each component data length:
            e.g. 1000

        Example:
            python3 gen_synthetic_data.py 4 1000

    Result:
        Will be saving to ../data/synthetic_data.npy
        Size of each row is 501, first 500 is X, last dimention is y


2. sepr_train_test.py
    Usage:
        First parameter (path): path to data file:
            e.g. ../data/synthetic_data.385_1.npy 
        Second parameter (int): length of label:
            e.g. 1
        Third parameter (float): ratio of training set size:
            e.g. 0.8

        Example:
            python3 sepr_train_test.py ../data/synthetic_data.385_1.npy 1 0.8

    Result:
        Will be saving to same path of input file


3. extr_feature.py
    Usage:
        First parameter (path): path to data file:
            e.g. ../data/synthetic_data.npy 
        Second parameter (int): length of label:
            e.g. 1
        Third parameter (str): extraction method:
            e.g. tsfel/tsfel_win/spectral

        Example:
            python3 extr_feature.py ../../data/synthetic_data.npy 1 tsfel

    Result:
        Will be saving to same path of input file
        name: ../data/synthetic_data.385_1.npy
            385 indicates 385 extracted features 
            1 indicates last dimension is label

4. algo_generic_regressors.py
    Usage:
        First parameter (path): path to data file:
            e.g. ../data/synthetic_data_train.385_1.npy 
        Second parameter (int): length of label:
            e.g. 1
        Third parameter (str): mode:
            e.g. fit/predict/score

        Example:
            python3 algo_generic_regressors.py ../data/synthetic_data.385_1.npy 1 fit
            python3 algo_generic_regressors.py ../data/synthetic_data.385_1.npy 1 predict
            python3 algo_generic_regressors.py ../data/synthetic_data.385_1.npy 1 score

    Result:
        Will be saving to same path of input file
