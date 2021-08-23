import pandas as pd
import numpy as np
import os, sys
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import math
import scipy.integrate as integrate
import scipy.special as special
# from varname import nameof

class CpEvaluationClass(object):
    u = 0
    d = 0
    v = 0
    L = 0
    result = 0
    integral = 0

    def __init__(self) -> None:
        pass
        # print('init CpEvaluationClass')
    def cpequation(self, x, u, v, d):
        return (1 + (x - u) ** 2 / (d ** 2 * (v - 2))) ** ((v + 1) / 2)

    def calculate(self):
        self.result = math.gamma((self.v + 1) / 2)
        self.result = self.result / self.d
        self.result = self.result / math.gamma(self.v / 2)
        self.result = self.result / np.sqrt((self.v - 2) * np.pi)
        ## print("self.result before try: ",self.result)
        try:
            # self.integral = integrate((1 + (x - self.u) ** 2 / (self.d ** 2 * (self.v - 2))) ** ((self.v + 1) / 2),
            #                          (x, -self.L, self.L))
            self.integral = integrate.quad(lambda  x: self.cpequation(x, self.u, self.v, self.d),-self.L,self.L)
            # cp5_integral = integrate((1+((x-cp5_u)**2)/(cp5_d**2 * (2)))**((5/2)), (x, -cp5_l, cp5_l))
            # print("integral answer: ",self.integral)
            self.result = self.result * self.integral[0]
            # self.result = self.result * 1

            # print('result', self.result)
            # print('integral', self.integral[0])
            # print('u', self.u)
            # print('d', self.d)
        except Exception as e:
            print('Exception', e)
        finally:
            return self.result

    def print(self):
        print('cp'+str(self.L)+" u:" + str(self.u))
        print('cp' + str(self.L) + " d:" + str(self.d))
        print('cp' + str(self.L) + " integral:" + str(self.integral))
        print('cp' + str(self.L) + " result:" + str(self.result))


# if  __name__ == '__main__':
#     cpEvaluationClass = CpEvaluationClass()
#     cpEvaluationClass.u = -2.567
#     cpEvaluationClass.d = 6.589489000000001
#     cpEvaluationClass.v = 4
#     cpEvaluationClass.L = 5
#     cpEvaluationClass.result = 0
#     cpEvaluationClass.integral = 0
#     result = cpEvaluationClass.calculate()
#     print(result)


def calc_md(gt, pred):
    return np.mean(gt-pred,0)

def calc_mad(gt, pred):
    return mean_absolute_error(gt, pred)

def calc_mapd(gt, pred):
    return mean_absolute_percentage_error(gt,pred)

def calc_cp5(gt, pred):
    cp5_evaluation = CpEvaluationClass()
    try:
        cp5_evaluation.u = np.mean((gt-pred),0)
        cp5_evaluation.d = np.mean(gt-pred,0) ** 2
        cp5_evaluation.v = 4
        cp5_evaluation.L = 5
        cp5_evaluation.result = 0
        cp5_evaluation.integral = 0
        cp5_evaluation.calculate()
    except Exception as e:
        print('Exception', e)

    return cp5_evaluation.result

def calc_cp10(gt, pred):
    cp10_evaluation = CpEvaluationClass()
    try:
        cp10_evaluation.u = np.mean((gt-pred),0)
        cp10_evaluation.d = np.mean(gt-pred,0) ** 2
        cp10_evaluation.v = 4
        cp10_evaluation.L = 10
        cp10_evaluation.result = 0
        cp10_evaluation.integral = 0
        cp10_evaluation.calculate()
    except Exception as e:
        print('Exception', e)

    return cp10_evaluation.result

def calc_cp15(gt, pred):
    cp15_evaluation = CpEvaluationClass()
    try:
        cp15_evaluation.u = np.mean((gt-pred),0)
        cp15_evaluation.d = np.mean(gt-pred,0) ** 2
        cp15_evaluation.v = 4
        cp15_evaluation.L = 15
        cp15_evaluation.result = 0
        cp15_evaluation.integral = 0
        cp15_evaluation.calculate()
    except Exception as e:
        print('Exception', e)

    return cp15_evaluation.result

def eval_result_stats(gt, pred):
    return [calc_md(gt,pred), calc_mad(gt,pred), calc_mapd(gt,pred), calc_cp5(gt,pred),calc_cp10(gt,pred),calc_cp15(gt,pred)]




def main():
    # example
    csv = pd.read_csv('../data/comparison_regressor.csv').drop("Unnamed: 0", axis= 1)

    H_gt = csv['H']
    H_pred = csv['KNeighborsRegressor_H']
    R_gt = csv['R']
    R_pred = csv['KNeighborsRegressor_R']
    S_gt = csv['S']
    S_pred = csv['KNeighborsRegressor_S']
    D_gt = csv['D']
    D_pred = csv['KNeighborsRegressor_D']

    vital_list = [(H_gt, H_pred, 'H'), (R_gt, R_pred,'R'), (S_gt, S_pred,'S'), (D_gt, D_pred,'D')]
    for each_ in vital_list:
        vital_eval = eval_result_stats(each_[0],each_[1])
        vital_name = each_[2] #nameof(vital_eval).split('_')[0]
        # import pdb; pdb.set_trace()
        print('------------------------')
        print(f'{vital_name}: \n MD: {vital_eval[0]} \n MAD: {vital_eval[1]} \n MAPE: {vital_eval[2]} \n CP5: {vital_eval[3]} \n CP10: {vital_eval[4]} \n CP15: {vital_eval[5]}')
        print('------------------------')
        # import pdb; pdb.set_trace()

if __name__== '__main__':
    main()
