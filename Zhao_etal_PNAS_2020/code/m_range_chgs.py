#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yang Z.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append("./code")
import data_load_01
import data_load_02

# build a fitting model
class model_fit_poly_1d(object):
    def __init__(self,deg):
        self.f_ = None
        self.p_ = None
        self.deg = deg
    def fit(self,x,y):
        self.f_ = np.polyfit(x,y,self.deg)
        self.p_ = np.poly1d(self.f_)
    def predict(self,x_pre):
        poly_fit = self.p_(x_pre)
        return poly_fit
    def para(self):
        return self.f_
    
#%%
if __name__ == '__main__':
    # Loading dynamic vehicular operating data
    data_op = data_load_01.get_operating_data_auto()
    data_info = data_load_02.get_veh_info() # data specification data
    data_op_p1 = pd.merge(data_op, data_info, 
                          left_on='b.vin', right_on='vin') # merge dynamic and static data
    data_op_p2 = data_op_p1.copy() #copy

#%%
    # Data screening
    msk1 = data_op_p2.city == '北京市'#'北京' Beijing, '广州市' Guangzhou，‘上海市’ Shanghai
    # vehicle model can be selected as specific requirements.
    msk2 = data_op_p2.veh_model == '1a5afe0eb2520bdfa00134b987268a8b' #choose vehicle model
    msk3 = data_op_p2['b.category'] == 10 # 10 and 30 for driving and charging sessions.
    msk =  msk2 & msk3 & msk1
    data_op_p3 = data_op_p2.loc[msk, :] 
    data_op_p4 = data_op_p3.copy() #copy
    # To obtain the driving distances of individual sesssions.
    data_op_p4['dert_dist'] = (data_op_p4['b.stop_mileage'] - 
                               data_op_p4['b.start_mileage'])
    data_op_p4['dert_soc'] = data_op_p4['b.start_soc'] - data_op_p4['b.end_soc']
    data_op_p4['normalized_range'] = 100 * data_op_p4['dert_dist']/data_op_p4['dert_soc']   
    msk1 = data_op_p4['b.start_temp'] > -20
    msk2 = data_op_p4['dert_soc'] > 5
    msk3 = data_op_p4['normalized_range'] > 40
    msk4 = (data_op_p4['b.start_soc'] <= 95) & (data_op_p4['b.start_soc'] >= 35)
    msk5 = (data_op_p4['b.end_soc'] <= 95) & (data_op_p4['b.end_soc'] >= 35)
    msk =  msk1 & msk2 & msk3 & msk4 & msk5 
    data_op_p4 = data_op_p4.loc[msk, :]
    del msk1, msk2, msk3, msk
#%% 
    # Rates of range declines
    x = data_op_p4['b.start_temp']
    x_np = np.linspace(-11, 40, 100)
    y1 = data_op_p4['normalized_range']
    model_fit1 = model_fit_poly_1d(1)
    model_fit1.fit(x, y1)
    model_fit3 = model_fit_poly_1d(3)
    model_fit3.fit(x, y1)
    model_fit5 = model_fit_poly_1d(5)
    model_fit5.fit(x, y1)
    model_fitx = model_fit_poly_1d(10)
    model_fitx.fit(x, y1)
    optimal_avg_range = model_fit3.predict(22)
    y = 100 * (data_op_p4['normalized_range'] - optimal_avg_range)/optimal_avg_range
    model_fit = model_fit_poly_1d(3) # polynomial fitting model construction
    model_fit.fit(x, y) # line fitting
    para = model_fit.para() # parameters for the fitted line
    optimal_avg_range = model_fit.predict(22)
    fig = plt.figure(figsize=((7.5, 3))) # plot
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.96, 
                        top=0.96, wspace=0.3, hspace=0.30)
    ax1 = fig.add_subplot(121)
    ax1.scatter(x, y, s=3)
    ax1.plot(x_np, model_fit.predict(x_np), c='k')
    ax1.set_ybound(-90, 100)
    ax1.set_xlabel('Ambient temperature ($^\circ$C)')
    ax1.set_ylabel('Range decline ratio (%)')
    ax2 = fig.add_subplot(122)
    cmap = sns.cubehelix_palette(start=2, light=1, as_cmap=True)
    sns.kdeplot(x=x, y=y,cmap=cmap, fill=True,levels=15,ax=ax2,)
    ax2.set_ybound(-90, 100)
    ax2.set_xlabel('Ambient temperature ($^\circ$C)')
    ax2.set_ylabel('Range decline ratio (%)')
#%%
    # Statistical changes of ranges across varied regions and during twelve months.
    Month = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11','12']
    data_op_p4['start_time'] = pd.to_datetime(data_op_p4['b.st_time_e'],unit='ms') + pd.DateOffset(hours=8)#unix time
    data_op_p4['end_time'] = pd.to_datetime(data_op_p4['b.et_time_e'],unit='ms')+ pd.DateOffset(hours=8)
    norm_rg_lst = []
    for i in range(1, 13, 1):
        msk1 = data_op_p4['start_time'].dt.year == 2019
        msk2 = data_op_p4['start_time'].dt.month == i
        block_tmp = data_op_p4.loc[msk1 & msk2, 'normalized_range'].copy()
        norm_rg_lst.append(block_tmp.mean())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Month, norm_rg_lst, '-X', color='k', label='Stat.')
    ax.legend(loc='upper right')