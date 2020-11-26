#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yang Z.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
sys.path.append("./code")
import data_load_01, data_load_02

def get_drv_chg_tbl(veh_model_tmp):
    # Loading dynamic vehicular operating data
    data_op = data_load_01.get_operating_data_auto()
    data_info = data_load_02.get_veh_info() # data specification data
    # merge dynamic and static data
    data_op_p1 = pd.merge(data_op, data_info, 
                          left_on='b.vin', right_on='vin') 
    data_op_p2 = data_op_p1.copy() #copy
    # Data screening of vehicle models
    # vehicle model can be selected as specific requirements.
    # a vehicle model can be specifically modified accorading to study requirements.
    msk =  data_op_p2.veh_model == veh_model_tmp 
    data_op_p3 = data_op_p2.loc[msk, :]
    data_op_p4 = data_op_p3.copy() #copy
    # To obtain the driving distances of individual sesssions.
    data_op_p4['dert_dist'] = (data_op_p4['b.stop_mileage'] - 
                               data_op_p4['b.start_mileage'])
    data_op_p4['dert_soc'] = data_op_p4['b.start_soc'] - data_op_p4['b.end_soc']
    data_op_p4['start_time'] = pd.to_datetime(data_op_p4['b.st_time_e'],unit='ms') + pd.DateOffset(hours=8)#unix time
    data_op_p4['end_time'] = pd.to_datetime(data_op_p4['b.et_time_e'],unit='ms')+ pd.DateOffset(hours=8)
    msk1 = data_op_p4['b.start_temp'] > -20
    data_op_p4 = data_op_p4.loc[msk1, :]
    # Get individual charging and driving datasets.
    msk1 = data_op_p4['start_time'].dt.year == 2019
    msk2 = data_op_p4['b.category'] == 10 # 10 and 30 for driving and charging sessions.
    msk3 = data_op_p4['b.category'] == 30 # 10 and 30 for driving and charging sessions.
    data_op_p4_drv = data_op_p4[msk1 & msk2]        
    data_op_p4_chg = data_op_p4[msk1 & msk3]
    data_op_p4_drv['dert_soc'] = data_op_p4_drv['b.start_soc'] -  data_op_p4_drv['b.end_soc']
    chg_cols = ['vin', 'veh_model', 'start_time', 'end_time',
                'b.start_soc', 'b.end_soc', 'dert_soc', 'b.start_mileage', 'b.stop_mileage', 
                'b.power', 'b.volume', 'b.charge_c', 'b.start_temp', 
                'b.end_temp', 'city', 'fleet_type', ]
    drv_cols = ['vin', 'veh_model', 'start_time', 'end_time',
                'b.start_soc', 'b.end_soc', 'dert_soc',
                'b.start_mileage', 'b.stop_mileage', 'dert_dist',
                'b.start_temp', 'b.end_temp', 'city', 'fleet_type',]
    data_op_p5_chg = data_op_p4_chg[chg_cols]
    data_op_p5_drv = data_op_p4_drv[drv_cols]
    del msk1, msk2, msk3, chg_cols, drv_cols
    return data_op_p5_chg, data_op_p5_drv

def get_ecr_func_01(data_op_p5_drv, data_op_p5_chg, vin_list):
    data_op_p7_drv = pd.DataFrame()
    for vin_tmp in vin_list:
        # Separate individual vehicles by using key 'vin'.
        msk1 = data_op_p5_chg.vin == vin_tmp
        msk2 = data_op_p5_drv.vin == vin_tmp
        # Sort start time and reset indices.
        data_op_p6_chg = data_op_p5_chg[msk1].sort_values('start_time').reset_index(drop=True)
        data_op_p6_drv = data_op_p5_drv[msk2].sort_values('start_time').reset_index(drop=True)
        # Calculate the energy consumption rate
        data_op_p6_drv['ecr'] = -1
        data_op_p6_drv['e_per_soc'] = -1
        data_op_p6_drv2 = pd.DataFrame(columns=data_op_p6_drv.columns)
        for i in range(len(data_op_p6_drv)):
            if i == 0:
                row_last = data_op_p6_drv.iloc[0, :]
            else:
                row = data_op_p6_drv.iloc[i, :]
                if (row['b.start_mileage'] == row_last['b.stop_mileage']) & \
                    (row['b.start_soc'] == row_last['b.end_soc']):
                    row_last['b.stop_mileage'] = row['b.stop_mileage']
                    row_last['dert_dist'] = (row_last['b.stop_mileage'] - row_last['b.start_mileage'])
                    row_last['end_time'] = row['end_time']
                    row_last['b.end_soc'] = row['b.end_soc']
                    row_last['dert_soc'] = row_last['b.start_soc'] - row_last['b.end_soc']
                    row_last['b.end_temp'] = row['b.end_temp']
                else:
                    data_op_p6_drv2 = pd.concat([data_op_p6_drv2, pd.DataFrame(row_last).T])
                    row_last = data_op_p6_drv.iloc[i, :]
        # data_op_p6_drv2
        for i in range(len(data_op_p6_drv2)-1):
            # Select driving sessions occuring between two charging sessions.
            t1 = data_op_p6_drv2.start_time.iloc[i] - pd.DateOffset(days=1)
            t2 = data_op_p6_drv2.start_time.iloc[i] + pd.DateOffset(days=1)
            msk1 = (data_op_p6_chg.start_time >= t1) & (data_op_p6_chg.start_time <= t2)
            data_block = data_op_p6_chg.loc[msk1, :]
            if len(data_block) == 0:
                continue
            e_per_soc_se = data_block['b.power']/ (-1 * data_block.dert_soc)
            e_per_soc = e_per_soc_se.mean()
            e_tmp = (e_per_soc * data_op_p6_drv2['dert_soc'].iloc[i])
            d_tmp = data_op_p6_drv2['dert_dist'].iloc[i]
            data_op_p6_drv2['ecr'].iloc[i] = e_tmp / d_tmp
            data_op_p6_drv2['e_per_soc'].iloc[i] = e_per_soc
        data_op_p7_drv = pd.concat([data_op_p7_drv, data_op_p6_drv2])
    data_op_p7_drv['dist_per_soc'] = data_op_p7_drv['dert_dist'] / data_op_p7_drv['dert_soc']
    return data_op_p7_drv

def get_data_ecr(veh_model_tmp='1a5afe0eb2520bdfa00134b987268a8b'):# vehicle models can be selected.
    dict_fleet = {'私人乘用车':'private', '公务乘用车':'official', 
                  '出租乘用车':'taix', '租赁乘用车':'rental'}
    dict_reg = {'上海市':'Shanghai', '北京市':'Beijing', '广州市':'Guangzhou'}
    print('Getting chg and drv tbls...\n')
    data_op_p5_chg, data_op_p5_drv = get_drv_chg_tbl(veh_model_tmp)
    vin_list = data_op_p5_chg.vin.unique()
    print('Computing ECRs...\n')
    data_op_p7_drv = get_ecr_func_01(data_op_p5_drv, data_op_p5_chg, vin_list)
    msk1 = data_op_p7_drv.ecr > 0.1
    msk2 = data_op_p7_drv.ecr < 0.5
    data_op_p8_drv = data_op_p7_drv.loc[msk1 & msk2, :]
    data_op_p8_drv['region'] = data_op_p8_drv['city'].apply(lambda x: dict_reg[x])
    data_op_p8_drv['fleet_type'] = data_op_p8_drv['fleet_type'].apply(lambda x: dict_fleet[x])
    del data_op_p8_drv['city']
    print('Finish.\n')
    return data_op_p8_drv
#%%
if __name__ == '__main__':
    veh_model_tmp = '1a5afe0eb2520bdfa00134b987268a8b' # choose  vehicle models
    # data_op_p8_drv = get_data_ecr(veh_model_tmp)
    data_op_p5_chg, data_op_p5_drv = get_drv_chg_tbl(veh_model_tmp)
    vin_list = data_op_p5_chg.vin.unique()
#%%
    # Calculate energy consumpiton rates by using adjacent charging sessions of driving sessions.
    data_op_p7_drv = get_ecr_func_01(data_op_p5_drv, data_op_p5_chg, vin_list)
    msk1 = data_op_p7_drv.ecr > 0.1
    msk2 = data_op_p7_drv.ecr < 0.5
    data_op_p8_drv = data_op_p7_drv.loc[msk1 & msk2, :]
#%%
    # Calculate energy consumption rates by using energy predictions.
    data_op_p5_chg2 = data_op_p5_chg.copy()
    data_op_p5_chg2['dert_soc'] = -1 * data_op_p5_chg2['dert_soc']
    cols = ['dert_soc', 'b.start_temp', 'b.start_mileage']
    X_train = data_op_p5_chg2.loc[:, cols].copy()
    y_train = data_op_p5_chg2.loc[:, 'b.power'].copy()
    X_test = data_op_p5_drv.loc[:, cols].copy()
    #initiate model
    model_rf = RandomForestRegressor(n_estimators=200,random_state=0)
    model_rf.fit(X_train.values, y_train.values.ravel())
    y_predict = model_rf.predict(X_test.values)
    data_op_p9_drv = data_op_p5_drv.copy()
    data_op_p9_drv['ec'] = y_predict
    data_op_p9_drv['ecr'] =  data_op_p9_drv['ec'] /  data_op_p9_drv['dert_dist']
    msk1 = data_op_p9_drv.ecr > 0.1
    msk2 = data_op_p9_drv.ecr < 0.5
    data_op_p9_drv = data_op_p9_drv.loc[msk1 & msk2, :]
#%%
    # plt.scatter(data_op_p9_drv['b.start_temp'], data_op_p9_drv['ecr'], c='r')
    plt.scatter(data_op_p8_drv['b.start_temp'], data_op_p8_drv['ecr'], c='b')  