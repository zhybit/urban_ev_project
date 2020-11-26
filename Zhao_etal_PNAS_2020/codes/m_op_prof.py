#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: YZ
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append("./codes")

def mat_pow(start_time_array, end_time_array, powe, num):
    l = len(start_time_array)
    record_mat = np.zeros((l, num))
    for i in range(l):
        if powe[i] != powe[i]:
            po = 0
        else:
            po = powe[i]
        if start_time_array[i] <= end_time_array[i]:
            record_mat[i, start_time_array[i]:end_time_array[i]] = po
        if start_time_array[i] > end_time_array[i]:
            record_mat[i, start_time_array[i]:] = po
            record_mat[i, :end_time_array[i]] = po
    return record_mat

def mon_chg_power(data_chg_mon_tmp):
    vids_chg = data_chg_mon_tmp.veh_id.unique()
    vids_chg_mat_list = []
    for vid_tmp in vids_chg:
        data_chg_mon_vid_tmp = data_chg_mon_tmp.loc[data_chg_mon_tmp.veh_id == vid_tmp, :]
        mon_vid_unique_days = data_chg_mon_vid_tmp.start_time_dt.dt.day.unique()
        # print(len(mon_vid_unique_days), mon_vid_unique_days.tolist())
        start_time_array = data_chg_mon_vid_tmp.start_minute.values
        end_time_array =  data_chg_mon_vid_tmp.end_minute.values
        powe = data_chg_mon_vid_tmp.average_power.values
        chg_mat_wd = mat_pow(start_time_array, end_time_array, powe, 1440)
        chg_mat_sum_wd = chg_mat_wd.sum(axis=0)
        chg_mat_sum_wd = chg_mat_sum_wd / len(mon_vid_unique_days)
        vids_chg_mat_list.append(list(chg_mat_sum_wd))
    vids_chg_mat_arr = np.array(vids_chg_mat_list)
    chg_power_perCar_perDay = vids_chg_mat_arr.sum(axis=0) / len(vids_chg)
    return chg_power_perCar_perDay

def mat_minu(start_time_array, end_time_array, sta, num):
    l = len(start_time_array)
    record_mat = np.zeros((l, num))
    for i in range(l):
        if start_time_array[i] <= end_time_array[i]:
            record_mat[i, start_time_array[i]:end_time_array[i]] = 1
        if (start_time_array[i] > end_time_array[i]) & (sta==30):
            record_mat[i, start_time_array[i]:] = 1
            record_mat[i, :end_time_array[i]] = 1
    return record_mat

def month_oper_num(gb_drv_mon, gb_chg_mon):
    'drive'
    vins_drv = gb_drv_mon.veh_id.unique()
    drv_week_num_list = []
    vins_drv_num = len(vins_drv)
    weekday_num_dict = {0:vins_drv_num* 7, 1:vins_drv_num* 7, 
                        2:vins_drv_num* 7, 3:vins_drv_num* 7,
                        4:vins_drv_num* 7, 5:vins_drv_num* 7, 6:vins_drv_num* 7}
    for vin_tmp in vins_drv:
        data_tmp = gb_drv_mon.loc[gb_drv_mon.veh_id == vin_tmp]
        drv_week = []
        weekday_num_list = []
        for wd in np.arange(0, 7, 1):
            msk = data_tmp.start_time_dt.dt.weekday == wd
            data_tmp2 = data_tmp.loc[msk, :]
            if len(data_tmp2) == 0:
                drv_mat_sum_wd = np.zeros(1440)
                drv_week = drv_week + list(drv_mat_sum_wd)
                weekday_num_dict[wd] = weekday_num_dict[wd] - 1
                weekday_num_list = [1]
                continue
            weekday_num = len(data_tmp2.start_time_dt.dt.day.unique())
            weekday_num_list.append(weekday_num)
            drv_mat_wd = mat_minu(data_tmp2.start_minute.values, data_tmp2.end_minute.values, 10, 1440)
            drv_mat_sum_wd = drv_mat_wd.sum(axis=0)/weekday_num
            drv_week = drv_week + list(drv_mat_sum_wd)
        drv_week = list(np.array(drv_week) * np.array(weekday_num_list).mean())
        drv_week_num_list.append(drv_week)
    drv_week_num_arr = np.array(drv_week_num_list)
    drv_week_num_mean = drv_week_num_arr.sum(axis=0) / vins_drv_num
    for wd_tmp in np.arange(0, 7, 1):
        drv_week_num_mean[wd_tmp*1440:(wd_tmp+1)*1440] = \
            drv_week_num_mean[wd_tmp*1440:(wd_tmp+1)*1440] / weekday_num_dict[wd_tmp]
    drv_week_num_mean = drv_week_num_mean * np.array(list(weekday_num_dict.values())).mean()
    'charge'
    vins_chg = gb_chg_mon.veh_id.unique()
    chg_week_num_list = []
    vins_chg_num = len(vins_chg)
    weekday_num_dict2 = {0:vins_chg_num * 7, 1:vins_chg_num* 7, 
                         2:vins_chg_num* 7, 3:vins_chg_num* 7,
                        4:vins_chg_num* 7, 5:vins_chg_num* 7, 6:vins_chg_num* 7}
    for vin_tmp in vins_drv:
        data_tmp = gb_chg_mon.loc[gb_chg_mon.veh_id == vin_tmp]
        chg_week = []
        weekday_num_list = []
        for wd in np.arange(0, 7, 1):
            msk = data_tmp.start_time_dt.dt.weekday == wd
            data_tmp2 = data_tmp.loc[msk, :]
            if len(data_tmp2) == 0:
                chg_mat_sum_wd = np.zeros(1440)
                chg_week = chg_week + list(chg_mat_sum_wd)
                weekday_num_dict2[wd] = weekday_num_dict2[wd] - 1
                weekday_num_list = [1]
                continue
            weekday_num_tmp = len(data_tmp2.start_time_dt.dt.day.unique())
            weekday_num_list.append(weekday_num_tmp)
            chg_mat_wd = mat_minu(data_tmp2.start_minute.values, data_tmp2.end_minute.values, 30, 1440)
            chg_mat_sum_wd = chg_mat_wd.sum(axis=0)/weekday_num_tmp
            chg_week = chg_week + list(chg_mat_sum_wd)
        chg_week = list(np.array(chg_week) * np.array(weekday_num_list).mean())
        chg_week_num_list.append(chg_week)
    chg_week_num_arr = np.array(chg_week_num_list)
    chg_week_num_mean = chg_week_num_arr.sum(axis=0) / vins_chg_num
    for wd_tmp in np.arange(0, 7, 1):
        chg_week_num_mean[wd_tmp*1440:(wd_tmp+1)*1440] = \
            chg_week_num_mean[wd_tmp*1440:(wd_tmp+1)*1440] / weekday_num_dict2[wd_tmp]
    chg_week_num_mean = chg_week_num_mean * np.array(list(weekday_num_dict2.values())).mean()
    return drv_week_num_mean, chg_week_num_mean


#%%
if __name__ == '__main__':
    # Load data
    data_r = pd.read_csv('./data/ev_data1/data_sta_sets/operate_prof.csv')
    # Reformat and create features
    data_p1 = data_r.copy()
    data_p1['start_time_dt'] = pd.to_datetime(data_p1['start_time'])
    data_p1['end_time_dt'] = pd.to_datetime(data_p1['end_time'])
    data_p1['start_minute'] = 60 * data_p1['start_time_dt'].dt.hour + \
    data_p1['start_time_dt'].dt.minute
    data_p1['end_minute'] = 60 * data_p1['end_time_dt'].dt.hour + \
    data_p1['end_time_dt'].dt.minute
    # Get regional partitions
    data_bj = data_p1.loc[data_p1.region=='Beijing', :]
    data_sh = data_p1.loc[data_p1.region=='Shanghai', :]
    data_gz = data_p1.loc[data_p1.region=='Guangzhou', :]
    # Separate records with different operating states
    data_bj_drv = data_bj.loc[data_bj.state=='drv', :]
    data_bj_chg = data_bj.loc[data_bj.state=='chg', :]
    data_sh_drv = data_sh.loc[data_sh.state=='drv', :]
    data_sh_chg = data_sh.loc[data_sh.state=='chg', :]
    data_gz_drv = data_gz.loc[data_gz.state=='drv', :]
    data_gz_chg = data_gz.loc[data_gz.state=='chg', :]
#%%
    #Calculate numbers of BJ EVs
    gb_drv = data_bj_drv.copy()
    gb_chg = data_bj_chg.copy()
    drv_week_arr = []
    chg_week_arr = []
    ratio_list = []
    for mon in np.arange(1, 13, 1):
        gb_drv_mon = gb_drv.loc[gb_drv.start_time_dt.dt.month==mon, :]
        gb_chg_mon = gb_chg.loc[gb_chg.start_time_dt.dt.month==mon, :]
        drv_week_num_mean, chg_week_num_mean = month_oper_num(gb_drv_mon, gb_chg_mon)
        drv_week_arr.append(list(drv_week_num_mean))
        chg_week_arr.append(list(chg_week_num_mean))
        ratio = round(sum(chg_week_num_mean)/sum(drv_week_num_mean), 3)
        ratio_list.append(ratio)
    drv_week_num_mean_arr_bj = np.array(drv_week_arr)
    chg_week_num_mean_arr_bj = np.array(chg_week_arr)
    # Calculate seasons
    x1 = drv_week_num_mean_arr_bj[[2, 3, 4], :].sum(axis=0)
    x2 = drv_week_num_mean_arr_bj[[5, 6, 7], :].sum(axis=0)
    x3 = drv_week_num_mean_arr_bj[[8, 9, 10], :].sum(axis=0)
    x4 = drv_week_num_mean_arr_bj[[11, 0, 1], :].sum(axis=0)
    y1 = chg_week_num_mean_arr_bj[[2, 3, 4], :].sum(axis=0)
    y2 = chg_week_num_mean_arr_bj[[5, 6, 7], :].sum(axis=0)
    y3 = chg_week_num_mean_arr_bj[[8, 9, 10], :].sum(axis=0)
    y4 = chg_week_num_mean_arr_bj[[11, 0, 1], :].sum(axis=0)
#%%    
    #Calculate numbers of SH EVs
    gb_drv = data_sh_drv.copy()
    gb_chg = data_sh_chg.copy()
    drv_week_arr = []
    chg_week_arr = []
    ratio_list_sh = []
    for mon in np.arange(1, 13, 1):
        gb_drv_mon = gb_drv.loc[gb_drv.start_time_dt.dt.month==mon, :]
        gb_chg_mon = gb_chg.loc[gb_chg.start_time_dt.dt.month==mon, :]
        drv_week_num_mean, chg_week_num_mean = month_oper_num(gb_drv_mon, gb_chg_mon)
        drv_week_arr.append(list(drv_week_num_mean))
        chg_week_arr.append(list(chg_week_num_mean))
        ratio = round(sum(chg_week_num_mean)/sum(drv_week_num_mean), 3)
        ratio_list_sh.append(ratio)
    drv_week_num_mean_arr_sh = np.array(drv_week_arr)
    chg_week_num_mean_arr_sh = np.array(chg_week_arr)
    #Calculate numbers of GZ EVs
    gb_drv = data_gz_drv.copy()
    gb_chg = data_gz_chg.copy()
    drv_week_arr = []
    chg_week_arr = []
    ratio_list_gz = []
    for mon in np.arange(1, 13, 1):
        gb_drv_mon = gb_drv.loc[gb_drv.start_time_dt.dt.month==mon, :]
        gb_chg_mon = gb_chg.loc[gb_chg.start_time_dt.dt.month==mon, :]
        drv_week_num_mean, chg_week_num_mean = month_oper_num(gb_drv_mon, gb_chg_mon)
        drv_week_arr.append(list(drv_week_num_mean))
        chg_week_arr.append(list(chg_week_num_mean))
        ratio = round(sum(chg_week_num_mean)/sum(drv_week_num_mean), 3)
        ratio_list_gz.append(ratio)
    drv_week_num_mean_arr_gz = np.array(drv_week_arr)
    chg_week_num_mean_arr_gz = np.array(chg_week_arr)
#%%
    from matplotlib.ticker import MultipleLocator
    'plot Beijinng seasons'
    weekdays = ['Mon.', 'Tue.', 'Wed.', 'Thu.', 'Fri.', 'Sat.', 'Sun.']
    f_size = 8
    plt.rcParams['font.family'] = 'Arial'
    font = {'size': f_size} 
    font2 = {'size': f_size-2}
    fx, fy = 4, 1
    fig = plt.figure(figsize=(5, 4))
    plt.subplots_adjust(left=0.1, bottom=0.06, right=0.99, top=0.98, wspace=0.3, hspace=0.4)
    ax1 = fig.add_subplot(fx, fy, 1)
    ax2 = fig.add_subplot(fx, fy, 2)
    ax3 = fig.add_subplot(fx, fy, 3)
    ax4 = fig.add_subplot(fx, fy, 4)
    al = 0.95
    for ax, x, y, sea, figorder in zip([ax1, ax2, ax3, ax4], [x1, x2, x3, x4], 
                             [y1, y2, y3, y4], ['Spring', 'Summer', 'Autumn', 'Winter'], 
                             ['a', 'b', 'c', 'd']):
        ax.fill_between(np.arange(0, 1440*7)/1440, 1000 * x / x.max(), 
                         label='Driving', alpha=al, color='royalblue', linewidth=0.1, edgecolor='k')
        ax.fill_between(np.arange(0, 1440*7)/1440, 1000 * y / x.max(), 
                         label='Charging', alpha=al, color='tomato', linewidth=0.1, edgecolor='k')
        ratio = y.mean()/x.mean()
        tx, ty = 0.83, 0.84
        ax.text(tx, ty, 'R$_{\mathregular{dc}}$=%s'%round(ratio,2), fontsize=f_size-1, transform=ax.transAxes) 
        tx, ty = 0.65, 0.84
        ax.text(tx, ty, sea, fontsize=f_size-1, transform=ax.transAxes) 
        ax.yaxis.set_major_locator(MultipleLocator(500))
        ax.set_yticklabels(['{:,.0f}'.format(k) for k in ax.get_yticks()])
#%%
    weekdays = ['Mon.', 'Tue.', 'Wed.', 'Thu.', 'Fri.', 'Sat.', 'Sun.']
    f_size = 9
    plt.rcParams['font.family'] = 'Arial'
    font = {'size': f_size} 
    font2 = {'size': f_size-2}
    fig = plt.figure(figsize=(7.2, 7))
    plt.subplots_adjust(left=0.068, bottom=0.03, right=0.99, top=0.99, wspace=0.05, hspace=0.1)
    fy = 2
    fx = 12 / fy
    ii=0
    for mon in np.arange(1, 13, 1):
        ii = ii + 1
        ax = fig.add_subplot(fx, fy, ii)
        y_drv = drv_week_num_mean_arr_bj[mon-1, :]
        y_chg = chg_week_num_mean_arr_bj[mon-1, :]
        ax.fill_between(np.arange(0, 1440*7)/1440, 1000 * y_drv/y_drv.max(), color='b', linewidth=0.1, edgecolor='k')
        ax.fill_between(np.arange(0, 1440*7)/1440, 1000 * y_chg/y_drv.max(), color='tomato', linewidth=0.1, edgecolor='k')
        ax.set_ybound(0, 1300)
        ax.set_xbound(0, 7)
        ax.tick_params(labelsize=f_size-1, direction='out')
        ax.set_xticklabels([])
        ax.set_ylabel('Number', font, labelpad=0)
        if mon not in np.arange(1, 13, 2):
            ax.set_yticklabels([])
            ax.set_ylabel('', font, labelpad=0)
#%%
    f_size = 8
    plt.rcParams['font.family'] = 'Arial'
    font = {'size': f_size} 
    # font2 = {'size': f_size-2}
    fig = plt.figure(figsize=(7.2, 4))
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.99, wspace=0.3, hspace=0.5)
    fx, fy = 3, 4
    ii=0
    chg_power_perCar_perDay_cum = pd.DataFrame()
    for mon in np.arange(1, 13, 1):
        ii = ii + 1
        ax = fig.add_subplot(fx, fy, ii)
        data_bj_chg_mon = data_bj_chg.loc[data_bj_chg.start_time_dt.dt.month == mon, :]
        chg_power_perCar_perDay = mon_chg_power(data_bj_chg_mon)
        p = chg_power_perCar_perDay.mean()
        ax.fill_between(np.arange(0, 1440)/60, chg_power_perCar_perDay)
        ax.set_xbound(0, 24)
        ax.set_ybound(0, 2)
        ax.tick_params(labelsize=f_size-1)
        ax.set_ylabel('Power (MW)', font, labelpad=0)
        ax.set_xlabel('Hour of day', font, labelpad=0)