#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yang Z.
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator #,LinearLocator, FormatStrFormatter
import seaborn as sns
# from sklearn import linear_model
# from sklearn import metrics
os.chdir('/Users/zhaoyang/Nutstore Files/Nutstore/codes_python/project_03/')
sys.path.append("./code_upload")
import data_load_01
import data_load_02

# cumulative distribution func
def cdf_se(se):
    se = se[se<=600]
    x = np.linspace(0,500,500) 
    length = len(se)
    y = [len(se[se<=x_values])/length for x_values in tqdm(x)]
    y_np = np.array(y)
    para = np.polyfit(x,y_np,deg = 10)
    f = np.poly1d(para)
    return x,y,f

# fitting by using np
def get_fit(x, y): # get para
    p = np.polyfit(x, y, deg=300)
    f = np.poly1d(p)
    return p, f

# reclassify fleet types
def fleet_class(cell): # classify fleet type into pri/pub
    if cell == 'private':
        return 'pr'
    elif cell in ['taix', 'rental']:
        return 'pu'
    else: # official
        return cell
   
# make cdf dataframe
def cdf_df_make(daily_dist): # fleet types should include pr and pu
    cdf_list = []
    for reg in ['BJ', 'SH', 'GZ']:
        for ft in ['pr', 'pu']:
            # print('Start prossessing: %s, %s...'%(reg,ft))
            msk1 = daily_dist.region == reg
            msk2 = daily_dist.fleet_type2 == ft
            x,y,f = cdf_se(daily_dist.loc[(msk1 & msk2), 'dert_dist(km)'])
            cdf_list.append([reg, ft, x, y, f])
    cols = ['region', 'fleet_type', 'x_arr', 'y_arr', 'func']
    cdf_df = pd.DataFrame(cdf_list, columns=cols)
    return cdf_df

# get the function of probability -> daily drivein distance
def get_reverse_func(cdf_df): 
    cdfr_list = []
    for reg in ['BJ', 'SH', 'GZ']:
        for ft in ['pr', 'pu']:
            # print('Start prossessing: %s, %s...'%(reg,ft))
            msk1 = cdf_df.region == reg
            msk2 = cdf_df.fleet_type == ft
            cdf_tmp = cdf_df.loc[msk1 & msk2, :]
            x = cdf_tmp.loc[:,'x_arr'].iloc[0]
            y = cdf_tmp.loc[:,'y_arr'].iloc[0]
            para, func = get_fit(y, x)
            cdfr_list.append([reg, ft, para, func])
    cols = ['region', 'fleet_type', 'p', 'f']
    cdfr_df = pd.DataFrame(cdfr_list, columns=cols)
    return cdfr_df

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

# calculate the unavailable battery energy
def waste_energy(e, max_temp_waste, max_age_waste, x1, x2):
    y1 = e * (1 - (1 - max_temp_waste*(1-x1)) * (1 - max_age_waste*(1-x2)))
    return y1

# get color palette
def get_ggplot_palette():
    plt.style.use('ggplot')
    color_palette_ggplot = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # sns.palplot(color_palette_ggplot)
    plt.style.use('default')
    return color_palette_ggplot

# setup the Months list
Month = ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'Jun.',
         'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.','Dec.']
Month = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11','12']
Month2 = ['', 'Feb.', '', 'Apr.', '', 'Jun.', 
          '', 'Aug.', '', 'Oct.', '','Dec.']
#%%
if __name__ == '__main__':
    # Loading daily driving datasets
    daily_dist = data_load_02.get_daily_dist() # load data
    # fleet type reclassifications
    daily_dist['fleet_type2'] = daily_dist['fleet_type'].apply(fleet_class) 
    # make matrices of cumulative distribution functions (cdf) 
    cdf_df = cdf_df_make(daily_dist) 
    cdfr_df = get_reverse_func(cdf_df)
    color_palette_ggplot = get_ggplot_palette()
#%%
    # distributions of daily driving distance
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    bins = np.linspace(0,400, 50)
    for reg in ['BJ', 'SH', 'GZ']:
        for ft in ['pr', 'pu']:
            msk1 = daily_dist.region == reg
            msk2 = daily_dist.fleet_type2 == ft
            data_tmp = daily_dist.loc[msk1 * msk2, 'dert_dist(km)']
            sns.distplot(data_tmp, hist=True, #color='b',
                          kde_kws={"shade": True,'linewidth':1,'linestyle':'--'},
                          bins=bins, 
                          ax=ax1, label='%s, %s'%(reg, ft))#, color="#3C5488FF", color="g"
    ax1.set_xbound(0, 400)
    plt.show()
    del bins
#%%
    # cdf curves of varied regions and fleet types
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for reg in ['BJ', 'SH', 'GZ']:
        for ft in ['pr', 'pu']:
            msk1 = cdf_df.region == reg
            msk2 = cdf_df.fleet_type == ft
            se = cdf_df.loc[msk1 & msk2, :]
            ax1.plot(se['x_arr'].iloc[0], se['y_arr'].iloc[0], label='%s, %s'%(reg, ft))
            se_r = cdfr_df.loc[msk1 & msk2, :]
            x = np.linspace(0, 0.999, 100)
            f = se_r['f'].iloc[0]
            ax2.plot(x, f(x), '--', label='%s, %s'%(reg, ft))
    ax1.legend(loc='lower right')
    ax2.legend(loc='upper left')
    plt.show()
#%%
    # Battery utilization rates for different regions and fleet types
    f_size = 10
    font = {'size': f_size} 
    plt.rcParams['font.family'] = 'Arial'
    fig = plt.figure(figsize=(4.5,3))
    plt.subplots_adjust(left=0.12, bottom=0.15, right=0.96, 
                        top=0.99, wspace=0.17, hspace=0.30)
    ax_sup1 = fig.add_subplot(111)
    ax_sup1.set_prop_cycle(color=color_palette_ggplot)
    xp = np.linspace(0, 0.99, 21)
    lw_line = 1
    ms = 3
    for reg, marker_style in zip(['BJ', 'SH', 'GZ'], ['o', '*', '^']):
        for ft in ['pr', 'pu']:
            for dist in [300, 450, 600]:
                msk1 = cdfr_df.region == reg
                msk2 = cdfr_df.fleet_type == ft
                f = cdfr_df.loc[msk1 & msk2, 'f'].iloc[0]
                ax_sup1.plot(xp,f(xp)/dist,'--o',c=None,linewidth=lw_line, 
                             markersize=ms,marker=marker_style, 
                             label='%s, %s, %s'%(reg, ft, dist))#, '-',c='tomato'
    ax_sup1.legend(loc='upper left', fontsize=8, ncol=3)
    ax_sup1.set_ybound(0,1.7)
    ax_sup1.set_xbound(0,1.01)
    ax_sup1.tick_params(labelsize=f_size)
    xmajorLocator = MultipleLocator(0.2)
    ax_sup1.xaxis.set_major_locator(xmajorLocator)
    vals = ax_sup1.get_xticks()
    # ax_sup1.set_xticks(ax_sup1.get_xticks().tolist())
    ax_sup1.set_xticklabels(['{:,.0f}'.format(x*100) for x in vals])
    vals = ax_sup1.get_yticks()
    ax_sup1.set_yticklabels(['{:,.0f}'.format(x*100) for x in vals])
    ax_sup1.set_xlabel('$P$ (%)',font)
    ax_sup1.set_ylabel('$R_\mathregular{b}$ (%)',font)
    del reg, ft, dist
#%%
    # Loading dynamic vehicular operating data
    data_op = pd.DataFrame()
    for i in range(4): # for 12 partitions
        d_tmp = data_load_01.get_operating_data(i)
        data_op = pd.concat([data_op, d_tmp])
    data_info = data_load_02.get_veh_info() # data specification data
    data_op_p1 = pd.merge(data_op, data_info, 
                          left_on='b.vin', right_on='vin') # merge dynamic and static data
    data_op_p2 = data_op_p1.copy() #copy
#%%
    # Data screening
    #filter for '北京' Beijing, '广州市' Guangzhou，‘上海市’ Shanghai
    msk1 = data_op_p2.city == '北京市'
    # vehicle model can be selected as specific requirements.
    msk2 = data_op_p2.veh_model == '1a5afe0eb2520bdfa00134b987268a8b' #choose model
    msk3 = data_op_p2['b.category'] == 10 # 10 and 30 for driving and charging sessions.
    msk = msk1 & msk2 & msk3
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
    msk =  msk1 & msk2 & msk3
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
    # optimal_avg_range = model_fit.predict(22)
#%%
    # Loading regional temperature data.
    data_city_temp = data_load_02.get_ambient_temp()
    msk1 = data_city_temp.Region == 'Beijing'
    msk2 = data_city_temp.Item == 'Avg'
    msk3 = data_city_temp.Item == 'Max'
    msk4 = data_city_temp.Item == 'Min'
    data_city_temp_p1 = data_city_temp.loc[msk1 & msk2, 'January':'December']
    data_city_temp_p2 = data_city_temp.loc[msk1 & msk3, 'January':'December']
    data_city_temp_p3 = data_city_temp.loc[msk1 & msk4, 'January':'December']
    months = data_city_temp_p1.columns
    # months = [m[0:3]+'.' for m in months]
    months = [str(int(s)) for s in range(1, 13, 1)]
    temps_avg = data_city_temp_p1.values.reshape(-1)
    temps_max = data_city_temp_p2.values.reshape(-1)
    temps_min = data_city_temp_p3.values.reshape(-1)
    del msk1, msk2, msk3, msk4
#%%
    # Declines in regions and during months generated by a general model.
    fig = plt.figure(figsize=((5, 3))) # plot
    plt.subplots_adjust(left=0.13, bottom=0.2, right=0.6, 
                        top=0.96, wspace=0.3, hspace=0.30)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(months, temps_avg, '--o', c='k', label='T. (Avg)')
    ax1.plot(months, temps_max, '--o', c='tomato', label='T. (Max)')
    ax1.plot(months, temps_min, '--o', c='royalblue', label='T. (Min)')
    ax2.plot(months, model_fit.predict(temps_avg), '-^', c='k', label='R (avgT)')
    ax2.plot(months, model_fit.predict(temps_max), '-s', c='tomato', label='R (maxT)')
    ax2.plot(months, model_fit.predict(temps_min), '-x', c='royalblue', label='R (minT)')
    ax1.set_xlabel('Month') 
    ax1.set_ylabel('Temperature ($^\circ$C)')
    ax2.set_ylabel('Range decline ratio (%)')
    fig.legend()
#%%
    # Averge and worst case calculations
    region_temp2 = data_city_temp.T
    region_bu = region_temp2.loc['January':, :].applymap(model_fit.predict) # model_fit.predic can be any model refecting performance changes
    region_bu = pd.concat([region_temp2.loc[['Region','Item'],:], region_bu])
    region_plot = pd.DataFrame()
    regions = data_city_temp.Region.unique()
    for region in regions:
        maskcol1 = (region_bu.loc['Region', :]==region)
        maskcol2 = (region_bu.loc['Item', :]=='Avg')
        temp = region_bu.loc[:, maskcol1]
        min_col = temp.loc['January':'December', :].astype(float).min().idxmin() #min df in the three cols，use idmin to get the col of mins
        min_month = temp.loc['January':'December', min_col].astype(float).idxmin()
        min_value = temp.loc[min_month, min_col]
        avg_value = temp.loc['January':'December', temp.loc['Item',:]=='Avg'].mean()
        dftemp = pd.DataFrame({'Region':region, 'Avg': avg_value, 
                               'min':min_value, 'min month': min_month})
        region_plot = pd.concat([region_plot, dftemp])
#%%
    # Changes of unavailable battery energy with improvements of battery technology.
    x = np.linspace(0, 1, 6)
    region_bu2 = region_bu.T.set_index(['Region', 'Item'])
    _block = region_bu.loc['January':'December',0:2]
    # max_temp_waste = 1 - _block.min().min()
    max_temp_waste=0.35 # derive from max range declines in varied regions.
    max_age_waste=0.2
    e = 70 #'70 kWh'
    x1, x2 = x, 0
    y1 = waste_energy(e, max_temp_waste, max_age_waste, x1, x2)
    x1, x2 = 0, x
    y2 = waste_energy(e, max_temp_waste, max_age_waste, x1, x2)
    x1, x2 = x, x
    y3 = waste_energy(e, max_temp_waste, max_age_waste, x1, x2)
    x1, x2 = x*0.5, x
    y4 = waste_energy(e, max_temp_waste, max_age_waste, x1, x2)
    x1, x2 = x, x*0.5
    y5 = waste_energy(e, max_temp_waste, max_age_waste, x1, x2)
    #'unavailable energy'
    x1, x2 = 0, 0
    output0 = waste_energy(e, max_temp_waste, max_age_waste, x1, x2)
    x1, x2 = 0.8, 0 #temperature 80%
    output1 = waste_energy(e, max_temp_waste, max_age_waste, x1, x2)
    x1, x2 = 0, 0.8 #aging 80%
    output2 = waste_energy(e, max_temp_waste, max_age_waste, x1, x2)
    print(output0, output1, output2)
    fig = plt.figure() # plot
    _ax1 = fig.add_subplot(111)
    _ax1.plot(x, y1,'-H', c = 'b', label='P1 T.', linewidth=1.5, markersize=5)
    _ax1.plot(x, y2,'-s', c = 'b', label='P1 A.', linewidth=1.5, markersize=5)
    _ax1.plot(x, y3,'-X', c = 'b', label='P1 T. & A.', linewidth=1.5, markersize=5)
    e3=100 #'100 kWh'
    e, x1, x2 = e3, x, 0
    y1 = waste_energy(e, max_temp_waste, max_age_waste, x1, x2)
    e, x1, x2 = e3, 0, x
    y2 = waste_energy(e, max_temp_waste, max_age_waste, x1, x2)
    e, x1, x2 = e3, x, x
    y3 = waste_energy(e, max_temp_waste, max_age_waste, x1, x2)
    e, x1, x2 = e3, x*0.5, x
    y4 = waste_energy(e, max_temp_waste, max_age_waste, x1, x2)
    e, x1, x2 = e3, x, x*0.5
    y5 = waste_energy(e, max_temp_waste, max_age_waste, x1, x2)
    _ax1.plot(x, y1,'--H', c = 'r', label='P2 T.', linewidth=1.5, markersize=5)
    _ax1.plot(x, y2,'--s', c = 'r', label='P2 A.', linewidth=1.5, markersize=5)
    _ax1.plot(x, y3,'--X', c = 'r', label='P2 T. & A.', linewidth=1.5, markersize=5)
    _ax1.legend(fontsize=8,loc=(0.02,0),ncol=1,frameon =False,
                columnspacing=0.9, labelspacing=0.2)
    _ax1.set_ybound(-2,50)
    _ax1.set_xlabel('Level (%)')
    _ax1.set_ylabel('$Q_\mathregular{m}$ (kWh)')
    vals = _ax1.get_xticks()
    _ax1.set_xticklabels(['{:,.0f}'.format(x*100) for x in vals])
    _ax2 = _ax1.twinx() #'100 kWh'
    unit_cost = 180
    _ax2.set_ybound(np.array(_ax1.get_ybound())*unit_cost)
    _ax2.set_ylabel('Equivalent cost (10$^\mathregular{3}$ US\$)')
    _ax2.set_ybound(np.array(_ax1.get_ybound())*unit_cost/1000)
    _ax2.set_yticklabels(['{:.1f}'.format(val) for val in _ax2.get_yticks()])
    
    del x, x1, x2, e, e3, y1, y2, y3, y4, y5, \
    max_temp_waste, max_age_waste, _block, vals
#%%    
    # Display temperatures during 12 months and across regions.
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Arial'
    f_size = 10
    font = {'size': f_size}
    font2 = {'size': f_size-2} 
    ms=4.5
    color_c = ['#292D64','#334382','#3D538D', '#598AB9','#7FC2D2','#8ABF95',
               '#B5CC5C','#F9EC58','#E09A46', '#CD4E36','#BB3633','#5F2120']
    color_c.reverse()
    color_ = [color_c[0]] + color_c[2:-2]
    order_list = [4, 2, 8, 5, 1, 6, 7, 3, 0]
    ii=0
    fig = plt.figure(figsize=(7.5,6))
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, 
                        top=0.95, wspace=0.3, hspace=0.350)
    region_temp = data_city_temp.copy()
    regions = data_city_temp.Region.unique()
    for i in order_list:
        ax = fig.add_subplot(3, 3, ii + 1)
        'Draw lines in 3 categories'
        mask1 = region_temp.Item=='Max'
        mask2 = region_temp.Item=='Avg'
        mask3 = region_temp.Item=='Min'
        msk = region_temp.Region==regions[i]
        temp1 = region_temp.loc[mask1 & msk, 'January':'December'].values.ravel()
        temp2 = region_temp.loc[mask2 & msk, 'January':'December'].values.ravel()
        temp3 = region_temp.loc[mask3 & msk, 'January':'December'].values.ravel()
        ax.plot(Month, temp1,':o', label = 'Max', markersize=ms,
                  linewidth=1.5, c=color_[ii],)
        ax.plot(Month, temp2,'-o', label = 'Avg.', markersize=ms,
                  linewidth=1.5, c=color_[ii],)
        ax.plot(Month, temp3,'--o', label = 'Min', markersize=ms,
                  linewidth=1.5, c=color_[ii],)
        ax.set_ybound(-20, 42)
        ax.tick_params(labelsize=f_size-1)
        # ax.set_xticklabels(Month2)
        ax.set_title(regions[i], font)
        if ii == 7:
            ax.set_xlabel('Month',font)
        if ii == 3:
            ax.set_ylabel('Ambient temperature ($^\circ$C)',font)
        # ax1_.set_yticklabels([ '{:,.0f}'.format(x*100) for x in ax1_.get_yticks()])
        ax.legend(fontsize=f_size-2,loc=(0.25, 0.01),ncol=2,frameon =False,
                  columnspacing=0.2, labelspacing=0.3)
        ii = ii + 1
        del temp1, temp2, temp3, mask1, mask2, mask3, msk
        