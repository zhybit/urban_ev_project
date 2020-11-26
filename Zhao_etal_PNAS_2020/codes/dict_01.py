#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: YZ
"""
#%%
#vehicle operating data
# data_op_xx
cols = [
        'b.vid',
        'b.vin',
        'b.st_time_e',
        'b.et_time_e',
        'b.category',
        'b.start_mileage',
        'b.stop_mileage',
        'b.start_soc',
        'b.end_soc',
        'b.avg_speed',
        'b.max_speed',
        'b.max_total_voltage',
        'b.min_total_voltage',
        'b.max_total_current',
        'b.min_total_current',
        'b.max_cell_volt',
        'b.min_cell_volt',
        'b.max_acquisition_point_temp',
        'b.min_acquisition_point_temp',
        'b.start_max_cell_volt',
        'b.start_min_cell_volt',
        'b.end_max_cell_volt',
        'b.end_min_cell_volt',
        'b.start_max_acquisition_point_temp',
        'b.start_min_acquisition_point_temp',
        'b.end_max_acquisition_point_temp',
        'b.end_min_acquisition_point_temp',
        'b.start_longitude',
        'b.start_latitude',
        'b.end_longitude',
        'b.end_latitude',
        'b.start_total_current',
        'b.start_total_volt',
        'b.end_total_current',
        'b.end_total_volt',
        'b.max_power',
        'b.min_power',
        'b.avg_current',
        'b.avg_volt',
        'b.charge_c',
        'b.power',
        'b.volume',
        'b.today_start_mileage',
        'b.today_end_mileage',
        'b.avg_acquisition_point_temp',
        'b.start_avg_temp',
        'b.end_avg_temp',
        'b.max_avg_temp',
        'b.min_avg_temp',
        'b.max_min_temp',
        'b.min_max_temp',
        'b.start_temp',
        'b.end_temp',
        'b.avg_motor_temp',
        'b.max_motor_temp',
        'b.min_motor_temp',
        'b.avg_motor_rpm',
        'b.max_motor_rpm',
        'b.min_motor_rpm',
        ]

data_info = [
         'vin',
         'fleet_type', 
         'city', 
         'province', 
         'veh_model', 
         'common_name',
         'brand', 
         'fuel_type'
         ]

dict_fuel_type = {'汽油/电混合动力':'hybrid', 
                  '纯电动':'BEV',}

dict_fleet = {'私人乘用车':'private', 
              '公务乘用车':'official',
              '出租乘用车':'taix',
              '租赁乘用车':'rental', 
              '公交客车':'bus',
              '物流特种车':'special_uses',
              }

dict_region = {'北京市':'Beijing', 
              '上海市':'Shanghai',
              '广州市':'Guangzhou',}            
