# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:30:59 2019

@author: Txin
"""

import pandas as pd
from pandas import DataFrame as df
import numpy as np
import os

method_list=['CDL','CF']
#method_list=['CF']

dataset_list=['citeulike-a','citeulike-t']
for method in method_list:
    data_path='result_data_'+method
    for  dataset in dataset_list:
        ##Three_way_recommendation#####
        three_way_file_name=dataset+'_three_way_info.csv'
        three_way_file=os.path.join(data_path,three_way_file_name)
        three_result_info_df = pd.read_csv(three_way_file)
        three_groups=three_result_info_df.groupby(['cost_num'])
        three_bou_num=df()
        three_cost_info=df()
        for cost_num,group in three_groups:
            group_one=group.groupby('n_topic').mean().astype(int)
            del group_one['test_fold']
            group_one.reset_index(inplace=True)
            three_bou_num['n_topic']=group_one['n_topic']
            num_name='num_'+str(int(cost_num))+'_bou'
            three_bou_num[num_name] = group_one['BL'] + group_one['BD']
            cost_info_df=group.groupby('n_topic').mean()[['d_cost','t_cost']]
            cost_info_df.reset_index(inplace=True)
            d_name='cost_'+str(int(cost_num))+'_d'
            t_name='cost_'+str(int(cost_num))+'_t'
            three_cost_info['n_topic']=cost_info_df['n_topic']
            three_cost_info[d_name]=cost_info_df['d_cost']
            three_cost_info[t_name]=cost_info_df['t_cost']*0.01
        three_bou_file_name=dataset+'_three_way_bou_num.csv'
        three_bou_file=os.path.join(data_path,three_bou_file_name)
        three_bou_num.to_csv(three_bou_file,index=None)
        three_cost_file_name=dataset+'_three_way_cost.csv'
        three_cost_file=os.path.join(data_path,three_cost_file_name)
        three_cost_info.to_csv(three_cost_file,index=None)
        
        ###Two_way_recommendation
        two_way_file_name=dataset+'_two_way_info.csv'
        two_way_file=os.path.join(data_path,two_way_file_name)
        two_result_info_df = pd.read_csv(two_way_file)
        two_groups=two_result_info_df.groupby(['cost_num'])
        two_bou_num=df()
        two_cost_info=df()
        for cost_num,group in two_groups:
            cost_info_df=group.groupby('n_topic').mean()[['d_cost','t_cost']]
            cost_info_df.reset_index(inplace=True)
            d_name='cost_'+str(int(cost_num))+'_d'
            t_name='cost_'+str(int(cost_num))+'_t'
            two_cost_info['n_topic']=cost_info_df['n_topic']
            two_cost_info[d_name]=cost_info_df['d_cost']
            two_cost_info[t_name]=cost_info_df['t_cost']*0.01
      
        two_cost_file_name=dataset+'_two_way_cost.csv'
        two_cost_file=os.path.join(data_path,two_cost_file_name)
        two_cost_info.to_csv(two_cost_file,index=None)
