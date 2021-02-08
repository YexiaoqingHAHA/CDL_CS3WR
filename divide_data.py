# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 10:25:20 2018

@author: Xiaoqing Ye
"""

import numpy as np
import os


def divide_data(data_name,divide_num,test_num_one):
    data_file='ctrsr_datasets\\'+data_name+'\\users.dat'
    if data_name=='citeulike-a':     
        for i in range(1,divide_num+1):
            test_file='data\\cf-a\\cf-a-test-'+str(i)+'-users.dat'
            train_file='data\\cf-a\\cf-a-train-'+str(i)+'-users.dat'
      
            if os.path.exists(test_file):
                os.remove(test_file)
            if os.path.exists(train_file):
                os.remove(train_file)
            
            test_w=open(test_file,'a')
            train_w=open(train_file,'a')
            fp=open(data_file)
            lines=fp.readlines()
            for i,line in enumerate(lines):
                cite_content=line.strip().split(' ')
                cite_num=int(cite_content[0])
                cite_item=np.array(cite_content[1:])
                random_index=np.random.permutation(cite_num).tolist()
                   
                if cite_num>test_num_one:
                    test_index=random_index[:test_num_one]
                    test_item=cite_item[test_index].tolist()
                    train_index=random_index[test_num_one:]
                    train_item=cite_item[train_index].tolist()
                    test_num=test_num_one
                    train_num=cite_num-test_num_one
                else:
                    test_item=cite_item.tolist()
                    test_num=cite_num
                    train_item=[]
                    train_num=0
                test_w.write(str(test_num))
                test_w.write(' ')
                for item in test_item:
                    test_w.write(item)
                    test_w.write(' ')
                test_w.write('\n')
                
                train_w.write(str(train_num))
                train_w.write(' ')
                for item in train_item:
                    train_w.write(item)
                    train_w.write(' ')
                train_w.write('\n')
            test_w.close()
            train_w.close()
            fp.close()
        
    elif data_name=='citeulike-t':
        for i in range(1,divide_num+1):
            test_file='data\\cf-t\\cf-t-test-'+str(i)+'-users.dat'
            train_file='data\\cf-t\\cf-t-train-'+str(i)+'-users.dat'
            #num_user=7947
            #num_item=25975
            num_ratings=134860
            n_test=int(num_ratings*0.2)
            n_test_user=int(n_test/test_num_one)
            
            if os.path.exists(test_file):
                os.remove(test_file)
            if os.path.exists(train_file):
                os.remove(train_file)
            fp=open(data_file)
            lines=fp.readlines()
            user_rating_num=[]
            for i,line in enumerate(lines):
                cite_content=line.strip().split(' ')
                cite_num=int(cite_content[0])
                user_rating_num.append(cite_num)
            user_rating_num=np.array(user_rating_num)
            test_user=np.argsort(user_rating_num).tolist()[::-1][:n_test_user]
    
            test_w=open(test_file,'a')
            train_w=open(train_file,'a')
            for i,line in enumerate(lines):
                cite_content=line.strip().split(' ')
                cite_num=int(cite_content[0])
                cite_item=np.array(cite_content[1:])
                if i in test_user:
                    random_index=np.random.permutation(cite_num).tolist()
                    if cite_num>test_num_one:
                        test_index=random_index[:test_num_one]
                        test_item=cite_item[test_index].tolist()
                        train_index=random_index[test_num_one:]
                        train_item=cite_item[train_index].tolist()
                        test_num=test_num_one
                        train_num=cite_num-test_num_one
                    else:
                        test_item=cite_item.tolist()
                        test_num=cite_num
                        train_item=[]
                        train_num=0
                else:
                    test_num=0
                    test_item=[]
                    train_num=cite_num
                    train_item=cite_item.tolist()
                test_w.write(str(test_num))
                test_w.write(' ')
                for item in test_item:
                    test_w.write(item)
                    test_w.write(' ')
                test_w.write('\n')
                
                train_w.write(str(train_num))
                train_w.write(' ')
                for item in train_item:
                    train_w.write(item)
                    train_w.write(' ')
                train_w.write('\n')
            test_w.close()
            train_w.close()
            fp.close()
                    
            
        
if __name__=='__main__':
    data_names=['citeulike-a','citeulike-t']
    divide_num=20
    test_num_one=10
    for data_name in data_names:
        divide_data(data_name,divide_num,test_num_one)
        