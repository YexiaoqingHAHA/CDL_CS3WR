# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 10:56:48 2018

@author: ifbd
"""

from CDL import CDL
from DAE import DAE
import tensorflow as tf
import time
import argparse
import numpy as np
import data
import os
import pandas as pd
from multiprocessing import Pool
from functools import partial


current_time=time.time()

parser=argparse.ArgumentParser(description='Collaborative Deep Learning')
parser.add_argument('--model_name',choices=['CDL'],default='CDL')



##variation
parser.add_argument('--data_name',choices=['citeulike-a','citeulike-t'],default='citeulike-a')
parser.add_argument('--n_process',type=int,default=5)
parser.add_argument('--test_folds',type=int,default=20)
parser.add_argument('--train_epoch',type=int,default=100)
parser.add_argument('--max_hidden_neuron',type=int,default=100)



parser.add_argument('--random_seed',type=int,default=1000)
parser.add_argument('--display_step',type=int,default=1)
parser.add_argument('--lr',type=float,default=1e-3) #learning_rate
parser.add_argument('--optimizer_method',choices=['Adam','Adadelta','Adagrad','RMSProp','GradientDescent','Momentum'],default='Adam')
parser.add_argument('--keep-prob',type=float,default=0.9)  #dropout
parser.add_argument('--a',type=float,default=1) #observed ratings
parser.add_argument('--b',type=float,default=0.01) #unseen ratings
parser.add_argument('--grad_clip',choices=['True','False'],default='True')
parser.add_argument('--batch_normalization', choices=['True','False'], default = 'False')
parser.add_argument('--is_dummy',choices=['True','False'],default='False')


parser.add_argument('--hidden_neuron',type=int,default=50)
parser.add_argument('--corruption_level',type=float,default=0.3) # input corruption ratio

parser.add_argument('--f_act', choices=['Sigmoid','Relu','Elu','Tanh',"Identity"], default = 'Sigmoid') # Encoder Activation
parser.add_argument('--g_act', choices=['Sigmoid','Relu','Elu','Tanh',"Identity"], default = 'Sigmoid') # Decoder Activation


parser.add_argument('--encoder_method',choices=['SDAE'],default='SDAE')
parser.add_argument('--lambda_v',type=float,default=10) # xi_dk prior std / cost3
parser.add_argument('--lambda_u',type=float , default = 0.1) # x_uk prior std / cost5
parser.add_argument('--lambda_w',type=float , default = 0.1) # SDAE weight std. / weight , bias regularization / cost1
parser.add_argument('--lambda_n',type=float , default = 1000) # SDAE output (cost2)
args = parser.parse_args()


random_seed=args.random_seed
tf.reset_default_graph()
np.random.seed(random_seed)
n_process=args.n_process
max_hidden_neuron=args.max_hidden_neuron

model_name=args.model_name
data_name=args.data_name

a=args.a
b=args.b

test_folds=args.test_folds
hidden_neuron=args.hidden_neuron

#dropout
keep_prob=args.keep_prob
batch_normalization=args.batch_normalization

batch_size=256
lr=args.lr
train_epoch=args.train_epoch
optimizer_method=args.optimizer_method
display_step=args.display_step
decay_epoch_step=10000
decay_rate=0.96
grad_clip=args.grad_clip

if args.f_act == "Sigmoid":
    f_act = tf.nn.sigmoid
elif args.f_act == "Relu":
    f_act = tf.nn.relu
elif args.f_act == "Tanh":
    f_act = tf.nn.tanh
elif args.f_act == "Identity":
    f_act = tf.identity
elif args.f_act == "Elu":
    f_act = tf.nn.elu
else:
    raise NotImplementedError("ERROR")

if args.g_act == "Sigmoid":
    g_act = tf.nn.sigmoid
elif args.g_act == "Relu":
    g_act = tf.nn.relu
elif args.g_act == "Tanh":
    g_act = tf.nn.tanh
elif args.g_act == "Identity":
    g_act = tf.identity
elif args.g_act == "Elu":
    g_act = tf.nn.elu
else:
    raise NotImplementedError("ERROR")
    
    
def one_process_CDL(parameterlist,train_info_file,model_name,data_name):
    test_fold=parameterlist[0]
    hidden_neuron=parameterlist[1]   
    #result_path='results/'+str(current_time)+'/'+model_name+'/'+data_name+'/'+str(test_fold)+'/'+str(hidden_neuron)+'/'
    result_path='results/'+model_name+'/'+data_name+'/'

    if data_name=="citeulike-a":
        num_user=5551
        num_item=16980
        num_voca=8000   
        train_file='data/cf-a/cf-a-train-'+str(test_fold)+'-users.dat'
        test_file='data/cf-a/cf-a-test-'+str(test_fold)+'-users.dat'
        item_info='data/cf-a/cf-a-mult.dat'
    elif data_name=="citeulike-t":
        num_user=7947
        num_item=25975
        num_voca=20000   
        train_file='data/cf-t/cf-t-train-'+str(test_fold)+'-users.dat'
        test_file='data/cf-t/cf-t-test-'+str(test_fold)+'-users.dat'
        item_info='data/cf-t/cf-t-mult.dat'
        
        
    train_R,train_user_set,train_item_set,num_train_ratings,train_C=data.read_user(train_file,num_user,num_item,a,b)
    test_R,test_user_set,test_item_set,num_test_ratings,test_C=data.read_user(test_file,num_user,num_item,a,b)
    X_dw=data.read_mult(item_info,num_voca)
    
    train_mask_R=train_R
    test_mask_R=test_R
    
    tf.reset_default_graph()
    config = tf.ConfigProto() 
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        if model_name=="CDL":
            lambda_u=args.lambda_u
            lambda_v=args.lambda_v
            lambda_w=args.lambda_w
            lambda_n=args.lambda_n
            lambda_list=[lambda_u,lambda_v,lambda_w,lambda_n]
            corruption_level=args.corruption_level
            
            layer_structure=[num_voca,200,hidden_neuron,200,num_voca]
            n_layer=len(layer_structure)
            pre_W=dict()
            pre_b=dict()
            
            
            for itr in range(n_layer-1):
                initial_DAE=DAE(layer_structure[itr],layer_structure[itr+1],num_item,num_voca,itr,"sigmoid")
                pre_W[itr],pre_b[itr]=initial_DAE.do_not_pretrain()
            
            model=CDL(sess,num_user,num_item,num_voca,hidden_neuron,current_time,
                      batch_size,lambda_list,layer_structure,train_epoch,
                      pre_W,pre_b,f_act,g_act,corruption_level,keep_prob,
                      num_train_ratings,num_test_ratings,X_dw,train_R,test_R,
                      train_C,train_mask_R,test_mask_R,grad_clip,display_step,a,b,
                      optimizer_method,lr,result_path,decay_rate,decay_epoch_step,args,
                      random_seed,model_name,test_fold,data_name,test_fold)
            #model.run()
            model.run_fin()
    del model
    print("############record#########")
    time.sleep(5)
    train_info_df=pd.read_csv(train_info_file)
    train_info_df.loc[((train_info_df['test_fold']==test_fold) & (train_info_df['hidden_neuron']==hidden_neuron)),'is_train']=1
    train_info_df.to_csv(train_info_file,index=False)
    

def multi_process_CDL(n_process,not_train_list,train_info_file,model_name,data_name):
    pool=Pool(processes=n_process)
    pool.map(partial(one_process_CDL,train_info_file=train_info_file,model_name=model_name,data_name=data_name),not_train_list)
    pool.close()
    

       
if __name__=='__main__':


    #train CDL
    model_name='CDL'
    for data_name in ["citeulike-a","citeulike-t"]:
        train_info_path='results/'+model_name+'/'+data_name+'/'
        if not os.path.exists(train_info_path):
            os.makedirs(train_info_path)
        train_info_file=train_info_path+'train_info.csv'
        if not os.path.exists(train_info_file):
            train_info_df=pd.DataFrame(columns=['test_fold','hidden_neuron','is_train'])
            for test_fold in range(1,test_folds+1):
                for hidden_neuron in range(5, max_hidden_neuron+1,5):
                    train_info_df=train_info_df.append({'test_fold':test_fold,'hidden_neuron':hidden_neuron,'is_train':0},ignore_index=True)
            train_info_df.to_csv(train_info_file,index=False)
        train_info_df=pd.read_csv(train_info_file)
        not_train_info=train_info_df[train_info_df['is_train']==0][['test_fold','hidden_neuron']]
        not_train_list=np.array(not_train_info).tolist()
        multi_process_CDL(n_process,not_train_list,train_info_file,model_name,data_name)











