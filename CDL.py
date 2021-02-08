# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 21:24:29 2018

@author: Xiaoqing Ye
"""



import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import time
import collections
from utils import make_records,SDAE_calculate,variable_save
from tensorflow.contrib.layers import batch_norm

class CDL():
    def __init__(self, sess, num_user, num_item, num_voca, hidden_neuron,current_time,
               batch_size, lambda_list, layer_structure, train_epoch,
               pre_W, pre_b, f_act,g_act,
               cdl_corruption_level, cdl_keep_prob,
               num_train_ratings, num_test_ratings,
               item_data,train_R, test_R, C,
               train_mask_R, test_mask_R,
               grad_clip, display_step, a, b,
               cdl_optimizer, cdl_learning_rate,
                 result_path,
               decay_rate, decay_epoch_step, args,random_seed,model_name,train_ratio,data_name,test_fold):
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.num_voca = num_voca
        self.hidden_neuron = hidden_neuron
        self.batch_size = batch_size
      
        self.current_time = current_time

        self.lambda_u = lambda_list[0]
        self.lambda_w = lambda_list[1]
        self.lambda_v = lambda_list[2]
        self.lambda_n = lambda_list[3]

        self.layer_structure = layer_structure
        self.train_epoch = train_epoch
        self.Weight = pre_W
        self.bias = pre_b
        self.do_batch_norm = False

        self.f_act = f_act
        self.g_act = g_act

        self.cdl_corruption_level = cdl_corruption_level
        self.cdl_keep_prob = cdl_keep_prob

        self.num_train_ratings = num_train_ratings
        self.num_test_ratings = num_test_ratings

        self.item_data_dw = item_data
        
        self.train_R = train_R
        self.test_R = test_R
        self.C = C

        self.train_mask_R = train_mask_R
        self.test_mask_R = test_mask_R

        self.a = a
        self.b = b
        self.optimizer_method = cdl_optimizer

        self.grad_clip = grad_clip
        self.display_step = display_step

        self.cdl_optimizer = cdl_optimizer
        # self.cdl_learning_rate = cdl_learning_rate

        self.result_path = result_path
        self.args = args

        self.train_cost_list = []
        self.test_cost_list = []
        self.test_rmse_list = []
        self.test_mae_list = []
        self.test_acc_list = []
        self.test_avg_loglike_list = []
        self.test_recall_list=[]
        self.test_precision_list=[]
        self.test_map_list=[]
        self.test_ndcg_one_list=[]
        self.test_ndcg_two_list=[]
        self.test_converage_list=[]
        self.test_diversity_list=[]

        self.test_recall_dict = collections.OrderedDict()
        self.test_map_dict = collections.OrderedDict()

        self.decay_rate = decay_rate
        self.decay_epoch_step = decay_epoch_step

        self.step = tf.Variable(0, trainable=False)

        self.decay_step = self.decay_epoch_step * int(self.num_item / self.batch_size)

        self.lr = tf.train.exponential_decay(
            cdl_learning_rate, self.step, self.decay_step, self.decay_rate, staircase=True, name="lr"
        )

        self.train_var_list1 = [] # U , V
        self.train_var_list2 = [] # W , b

        self.random_seed = random_seed
        self.model_name = model_name
        self.data_name = data_name
        self.train_ratio = train_ratio
        self.test_fold=test_fold
        self.mask_corruption_np = np.random.binomial(1, 1-self.cdl_corruption_level,
                                                (self.num_item, self.num_voca))

        self.earlystop_switch = False
        self.min_RMSE = 99999
        self.min_epoch = -99999
        self.patience = 0
        self.total_patience = 20

    def __del__(self):
        class_name=self.__class__.__name__
        print(class_name,"del")
        
    def prepare_model(self):
        self.model_mask_corruption = tf.placeholder(dtype=tf.float32, shape=[None, self.num_voca], name="model_mask_corruption")
        self.model_X = tf.placeholder(dtype=tf.float32, shape=[None, self.num_voca], name="model_X")
        self.model_input_R = tf.placeholder(dtype=tf.float32, shape=[self.num_user,None], name="model_input_R")
        # self.model_input_mask_R = tf.placeholder(dtype=tf.float32, shape=[self.num_user, None], name="model_input_mask_R")
        self.model_C = tf.placeholder(dtype=tf.float32, shape=[self.num_user,None], name="model_C")

        self.model_num_voting = tf.placeholder(dtype=tf.float32)
        self.model_keep_prob = tf.placeholder(dtype=tf.float32)
        self.model_batch_data_idx = tf.placeholder(dtype=tf.int32)

        X_corrupted = tf.multiply(self.model_mask_corruption, self.model_X)
        real_batch_size = tf.cast(tf.shape(self.model_X)[0], tf.int32)
        # X_c, layer_structure, W, b, batch_normalization, post_activation_bn, activation, model_keep_prob
        Encoded_X, sdae_output = SDAE_calculate(self.model_name,X_corrupted, self.layer_structure, self.Weight, self.bias,
                                               self.do_batch_norm, self.f_act,self.g_act, self.model_keep_prob)

        with tf.variable_scope("CDL_Variable"):
            self.v_jk = tf.get_variable(name="item_factor", shape=[self.num_item, self.hidden_neuron], dtype=tf.float32)
            self.u_ik = tf.get_variable(name="user_factor", shape=[self.num_user, self.hidden_neuron], dtype=tf.float32)
        batch_v_jk = tf.reshape(tf.gather(self.v_jk, self.model_batch_data_idx), shape=[real_batch_size, self.hidden_neuron])

        tmp_likelihood1 = tf.constant(0, dtype=tf.float32)
        for itr in range(len(self.Weight.keys())):
            tmp_likelihood1 = tf.add(tmp_likelihood1,
                                     tf.add(tf.nn.l2_loss(self.Weight[itr]), tf.nn.l2_loss(self.bias[itr])))

        loss_1 = self.lambda_u * tf.nn.l2_loss(self.u_ik) + self.lambda_w * tmp_likelihood1
        loss_2 = self.lambda_v * tf.nn.l2_loss(batch_v_jk - Encoded_X)
        loss_3 = self.lambda_n * tf.nn.l2_loss(sdae_output - self.model_X)
        loss_4 = tf.reduce_sum(tf.multiply(self.model_C,
                                      tf.square(self.model_input_R - tf.matmul(self.u_ik, batch_v_jk, transpose_b=True))))

        self.cost = loss_1 + loss_2 + loss_3 + loss_4

        for var in tf.trainable_variables():
            if ("CDL_Variable" in var.name):
                self.train_var_list1.append(var)
            elif ("SDAE_Variable" in var.name):
                self.train_var_list2.append(var)

        if self.optimizer_method == "Adam":
            optimizer1 = tf.train.AdamOptimizer(self.lr)
            optimizer2 = tf.train.AdamOptimizer(self.lr)
        elif self.optimizer_method == "Momentum":
            optimizer1 = tf.train.MomentumOptimizer(self.lr,0.9)
            optimizer2 = tf.train.MomentumOptimizer(self.lr,0.9)
        elif self.optimizer_method == "Adadelta":
            optimizer1 = tf.train.AdadeltaOptimizer()
            optimizer2 = tf.train.AdadeltaOptimizer()
        elif self.optimizer_method == "Adagrad":
            optimizer1 = tf.train.AdagradOptimizer(self.lr)
            optimizer2 = tf.train.AdagradOptimizer(self.lr)

        gvs = optimizer1.compute_gradients(self.cost, var_list=self.train_var_list1)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        self.optimizer1 = optimizer1.apply_gradients(capped_gvs, global_step=self.step)
        gvs = optimizer2.compute_gradients(self.cost, var_list=self.train_var_list2)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        self.optimizer2 = optimizer2.apply_gradients(capped_gvs, global_step=self.step)

    def run(self):
        self.max_R = -1.0
        self.prepare_model()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch_itr in range(self.train_epoch):
            if self.earlystop_switch:
                break
            else:
                self.train_model(epoch_itr)
                self.test_model(epoch_itr)
    
        make_records(self.result_path,self.test_acc_list,self.test_rmse_list,self.test_mae_list,self.test_avg_loglike_list,self.test_recall_list,
                     self.test_precision_list,self.test_map_list,self.test_ndcg_one_list,self.test_ndcg_two_list,self.test_converage_list,self.test_diversity_list,
                     self.current_time,self.args,self.model_name,self.data_name,self.train_ratio,self.hidden_neuron,self.random_seed,self.optimizer_method,self.lr)
        variable_save(self.result_path, self.model_name,self.train_var_list1, self.train_var_list2, self.Estimated_R, self.test_R,
                      self.test_mask_R,self.hidden_neuron)
        
        
    def run_fin(self):
        self.max_R = -1.0
        self.prepare_model()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch_itr in range(self.train_epoch):
            if self.earlystop_switch:
                break
            else:
                self.train_model(epoch_itr)
                #self.test_model(epoch_itr)
                print (self.hidden_neuron, epoch_itr)
        self.U = self.u_ik.eval()
        self.V = self.v_jk.eval()

        '''
        calculate Encoder X
        '''
        mask_corruption_value= np.multiply(self.mask_corruption_np,self.item_data_dw)
        mask_corruption_X=tf.placeholder(dtype=tf.float32,shape=[self.num_item,self.num_voca])
        hidden_value = mask_corruption_X
        for itr1 in range(len(self.layer_structure) - 1):
            ''' Encoder '''
            if itr1 <= int(len(self.layer_structure) / 2) - 1:
                if (itr1 == 0) and (self.model_name == "CDAE"):
                    ''' V_u '''
                    before_activation = tf.add(tf.add(tf.matmul(hidden_value,self.Weight[itr1]), self.V_jk), self.bias[itr1])
                else:
                    before_activation = tf.add(tf.matmul(hidden_value,self.Weight[itr1]), self.bias[itr1])

                if self.do_batch_norm == "True":
                    before_activation = batch_norm(before_activation)
                hidden_value = self.f_act(before_activation)
        Encoded_X=self.sess.run(hidden_value,feed_dict={mask_corruption_X:mask_corruption_value})


        variable_save(self.result_path, self.model_name,self.train_var_list1, self.train_var_list2,self.U,self.V,Encoded_X,self.test_R,
                      self.test_mask_R,self.hidden_neuron,self.test_fold)
        
    def train_model(self, epoch_itr):
        start_time = time.time()
        total_batch = int(self.num_item /float(self.batch_size)) + 1
        '''
        mask_corruption_np = np.random.binomial(1, 1-self.cdl_corruption_level,
                                                (self.num_item, self.num_voca))
        '''
        random_perm_doc_idx = np.random.permutation(self.num_item)
        batch_cost = 0

        for i in range(total_batch):
            if i == total_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size:]
            elif i < total_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size : (i+1) * self.batch_size]

            _, Cost = self.sess.run(
                [self.optimizer1, self.cost],
                feed_dict={self.model_mask_corruption: self.mask_corruption_np[batch_set_idx,:],
                           self.model_X: self.item_data_dw[batch_set_idx,:],
                           self.model_input_R: self.train_R[:, batch_set_idx],
                           self.model_C: self.C[:, batch_set_idx],
                           self.model_num_voting: self.num_train_ratings,
                           self.model_keep_prob: self.cdl_keep_prob,
                           self.model_batch_data_idx: batch_set_idx}
            )

            _, Cost = self.sess.run(
                [self.optimizer2, self.cost],
                feed_dict={self.model_mask_corruption: self.mask_corruption_np[batch_set_idx,:],
                           self.model_X: self.item_data_dw[batch_set_idx,:],
                           self.model_input_R: self.train_R[:, batch_set_idx],
                           self.model_C: self.C[:, batch_set_idx],
                           self.model_num_voting: self.num_train_ratings,
                           self.model_keep_prob: self.cdl_keep_prob,
                           self.model_batch_data_idx: batch_set_idx}
            )

            batch_cost = batch_cost + Cost

        self.train_cost_list.append(batch_cost)
        if epoch_itr % self.display_step == 0:
            print ("Training //", "Epoch %d //" % (epoch_itr), " Total cost = {:.2f}".format(batch_cost),
                   "Elapsed time : %d sec" % (time.time() - start_time))

   
