#!/usr/bin/env python
# coding: utf-8

# In[7]:


from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.synthesizers.timeseries import TimeGAN
from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading
import tensorflow as tf

import sys
sys.path.insert(0, '../../data_process')
from my_data_pre_intdash import DataPre


# In[8]:


def loadDp(random):
    dp = DataPre()
    dp.loadDataSet(path32_int='../../../datasets/log_INT_TD-32_100.csv', 
                   path64_int='../../../datasets/log_INT_TD-64_100.csv', 
                   path32_dash='../../../datasets/dash_TD-32_100.csv', 
                   path64_dash='../../../datasets/dash_TD-64_100.csv')
    
    sorted_cols  = ['enq_qdepth1','deq_timedelta1', 'deq_qdepth1',
                    ' enq_qdepth2', ' deq_timedelta2', ' deq_qdepth2',
                    'enq_qdepth3', 'deq_timedelta3', 'deq_qdepth3',
                    'Buffer', 'ReportedBitrate', 'FPS', 'CalcBitrate',
                    'q_size'] 
    cat_cols = ['Resolution']
    
    dp.preProcessData(sorted_cols, cat_cols=cat_cols, cond_col='q_size', random=random)
    dp.removeAtributoscomMesmoValor
    
    return dp


# In[9]:


def train(dp, seq_len, n_seq, hidden_dim, noise_dim, dim, batch_size, clas, model):        
    log_step = 100
    learning_rate = 5e-4

    gan_args = ModelParameters(batch_size=batch_size,
                            lr=learning_rate,
                            noise_dim=noise_dim,
                            layers_dim=dim)

    dp.processed_data = dp.processed_data.loc[dp.processed_data['q_size'] == clas].copy()
    processed_data = real_data_loading(dp.processed_data.values, seq_len=seq_len)
    
    synth = TimeGAN(model_parameters=gan_args, hidden_dim=hidden_dim, seq_len=seq_len, n_seq=n_seq, gamma=1)
    synth.train(processed_data, train_steps=3000)
    synth.save(model)


# In[10]:


dp = loadDp(random=False)
dp.processed_data


# In[11]:


dp.removeOutliers()
dp.processed_data


# In[14]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#tf.debugging.set_log_device_placement(True)

try:
  # Specify an invalid GPU device
  with tf.device('/device:GPU:1'):
    for i in range(0,1):
      for j in range(0,3):
        for k in range(0,3):
          train(dp,
            seq_len=(50*(i)+50), 
            n_seq=17, 
            hidden_dim=(20*(j)+20), 
            noise_dim=32, 
            dim=128, 
            batch_size=(28*(k) + 100),  
            clas=1, 
            model=str('../saved_models/so64_seqlen_'+ str((50*(i) + 50)) + '_hidim_' + str(20*(j)+20) + '_batch_' +  str(28*(k) + 100) + '.pkl'))
except RuntimeError as e:
  print(e)

