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
sys.path.insert(1, '../')
from preprocess_data import DataPre
import params

def loadDp(random, outliers):
    dp = DataPre()
    
    #Loading and mergint INT and DASH datasets
    dp.loadDataSet(path_int='../../../datasets/log_INT_TD-32_100.csv', 
                   path_dash='../../../datasets/dash_TD-32_100.csv')
    
    #defining columns to process
    sorted_cols  = ['enq_qdepth1','deq_timedelta1', 'deq_qdepth1',
                    ' enq_qdepth2', ' deq_timedelta2', ' deq_qdepth2',
                    'enq_qdepth3', 'deq_timedelta3', 'deq_qdepth3',
                    'Buffer', 'ReportedBitrate', 'FPS', 'CalcBitrate'] 
    cat_cols = ['Resolution']
    
    #preprocessing data
    dp.preProcessData(sorted_cols, cat_cols=cat_cols, random=random)
    
    #removing columns with same values
    dp.removeSameValueAttributes()
    
    if outliers == False:
        dp.removeOutliers()
    
    #printing processed data
    dp.processed_data
    
    return dp

def train(dp, seq_len, n_seq, hidden_dim, noise_dim, dim, batch_size, model, train_steps):        
    learning_rate = 5e-4

    gan_args = ModelParameters(batch_size=batch_size,
                              lr=learning_rate,
                              noise_dim=noise_dim,
                              layers_dim=dim)

    #normalizing the data
    processed_data = real_data_loading(dp.processed_data.values, seq_len=seq_len)
    
    synth = TimeGAN(model_parameters=gan_args, hidden_dim=hidden_dim, seq_len=seq_len, n_seq=n_seq, gamma=1)
    
    synth.train(processed_data, train_steps=train_steps)
    synth.save(model)

def fatNum(N):
    # Find the closest possible factor to N^(1/3) to balance i, j and k
    closest_cube_root = round(N ** (1/3))

    # Find the closest factors starting from the approximate cube root
    for i in range(closest_cube_root, 0, -1):
        if N % i == 0:
            # N/i is now the product of j * k
            remaining = N // i
            for j in range(int(remaining ** 0.5), 0, -1):
                if remaining % j == 0:
                    k = remaining // j
                    return i, j, k  # Returns factors as soon as they are found

    return None  # Returns None if factoring is not possible

#Loanding real data from the datasets
dp = loadDp(random=False, outliers=False)

# In the training segment that follows, the quantity of models dictates the maximal values for `i`, `j`, and `k` 
# within the nested triple-loop structure. The `fatNum` function will be employed to compute these maximum values. 
# For instance, if the total number of models created equals 3, then the upper limits for `i`, `j`, and `k` would be 
# respectively set to 1, 1, and 3.
iMax, jMax, kMax = fatNum(params.amount_of_models) # Change the file params.py 

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In this part of the code we actually execute the training of the models by varying the following hyperparameters:
# seq_len: the sequence length would be the size of the temporal window of each sequence used to train the model, 
#          that is, how many time steps (lines) each sequence contains.
# hidden_dim: Number of units or neurons in each hidden layer
# batch_size: The batch size determines how many temporal sequences (or how many data examples/lines) are included in a single batch for training.
# train_steps: Refers to the total number of training iterations
try:
  # Specify an valid GPU device
  with tf.device('/device:GPU:0'):
    for i in range(0,iMax):
      for j in range(0,jMax):
        for k in range(0,kMax):
          train(dp,
            seq_len=(50*(i)+50), 
            n_seq=16, 
            hidden_dim=(20*(j)+20), 
            noise_dim=32, 
            dim=128, 
            batch_size=(28*(k) + 100), 
            model=str('../saved_models/so32_seqlen_'+ str((50*(i) + 50)) + '_hidim_' + str(20*(j)+20) + '_batch_' +  str(28*(k) + 100) + '.pkl'),
            train_steps=params.train_steps)
except RuntimeError as e:
  print(e)