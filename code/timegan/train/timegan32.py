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
from model_utility import ModelUtility
import params


def loadDp(random, outliers):
    """
      Load the datasets and merge the INT and DASH data

      Args:
          random (boolean): True to randomize the dataset
          outliers (boolean): False to remove the outliers
          
      Returns:
          dp: Dataset with INT and DASH data loaded
    """
    dp = DataPre()
    
    #Loading and mergint INT and DASH datasets
    dp.loadDataSet(path_int='../../../datasets/log_INT_TD-32_100.csv', 
                   path_dash='../../../datasets/dash_TD-32_100.csv')
    
 
    #preprocessing data
    dp.preProcessData(params.num_cols, cat_cols=params.cat_cols, random=random)
    
    #removing columns with same values
    dp.removeSameValueAttributes()
    
    if outliers == False:
        dp.removeOutliers()
    
    #printing processed data
    dp.processed_data
    
    return dp

def train(dp, seq_len, n_seq, hidden_dim, noise_dim, dim, batch_size, model, train_steps):
    """
      Method that implements TimeGAN training

      Args:
          dp: dataset to be used in the training 
          seq_len (int): the sequence length would be the size of the temporal window of each sequence used to train the model, 
                         that is, how many time steps (lines) each sequence contains. 
          n_seq (int): amount of columns in the dataset (features)
          hidden_dim (int): Number of units or neurons in each hidden layer
          noise_dim (int): refers to the size or dimensionality of the random noise input that is fed into the generator network. 
                     TimeGAN is designed to generate realistic time-series data, and the noise vector serves as a source of 
                     randomness that helps the generator create diverse and plausible time-series samples.
          dim (int): unused
          batch_size (int): The batch size determines how many temporal sequences (or how many data examples/lines) are included in a single batch for training. 
          model: model name to be saved after training
          train_steps: Refers to the total number of training iterations
          
      Returns:
          There are no return objects since models are saved in the saved folder
    """        
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


#Loanding real data from the datasets
dp = loadDp(random=False, outliers=False)

# In the training segment that follows, the quantity of models dictates the maximal values for `i`, `j`, and `k` 
# within the nested triple-loop structure. The `fatNum` function will be employed to compute these maximum values. 
# For instance, if the total number of models created equals 3, then the upper limits for `i`, `j`, and `k` would be 
# respectively set to 1, 1, and 3.
iMax, jMax, kMax = ModelUtility.fatNum(params.amount_of_models) # Change the file params.py 
print("\nNumber of models" + str(params.amount_of_models) + ' iMax: ' + str(iMax) + ' jMax: ' + str(jMax) + ' kMax: ' + str(kMax))

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
            n_seq=params.merged_columns_len, 
            hidden_dim=(20*(j)+20), 
            noise_dim=32, 
            dim=128, 
            batch_size=(28*(k) + 100), 
            model=str('../saved_models/so32_seqlen_'+ str((50*(i) + 50)) + '_hidim_' + str(20*(j)+20) + '_batch_' +  str(28*(k) + 100) + '.pkl'),
            train_steps=params.train_steps)
except RuntimeError as e:
  print(e)