from sklearn.preprocessing import MinMaxScaler
from ydata_synthetic.synthesizers.timeseries import TimeGAN
from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading
import numpy as np
import tensorflow as tf
import sys
import pickle
sys.path.insert(0, '../../data_process/')
sys.path.insert(1, '../')
from preprocess_data import DataPre
from model_utility import ModelUtility
import params

dataset_directory = '../../../datasets/'

def loadSynthData(model32, model64, number_of_windows):
    '''
    Method to generate synthetic data based on models trained in the timegan32 to timegan64 scripts.
    The object returned when generating synthetic data is a three-dimensional object ([i][j][k]), 
    where the variable i controls the number of windows generated and defined by the variable seq_len 
    (explained in the training script). In turn, the variable j controls the index of the lines in 
    each window and the variable k controls the columns. In other words, a number of time windows are 
    returned. Therefore, you need to define in the loadSynthData function the number of windows you 
    want to generate.
    
    Args:
        model32 (string): Name of the TimeGAN model trained on 32-bit buffer size data
        model64 (string): Name of the TimeGAN model trained on 64-bit buffer size data
        seq_len (int):  the number of windows you want to generate. Remember that each 
                        window has the size defined in the seq_len variable of each model.
    
    Returns:
        Returns numpy arrays for 32-bit and 64-bit buffer sizes

    '''
    synth_32 = TimeGAN.load(model32)
    synth_data_32 = synth_32.sample(number_of_windows)
    
    synth_64 = TimeGAN.load(model64)
    synth_data_64 = synth_64.sample(number_of_windows)
    
    synth_data_32[:,:,13:17][synth_data_32[:,:,13:17] >= 0.5] = 1
    synth_data_32[:,:,13:17][synth_data_32[:,:,13:17] < 0.5] = 0
    synth_data_64[:,:,13:17][synth_data_64[:,:,13:17] >= 0.5] = 1
    synth_data_64[:,:,13:17][synth_data_64[:,:,13:17] < 0.5] = 0
        
    return synth_data_32, synth_data_64

def loadRealData(dsint32, dsint64, dsdash32, dsdash64, num_cols, cat_cols, sample_size, randon, outliers):
    '''
    Method to load the real datasets to be able to use them to calculate statistical metrics. 
    
    Args:
        dsint32 (string): path to INT dataset for 32-bit buffer size   
        dsint64 (string): path to INT dataset for 64-bit buffer size
        dsdash32 (string): path to DASH dataset for 32-bit buffer size
        dsdash64 (string): path to DASH dataset for 64-bit buffer size
        num_cols (list): numercial features that will be read
        cat_cols (list): categorical features that will be read
        sample_size (int): set the number of rows that will be read in the real dataset
        random (boolean): True to randomize the dataset
        outliers (boolean): False to remove the outliers
    
    Returns:
        Returns numpy arrays for 32-bit and 64-bit buffer sizes.

    '''
    dp32 = DataPre()
    dp32.loadDataSet(path_int=dsint32, path_dash=dsdash32)
    dp32.preProcessData(num_cols, cat_cols=cat_cols, random=randon)
    if outliers == False:
        dp32.removeOutliers()
    
    real_data_32 = dp32.processed_data
    real_data_32 = real_data_32[0:sample_size].copy()
    real_data_32 = real_data_32.values
    
    #loading 64 bit buffer dataset
    dp64 = DataPre()
    dp64.loadDataSet(path_int=dsint64, path_dash=dsdash64)
    dp64.preProcessData(num_cols, cat_cols=cat_cols, random=randon)
    if outliers == False:
        dp64.removeOutliers()
    
    real_data_64 = dp64.processed_data
    real_data_64 = real_data_64[0:sample_size].copy()
    real_data_64 = real_data_64.values
    
    return real_data_32, real_data_64

def getStatistics(data):
    '''
    Method that calculates the quartiles and median for a given feature
    
    Args:
        data (numpy array): array of values of a given feature
    
    Returns:
        Returns a list containing percentiles and medians. Positions 0, 1, 2 and 3 store the 1st percentile,
           median and 3rd percentile, respectively

    '''
    median = np.median(data)
    percentile_25 = np.percentile(data, 25)
    percentile_75 = np.percentile(data, 75)
    
    return [percentile_25, median, percentile_75]

def genStatisctics(real_32, synth_32, real_64, synth_64, sample_size, num_cols): 
    '''
    Method that generates statistical data for each of the columns of a given set of real and synthetic model.
    
    Args:
        real_32 (numpy array): contains the real values for 32-bit buffer size  
        synth_32 (numpy array): contains the synthetic values for 32-bit buffer size  
        real_64 (numpy array): contains the real values for 64-bit buffer size  
        synth_64 (numpy array): contains the synthetic values for 64-bit buffer size  
        sample_size (int): define the number of rows in the synthetic dataset
        num_cols (list): List of numeric columns (features) of the dataset
    
    Returns:
        Returns a dictionary for a given set of real and synthetic models,
        comprising lists of column statistics (features). Each list contains
        statistical values in positions 0, 1, 2 and 3, corresponding to real data from 32
        bits real, 32-bit synthetic, 64-bit real, and 64-bit synthetic respectively,
        presenting percentiles and medians.

    '''
    
    dict = {}

    for j, col in enumerate(num_cols): 
        dict[col] = [getStatistics(real_32[:,j][:sample_size]), 
                     getStatistics(synth_32[:,j][:sample_size]), 
                     getStatistics(real_64[:,j][:sample_size]), 
                     getStatistics(synth_64[:,j][:sample_size])]
    
    return dict

def createDataSet(seq_len, data):
    '''
    Method that converts three-dimensional synthetic data objects into a two-dimensional dataset by 
    flattening the windows into rows and columns
    
    Args:
        seq_len (int): the sequence length would be the size of the temporal window of each sequence used to train the model, 
                       that is, how many time steps (lines) each sequence contains.
        data (numpy array): three-dimensional object containing the windows of temporal data
    
    Returns:
        Returns a bidimensional numpy array 

    '''
    lines =  int(params.synth_sample_size/seq_len)
    dataset = np.zeros(lines * seq_len * params.merged_columns_len).reshape(lines*seq_len, params.merged_columns_len)
    lines

    for i in range(0,lines):
        for j in range(0, seq_len):
            dataset[(i*seq_len) + j] = data[i][j][:]
            
    return dataset

def getMetrics(statistic_data):
    '''
    Method called by the get_alfeatures_metrics method that effectively performs the calculation of equation 1 defined in the tutorial text.
    
    Args:
        data (list): list containing statistical data for a given feature (column) of the dataset
    
    Returns:
        Returns the two metrics (for 32 and 64-bit buffer size) values of calculated metric

    '''
    metric32 = (abs( (statistic_data[0][2] - statistic_data[0][0]) - (statistic_data[1][2] - statistic_data[1][0])) +
               abs(  statistic_data[0][1] - statistic_data[1][1] ) )
    
    metric64 = (abs( (statistic_data[2][2] - statistic_data[2][0]) - (statistic_data[3][2] - statistic_data[3][0])) +
               abs(  statistic_data[3][1] - statistic_data[2][1] ) )

    return metric32, metric64

def get_allfeatures_metrics(metrics, model_index, statistic_data):
    '''
    Generates metrics based on equation 1 showed in our tutorial text document, which is derived 
    from the discrepancy in the interquartile ranges between each feature of the real and synthetic data. 
    Additionally, the variance in the medians of these features is incorporated into this computation. 
    Consequently, an object named 'metrics' is instantiated, encapsulating the results of this equation 
    across all models and their corresponding features. Subsequently, in the evaluation script, these metrics 
    are utilized to determine the optimal model. It should be emphasized that the total of the metrics for all 
    features of a given model, denoted by the variable M in the equation, is calculated in the 
    analyze_data_models.ipynb script in the method getFeaturesBestMetricsOfModels. This method 
    identifies the best and worst models from the trained models."
    
    Args:
        statistic_data (list): list containing statistical data for each feature (column) of the model 
        model_indice (int): model index where the metrics calculated for the model's buffer size will be stored 
    
    Returns:
        None

    ''' 
    for j, col in enumerate(params.num_cols): 
        metrics[0][model_index][j], metrics[1][model_index][j] = getMetrics(statistic_data.get(col))
        
        

####################### GENERATION #######################

iMax, jMax, kMax = ModelUtility.fatNum(params.amount_of_models)


# Loading data from real datasets
real_32, real_64 = loadRealData(dsint32= dataset_directory + 'log_INT_TD-32_100.csv',
                                dsint64= dataset_directory + 'log_INT_TD-64_100.csv',
                                dsdash32= dataset_directory + 'dash_TD-32_100.csv',
                                dsdash64= dataset_directory + 'dash_TD-64_100.csv',
                                num_cols=params.num_cols,
                                cat_cols=params.cat_cols,
                                sample_size=4000, 
                                randon=False, 
                                outliers=False) 


# Creating scalers in order to return the data to the original scales, once they have been normalized
scaler32 = MinMaxScaler().fit(real_32)
scaler64 = MinMaxScaler().fit(real_64)

'''
Creates a dictionary to store the synthetic data generated from each of the models
Model dictionary where the key is the name of the trained model file and content is 
a list of two numpy arrays, where the first element (index 0) stores the 32-bit model 
data and the second element (index 1) stores the data of the 4-bit model.
'''
models = {}

'''
Numpy array to store statistics for each column across different models and buffer sizes 
(32 and 64-bit). This array is three-dimensional, indexed by [i][j][k], where 'i' indexes 
the buffer sizes (32 and 64-bit sizes), 'j' indexes the models, and 'k' indexes the columns 
within each model.

'''
metrics = np.zeros(2*params.amount_of_models * len(params.num_cols)).reshape(2, params.amount_of_models, len(params.num_cols))

try:
  # Specify an invalid GPU device
  with tf.device('/device:GPU:0'):
    for i in range(0,iMax):
        for j in range(0,jMax):
            for k in range(0,kMax):
                seq_len=(50*(i)+50) 
                
                synth_32_norm, synth_64_norm = loadSynthData(model32 = str('../saved_models/so32_seqlen_'
                                                                          + str((50*(i) + 50)) 
                                                                          + '_hidim_' + str(20*(j)+20) 
                                                                          + '_batch_' + str(28*(k) + 100) 
                                                                          + '.pkl'), 
                                                            model64 = str('../saved_models/so64_seqlen_'
                                                                          + str((50*(i) + 50)) 
                                                                          + '_hidim_' + str(20*(j)+20) 
                                                                          + '_batch_' + str(28*(k) + 100) 
                                                                          + '.pkl'), 
                                                            number_of_windows=int(params.synth_sample_size/seq_len))                                              
                
                
                print('\n\n DATA TYPE' + str(type(synth_32_norm)) + '\n\n')
                # Normalizing the real data
                real_32_norm =  real_data_loading(real_32, seq_len)
                real_64_norm =  real_data_loading(real_64, seq_len)
                
                
                # The createDataSet function transforms three-dimensional synthetic data objects by serializing the 
                # windows into a two-dimensional dataset with only rows and columns.
                real_32_norm = createDataSet(seq_len, real_32_norm)
                real_64_norm = createDataSet(seq_len, real_64_norm)
                synth_32_norm = createDataSet(seq_len, synth_32_norm)
                synth_64_norm = createDataSet(seq_len, synth_64_norm)
                
                
                # Reverting the normalization to the original scale is performed to produce a dataset that aligns 
                # with the scale of data collected in the actual environment
                data_synth_32 = scaler32.inverse_transform(synth_32_norm)
                data_synth_64 = scaler64.inverse_transform(synth_64_norm)
                
                # Call roundFeatures method if you want to round the dataset values
                ModelUtility.roundFeatures(data_synth_32)
                ModelUtility.roundFeatures(data_synth_64)
                       
                models['so_seqlen_'+str((50*(i) + 50))+'_hidim_'+str(20*(j)+20)+'_batch_'+str(28*(k)+100)+'.pkl'] = [data_synth_32, 
                                                                                                                    data_synth_64]  
                statistic_data = genStatisctics(real_32_norm, synth_32_norm, 
                                                real_64_norm, synth_64_norm, 
                                                params.statistic_sample_size, params.num_cols)

                print(statistic_data)
                # Saving the metrics of each feature in object data, creating a general
                # object summarizing the statistics of all models.
                get_allfeatures_metrics(metrics, (i*iMax) + (j*jMax) + (k) , statistic_data)

    directory_path = '../saved_objects/'
                                
    with open(directory_path + params.realdata_obj, 'wb') as file:
        pickle.dump([real_32, real_64], file)
    
    with open(directory_path + params.models_obj, 'wb') as file:
        pickle.dump(models, file) 
        
    with open(directory_path + params.metrics_obj, 'wb') as file:
        pickle.dump(metrics, file)

except RuntimeError as e:
  print(e)