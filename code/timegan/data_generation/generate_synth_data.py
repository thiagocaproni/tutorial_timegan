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
import params

dataset_directory = '../../../datasets/'

# Function to generate synthetic data based on models created in
# training scripts
def loadSynthData(model32, model64, seq_len):
    synth_32 = TimeGAN.load(model32)
    synth_data_32 = synth_32.sample(seq_len)
    
    synth_64 = TimeGAN.load(model64)
    synth_data_64 = synth_64.sample(seq_len)
    
    synth_data_32[:,:,13:17][synth_data_32[:,:,13:17] >= 0.5] = 1
    synth_data_32[:,:,13:17][synth_data_32[:,:,13:17] < 0.5] = 0
    
    synth_data_64[:,:,13:17][synth_data_64[:,:,13:17] >= 0.5] = 1
    synth_data_64[:,:,13:17][synth_data_64[:,:,13:17] < 0.5] = 0
        
    return synth_data_32, synth_data_64

def loadRealData(dsint32, dsint64, dsdash32, dsdash64, num_cols, cat_cols, sample_size, randon, outliers):
    #loading 32 bit buffer dataset
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
    median = np.median(data)
    percentile_25 = np.percentile(data, 25)
    percentile_75 = np.percentile(data, 75)
    
    return [percentile_25, median, percentile_75]

def genStatisctics(real_32, synth_32, real_64, synth_64, sample_size, num_cols):    
    dict = {}

    for j, col in enumerate(num_cols): 
        dict[col] = [getStatistics(real_32[:,j][:sample_size]), 
                     getStatistics(synth_32[:,j][:sample_size]), 
                     getStatistics(real_64[:,j][:sample_size]), 
                     getStatistics(synth_64[:,j][:sample_size])]
    
    return dict

def createDataSet(lines, seq_len, data):
    dataset = np.zeros(lines*seq_len*16).reshape(lines*seq_len,16)

    for i in range(0,lines):
        for j in range(0, seq_len):
            dataset[(i*seq_len) + j] = data[i][j][:]
            
    return dataset

def roundFeatures(dataset):
    for i in range(0, len(dataset)):
        dataset[i] = np.round(dataset[i])

def getMetrics(data): 
    # [ (Upper_Quartile(Real) - Lower_Quartile(Real)) - (Upper_Quartile(Synth) - Lower_Quartile(Synth) ] + (Median(Real) - Median(Synth))
    
    metric32 = (abs( (data[0][2] - data[0][0]) - (data[1][2] - data[1][0])) +
               abs(  data[0][1] - data[1][1] ) )
    
    metric64 = (abs( (data[2][2] - data[2][0]) - (data[3][2] - data[3][0])) +
               abs(  data[3][1] - data[2][1] ) )

    return metric32, metric64

def get_allfeatures_metrics(num_cols, arr, m, metrics):
    dicts = {}
    for j, col in enumerate(num_cols): 
        arr[0][m][j], arr[1][m][j] = getMetrics(metrics.get(col))
        
    return dicts

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

####################### GENERATION #######################

# Amount of line which will be considered to generate the statistics
# in both datasets: real and synthetic
iMax, jMax, kMax = fatNum(params.amount_of_models)

num_cols = ['enq_qdepth1','deq_timedelta1', 'deq_qdepth1',
            ' enq_qdepth2', ' deq_timedelta2', ' deq_qdepth2',
            'enq_qdepth3', 'deq_timedelta3', 'deq_qdepth3',
            'Buffer', 'ReportedBitrate', 'FPS', 'CalcBitrate',] 
cat_cols = ['Resolution']

# Loading data from real datasets
real_32, real_64 = loadRealData(dsint32= dataset_directory + 'log_INT_TD-32_100.csv',
                                dsint64= dataset_directory + 'log_INT_TD-64_100.csv',
                                dsdash32= dataset_directory + 'dash_TD-32_100.csv',
                                dsdash64= dataset_directory + 'dash_TD-64_100.csv',
                                num_cols=num_cols,
                                cat_cols=cat_cols,
                                sample_size=4000, 
                                randon=False, 
                                outliers=False) 

# Creating scalers in order to return the data to the original scales, once they have been normalized
scaler32 = MinMaxScaler().fit(real_32)
scaler64 = MinMaxScaler().fit(real_64)

models = {}


data = np.zeros(2*params.amount_of_models*len(num_cols)).reshape(2,params.amount_of_models,len(num_cols))

try:
  # Specify an invalid GPU device
  with tf.device('/device:GPU:0'):
    for i in range(0,iMax):
        for j in range(0,jMax):
            for k in range(0,kMax):
                seq_len=(50*(i)+50) 
                
                """
                The object returned when generating synthetic data is a three-dimensional object ([i][j][k]), 
                where the variable i controls the number of windows generated and defined by the variable seq_len 
                (explained in the training script). In turn, the variable j controls the index of the lines in 
                each window and the variable k controls the columns. In other words, a number of time windows are 
                returned. Therefore, you need to define in the loadSynthData function the number of windows you 
                want to generate.
                """
                synth_32_norm, synth_64_norm = loadSynthData(model32= str('../saved_models/so32_seqlen_'+ str((50*(i) + 50)) + '_hidim_' + str(20*(j)+20) + '_batch_' +  str(28*(k) + 100) + '.pkl'), 
                                                model64= str('../saved_models/so64_seqlen_'+ str((50*(i) + 50)) + '_hidim_' + str(20*(j)+20) + '_batch_' +  str(28*(k) + 100) + '.pkl'), 
                                                seq_len=int(params.synth_sample_size/seq_len))                                              
                
                # Normalizing the real data
                real_32_norm =  real_data_loading(real_32, seq_len=seq_len)
                real_64_norm =  real_data_loading(real_64, seq_len=seq_len)
                
                
                # The createDataSet function transforms three-dimensional synthetic data objects by serializing the 
                # windows into a two-dimensional dataset with only rows and columns.
                real_32_norm = createDataSet(int(params.synth_sample_size/seq_len),seq_len, real_32_norm)
                real_64_norm = createDataSet(int(params.synth_sample_size/seq_len),seq_len, real_64_norm)
                synth_32_norm = createDataSet(int(params.synth_sample_size/seq_len),seq_len, synth_32_norm)
                synth_64_norm = createDataSet(int(params.synth_sample_size/seq_len),seq_len, synth_64_norm)
                
                
                # Inverting the normalization to the original scale in order to generate a data set on the same scale 
                # as that collected in the real environment.
                data_synth_32 = scaler32.inverse_transform(synth_32_norm)
                data_synth_64 = scaler64.inverse_transform(synth_64_norm)
                
                # roundFeatures(data_synth_32)
                # roundFeatures(data_synth_64)
                
                
                
                models['so_seqlen_'+str((50*(i) + 50))+'_hidim_'+str(20*(j)+20)+'_batch_'+str(28*(k)+100)+'.pkl'] = [data_synth_32, 
                                                                                                                    data_synth_64]    
                        
                metrics = genStatisctics(real_32_norm, synth_32_norm, real_64_norm, synth_64_norm, params.statistic_sample_size, num_cols)

                print(metrics)
                # Saving the metrics of each feature in object data, creating a general
                # object summarizing the statistics of all models.
                get_allfeatures_metrics(num_cols, data, (i*iMax) + (j*jMax) + (k) , metrics)

    directory_path = '../saved_objects/'
                                
    with open(directory_path + params.realdata_obj, 'wb') as file:
        pickle.dump([real_32, real_64], file)
    
    with open(directory_path + params.models_obj, 'wb') as file:
        pickle.dump(models, file) 
        
    with open(directory_path + params.metrics_obj, 'wb') as file:
        pickle.dump(data, file)

except RuntimeError as e:
  print(e)