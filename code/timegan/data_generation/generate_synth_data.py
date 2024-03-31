from sklearn.preprocessing import MinMaxScaler
from ydata_synthetic.synthesizers.timeseries import TimeGAN
from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading
import numpy as np
import tensorflow as tf
import sys
import pickle
sys.path.insert(0, '../../data_process/')
from my_data_pre_intdash import DataPre

dataset_directory = '../../datasets/'

def loadSynthData(model32, model64, seq_len):
    synth_32 = TimeGAN.load(model32)
    synth_data_32 = synth_32.sample(seq_len)
    
    synth_64 = TimeGAN.load(model64)
    synth_data_64 = synth_64.sample(seq_len)
    
    synth_data_32[:,:,14:18][synth_data_32[:,:,14:17] >= 0.5] = 1
    synth_data_32[:,:,14:18][synth_data_32[:,:,14:17] < 0.5] = 0
    
    synth_data_64[:,:,14:18][synth_data_64[:,:,14:17] >= 0.5] = 1
    synth_data_64[:,:,14:18][synth_data_64[:,:,14:17] < 0.5] = 0
        
    return synth_data_32, synth_data_64

def loadRealData(dsint32, dsint64, dsdash32, dsdash64, num_cols, cat_cols, sample_size, randon, outliers):
    dp = DataPre()
    dp.loadDataSet(path32_int=dsint32, path64_int=dsint64, path32_dash=dsdash32, path64_dash=dsdash64)
    
    dp.preProcessData(num_cols, cat_cols=cat_cols, cond_col='q_size', random=randon)
    if outliers == False:
        dp.removeOutliers()
    
    real_data_32 = dp.processed_data.loc[dp.processed_data['q_size'] == 0].copy()
    real_data_32 = real_data_32[0:sample_size].copy()
    real_data_32 = real_data_32.values
    
    real_data_64 = dp.processed_data.loc[dp.processed_data['q_size'] == 1].copy()
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
    dataset = np.zeros(lines*seq_len*17).reshape(lines*seq_len,17)

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

####################### GENERATION #######################

sample_size = 2400

num_cols = ['enq_qdepth1','deq_timedelta1', 'deq_qdepth1',
            ' enq_qdepth2', ' deq_timedelta2', ' deq_qdepth2',
            'enq_qdepth3', 'deq_timedelta3', 'deq_qdepth3',
            'Buffer', 'ReportedBitrate', 'FPS', 'CalcBitrate',
            'q_size'] 
cat_cols = ['Resolution']

real_32, real_64 = loadRealData(dsint32= dataset_directory + 'log_INT_TD-32_100.txt',
                                dsint64= dataset_directory + 'log_INT_TD-64_100.txt',
                                dsdash32= dataset_directory + 'dash_TD-32_100.txt',
                                dsdash64= dataset_directory + 'dash_TD-64_100.txt',
                                num_cols=num_cols,
                                cat_cols=cat_cols,
                                sample_size=4000, 
                                randon=False, 
                                outliers=False) 

scaler32 = MinMaxScaler().fit(real_32)
scaler64 = MinMaxScaler().fit(real_64)

models = {}

size = 3600
data = np.zeros(2*27*len(num_cols)).reshape(2,27,len(num_cols))

try:
  # Specify an invalid GPU device
  with tf.device('/device:GPU:0'):
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                seq_len=(50*(i)+50) 
                synth_32_norm, synth_64_norm = loadSynthData(model32= str('../models_dash_int/so32_seqlen_'+ str((50*(i) + 50)) + '_hidim_' + str(20*(j)+20) + '_batch_' +  str(28*(k) + 100) + '.pkl'), 
                                                model64= str('../models_dash_int/so64_seqlen_'+ str((50*(i) + 50)) + '_hidim_' + str(20*(j)+20) + '_batch_' +  str(28*(k) + 100) + '.pkl'), 
                                                seq_len=int(size/seq_len))                                              
                
                real_32_norm =  real_data_loading(real_32, seq_len=seq_len)
                real_64_norm =  real_data_loading(real_64, seq_len=seq_len)
                
                
                real_32_norm = createDataSet(int(size/seq_len),seq_len, real_32_norm)
                real_64_norm = createDataSet(int(size/seq_len),seq_len, real_64_norm)
                
                synth_32_norm = createDataSet(int(size/seq_len),seq_len, synth_32_norm)
                synth_64_norm = createDataSet(int(size/seq_len),seq_len, synth_64_norm)
                
                data_synth_32 = scaler32.inverse_transform(synth_32_norm)
                data_synth_64 = scaler64.inverse_transform(synth_64_norm)
                
                roundFeatures(data_synth_32)
                roundFeatures(data_synth_64)
                    
                models['so_seqlen_'+str((50*(i) + 50))+'_hidim_'+str(20*(j)+20)+'_batch_'+str(28*(k)+100)+'.pkl'] = [data_synth_32, 
                                                                                                                    data_synth_64]            
                metrics = genStatisctics(real_32_norm, synth_32_norm, real_64_norm, synth_64_norm, sample_size, num_cols)

                print(metrics)
                get_allfeatures_metrics(num_cols, data, (i*9) + (j*3) + (k) , metrics)

    directory_path = '../saved_objects/'
    
    with open(directory_path + 'real9_50_3600_norm_27.pkl', 'wb') as file:
        pickle.dump([real_32, real_64], file)
    
    with open(directory_path + 'models9_50_3600_norm_27.pkl', 'wb') as file:
        pickle.dump(models, file) 
        
    with open(directory_path + 'metrics9_50_3600_norm_27.pkl', 'wb') as file:
        pickle.dump(data, file)

except RuntimeError as e:
  print(e)