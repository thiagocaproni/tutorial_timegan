import pandas as pd
from sklearn import cluster

import numpy as np
from sklearn.preprocessing import OneHotEncoder

class DataPre:
    def __init__(self):
        pass
    
    
    def transformTimeStamp(self, df):
        df['timestamp'] = df['timestamp'] / 1000
        df['timestamp'] = df['timestamp'].astype(int)
        df.set_index('timestamp', inplace=True)
            
    def loadDataSet(self, path_int, path_dash):
        
        df_int = pd.read_csv(path_int, sep = ',')
        df_dash = pd.read_csv(path_dash, sep = ';')
        
        self.transformTimeStamp(df_dash)
        df_int = df_int.loc[df_int.groupby('timestamp')['deq_timedelta1'].idxmax()]

        #df32_int = df32_int.drop_duplicates(subset=['timestamp'])
        #df64_int = df64_int.drop_duplicates(subset=['timestamp'])
        
        df_int.set_index('timestamp', inplace=True)     
        self.dataset = pd.merge(df_int, df_dash, left_index=True, right_index=True).reset_index()


    def hotEncode(self):
        #creating instance of one-hot-encoder
        self.encoder = OneHotEncoder(handle_unknown='ignore')

        #perform one-hot encoding on 'team' column 
        encoder_df = pd.DataFrame(self.encoder.fit_transform(self.dataset[['Resolution']]).toarray())

        #merge one-hot encoded columns back with original DataFrame
        self.dataset = self.dataset.join(encoder_df).copy()
        self.dataset.drop('Resolution', axis=1, inplace=True)
    
    def inverseHotEncode(self, matrix):
        return self.encoder.inverse_transform(matrix)
    
    
    def showColumns(self):
         for i in self.dataset.columns:
            print(i)
        
    def preProcessData(self, sorted_cols, cat_cols, random): 
        
        #self.dataset.drop('LOG', axis=1, inplace=True)
        #self.dataset.drop('Resolution', axis=1, inplace=True) 
        for i in sorted_cols:
            self.dataset[i].fillna(self.dataset[i].mean(), inplace=True)
        
        if len(cat_cols) > 0:
            self.hotEncode()
            cat_cols = [0,1,2]
        
        self.processed_data = self.dataset[ sorted_cols + cat_cols ].copy()
        self.cat_cols = cat_cols
        self.num_cols = sorted_cols

        if random == True:
            idx = np.random.permutation(self.processed_data.index)
            self.processed_data = self.processed_data.reindex(idx)
        
        
    def removeOutliers(self):
        #for i in self.processed_data.columns:
        for i in self.num_cols:
            q1 = self.processed_data[i].quantile(.25)
            q3 = self.processed_data[i].quantile(.75)
            iqr = q3 - q1
            lim_inf = q1 - (1.5*iqr) 
            lim_sup = q3 + (1.5*iqr)
            for (j, k) in zip(self.processed_data[i].values, self.processed_data.index):
                if ( (j < lim_inf) or (j > lim_sup) ):
                    self.processed_data = self.processed_data.drop(k) 
                    
    def removeSameValueAttributes(self):
        self.processed_data = self.processed_data.drop(columns=self.processed_data.columns[self.processed_data.nunique()==1], inplace=False)

    def clusterData(self, number_clusters):
        #For the purpose of this example we will only synthesize the minority class
        #Create a new class column using KMeans - This will mainly be useful if we want to leverage conditional GAN
        print("Dataset info: Number of records - {} Number of variables - {}".format(self.processed_data.shape[0], self.processed_data.shape[1]))
        algorithm = cluster.KMeans
        args, kwds = (), {'n_clusters':number_clusters, 'random_state':0}
        labels = algorithm(*args, **kwds).fit_predict(self.processed_data[ self.num_cols ])
        
        self.cluster = self.processed_data.copy()
        self.cluster['Class'] = labels