import pandas as pd
from sklearn import cluster

import numpy as np
from sklearn.preprocessing import OneHotEncoder

class DataPre:
    def __init__(self):
        pass
    
    
    def transformTimeStamp(self, df):
        """
        Transforms the timestamp column in the DataFrame from milliseconds to seconds
        and sets it as the index of the DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame containing a 'timestamp' column with time values in milliseconds.

        Modifies:
            df (pd.DataFrame): The input DataFrame is modified in-place with the 'timestamp' column 
                            converted to seconds and set as the index.

        Returns:
            None
        """
        df['timestamp'] = df['timestamp'] / 1000
        df['timestamp'] = df['timestamp'].astype(int)
        df.set_index('timestamp', inplace=True)
            
    def loadDataSet(self, path_int, path_dash):
        """
        Loads and merges datasets from specified file paths. The 'INT' dataset is merged with
        the 'DASH' dataset based on timestamp indices after preprocessing.

        Args:
            path_int (string): Path to the INT dataset file, expected to be a CSV with a comma separator.
            path_dash (string): Path to the DASH dataset file, expected to be a CSV with a semicolon separator.

        Modifies:
            self.dataset: The class attribute 'dataset' is set to the merged DataFrame.

        Detailed Steps:
            1. Reads the INT and DASH datasets from their respective paths.
            2. Transforms the timestamp in the DASH dataset from milliseconds to seconds.
            3. Filters the INT dataset to retain rows with the maximum 'deq_timedelta1' value per timestamp.
            4. Sets 'timestamp' as the index for both datasets.
            5. Merges the two datasets on their timestamp indices and resets the index of the merged DataFrame.
        """
        
        # Load the INT and DASH datasets from specified paths
        df_int = pd.read_csv(path_int, sep = ',')
        df_dash = pd.read_csv(path_dash, sep = ';')
        
        # from milliseconds to seconds
        self.transformTimeStamp(df_dash)
        df_int = df_int.loc[df_int.groupby('timestamp')['deq_timedelta1'].idxmax()]

         # Set 'timestamp' as the index for the INT dataset
        df_int.set_index('timestamp', inplace=True)  
        
        # Merge the INT and DASH datasets on their timestamp indices and reset the merged DataFrame's index   
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
        
    def preProcessData(self, num_cols, cat_cols, random): 
        """
        Preprocesses the dataset by filling missing values, encoding categorical variables,
        and optionally shuffling the data.

        Parameters:
            num_cols (list): A list of column names representing numerical variables
                            in the dataset where missing values will be replaced with the column mean.
            cat_cols (list): A list of column names representing categorical variables
                            to be one-hot encoded.
            random (bool): A flag that determines if the dataset rows should be randomly shuffled.

        Modifies:
            self.processed_data (pd.DataFrame): A copy of the dataset with preprocessing applied
                                                on numerical and categorical columns specified.
            self.cat_cols (list): Updated list of categorical columns after encoding.
            self.num_cols (list): List of numerical columns used in preprocessing.

        Returns:
            None
        """
        # Fill missing values in numerical columns with their mean
        for i in num_cols:
            self.dataset[i].fillna(self.dataset[i].mean(), inplace=True)
        
        # Perform one-hot encoding if there are categorical columns
        if len(cat_cols) > 0:
            self.hotEncode()
            cat_cols = [0,1,2]
        
        # Create a copy of the dataset with only the processed columns
        self.processed_data = self.dataset[ num_cols + cat_cols ].copy()
        self.cat_cols = cat_cols
        self.num_cols = num_cols

         # Randomly shuffle the dataset if requested
        if random == True:
            idx = np.random.permutation(self.processed_data.index)
            self.processed_data = self.processed_data.reindex(idx)
        
        
    def removeOutliers(self):
        """
        Removes outliers from the numerical columns of the dataset.

        This method iterates through each numerical column specified in self.num_cols.
        For each column, it calculates the first and third quartiles, and subsequently the 
        Interquartile Range (IQR). It defines outliers as those values that fall below 
        Q1 - 1.5*IQR or above Q3 + 1.5*IQR. Rows containing outliers in any of the specified 
        columns are dropped from the dataset.

        Modifies:
            self.processed_data (pd.DataFrame): The dataset with outliers removed.
        """
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
        """
        Removes columns from the processed_data attribute that contain the same value in all rows.

        This method identifies and drops any columns within the processed_data DataFrame 
        where all rows have the same value, indicating no variability in the dataset for 
        those attributes. Such columns are not useful for most analytical and machine learning 
        tasks as they do not provide any distinguishing information.

        Modifies:
            self.processed_data (pd.DataFrame): The dataset with constant-value columns removed.
        """
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