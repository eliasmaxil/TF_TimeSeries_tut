""" Split and scale a DF """

import pandas as pd
import numpy as np

class SplitScale(object):
    def __init__(self, data=None, train_ratio=0.7, valid_ratio=0.2, use_train=True):
        self.df = data
        self.use_train = use_train
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self._train_df = None
        self._val_df = None
        self._test_df = None
        self.orig_min = None
        self.orig_max = None
        self.orig_mean = None
        self.orig_std = None
    
    def train_test_valid_split(self):
            """
                Splits the data frame into 3 datasets: train, validation and test without shuffling the data. 
                
                Returns a tuple of the whole data (not yet scaled) in the form:

                train_df, val_df, test_df
                
            """
            n = self.df.shape[0]
            self.train_df = self.df[0:int(n*self.train_ratio)]
            self.val_df = self.df[int(n*self.train_ratio):int(n* (self.train_ratio + self.valid_ratio) )]
            self.test_df = self.df[int(n* (self.train_ratio + self.valid_ratio)):]

            return (self.train_df, self.val_df, self.test_df)

    def minmax_scale(self):
        """ Scale the train, validation & test matrices to a [0,1] range.
        The train, validation and test matrices of the features must already exist.

        When use_train==True:

            The functions follows the guidelines of:
            https://www.tensorflow.org/tutorials/structured_data/time_series#split_the_data
            
            The min & max are computed using the training data so that the model has no 
            access to the values in the validation and test sets.

        Returns min and max values of each feature
        """

        if self.use_train:
            self.orig_min = self.train_df.min()
            self.orig_max = self.train_df.max()
        else:
            self.orig_min = self.df.min()
            self.orig_max = self.df.max()
        
        self._train_df = (self.train_df-self.orig_min)/(self.orig_max-self.orig_min)
        self._val_df = (self.val_df-self.orig_min)/(self.orig_max-self.orig_min)
        self._test_df = (self.test_df-self.orig_min)/(self.orig_max-self.orig_min)

        return self.orig_min, self.orig_max

    def z_normalize(self):
        """ Normalizes the values of the dataframe so that each column has a mean = 0 and 
            standard deviation = 1. 
            The train, validation and test matrices of the features must already exist. 

            When use_train =True:

                The functions follows the guidelines of:
                https://www.tensorflow.org/tutorials/structured_data/time_series#split_the_data
                
                The mean and standard deviation are computed using the training data so that 
                the model have no access to the values in the validation and test sets.

        Returns mean & std of each feature
        """

        if self.use_train:
            self.orig_mean = self.train_df.mean()
            self.orig_std = self.train_df.std()
        else:
            self.orig_mean = self.df.mean()
            self.orig_std = self.df.std()

        self._train_df = (self.train_df - self.orig_mean) / self.orig_std
        self._val_df = (self.val_df - self.orig_mean) / self.orig_std
        self._test_df = (self.test_df - self.orig_mean) / self.orig_std

        return self.orig_mean, self.orig_std

    def get_scaled_data(self):
        """ Returns the scaled train, val & test dfs """
        return self._train_df, self._val_df, self._test_df


if __name__ == "__main__":

    np.random.seed(1)
    n = 40
    v = np.array([
            np.around(np.random.default_rng(0).random(n) - 1, 1),
            np.around(np.random.default_rng(0).random(n), 1),
            np.around(np.random.default_rng(0).random(n) + 1, 1),
            np.around(np.random.default_rng(0).random(n) + 2, 1)
            ]) 
    idx = pd.date_range("2018-01-01", periods=n, freq="60T")
    df = pd.DataFrame(v.T, columns=['c', 'd', 'b', 'a'], index=idx)
    df.iloc[3:5,:] = np.nan
    df.iloc[12:14,:] = np.nan
    df.iloc[21:25,:] = np.nan
    df.iloc[30:34,:] = np.nan

    print('Data frame:')
    print(df)

    data_obj = SplitScale(df)
    train_df, val_df, test_df = data_obj.train_test_valid_split()
    print('train_df')
    print(train_df.head())
    print('val_df')
    print(val_df.head())


    orig_min, orig_max = data_obj.minmax_scale()
    train_df, val_df, test_df = data_obj.get_scaled_data()
    print('Scaled data')
    print('train_df')
    print(train_df.head())
    print('val_df')
    print(val_df.head())
    print('orig_min')
    print(orig_min)
    print('orig_max')
    print(orig_max)