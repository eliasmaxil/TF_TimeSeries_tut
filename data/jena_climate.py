""" Gets the data of jena_climate to work with """

import os
import numpy as np
import pandas as pd
import tensorflow as tf

def jena_climate(with_nans=True):
    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True)
    csv_path, _ = os.path.splitext(zip_path)
    df = pd.read_csv(csv_path)
    

    # Slice [start:stop:step], starting from index 5 take every 6th record.
    df = df[5::6]
    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

    # Wind velocity
    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    wv[bad_wv] = 0.0

    max_wv = df['max. wv (m/s)']
    bad_max_wv = max_wv == -9999.0
    max_wv[bad_max_wv] = 0.0

    # The above inplace edits are reflected in the DataFrame.
    # df['wv (m/s)'].min()

    # Convertion of widh direction & velocity to wind vectors
    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = df.pop('wd (deg)')*np.pi / 180

    # Calculate the wind x and y components.
    df['Wx'] = wv*np.cos(wd_rad)
    df['Wy'] = wv*np.sin(wd_rad)

    # Calculate the max wind x and y components.
    df['max Wx'] = max_wv*np.cos(wd_rad)
    df['max Wy'] = max_wv*np.sin(wd_rad)

    # Time
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24*60*60
    year = (365.2425)*day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    if with_nans:
        df0 = df.iloc[:2]
        df1 = df.iloc[6:1234]
        df2 = df.iloc[2290:5076]
        df3 = df.iloc[10645:23523]
        df4 = df.iloc[28280:56978]
        df5 = df.iloc[63109:]
        dfb = pd.concat([df0, df1,df2,df3,df4,df5], axis=0)

        # Create NaNs
        dflong = pd.Series(data=np.ones(df.shape[0]), index=df.index, name='Dummy')
        df = pd.concat([dflong, dfb], axis=1).drop(columns=['Dummy'])
    
    return df

