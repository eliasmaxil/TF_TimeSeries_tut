
""" make_dataset for the WindowGenerator class """

import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def _make_dataset(self, data):
    """ 
    This function takes in a sequence of data-points gathered at equal intervals, along with 
    time series parameters such as length of the sequences/windows, spacing between two 
    sequence/windows, etc., 
    
    https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    
    Returns batches of timeseries inputs and targets.
 
    """
    data = np.array(data, dtype=np.float32)
    # https://www.tensorflow.org/api_docs/python/tf/keras/utils/timeseries_dataset_from_array
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=self.shuffle,
        batch_size=self.batch_size,)
    # ds is a tf.data.Dataset instance. Represents a potentially large set of elements.

    # .filter(): Filters this dataset according to predicate.
    # .rebatch(): Creates a Dataset that rebatches the elements from this dataset
        # It is functionally equivalent to unbatch().batch(N), but is more
        # efficient, performing one copy instead of two.
    ds = ds.filter(lambda x: ~tf.reduce_any(tf.math.is_nan(x))).\
        rebatch(self.batch_size)
    # ds = ds.unbatch().\
    #         filter(lambda x: ~tf.reduce_any(tf.math.is_nan(x))).\
    #         batch(self.batch_size)

    # map(): Applies map_func to each element of this dataset, 
        # and returns a new dataset containing the transformed elements, 
        # in the same order as they appeared in the input.
    ds = ds.map(self.split_window)

    return ds


if __name__ == "__main__":
    
    # np.random.seed(1)

    import pandas as pd
    from window_generator import WindowGenerator

    # A random df
    n = 8
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(np.around(rng.random((n, 4)), 1), columns=['a','b','c','y'])
    # df.iloc[1:2,:] = np.nan
    # df.iloc[7:9,:] = np.nan

    # Add the function _make_dataset to WindowGenerator
    WindowGenerator.make_dataset = _make_dataset

    OUT_STEPS = 2
    input_width= 2
    label_width=OUT_STEPS
    shift=OUT_STEPS
    batch_size=1
    label_column='y'
    my_window = WindowGenerator(
                    input_width=input_width, 
                    label_width=label_width, 
                    shift=shift, 
                    batch_size=batch_size,
                    train_df=df, val_df=df, test_df=df,
                    label_columns=[label_column],
                    use_label_columns=True, shuffle=False 
                    )

    print('Executing _make_dataset ')
    print(f'\nThe data looks like:\n{df}\n')

    print('ds = my_window.make_dataset(df)\n')
    ds = my_window.make_dataset(df)
    print(f'{ds}')
    print(f'inputs:\n {next(iter(ds))[0]}')
    print(f'outputs:\n {next(iter(ds))[1]}')

