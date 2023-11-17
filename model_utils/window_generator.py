""" Window generator class """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
import tensorflow as tf
import numpy as np

from _split_window import _split_window
from _make_dataset import _make_dataset
from _plot_example import _plot_example

class WindowGenerator():
    """ 
    Main reference: https://www.tensorflow.org/tutorials/structured_data/time_series
    """
    def __init__(self, input_width, label_width, shift, batch_size=32,
                train_df=None, val_df=None, test_df=None,
                label_columns=None, use_label_columns=True,
                shuffle=True):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                            enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                            enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        # Specifications
        self.batch_size = batch_size
        self.use_label_columns = use_label_columns
        self.columns = list(self.train_df.columns)
        self.shuffle = shuffle
        
        # Cache
        self._example = None
    
    def __repr__(self):
        return '\n'.join([
            f'input_width: {self.input_width}',
            f'label_width: {self.label_width}',
            f'batch_size: {self.batch_size}',
            f'shift: {self.shift}',
            f'total_window_size: {self.total_window_size} (input_width+shift)',
            f'input_indices: {self.input_indices}',
            f'label_columns: {self.label_columns}',
            f'label_indices: {self.label_indices}',
            f'colum_indices: {self.column_indices}',
            f'use_label_columns: {self.use_label_columns}\n'
            'All shapes are: (batch, time, features)'
            ])


    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

WindowGenerator.split_window = _split_window
WindowGenerator.make_dataset = _make_dataset
WindowGenerator.plot_example = _plot_example

if __name__ == "__main__":

    np.random.seed(1)
    import pandas as pd

    n =  13
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(np.around(rng.random((n, 4)), 1), columns=['a','b','c','y'])
    # df.iloc[1:2,:] = np.nan
    # df.iloc[7:9,:] = np.nan
    print(df.head())
    print(df.shape)


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

    print(f'\nGenerated window:\n {my_window}')

    example_window = tf.stack([
                        np.array(df[:my_window.total_window_size]),
                        np.array(df[1:1+my_window.total_window_size]),
                        ])


    print('Executing my_window.split_window(...)')
    inputs, labels = my_window.split_window(example_window)
    print('my_window.split_window() done')

    print('Executing my_window.make_dataset(...)')
    ds = my_window.make_dataset(df)
    print('Dataset sample:')
    for batch in ds:
        input, target = batch
        print(f'input: {input}')
        print(f'target: {target}')
    print(f'input.shape: {input.shape}')
    print(f'target.shape: {target.shape}')
 

    print('Executing @property my_window.train')
    ds = my_window.train
    print(f'@property my_window.train: {ds}')

    print('Executing the plot')
    fig = my_window.plot_example(plot_col='y')
    print(f'Object returned: {fig}')