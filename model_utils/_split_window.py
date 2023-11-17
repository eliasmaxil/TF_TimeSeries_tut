""" split_window for the WindowGenerator class """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def _split_window(self, features):

    if self.use_label_columns:
        inputs = features[:, self.input_slice, :]
    else:
        rest_cols = [item for item in self.columns if item not in self.label_columns]
        labels_slice = [inputs[:,:, self.column_indices[name]] for name in rest_cols]
        inputs = tf.stack(labels_slice, axis=-1)

    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels_slice = [labels[:,:, self.column_indices[name]] for name in self.label_columns]
        labels = tf.stack(labels_slice, axis=-1)
       
    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels


if __name__ == "__main__":

    import numpy as np
    np.random.seed(1)

    import pandas as pd
    from window_generator import WindowGenerator

    # A random df
    n = 8
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(np.around(rng.random((n, 4)), 1), columns=['a','b','c','y'])
    # df.iloc[1:2,:] = np.nan
    # df.iloc[7:9,:] = np.nan

    print(df)
    print(df.shape)

    # Add this function to WindowGenerator
    WindowGenerator.split_window = _split_window

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

    print('\n'.join([
        'Attributes from the window needed:',
        f'.input_slice: {my_window.input_slice}',
        f'.columns: {my_window.columns}',
        f'.label_columns: {my_window.label_columns}',
        f'.input_width: {my_window.input_width}',
        f'.label_width: {my_window.label_width}',
        f'.shift: {my_window.shift}',
        '.total_window_size = input_width + shift',
        f'.total_window_size: {my_window.total_window_size}',
        ]))


    # The example stack with total_window_size length.
    data = np.array(df.values, dtype=np.float32)
    def stack_data(data, total_window_size):
        batches = []
        start = 0
        end = total_window_size
        for start in range(data.shape[0]-1):
            batch = data[start:end]
            start = start + total_window_size + 1
            end = start
            if batch.shape[0] == total_window_size:
                batches.append(batch)
        return tf.stack(batches)
    example_window = stack_data(data, my_window.total_window_size)

    print('example_window: A tf.stack of n batches and total_window_size width')
    print(example_window)
    print(f'The tf.stack has a shape: {example_window.shape}\n')


    inputs, labels = my_window.split_window(example_window)

    print('inputs, labels = my_window.split_window(example_window)')
    print(f'\ninputs\n {inputs} \n inputs.shape: {inputs.shape}')
    print(f'\nlabels:\n {labels} \n labels.shape: {labels.shape}')
    print('\nAll shapes are: batch, time, features')