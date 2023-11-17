""" Baseline model: Repeat last value of the input ONCE  """
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class SingleStepBaseline(tf.keras.Model):
    """ 
    Repeats input_width times the last value of 
    * all columns of a row's input (label_index=None)
    * or a single colunm of the input (label_index=int)
    https://www.tensorflow.org/tutorials/structured_data/time_series#baseline
    """
    def __init__(self, label_index=None):
        """ 
        label_index --> an int with the number of the column to be evaluated 
        """
        super().__init__()
        self.label_index = label_index


    def call(self, inputs):
        """
        Calls the model on new inputs.
        Returns the outputs as tensors.

        This method should not be called directly. 
        It is only meant to be overridden when subclassing tf.keras.Model.
        
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#call
        """
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]

if __name__ == "__main__":

    import numpy as np
    import pandas as pd
    from window_generator import WindowGenerator

    n =  13
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(np.around(rng.random((n, 4)), 1), columns=['a','b','c','y'])
    df.iloc[1:2,:] = np.nan
    df.iloc[7:9,:] = np.nan
    print(df)

    # Window generator
    # When label_columns=None the naive_mae does not work
    # It works with batch_size != 1
    # input_width | shift | label_width can be also != 1
    # And the naive_mae will give the same results.
    input_width=2   # Like a wide_window
    label_width=2   # Like a wide_window
    shift=1         # Like a wide_window
    label_column='y'
    batch_size=1
    label_index = df.columns.get_loc(label_column)
    my_window = WindowGenerator(
                    input_width=input_width,
                    label_width=label_width,
                    shift=shift,
                    batch_size=batch_size,
                    train_df=df, val_df=df, test_df=df,
                    label_columns=[label_column],
                    # label_columns=None,
                    use_label_columns=True, shuffle=False
                    )
    print(f'Window:\n{my_window}')

    # Baseline instatiation
    baseline = SingleStepBaseline(label_index=label_index)
    baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])

    # Baseline evaluation of dataset
    dataset = my_window.val
    print('The dataset looks like:')
    for batch in dataset.take(3):
        inputs, targets = batch
        print(f'input: {inputs}')
        print(f'target: {targets}')
    print(f'input.shape: {inputs.shape}')
    print(f'target.shape: {targets.shape}')
    
    # Model evaluation with evaluate() & Naive
    print('model.evaluate() method:')
    evaluation = baseline.evaluate(dataset, verbose=0)
    # Naive, low-level evaluation 
    def naive_mae_single(model, dataset_, label_index_):
        targets = []
        for batch_ in dataset_:
            _, target_ = batch_
            targets.append(target_.numpy())
        targets = np.array(targets)   
        predictions = model.predict(dataset, verbose = 0)
        targets = targets.reshape([targets.shape[0], targets.shape[2], targets.shape[1]])
        mae = np.abs(predictions - targets).mean()
        return mae, predictions

    mae, predictions = naive_mae_single(baseline, dataset, label_index)

    # The MAE values should be the same
    print(f'Evaluation loss & MAE in dataset: {evaluation}')
    print(f'Naive MAE according to the dataset: {mae}')
    print('======')
    
    # The predicted values
    # predictions = baseline.predict(dataset, verbose=0)
    print('my predictions')
    print(predictions)