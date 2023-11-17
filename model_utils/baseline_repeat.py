""" Baseline model: Repeats the input values into te target velues """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class MultiStepRepeatBaseline(tf.keras.Model):
    """
    !!! It is the same model as SingleStepBaseline !!!
    Repeats the input into the target.
    https://www.tensorflow.org/tutorials/structured_data/time_series#baselines
    """
    def __init__(self, label_index=None):
        """ 
        label_index: an int with the number of the column to be evaluated
        OUT_STEPS: Number of times the last value (label_width) is repeated
        """
        super().__init__()
        self.label_index = label_index


    def call(self, inputs):
        """
        Calls the model on new inputs.
        Returns the outputs as tensors.
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
    OUT_STEPS = 2  
    input_width = 2   # Take two rows (or time steps) of all columns as input
    label_width = OUT_STEPS  # Size of the prediction (output)
    shift = 1  # Time (or rows) offset between input and output
    batch_size = 1
    label_column='y'
    label_index = df.columns.get_loc(label_column)
    my_window = WindowGenerator(
                    input_width=input_width,
                    label_width=label_width,
                    shift=shift,
                    batch_size=batch_size,
                    train_df=df, val_df=df, test_df=df,
                    label_columns=[label_column],
                    use_label_columns=True, shuffle=False
                    )
    print(f'Window:\n{my_window}')

    # Baseline instatiation
    baseline_multi = MultiStepRepeatBaseline(
                        label_index=label_index, 
                        )
    baseline_multi.compile(
                        loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()]
                        )

    # Evaluation of the dataset
    dataset = my_window.val
    print('The dataset looks like:')
    for batch in dataset.take(3):
        inputs, targets = batch
        print(f'input: {inputs}')
        print(f'target: {targets}')
    print(f'input.shape: {inputs.shape}')
    print(f'target.shape: {targets.shape}')

    # Model evaluation with evaluate() & Naive
    print('The keras model.evaluate() method:')
    evaluation = baseline_multi.evaluate(dataset, verbose=0)
    # Naive, low-level evaluation
    def naive_mae_multi(model , dataset_, label_index_):
        targets = []
        for batch_ in dataset_:
            _, target_ = batch_
            targets.append(target_.numpy())
        targets = np.array(targets)   
        predictions = model.predict(dataset, verbose = 0)
        targets = targets.reshape([targets.shape[0], targets.shape[2], targets.shape[1]])
        mae = np.abs(predictions - targets).mean()
        return mae, predictions
        
    mae, predictions = naive_mae_multi(baseline_multi, dataset, label_index)

    # The MAE values should be the same
    print(f'Evaluation loss & MAE in dataset: {evaluation}')
    print(f'Naive MAE according to the dataset: {mae}')
    print('======')
    
    # The predicted values
    # predictions = baseline.predict(dataset, verbose=0)
    print('my predictions')
    print(predictions)