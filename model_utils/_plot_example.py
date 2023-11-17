""" Plot function for the WindowGenerator class """

import matplotlib.pyplot as plt




def _plot_example(self, model=None, plot_col=None, max_subplots=1, xlabel='Time [h]'):
    inputs, labels = self.example
    # print(f'Cached example.inputs: {next(iter(self._example))}')
    
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    
    fig, _ = plt.subplots(max_n, 1, 
        figsize=(12, int(2*max_subplots)),
        sharex=True,
        gridspec_kw=dict(hspace=0.05),
        )
    for n in range(max_n):
        plt.subplot(max_n, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

    plt.xlabel(xlabel)

    return fig



if __name__ == "__main__":

    import pandas as pd
    import numpy as np
    # import matplotlib
    # matplotlib.use('module://matplotlib-sixel')
    np.random.seed(1)

    from window_generator import WindowGenerator

    # A random df
    n =  20
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(np.around(rng.random((n, 4)), 1), columns=['a','b','c','y'])
    df = pd.DataFrame(v.T, columns=['a','b','c','y'], index=idx)
    df.iloc[1:2,:] = np.nan
    df.iloc[7:9,:] = np.nan


    my_window = WindowGenerator(input_width=3, label_width=1, shift=1,
                    train_df=df, val_df=df, test_df=df,
                    batch_size=2, label_columns=['b'],
                    use_label_columns=True
                    )


    fig = _plot_example(my_window, plot_col='b')
    print(f'Object returned: {fig}')


