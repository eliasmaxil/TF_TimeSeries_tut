import tensorflow as tf

def compile_and_fit(model, window, basics):
    early_stopping = tf.keras.callbacks.\
            EarlyStopping(
                monitor='val_loss',
                patience=basics.patience,
                mode='min'
                )

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=basics.max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history



# TODO: Adapt this complie & fit to the def above
# def tf_compile_and_fit(window, m:ModelPayload, model, **kwargs):

#     try:
#         optimizer=kwargs['optimizer']
#     except KeyError:
#         optimizer = tf.keras.optimizers.Adam()
    
#     early_stopping = tf.keras.callbacks.EarlyStopping(
#         monitor='val_loss',
#         patience=m.patience,
#         restore_best_weights=True
#         )

#     model.compile(
#         loss=tf.keras.losses.MeanSquaredError(),
#         optimizer=optimizer,
#         metrics=[tf.keras.metrics.MeanAbsoluteError()]
#         )

#     try:
#         model.summary()
#     except ValueError:
#         print('The summary of the model has not yet been built.')

#     history = model.fit(
#         window.train, 
#         epochs=m.max_epochs,
#         batch_size=m.batch_size,
#         validation_data=window.val,
#         verbose=m.verbosity,
#         callbacks=[early_stopping],
#         shuffle=m.shuffle
#         )
#     return model, history



# if __name__ == "__main__":