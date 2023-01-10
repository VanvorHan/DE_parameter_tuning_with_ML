import os
import gc
import wandb
import numpy as np
import tensorflow as tf

from print_dict import pd
from wandb.keras import WandbCallback
from dataset_processing import DatasetDE, preprocess_dataset
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, InputLayer


classes = 3
epochs = 10
learning_rate=0.001


def train_model(model, epochs=epochs, learning_rate=learning_rate):
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    
    tf.config.run_functions_eagerly(True)
    print("\n>>>>>>>>>>>>>>>>>>>> Training Starts >>>>>>>>>>>>>>>>>>>>")
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        validation_data=(X_val, y_val),
                        callbacks=[WandbCallback()])
    print(">>>>>>>>>>>>>>>>>>>> Training Ends >>>>>>>>>>>>>>>>>>>>\n")
    
    return history


def predict(model):
    y_pred = model.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_count = np.zeros(classes)
    for each_y in y_pred_class:
        y_count[each_y] += 1

    print("\n>>>>>>>>>>>>>>>>>>>> Evaluating Starts >>>>>>>>>>>>>>>>>>>>")
    print(f"Prediction Class Distribution:\n[0:stay; 1:decrease; 2:increase]\n{y_count}\n")
    model.evaluate(X_test, y_test)
    print(">>>>>>>>>>>>>>>>>>>> Evaluating Ends >>>>>>>>>>>>>>>>>>>>\n")


thresholds = [0.2]
pca_ns = [500]
#pca_ns = [1000, 800, 700, 600, 500]
#layers_def = [512, 256, 128, 64, 32, 16, 8, 4]
layers = [ [200, 100, 50]]

for threshold in thresholds:
    index = thresholds.index(threshold)
    layer = layers[index]
    pca_n = pca_ns[index]
    
    dataset_DE = DatasetDE("mut", threshold)
    print("\n>>>>>>>>>>>>>>>>>>>>>>>>>> GC >>>>>>>>>>>>>>>>>>>>>>>")
    n = gc.collect()
    print(f"Number of unreachable objects collected by GC: {n}\n")

    X_train, X_test, X_val, y_train, y_test, y_val = preprocess_dataset(dataset_DE.create_dataset(), pca_n=pca_n)
            
    config = {
        "dataset": "mut",
        "threshold": threshold,
        "model_architecture": "fully-dense",
        "kernel_size": layer,
        "hidden_layer": len(layer),
        "epochs": epochs,
        "pca_n": pca_n,
        "use_batch_norm": False,
        "regularization": "L1_L2",
        "learning_rate": learning_rate,
    }

    print("\n>>>>>>>>> CONFIG STARTS >>>>>>>>>")
    pd(config)
    print(">>>>>>>>> CONFIG ENDS >>>>>>>>>\n")
    
    wandb.init(project='gpt-3', entity='wangweihan-honourproject', config=config, sync_tensorboard=True)

        
    model = tf.keras.Sequential()
    model.add(InputLayer(input_shape=(pca_n,)))
    for kernel in layer:
        model.add(Dense(kernel, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    
    train_model(model)
    predict(model)

    model_name = "saved_model/mut_NN_t" + str(threshold)
    model.save(model_name)
    
    wandb.finish()
