#Creates the DTW model
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
def TS_kNN_DTW(n_neighbors, weights):
    return_errors = []
    #Checks the validity of the parameter values
    if(isinstance(n_neighbors, int) == False):
        return_errors.append("Invalid n_neighbors value!")
    if(weights not in ["uniform", "distance"]):
        return_errors.append("Invalid weights value!")
        
    if(len(return_errors) == 0):
        return KNeighborsTimeSeriesClassifier(n_neighbors = n_neighbors, weights = weights, distance="dtw")
    else:
        print(return_errors)
        return None
    
#Creates the NN model    
from tensorflow import keras
def TS_NN_make_model(input_shape, activation_function, drop_out, filters, kernel, num_classes):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters = filters, kernel_size = kernel, padding = "same", activation = activation_function)(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Dropout(drop_out)(conv1)

    conv2 = keras.layers.Conv1D(filters = filters, kernel_size = kernel, padding = "same", activation = activation_function)(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Dropout(drop_out)(conv2)

    conv3 = keras.layers.Conv1D(filters = filters, kernel_size = kernel, padding = "same", activation = activation_function)(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Dropout(drop_out)(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)
    
    if num_classes == 2: # Binary classification
        output_layer = keras.layers.Dense(1, activation="sigmoid")(gap)
    elif num_classes > 2: # Multiclass classification
        output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs = input_layer, outputs = output_layer)


#Traines the NN model  
import numpy as np
import pickle
import tensorflow as tf
def TS_NN_model(folder_route, epoch, batch_size, activation_function, drop_out, filters, kernel, num_classes, X_train, Y_train, metric, validation_split):
    return_errors = []
    tf.random.set_seed(0) # for reproducibility

    if(isinstance(epoch, int) == False):
        return_errors.append("Invalid epoch value!")
    if(isinstance(batch_size, int) == False):
        return_errors.append("Invalid batch_size value!")
    if(activation_function not in ["relu", "elu"]):
        return_errors.append("Invalid activation_function value!")
        
    
    if(len(return_errors) == 0):
        NN_model = TS_NN_make_model(X_train.shape[1:], activation_function, drop_out, filters, kernel, num_classes)
        
        # The loss function is different based on the classification type
        if num_classes == 2: # Binary classification
            compile_loss = "binary_crossentropy"
        elif num_classes > 2: # Multi classification
            compile_loss = "sparse_categorical_crossentropy"
            
        NN_model.compile(
            optimizer = "adam",
            loss = compile_loss,
            metrics = [metric],
        )
        
        callbacks = [
            keras.callbacks.ModelCheckpoint(folder_route + "/"+ str(epoch) + ".h5", save_best_only = True, monitor = "val_loss"),
            keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 20, min_lr = 0.0001),
            keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 5, restore_best_weights=True, verbose = 1), #restore_best_weights=True, A legjobb modellt állítja vissza
        ]
        
        history = NN_model.fit(
            X_train,
            Y_train,
            batch_size = batch_size,
            epochs = epoch,
            callbacks = callbacks,
            validation_split = validation_split,
            verbose = 0,
        )
        
        # Save history separately
        with open(folder_route + "/" + str(epoch) + "_history.pkl", "wb") as f:
            pickle.dump(history.history, f)
    else:
        return return_errors

#Creates the NN model    
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
def TS_NN_make_model2(input_shape, activation_function, num_classes):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape), # elég csak a max, mert ez nem veszi figyelembe a nullákat
        LSTM(64, return_sequences=True, dropout=0.2),
        LSTM(32, dropout=0.2),
        Dense(32, activation=activation_function),
        Dropout(0.3),
        Dense(1 if num_classes == 2 else num_classes, activation='sigmoid' if num_classes == 2 else 'softmax')  # Helyes kimeneti réteg
    ])
    return model

#Traines the NN model  
import numpy as np
import pickle
def TS_NN_model2(folder_route, epoch, batch_size, activation_function, num_classes, X_train, Y_train, metric, validation_split):
    return_errors = []
    
    if(isinstance(epoch, int) == False):
        return_errors.append("Invalid epoch value!")
    if(isinstance(batch_size, int) == False):
        return_errors.append("Invalid batch_size value!")
    if(activation_function not in ["relu", "elu"]):
        return_errors.append("Invalid activation_function value!")
        
    
    if(len(return_errors) == 0):
        NN_model = TS_NN_make_model2(X_train.shape[1:], activation_function, num_classes)
                
        # The loss function is different based on the classification type
        if num_classes == 2: # Binary classification
            compile_loss = "binary_crossentropy"
        elif num_classes > 2: # Multi classification
            compile_loss = "sparse_categorical_crossentropy"

        NN_model.compile(
            optimizer = "adam",
            loss = compile_loss,
            metrics = [metric],
        )
        
        callbacks = [
            keras.callbacks.ModelCheckpoint(folder_route + "/"+ str(epoch) + ".h5", save_best_only = True, monitor = "val_loss"),
            keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 10, restore_best_weights=True, verbose = 1), #restore_best_weights=True, A legjobb modellt állítja vissza
        ]

        history = NN_model.fit(
            X_train,
            Y_train,
            batch_size = batch_size,
            epochs = epoch,
            callbacks = callbacks,
            validation_split = validation_split,
            verbose = 0,
        )
        
        # Save history separately
        with open(folder_route + "/" + str(epoch) + "_history.pkl", "wb") as f:
            pickle.dump(history.history, f)
    else:
        return return_errors