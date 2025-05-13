#Reads the kinematic data.
import pandas as pd
def Loading_data(files, separator, col_names):
    if type(files) == list:
        df_list = []
        for filename in files:
            if col_names == None:
                df = pd.read_csv(filename, index_col = None, sep = separator)
            elif type(col_names) == list:
                df = pd.read_csv(filename, index_col = None, sep = separator, names = col_names)
            df_list.append(df)
        return df_list
    elif type(files) == str:
        if col_names == None :
            df = pd.read_csv(files, index_col = None, sep = separator)
        elif type(col_names) == list:
            df = pd.read_csv(files, index_col = None, sep = separator, names = col_names)
        return df  
    

#Creates the 0,1,2 classes    
def Multi_class_target_variable(df_class_value, class_tresholds):
    if df_class_value < class_tresholds[0]:
        return 0
    elif df_class_value < class_tresholds[1]:
        return 1
    elif df_class_value < class_tresholds[2]:
        return 2


#Creates classes for multiple target variables (based on mean)
from statistics import mean
def Multi_target_variable(df, col_idx):
    for i in col_idx:
        df.iloc[:, i] = np.where(df.iloc[:, i] < round(mean(df.iloc[:, i])), 0, 1)
    return df


#Transforms the data into non-time series
#i: trials
#j: Columns of the given trial (0-76).
#NP_df_list[i].iloc[:,j]: One column of the given trial
import antropy as ant
def Approximate_entropy(df_list, return_ApEn):
    ApEn_rows = []

    for i in range(len(df_list)):
        for j in range(len(df_list[i].columns)):
            ApEn_rows.append(ant.app_entropy(df_list[i].iloc[:,j]))
        
        ApEn_series = pd.Series(ApEn_rows, index = return_ApEn.columns)
        return_ApEn = return_ApEn.append(ApEn_series, ignore_index=True)
        ApEn_rows.clear()
        ApEn_series.empty
    
    return return_ApEn


#Attaches the target value to the data.
def Target_append(df_class, return_ApEn, class_column_name):
    target = df_class[class_column_name]
    return_ApEn = return_ApEn.join(target)
    return return_ApEn


#Splits the data into features and target variable
def Features_target(df):
    # features
    X = df.iloc[:, 0:(df.shape[1]-1)].values 
    
    # target variable
    Y = df.iloc[:, (df.shape[1]-1)].values
    
    return X, Y


#Splits the data into train and test set
from sklearn.model_selection import train_test_split
def Train_test(X, Y, t_size):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = t_size, random_state = 0)
    return X_train, X_test, Y_train, Y_test


#Standardizes the data
from sklearn.preprocessing import StandardScaler
def Standardization(X_train, X_test):
    scaler = StandardScaler()
    standardized_X_train = scaler.fit_transform(X_train)
    standardized_X_test = scaler.transform(X_test)
    return standardized_X_train, standardized_X_test


#Converts the time series data to equal lengths (max)
def TS_df_append_to_max_len(df_list, feature_names):
    #Creating a new DataFrame to avoid overwriting the original one
    df_list_max_len = df_list.copy()
    
    max_len = 0
    
    for i in range(len(df_list_max_len)):
        if max_len < len(df_list_max_len[i]):
            max_len = len(df_list_max_len[i])
            
    print(f"The maximum length: {max_len}")
    
    #shape[1]: number of columns
    #Row contains zeros
    row = [0] * df_list_max_len[0].shape[1]
    series = pd.Series(row, index = feature_names)
    
    for j in range(len(df_list_max_len)):
        if max_len > len(df_list_max_len[j]):
            while max_len > len(df_list_max_len[j]):
                df_list_max_len[j] = df_list_max_len[j].append(series, ignore_index=True)
    
    return df_list_max_len


#Converts the time series data to equal lengths (avg)
def TS_df_append_truncate_to_avg_len(df_list, feature_names):
    df_list_avg_len = df_list.copy()
    
    lengths = []
    for i in range(len(df_list_avg_len)):
        lengths.append(len(df_list_avg_len[i]))
        
    avg_len = round(sum(lengths)/len(lengths))
    print(f"The average length: {avg_len}")
    
    row = [0] * df_list_avg_len[0].shape[1]
    series = pd.Series(row, index = feature_names)
    
    for j in range(len(df_list_avg_len)):
        if avg_len > len(df_list_avg_len[j]):
            while avg_len > len(df_list_avg_len[j]):
                df_list_avg_len[j] = df_list_avg_len[j].append(series, ignore_index=True)
        else:
            #-1: DataFrame indexed from 0
            df_list_avg_len[j] = df_list_avg_len[j].truncate(after = avg_len - 1)
    
    return df_list_avg_len


#Converts the time series data to equal lengths (min)
import sys
def TS_df_truncate_to_min_len(df_list):
    df_list_min_len = df_list.copy()
        
    min_len = sys.maxsize
    
    for i in range(len(df_list_min_len)):
        if(min_len > len(df_list_min_len[i])):
            min_len = len(df_list_min_len[i])
    
    print(f"The minimum length: {min_len}")
    
    for j in range(len(df_list_min_len)):
        if min_len < len(df_list_min_len[j]):
            df_list_min_len[j] = df_list_min_len[j].truncate(after = min_len - 1)
            
    return df_list_min_len


#Returns the target variable
def TS_Y(df_class, column_index):
    return np.array(df_class.iloc[:, column_index])


#Standardizes the time series data
def TS_Standardization(X_train, X_test, feature_names):
    X_train_concat = pd.DataFrame(columns = feature_names)

    for i in range(len(X_train)):
        X_train_concat = pd.concat([X_train_concat, X_train[i]], axis = 0, ignore_index = True)

    scaler = StandardScaler()
    scaler.fit(X_train_concat)

    standardized_X_train = []
    for i in range(len(X_train)):
        standardized_X_train.append(scaler.transform(X_train[i]))
        
    standardized_X_test = []
    for i in range(len(X_test)):
        standardized_X_test.append(scaler.transform(X_test[i]))
    
    standardized_X_train_df_list = []
    for i in range(len(standardized_X_train)):
        df_train = pd.DataFrame(standardized_X_train[i], columns = feature_names)
        standardized_X_train_df_list.append(df_train)
        
    standardized_X_test_df_list = []
    for i in range(len(standardized_X_test)):
        df_test = pd.DataFrame(standardized_X_test[i], columns = feature_names)
        standardized_X_test_df_list.append(df_test)

    return standardized_X_train_df_list, standardized_X_test_df_list


#Transformes the data into an approppriate format for DTW
import numpy as np
def TS_kNN_DTW_transform(X_train, X_test):
    X_train_n_instances = len(X_train)
    X_test_n_instances = len(X_test)
    
    X_train_n_dimensions = X_train[0].shape[1]
    X_test_n_dimensions = X_test[0].shape[1]
    
    X_train_series_length = X_train[0].shape[0]
    X_test_series_length = X_test[0].shape[0]
    
    # Initialize a 3D numpy array
    X_train_transformed_df = np.zeros((X_train_n_instances, X_train_n_dimensions, X_train_series_length))
    X_test_transformed_df = np.zeros((X_test_n_instances, X_test_n_dimensions, X_test_series_length))
    
    # Populate the 3D numpy array with data from the list of dataframes
    for x_train_i, x_train_df in enumerate(X_train):
        X_train_transformed_df[x_train_i] = x_train_df.values.T
        
    for x_test_i, x_test_df in enumerate(X_test):
        X_test_transformed_df[x_test_i] = x_test_df.values.T
        
    return X_train_transformed_df, X_test_transformed_df


#Transformes the data into an approppriate format for NN
def TS_NN_transform(X_train, X_test):
    X_train_n_instances = len(X_train)
    X_test_n_instances = len(X_test)
    
    X_train_n_dimensions = X_train[0].shape[1]
    X_test_n_dimensions = X_test[0].shape[1]
    
    X_train_series_length = X_train[0].shape[0]
    X_test_series_length = X_test[0].shape[0]
    
    # Initialize a 3D numpy array
    X_train_transformed_df = np.zeros((X_train_n_instances, X_train_series_length, X_train_n_dimensions))
    X_test_transformed_df = np.zeros((X_test_n_instances, X_test_series_length, X_test_n_dimensions))
    
    # Populate the 3D numpy array with data from the list of dataframes
    for x_train_i, x_train_df in enumerate(X_train):
        X_train_transformed_df[x_train_i] = x_train_df.values
        
    for x_test_i, x_test_df in enumerate(X_test):
        X_test_transformed_df[x_test_i] = x_test_df.values
        
    return X_train_transformed_df, X_test_transformed_df


#Splits the trials time series data into gestures
def Gestures_data_splitting(number_of_gestures, feature_names, gestures, gesture_df_list, df_list):    
    final_gestures = []
    
    #running through each trial
    for trial_idx in range(len(df_list)): 
        trial_gestures = [None] * number_of_gestures
        
        for g in range(number_of_gestures):
            trial_gestures[g] = pd.DataFrame(columns = feature_names)
        
        #running through each gesture annotation dataframe for a single trial
        for i in range(len(gesture_df_list[trial_idx])): 
            start = gesture_df_list[trial_idx].iloc[i, 0] - 1   #-1: DataFrame indexed from 0
            end = gesture_df_list[trial_idx].iloc[i, 1] #The -1 is not necessary, because iloc[start:end, :]
            
            #running through the possible gestures in the task
            for gesture_idx in gestures:
                if gesture_df_list[trial_idx].iloc[i, 2] == f"G{gesture_idx}":
                    #gesture_idx-1: gestures_idx starts with 1, while trial_gestures DataFrame starts with 0
                    trial_gestures[gesture_idx - 1] = trial_gestures[gesture_idx - 1].append(df_list[trial_idx].iloc[start:end, :])
        
        final_gestures.append(trial_gestures)#, ignore_index=True)
            
    return final_gestures


#Gestures: Transforms the time series data with ApEn, and attaches the target variable
import math
def Gestures_ApEn_transform_target_append(number_of_gestures, df_gestures, feature_names, df_classes, classes_target_idx):
    ApEn_rows = []
    final_gestures = [None] * number_of_gestures
    trial_idx = -1 #-1: DataFrame indexed from 0
    
    #Attaching the target variable to the kinematic data
    for j in range(number_of_gestures):
        final_gestures[j] = pd.DataFrame(columns = feature_names)
        final_gestures[j][df_classes.columns[classes_target_idx]] = ""

    #running through each trial
    for trial in df_gestures:
        trial_idx = trial_idx + 1
        
        #running through each gesture in a trial
        for gesture in range(len(trial)):
            #Only transform the gesture data if the DataFrame is not empty.
            isempty = trial[gesture].empty
            if isempty == False:
                for column in range(len(trial[gesture].columns)):
                    ApEn_rows.append(ant.app_entropy(trial[gesture].iloc[:, column]))

                ApEn_series = pd.Series(ApEn_rows, index = trial[gesture].columns)
                final_gestures[gesture] = final_gestures[gesture].append(ApEn_series, ignore_index=True)
                
                ApEn_rows.clear()
                ApEn_series.empty
        
        #Attaching the target variable for the trial gestures data
        for g in range(len(final_gestures)):
            final_gesture_isempty = final_gestures[g].empty

            #Attaching the target variable to the newly added rows
            #iloc[-1, -1]: Last row, last column, which represents the target value
            if final_gesture_isempty == False and math.isnan(final_gestures[g].iloc[-1, -1]):
                final_gestures[g].iloc[-1, -1] = int(df_classes.iloc[trial_idx, classes_target_idx])
                final_gestures[g] = final_gestures[g].astype({df_classes.columns[classes_target_idx]: 'int'}) #forcing the type to int
     
    return final_gestures


#Splits the data into features and target variable (gestures)
def Gestures_features_target(task_gestures, gestures_idxs, minimum_rows):
    return_errors = []
    
    #minimum_rows < 1 to not include empty dataframes
    if isinstance(minimum_rows, int) == False or minimum_rows < 1:
        return_errors.append("Invalid minimum_rows value!")
    
    if len(return_errors) == 0:
        gestures_idxs_temp = gestures_idxs.copy()
        gestures_X = []
        gestures_Y = []
        
        #running through each gesture
        for i in range(len(task_gestures)):
            #Enough row
            if len(task_gestures[i]) >= minimum_rows:
                temp_gestures_X, temp_gestures_Y = Features_target(task_gestures[i])

                gestures_X.append(temp_gestures_X)
                gestures_Y.append(temp_gestures_Y)
            #Not enough row
            else:
                next_i = i+1 #i+1 because the gesture_idx starts with 1 and the task_gestures starts with 0
                if next_i in gestures_idxs_temp:
                    print(f"There is not enough data in the G{next_i} gesture")
                    delete_idx = gestures_idxs_temp.index(next_i)
                    gestures_idxs_temp.pop(delete_idx)
        
        print(f"Final gestures for the task: {gestures_idxs_temp}")
        
        return gestures_X, gestures_Y, gestures_idxs_temp
    else:
        print(return_errors)
        return None, None, gestures_idxs_temp

    
#Gestures: Splits the data into training and testing datasets 
def Gestures_train_test(X, Y, t_size):
    gestues_X_train = []
    gestues_Y_train = []
    gestues_X_test = []
    gestues_Y_test = []
    
    for i in range(len(X)):
        X_train, X_test, Y_train, Y_test = Train_test(X[i], Y[i], t_size)

        gestues_X_train.append(X_train)
        gestues_Y_train.append(Y_train)

        gestues_X_test.append(X_test)
        gestues_Y_test.append(Y_test)
        
    return gestues_X_train, gestues_Y_train, gestues_X_test, gestues_Y_test


#Gestures: Standardizes the data 
def Gestures_standardization(gestures_X_train, gesture_X_test):
    return_standardized_X_train = []
    return_standardized_X_test = []
    
    #running through each gesture
    for i in range(len(gestures_X_train)):
        scaler = StandardScaler()
        return_standardized_X_train.append(scaler.fit_transform(gestures_X_train[i]))
        return_standardized_X_test.append(scaler.transform(gesture_X_test[i]))
    
    return return_standardized_X_train, return_standardized_X_test

    
#Gestures: converts the time series data to equal lengths (max)
def TS_gestures_df_append_to_max_len(df_list, feature_names):
    #Creating a new DataFrame to avoid overwriting the original one: deep=True
    df_list_max_len = [x.copy() for x in df_list]
    
    max_len = [0]*len(df_list_max_len[0])
    
    for i in range(len(df_list_max_len)):
        for j in range(len(df_list_max_len[i])):
            length = len(df_list_max_len[i][j])
            if max_len[j] < length:
                max_len[j] = length
                
    #shape[1]: number of columns
    #Row containing 76 zeros
    row = [0] * df_list_max_len[0][0].shape[1]
    series = pd.Series(row, index = feature_names)
    
    for i in range(len(df_list_max_len)):
        for j in range(len(df_list_max_len[i])):
            if len(df_list_max_len[i][j]) != 0: #do not append the not used gestures with zeros
                while max_len[j] > len(df_list_max_len[i][j]):
                    df_list_max_len[i][j] = df_list_max_len[i][j].append(series, ignore_index=True)
    
    return df_list_max_len


#Gestures: converts the time series data to equal lengths (avg)
def TS_gestures_df_append_truncate_to_avg_len(df_list, feature_names):
    df_list_avg_len = [x.copy() for x in df_list]
    lengths = []
 
    for i in range(len(df_list_avg_len[0])):
        temp_lengths = []
        for j in range(len(df_list_avg_len)):
            temp_lengths.append(len(df_list_avg_len[j][i]))
        lengths.append(temp_lengths)
        
    avg_len = []
    for i in range(len(lengths)):
        avg_len.append(round(sum(lengths[i])/len(lengths[i])))
        
    row = [0] * df_list_avg_len[0][0].shape[1]
    series = pd.Series(row, index = feature_names)
    
    for i in range(len(df_list_avg_len)):
        for j in range(len(df_list_avg_len[i])):
            if len(df_list_avg_len[i][j]) != 0:
                if avg_len[j] > len(df_list_avg_len[i][j]):
                    while avg_len[j] > len(df_list_avg_len[i][j]):
                        df_list_avg_len[i][j] = df_list_avg_len[i][j].append(series, ignore_index=True)
                else:
                    df_list_avg_len[i][j] = df_list_avg_len[i][j].iloc[:avg_len[j], :]
                    df_list_avg_len[i][j] = df_list_avg_len[i][j].reset_index(drop=True)

    return df_list_avg_len


#Gestures: converts the time series data to equal lengths (min)
def TS_gestures_df_truncate_to_min_len(df_list):
    df_list_min_len = [x.copy() for x in df_list]
    min_len = [sys.maxsize]*len(df_list_min_len[0])
    
    for i in range(len(df_list_min_len)):
        for j in range(len(df_list_min_len[i])):
            if len(df_list_min_len[i][j]) != 0:
                length = len(df_list_min_len[i][j])
                if min_len[j] > length:
                    min_len[j] = length
                                
    for i in range(len(df_list_min_len)):
        for j in range(len(df_list_min_len[i])):
            if len(df_list_min_len[i][j]) != 0:
                    df_list_min_len[i][j] = df_list_min_len[i][j].iloc[:min_len[j], :]
                    df_list_min_len[i][j] = df_list_min_len[i][j].reset_index(drop=True)
            
    return df_list_min_len


#Gestures: Returns the valid gesture X-Y
def TS_gestures_X_Y_transform(gestures_df, df_class, column_index):
    #gesture_X: Transforming the [trial][gesture] format to [gesture][trial]
    num_trials = len(gestures_df)  #Number of trials
    num_gestures = len(gestures_df[0])  #Number of gestures in one trial
    gestures_X = [[] for _ in range(num_gestures)]
    temp_gestures_X = []
    
    for trial in gestures_df:
        for gesture_idx, gesture_df in enumerate(trial):
            gestures_X[gesture_idx].append(gesture_df)
        
    #Selecting only the used gestures
    valid_gestures = []
    for i, gesture in enumerate(gestures_X):
        empty_trials = []
        not_null = False
        for j, trial in enumerate(gesture):
            if len(trial) > 0 and not_null == False: ##The gestures has trial(s)
                not_null = True
        if not_null:
            temp_gestures_X.append(gestures_X[i])
            valid_gestures.append(i)
    
    #Creating the same Y for each valid gesture
    gestures_Y = []
    for i in range(len(temp_gestures_X)):
        Y = TS_Y(df_class, column_index)
        gestures_Y.append(Y)
        
    #Selecting and deleting the empty trials
    final_gestures_X = []
    not_valid_trials = []
    for i, gesture in enumerate(temp_gestures_X):
        empty_trials = []
        for j, trial in enumerate(gesture):
            if len(trial) == 0: #Empty trials index
                empty_trials.append(j)
        not_valid_trials.append(empty_trials)

        for idx in reversed(empty_trials): #Reverse to not modify the indexes
            del gesture[idx] #Deleting a trial
            gestures_Y[i] = np.delete(gestures_Y[i], idx) #Deleting the Y of the deleted trial
        
    return temp_gestures_X, gestures_Y, valid_gestures, not_valid_trials


#Getures: standardizes the time series data
def TS_gesture_standardization(X_train, X_test, feature_names):
    return_standardized_X_train = []
    return_standardized_X_test = []
    
    for g_idx, x_train_gesture in enumerate(X_train):
        X_train_concat = pd.DataFrame(columns = feature_names)
        for t_train_idx, x_train_trial in enumerate(x_train_gesture):
            X_train_concat = pd.concat([X_train_concat, x_train_trial], axis = 0, ignore_index = True)
        
        scaler = StandardScaler()
        scaler.fit(X_train_concat)
        
        standardized_X_train = []
        for t_train_idx, x_train_trial in enumerate(x_train_gesture):
            standardized_X_train.append(scaler.transform(x_train_trial))
            
        standardized_X_test = []
        for t_test_idx in range(len(X_test[g_idx])):
            standardized_X_test.append(scaler.transform(X_test[g_idx][t_test_idx]))
        
        standardized_X_train_df_list = []
        for i in range(len(standardized_X_train)):
            df_train = pd.DataFrame(standardized_X_train[i], columns = feature_names)
            standardized_X_train_df_list.append(df_train)
        return_standardized_X_train.append(standardized_X_train_df_list)
        
        standardized_X_test_df_list = []
        for i in range(len(standardized_X_test)):
            df_test = pd.DataFrame(standardized_X_test[i], columns = feature_names)
            standardized_X_test_df_list.append(df_test)
        return_standardized_X_test.append(standardized_X_test_df_list)
        
    return return_standardized_X_train, return_standardized_X_test


#Gestures: transformes the data into an approppriate format for time series k-NN DTW
def TS_gesture_kNN_DTW_transform(X_train, X_test):
    return_X_train = []
    for i, gesture in enumerate(X_train):
        X_train_n_trial_instances = len(gesture) #Number of trials in one gesture
        X_train_n_trials_dimension = gesture[0].shape[1] #The dimension of the first trial
        X_train_trial_series_length = gesture[0].shape[0] #Time series length of the first trial
        
        X_train_transformed_df = np.zeros((X_train_n_trial_instances, X_train_n_trials_dimension, X_train_trial_series_length))
        
        for j, trial in enumerate(X_train[i]):
            X_train_transformed_df[j] = trial.values.T
        return_X_train.append(X_train_transformed_df)

    return_X_test = []
    for i, gesture in enumerate(X_test):
        X_test_n_trial_instances = len(gesture)
        X_test_n_trials_dimension = gesture[0].shape[1]
        X_test_trial_series_length = gesture[0].shape[0]
        
        X_test_transformed_df = np.zeros((X_test_n_trial_instances, X_test_n_trials_dimension, X_test_trial_series_length))
        
        for j, trial in enumerate(X_test[i]):
            X_test_transformed_df[j] = trial.values.T
        return_X_test.append(X_test_transformed_df)
    
    return return_X_train, return_X_test

#Gestures: transformes the data into an approppriate format for time series k-NN DTW
def TS_gesture_NN_transform(X_train, X_test):
    return_X_train = []
    for i, gesture in enumerate(X_train):
        X_train_n_trial_instances = len(gesture) #Number of trials in one gesture
        X_train_n_trials_dimension = gesture[0].shape[1] #The dimension of the first trial
        X_train_trial_series_length = gesture[0].shape[0] #Time series length of the first trial
        
        X_train_transformed_df = np.zeros((X_train_n_trial_instances, X_train_trial_series_length, X_train_n_trials_dimension))
        
        for j, trial in enumerate(X_train[i]):
            X_train_transformed_df[j] = trial.values
        return_X_train.append(X_train_transformed_df)

    return_X_test = []
    for i, gesture in enumerate(X_test):
        X_test_n_trial_instances = len(gesture)
        X_test_n_trials_dimension = gesture[0].shape[1]
        X_test_trial_series_length = gesture[0].shape[0]
        
        X_test_transformed_df = np.zeros((X_test_n_trial_instances, X_test_trial_series_length, X_test_n_trials_dimension))
        
        for j, trial in enumerate(X_test[i]):
            X_test_transformed_df[j] = trial.values
        return_X_test.append(X_test_transformed_df)
    
    return return_X_train, return_X_test


#Returns the target variable (gesture)
def TS_NN_gestures_Y(gestures_class):
    gestures_Y = []

    for i in range(len(gestures_class)):
        Y_binary = TS_Y(gestures_class[i], 2)
        gestures_Y.append(np.array(Y_binary))
        
    return gestures_Y
    
        
#Returns the best scores from hyperparameter tuning
import copy
from pandas import option_context 
def Best_hyperparameter_tuning_scores(score, estimator, cvs, test_size, class_alg):
    return_errors = []
    
    if len(score[0][0]) != len(test_size):
        return_errors.append("The specified test sizes do not mach the number of scores.")
    
    if len(return_errors) == 0:
        deep_copy_score = copy.deepcopy(score)
        deep_copy_estimator = copy.deepcopy(estimator)
        return_score = []

        for k in range(len(test_size)):
            maxi = []
            best_estimator = []
            best_cv = []

            for j in range(len(deep_copy_score[0])):
                temp_score = []
                temp_estimator = []

                for i in range(len(deep_copy_score)):
                    if deep_copy_score[i] != ["No measurable value"] * len(test_size): #The values for one cv can be No measurable value because of the high cv value
                        if deep_copy_score[i][j][k] != "No measurable value": # The values for one class can be No measurable value because of the high cv value
                            temp_score.append(deep_copy_score[i][j][k])
                            temp_estimator.append(deep_copy_estimator[i][j][k])

                if len(temp_score) != 0: #Do not calculate the max if all of the values are None
                    maximum = max(temp_score)              
                    maxi.append(maximum)

                    best_estimator.append(temp_estimator[temp_score.index(maximum)])
                    best_cv.append(cvs[temp_score.index(maximum)])
            
            if len(best_cv) != 0: #Do not print the results of if all of the values are None
                df = pd.DataFrame({'Classification algorithm': class_alg, 'Best test score': maxi, 'Best estimator': best_estimator, 'Best cv': best_cv})

                print(f"Test size: {test_size[k]}")

                with option_context('display.max_colwidth', 200):
                        display(df)
                        
            return_score.append(maxi)
        return return_score
            
    else:
        return return_errors
    
    
#Returns the best scores from dimred + hyperparameter tuning
import copy
from pandas import option_context 
def Best_dimRed_hyperparameter_tuning_scores(score, estimator, feature, threshold, cvs, test_size, class_alg):
    return_errors = []
    
    if len(score[0][0]) != len(test_size):
        return_errors.append("The specified test sizes do not mach the number of scores.")
    
    if len(return_errors) == 0:
        deep_copy_score = copy.deepcopy(score)
        deep_copy_estimator = copy.deepcopy(estimator)
        deep_copy_feature = copy.deepcopy(feature)
        deep_copy_threshold = copy.deepcopy(threshold)

        for k in range(len(test_size)):
            best_scores, best_estimators, best_features, best_thresholds = [], [], [], []
            best_cv = []

            for j in range(len(deep_copy_score[0])):
                temp_score, temp_estimator, temp_feature, temp_threshold = [], [], [], []

                for i in range(len(deep_copy_score)):
                    if deep_copy_score[i] != ["No measurable value"] * len(test_size): #The values for one cv can be No measurable value because of the high cv value
                        if deep_copy_score[i][j][k] != "No measurable value": # The values for one class can be No measurable value because of the high cv value
                            temp_score.append(deep_copy_score[i][j][k])
                            temp_estimator.append(deep_copy_estimator[i][j][k])
                            temp_feature.append(deep_copy_feature[i][j][k])
                            temp_threshold.append(deep_copy_threshold[i][j][k])

                if len(temp_score) != 0: #Do not calculate the max if all of the values are "No measurable value"
                    maximum = max(temp_score)   
                    
                    best_scores.append(maximum)
                    best_estimators.append(temp_estimator[temp_score.index(maximum)])
                    best_features.append(temp_feature[temp_score.index(maximum)])
                    best_thresholds.append(temp_threshold[temp_score.index(maximum)])
                    best_cv.append(cvs[temp_score.index(maximum)])

            if len(best_cv) != 0: #Do not print the results of if all of the values are "No measurable value"
                df = pd.DataFrame({'Classification algorithm': class_alg, 'Best test score': best_scores, 'Best estimator': best_estimators, \
                                   'Features': best_features, 'MIC_threshold': best_thresholds, 'Best cv': best_cv})

                print(f"Test size: {test_size[k]}")

                with option_context('display.max_colwidth', 200):
                        display(df)
    else:
        return return_errors

    
def Best_PCA_dimRed_hyperparameter_tuning_scores(score, estimator, feature, cvs, test_size, class_alg):
    return_errors = []
    
    if len(score[0][0]) != len(test_size):
        return_errors.append("The specified test sizes do not mach the number of scores.")
    
    if len(return_errors) == 0:
        deep_copy_score = copy.deepcopy(score)
        deep_copy_estimator = copy.deepcopy(estimator)
        deep_copy_feature = copy.deepcopy(feature)

        for k in range(len(test_size)):
            best_scores, best_estimators, best_features = [], [], []
            best_cv = []

            for j in range(len(deep_copy_score[0])):
                temp_score, temp_estimator, temp_feature = [], [], []

                for i in range(len(deep_copy_score)):
                    if deep_copy_score[i] != ["No measurable value"] * len(test_size): #The values for one cv can be No measurable value because of the high cv value
                        if deep_copy_score[i][j][k] != "No measurable value": # The values for one class can be No measurable value because of the high cv value
                            temp_score.append(deep_copy_score[i][j][k])
                            temp_estimator.append(deep_copy_estimator[i][j][k])
                            temp_feature.append(deep_copy_feature[i][j][k])

                if len(temp_score) != 0: #Do not calculate the max if all of the values are "No measurable value"
                    maximum = max(temp_score)   
                    
                    best_scores.append(maximum)
                    best_estimators.append(temp_estimator[temp_score.index(maximum)])
                    best_features.append(temp_feature[temp_score.index(maximum)])
                    best_cv.append(cvs[temp_score.index(maximum)])

            if len(best_cv) != 0: #Do not print the results of if all of the values are "No measurable value"
                df = pd.DataFrame({'Classification algorithm': class_alg, 'Best test score': best_scores, 'Best estimator': best_estimators, \
                                   'Features': best_features, 'Best cv': best_cv})

                print(f"Test size: {test_size[k]}")

                with option_context('display.max_colwidth', 200):
                        display(df)
    else:
        return return_errors

#Select tha data for that specific gesture
def TS_PCA_gesture_df_list_selection(df_list, gesture):
    return_errors = []
    
    number_of_gestures = len(df_list[0])
    if(gesture < 0 or gesture > number_of_gestures - 1):
        return_errors.append("Invalid gestures")
        
    if(len(return_errors) == 0):
        gesture_list = []
        for i in range(len(df_list)):
            if(df_list[i][gesture].empty == False):
                gesture_list.append(df_list[i][gesture])
                
        return gesture_list
    else:
        print(return_errors)
        return None