#Counts the target variables (0, 1, 2 = classes)
import numpy as np        
def Least_populated_class_calculation(Y):
    if len(np.unique(Y)) == 1: #Not enough class
        return None
    if len(np.unique(Y)) == 2: #Two unique class
        zero_count = 0
        one_count = 0

        for i in Y:
            if i == 0:
                zero_count = zero_count + 1
            else:
                one_count = one_count + 1

        if zero_count < one_count:
            return zero_count
        else:
            return one_count
    elif len(np.unique(Y)) == 3: #Three unique class
        zero_count = 0
        one_count = 0
        two_count = 0

        for i in Y:
            if i == 0:
                zero_count = zero_count + 1
            elif i == 1:
                one_count = one_count + 1
            else:
                two_count = two_count + 1

        if zero_count < one_count and zero_count < two_count:
            return zero_count
        elif one_count < zero_count and one_count < two_count:
            return one_count
        else:
            return two_count


#Checks the cv value
def CV_check(Y, cv):
    if isinstance(cv, int):
        least_populated_class_members = Least_populated_class_calculation(Y)
        if least_populated_class_members != None: #Enough class
            if cv > least_populated_class_members:
                if least_populated_class_members < 2:
                    print("The dataset lacks sufficient data to perform even a 2-fold cross-validation.")
                    return "not OK"
                else:
                    print("The specified cv value is too big! The maximum cv value: " + str(least_populated_class_members))
                    return "not OK"
            elif cv < 2:
                print("The specified cv value is too small! The minimum cv value: 2")
                return "not OK"
            else:
                return "OK"
        else: #Not enough class
            print("Y_train contains only a single unique class, which is not suitable for classification.")
            return "not OK"
    else:
        return "OK"
        
        
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_score
import seaborn as sn
#GridSearchCV
def GridSearch(models, X, Y, cv, n_jobs, metric):
    estimators = []
    validation_accs = []
    
    #n_jobs = -1 --> Utilizing all available CPU resources.
    for model_name, model_params in models.items():
        grid =  GridSearchCV(model_params['model'], model_params['params'], cv = cv, return_train_score = False, n_jobs = n_jobs, scoring = metric)
        grid.fit(X, Y)
        
        estimator = grid.best_estimator_
        estimators.append(estimator)
        
        validation_acc = grid.best_score_
        validation_accs.append(validation_acc) 
        
    return estimators, validation_accs


#Calculates the maximum number of neighbors for k-NN
def Max_neighbors(Y, cv):
    if isinstance(cv, int):
        #maximum number of neighbors: 
            #Y size / cv = value
            #if the value is an integer -> Y size - value = max_neighbors
            #if the value not an integer -> Y size - (value + 1) = max_neighbors
        length = len(Y)
        max_neighbors_temp = length / cv

        if (max_neighbors_temp - int(max_neighbors_temp)) == 0.0:
            max_neighbors = int(length - max_neighbors_temp)
        else:
            max_neighbors = length - (int(max_neighbors_temp) + 1)
                
        return max_neighbors
    else:
        #For LOOCV (Leave-One-Out cross-validation), the maximum number of neighbors is (Y size - 1).
        max_neighbors = len(Y) - 1
        return max_neighbors
    
LogReg_solver_penalty_pairs = {
    'newton-cg': {
        'solver': ['newton-cg'],
        'penalty': ['l2']
    },
    'lbfgs': {
        'solver': ['lbfgs'],
        'penalty': ['l2']
    },
    'liblinear': {
        'solver': ['liblinear'],
        'penalty': ['l1', 'l2']
    },
    'sag': {
        'solver': ['sag'],
        'penalty': ['l2']
    },
    'saga': {
        'solver': ['saga'],
        'penalty': ['l1', 'l2']
    }
}

#Parameter validation for Logistic Regression (solver-penalty)
def LogReg_parameter_validation(model_params):
    valid_params = []
    solvers = model_params['solver']
    penalties = model_params['penalty'] 
    
    for solver in solvers:
        if solver in LogReg_solver_penalty_pairs:
            valid_penalties = [penalty for penalty in penalties if penalty in LogReg_solver_penalty_pairs[solver]['penalty']]
            
            for penalty in valid_penalties:
                param_dict = {
                    'solver': [solver],
                    'penalty': [penalty],
                    'C': model_params['C'],
                    'max_iter': model_params['max_iter'],
                    'random_state': model_params['random_state']
                }
                valid_params.append(param_dict)
    
    return valid_params


#Hiperparameter tuning with one cv
from Modules import data_manipulation_module as dmm
from Modules import validation_and_results_module as vrm
from pandas import option_context
import copy
def Hyperparameter_tuning(models, metric, X, Y, test_sizes, cv, feature_names, standardize, calculate_maximum_neighbors, show_results, n_jobs):
    #Parameter check
    if metric not in ["accuracy", "recall_micro", "recall_macro", "recall_weighted", "precision_micro", "precision_macro", "precision_weighted", "f1_micro", "f1_macro", "f1_weighted"]: #Not valid metric
        print("Invalid metric value!")
        return None, None
    if standardize not in ["Non-time", "Time", "No"]: #Not valid standardize
        print("Invalid standardize value!")
        return None, None
    
    #Hyperparameter tuning
    best_estimators = []
    name_of_the_models = list(models.keys())
    scores = []
 
    for test_size in test_sizes:
        print(f"Test size ({test_size}):", end=" ")
        
        #Train-test split
        X_train, X_test, Y_train, Y_test = dmm.Train_test(X, Y, test_size) # This can be used for the full data and the gesture divided data as well, beacuse the user can only provide one gesture data
    
        #Standardization if needed
        # The standardization works different if it is working on time or non-time series data
        if standardize == "Non-time": 
            X_train, X_test = dmm.Standardization(X_train, X_test)
        elif standardize == "Time":
            X_train, X_test = dmm.TS_Standardization(X_train, X_test, feature_names)
        
        #Y_test validity check
        if len(np.unique(Y_test)) != len(np.unique(Y)):
            print(f"The test set is too small. The Y_test does not contain all of the classes. The classes in Y_test: {np.unique(Y_test)}. The required classes: {np.unique(Y)}")
            
            scores.append(["No measurable value"] * len(models)) # The models could have different size
            best_estimators.append(["No measurable value"] * len(models))
            continue #Go to the next iteration of test_sizes
                    
        #CV check
        check_result = CV_check(Y_train, cv)
        if check_result == "OK":
            models_temp = copy.deepcopy(models) #Deepcopy to not overrite the original one
            
            for model_name, model_params in models_temp.items():
                if model_name == "k-NN" or model_name == "Time_series_kNN":
                    if calculate_maximum_neighbors == True: #switching to calculating with all possible neighbors instead of the user-defined neighbor range
                        max_neighbors = Max_neighbors(Y_train, cv)
                        print(f"The n_neighbors parameter value has been modified. The new maximum value: {max_neighbors}")
                        models_temp[model_name]['params']['n_neighbors'] = range(1, max_neighbors + 1)
                
                if model_name == "Time_series_kNN": #Transforming the shape of the time series data
                    X_train, X_test = dmm.TS_kNN_DTW_transform(X_train, X_test)

                if model_name == "logistic_regression": #only using the usable solver-penalty pairs
                    models_temp[model_name] = {
                        'model': model_params['model'],
                        'params': LogReg_parameter_validation(model_params['params'])
                        }

            best_estimator, best_validation_score = GridSearch(models_temp, X_train, Y_train, cv, n_jobs, metric)
            best_estimators.append(best_estimator)

            score = vrm.Best_model_evaluation(best_estimator, metric, X_test, Y_test)
            scores.append(score)
            
            if show_results == True:              
                data = {
                    'Model': name_of_the_models,
                    'Validation score': best_validation_score,
                    'Test score': score,
                    'Parameter': best_estimator
                }

                print_df_score = pd.DataFrame(data)
                with option_context('display.max_colwidth', 200):
                    display(print_df_score)

                #Do not add space after the last test_size
                if test_sizes[-1] != test_size:
                    print("\n")
        else:
            scores.append(["No measurable value"] * len(models))
            best_estimators.append(["No measurable value"] * len(models))

    
    #Transforms the data for the bar plot
    return_scores = []
    return_estimators = []
    num_elements = len(scores[0])

    for idx in range(num_elements):
        score_column = [sublist[idx] for sublist in scores]
        estimator_column = [sublist[idx] for sublist in best_estimators]

        return_scores.append(score_column)
        return_estimators.append(estimator_column) 

    return return_scores, return_estimators


#Hyperparameter tuning with multiple cv
def Multiple_cv_hyperparameter_tuning(cv_start, cv_end, models, metric, X, Y, test_sizes, feature_names, standardize, calculate_maximum_neighbors, show_results, n_jobs):    
    if metric not in ["accuracy", "recall_micro", "recall_macro", "recall_weighted", "precision_micro", "precision_macro", "precision_weighted", "f1_micro", "f1_macro", "f1_weighted"]:
        print("Invalid metric value!")
        return None, None, None
        
    if cv_end < cv_start:
        print(f"Incorrect interval! cv_start: {cv_start} > cv_end: {cv_end}!")
        return None, None, None
    
    cvs = range(cv_start, cv_end + 1)
    scores = []
    estimators = []
    
    for cv in cvs:
        print(f"CV: {cv}")
        score, estimator = Hyperparameter_tuning(models, metric, X, Y, test_sizes, cv, feature_names, standardize, calculate_maximum_neighbors, show_results, n_jobs)
        scores.append(score)
        estimators.append(estimator)
        
    return scores, estimators, cvs
        
#Hyperparameter tuning for NN
from Modules import ts_classification_algorithms_module as ts_cam
from tensorflow import keras
import tensorflow as tf
def Hyperparameter_tuning_NN(X, Y, param_grid, epoch, batch_size, num_class, metric, validation_split, standardize, feature_names):
    if standardize not in ["Time", "No"]: #Not valid standardize
        print("Invalid standardize value!")
        return None
    
    tf.random.set_seed(0) # for reproducibility
    results = []
    
    X_train, X_test, Y_train, Y_test = dmm.Train_test(X, Y, 0.3)
    
    if standardize == "Time":
        X_train, X_test = dmm.TS_Standardization(X_train, X_test, feature_names)
        
    NN_X_train, NN_X_test = dmm.TS_NN_transform(X_train, X_test)
    
    for idx, (activation, dropout, filters, kernel) in enumerate(param_grid):
        print(f"Futtatás {idx+1}/{len(param_grid)}: {activation}, {dropout}, {filters}, {kernel}")

        folder = f"/tf/notebooks/SCARS/NN/grid_search/run_{idx+1}_{activation}_{dropout}_{filters}_{kernel}"

        ts_cam.TS_NN_model(
            folder_route = folder,
            epoch = epoch,
            batch_size = batch_size,
            activation_function = activation,
            drop_out = dropout,
            filters = filters,
            kernel = kernel,
            num_classes = num_class,
            X_train = NN_X_train,
            Y_train = Y_train,
            metric = metric,
            validation_split = validation_split
        )
        
        model = keras.models.load_model(str(folder) + "/" + str(epoch) + ".h5")
        test_loss, test_score = model.evaluate(NN_X_test, Y_test)
        
        results.append({
            "activation": activation,
            "dropout": dropout,
            "filters": filters,
            "kernel": kernel,
            "test_acc": test_score
        })
        
    return pd.DataFrame(results)