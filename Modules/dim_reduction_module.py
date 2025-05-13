from sklearn.feature_selection import mutual_info_classif as MIC
from Modules import data_manipulation_module as dmm
from Modules import param_tuning_module as ptm
from Modules import validation_and_results_module as vrm
import numpy as np
import pandas as pd
from pandas import option_context
import copy
def MIC_dimRed_and_hyperparameter_tuning(models, metric, X, Y, test_sizes, cv, MIC_step, feature_names, standardize, calculate_maximum_neighbors, show_results, n_jobs):
    #Parameter check
    if metric not in ["accuracy", "recall_micro", "recall_macro", "recall_weighted", "precision_micro", "precision_macro", "precision_weighted", "f1_micro", "f1_macro", "f1_weighted"]: #Not valid metric
        print("Invalid metric value!")
        return None, None, None, None
    if standardize not in ["Non-time", "No"]: #Not valid standardize
        print("Invalid standardize value!")
        return None, None, None, None
    
    #Hyperparameter tuning
    best_estimators, scores, best_features, best_thresholds = [], [], [], []
    name_of_the_models = list(models.keys())
    
    for test_size in test_sizes:
        print(f"Test size ({test_size}):", end=" ")
        
        #Train-test split
        X_train, X_test, Y_train, Y_test = dmm.Train_test(X, Y, test_size) # This can be used for the full data and the gesture divided data as well, beacuse the user can only provide one gesture data
    
        #Standardization if needed
        if standardize == "Non-time": 
            X_train, X_test = dmm.Standardization(X_train, X_test)
        
        #Y_test validity check
        if len(np.unique(Y_test)) != len(np.unique(Y)):
            print(f"The test set is too small. The Y_test does not contain all of the classes. The classes in Y_test: {np.unique(Y_test)}. The required classes: {np.unique(Y)}")
            
            scores.append(["No measurable value"] * len(models)) # The models could have different size
            best_estimators.append(["No measurable value"] * len(models))
            best_features.append(["No measurable value"] * len(models))
            best_thresholds.append(["No measurable value"] * len(models))
            continue #Go to the next iteration of test_sizes
                    
        #CV check
        check_result = ptm.CV_check(Y_train, cv)
        if check_result == "OK":
            models_temp = copy.deepcopy(models) #Deepcopy to not overrite the original one
            
            for model_name, model_params in models_temp.items():
                if model_name == "k-NN":
                    if calculate_maximum_neighbors == True: #switching to calculating with all possible neighbors instead of the user-defined neighbor range
                        max_neighbors = ptm.Max_neighbors(Y_train, cv)
                        print(f"The n_neighbors parameter value has been modified. The new maximum value: {max_neighbors}")
                        models_temp[model_name]['params']['n_neighbors'] = range(1, max_neighbors + 1)

                if model_name == "logistic_regression": #only using the usable solver-penalty pairs
                    models_temp[model_name] = {
                        'model': model_params['model'],
                        'params': ptm.LogReg_parameter_validation(model_params['params'])
                        }
            
            #MIC
            mi_score = MIC(X_train, Y_train, random_state = 0) #Random_state for reproducibility
            mini = min(mi_score)
            maxi = max(mi_score)
            
            dup_list = []
            best_threshold, best_feature = [], []
            best_estimator, best_validation_score, score = None, None, None
            
            # Iterating through each threshold
            for threshold in np.arange(mini, maxi, MIC_step):
                # Selecting features based on the thresholds
                selected_features = list(np.where(mi_score >= threshold)[0])

                if selected_features not in dup_list: #To not calculate with the already used features
                    dup_list.append(selected_features)
                    
                    #Only the selected fetaures are remaining in the X_train, and X_test
                    X_train_new_features = X_train[:, selected_features]
                    X_test_new_features = X_test[:, selected_features]
                
                    temp_best_estimator, temp_best_validation_score = ptm.GridSearch(models_temp, X_train_new_features, Y_train, cv, n_jobs, metric)
                    temp_score = vrm.Best_model_evaluation(temp_best_estimator, metric, X_test_new_features, Y_test)

                    if best_estimator is None: # Only for the first run
                        best_estimator = temp_best_estimator
                        best_validation_score = temp_best_validation_score
                        score = temp_score
                        
                        for k in range(len(models)):
                            best_feature.append(selected_features)
                            best_threshold.append(threshold)
                    else:
                        for j in range(len(temp_score)): #Selecting the best test scores
                            if(temp_score[j] > score[j]):
                                best_estimator[j] = temp_best_estimator[j]
                                best_validation_score[j] = temp_best_validation_score[j]
                                score[j] = temp_score[j]

                                best_feature[j] = selected_features
                                best_threshold[j] = threshold

            best_estimators.append(best_estimator)
            scores.append(score)
            best_features.append(best_feature)
            best_thresholds.append(best_threshold)
            
            if show_results == True:              
                data = {
                    'Model': name_of_the_models,
                    'Validation score': best_validation_score,
                    'Test score': score,
                    'Parameter': best_estimator,
                    'Features': best_feature,
                    'MIC_threshold': best_threshold
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
            best_features.append(["No measurable value"] * len(models))
            best_thresholds.append(["No measurable value"] * len(models))
            
    
    #Transforms the data for the bar plot
    return_scores, return_estimators, return_features, return_thresholds = [], [], [], []
    num_elements = len(scores[0])

    for idx in range(num_elements):
        score_column = [sublist[idx] for sublist in scores]
        estimator_column = [sublist[idx] for sublist in best_estimators]
        feature_column = [sublist[idx] for sublist in best_features]
        thresholds_column = [sublist[idx] for sublist in best_thresholds]

        return_scores.append(score_column)
        return_estimators.append(estimator_column)
        return_features.append(feature_column) 
        return_thresholds.append(thresholds_column) 

    return return_scores, return_estimators, return_features, return_thresholds


#Hyperparameter tuning with multiple cv
def Multiple_cv_MIC_dimRed_and_hyperparameter_tuning(cv_start, cv_end, models, metric, X, Y, test_sizes, MIC_step, feature_names, standardize, calculate_maximum_neighbors, show_results, n_jobs):    
    if metric not in ["accuracy", "recall_micro", "recall_macro", "recall_weighted", "precision_micro", "precision_macro", "precision_weighted", "f1_micro", "f1_macro", "f1_weighted"]:
        print("Invalid metric value!")
        return None, None, None, None, None
        
    if cv_end < cv_start:
        print(f"Incorrect interval! cv_start: {cv_start} > cv_end: {cv_end}!")
        return None, None, None, None, None
    
    cvs = range(cv_start, cv_end + 1)
    scores, estimators, features, thresholds = [], [], [], []
    
    for cv in cvs:
        print(f"CV: {cv}")
        score, estimator, feature, threshold = MIC_dimRed_and_hyperparameter_tuning(models, metric, X, Y, test_sizes, cv, MIC_step, feature_names, standardize, calculate_maximum_neighbors, show_results, n_jobs)
        
        scores.append(score)
        estimators.append(estimator)
        features.append(feature)
        thresholds.append(threshold)
        
    return scores, estimators, features, thresholds, cvs


#PCA dimensionality reduction and hyperparameter tuning
from sklearn.decomposition import PCA
def TS_PCA_dimRed_and_hyperparameter_tuning(models, metric, X, Y, test_sizes, cv, component, feature_names, standardize, calculate_maximum_neighbors, show_results, n_jobs):
    #Parameter check
    if metric not in ["accuracy", "recall_micro", "recall_macro", "recall_weighted", "precision_micro", "precision_macro", "precision_weighted", "f1_micro", "f1_macro", "f1_weighted"]: #Not valid metric
        print("Invalid metric value!")
        return None, None, None
    if standardize not in ["Time", "No"]: #Not valid standardize
        print("Invalid standardize value!")
        return None, None, None
    
    best_estimators, scores, best_features = [], [], []
    name_of_the_models = list(models.keys())

    for test_size in test_sizes:
        print(f"Test size ({test_size}):", end=" ")
        
        #Train-test split
        X_train, X_test, Y_train, Y_test = dmm.Train_test(X, Y, test_size) # This can be used for the full data and the gesture divided data as well, beacuse the user can only provide one gesture data
    
        #Standardization if needed
        if standardize == "Time":
            X_train, X_test = dmm.TS_Standardization(X_train, X_test, feature_names)
        
        #Y_test validity check
        if len(np.unique(Y_test)) != len(np.unique(Y)):
            print(f"The test set is too small. The Y_test does not contain all of the classes. The classes in Y_test: {np.unique(Y_test)}. The required classes: {np.unique(Y)}")
            
            scores.append(["No measurable value"] * len(models)) # The models could have different size
            best_estimators.append(["No measurable value"] * len(models))
            best_features.append(["No measurable value"] * len(models))
            continue #Go to the next iteration of test_sizes
                    
        #CV check
        check_result = ptm.CV_check(Y_train, cv)
        if check_result == "OK":
            models_temp = copy.deepcopy(models) #Deepcopy to not overrite the original one
            
            for model_name, model_params in models_temp.items():
                if model_name == "Time_series_kNN":
                    if calculate_maximum_neighbors == True: #switching to calculating with all possible neighbors instead of the user-defined neighbor range
                        max_neighbors = ptm.Max_neighbors(Y_train, cv)
                        print(f"The n_neighbors parameter value has been modified. The new maximum value: {max_neighbors}")
                        models_temp[model_name]['params']['n_neighbors'] = range(1, max_neighbors + 1)

            #PCA
            df_list = []
            for i in range(len(X_train)): #Transforming a data format into dataframe
                df_list.append(pd.DataFrame(X_train[i], columns = feature_names))
                
            df_concat = pd.concat(df_list, axis=0, ignore_index=True)
            
            pca = PCA(n_components = component)
            pca.fit(df_concat)
            
            """
            print("Components: " + str(pca.n_components_) + "\n") #the number of components to keep after applying PCA
            with pd.option_context('display.max_columns', None):  #Dislay all column of the dataframe
                display(pd.DataFrame(abs(pca.components_), columns = feature_names))
            print("Feature selection:")
            """
            
            most_important_indexes = []
            components = pca.components_
            
            for i in range(len(components)):
                most_important_feature = np.abs(components[i]).argmax() #Argmax: index where the maximum occurs
                most_important_feature_row = np.abs(components[i]).copy()
                
                while(most_important_feature in most_important_indexes): #If a feature already exists in the most_important list, it should not be repeated, it is possible that PCA selects the same feature twice
                    #print("Row: " + str(i))
                    #feature_dup = most_important_feature
                    most_important_feature_row[most_important_feature] = 0
                    most_important_feature = most_important_feature_row.argmax() #Next most important feature selection
                    #print("New best feature: " + str(feature_dup) + " --> "+ str(most_important_feature))
                    
                most_important_indexes.append(most_important_feature)  
                
            most_important_indexes = list(np.sort(most_important_indexes))
            
            #Selecting only the most important features
            X_train_new_features, X_test_new_features = [], []
            for i in range(len(X_train)):
                X_train_new_features.append(X_train[i].iloc[:, most_important_indexes])
            
            for j in range(len(X_test)):
                X_test_new_features.append(X_test[j].iloc[:, most_important_indexes])
            # End of PCA 
            
            X_train_new_features, X_test_new_features = dmm.TS_kNN_DTW_transform(X_train_new_features, X_test_new_features)
            
            best_estimator, best_validation_score = ptm.GridSearch(models_temp, X_train_new_features, Y_train, cv, n_jobs, metric)
            score = vrm.Best_model_evaluation(best_estimator, metric, X_test_new_features, Y_test)
            
            best_estimators.append(best_estimator)
            scores.append(score)
            best_features.append([most_important_indexes])

            #Results
            if(show_results == True):
                data = {
                    'Model': name_of_the_models,
                    'Validation score': best_validation_score,
                    'Test score': score,
                    'Parameter': best_estimator,
                    'Features': [most_important_indexes] #Lists needs to be in a same size
                }

                print_df_acc = pd.DataFrame(data)
                with option_context('display.max_colwidth', 200):
                    display(print_df_acc)

                #Do not add space after the last test_size
                if(test_sizes[-1] != test_size):
                    print("\n")
        else:
            scores.append(["No measurable value"] * len(models))
            best_estimators.append(["No measurable value"] * len(models))
            best_features.append(["No measurable value"] * len(models))          

    #Transforms the data for the bar plot
    return_scores, return_estimators, return_features = [], [], []
    num_elements = len(scores[0])
    
    for idx in range(num_elements):
        score_column = [sublist[idx] for sublist in scores]
        estimator_column = [sublist[idx] for sublist in best_estimators]
        feature_column = [sublist[idx] for sublist in best_features]

        return_scores.append(score_column)
        return_estimators.append(estimator_column)
        return_features.append(feature_column)
        
    return return_scores, return_estimators, return_features


#PCA dimensionality reduction and hyperparameter tuning with multiple cv
def TS_Multiple_cv_PCA_dimRed_and_hyperparameter_tuning(cv_start, cv_end, models, metric, X, Y, test_sizes, component, feature_names, standardize, calculate_maximum_neighbors, show_results, n_jobs):    
    if metric not in ["accuracy", "recall_micro", "recall_macro", "recall_weighted", "precision_micro", "precision_macro", "precision_weighted", "f1_micro", "f1_macro", "f1_weighted"]:
        print("Invalid metric value!")
        return None, None, None, None, None
        
    if cv_end < cv_start:
        print(f"Incorrect interval! cv_start: {cv_start} > cv_end: {cv_end}!")
        return None, None, None, None, None
    
    cvs = range(cv_start, cv_end + 1)
    scores, estimators, features, thresholds = [], [], [], []
    
    for cv in cvs:
        print(f"CV: {cv}")
        score, estimator, feature = TS_PCA_dimRed_and_hyperparameter_tuning(models, metric, X, Y, test_sizes, cv, component, feature_names, standardize, calculate_maximum_neighbors, show_results, n_jobs)
        
        scores.append(score)
        estimators.append(estimator)
        features.append(feature)
        
    return scores, estimators, features, cvs
