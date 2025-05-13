#Calculates the accuracy, precision, recall for the model
from sklearn.metrics import classification_report
import warnings
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
def Model_classification_report(model, X_train, X_test, Y_train, Y_test):
    #Modifies the warning handling mechanism to enable catching warnings within the except block
    warnings.simplefilter('error', UndefinedMetricWarning)
    warnings.simplefilter('error', ConvergenceWarning)
    
    try:
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        
        print(f"Actual classes: {Y_test}, Predicted classes: {y_pred}")
        print()
        print('                 Classification Report')
        print('-------------------------------------------------------')
        print(classification_report(Y_test, y_pred, zero_division=0))
        
    except UndefinedMetricWarning as UMWarning:
        print("UndefinedMetricWarning:", UMWarning)
    except ConvergenceWarning as CWarning:
        print("ConvergenceWarning:", CWarning)
    except ValueError as VError:
        print("ValueError:", VError)
    

#Cross-validation (K-fold, or Leave-One-Out)
from Modules import param_tuning_module as ptm
from sklearn.model_selection import LeaveOneOut, cross_val_score
def Cross_validation(model, X_train, Y_train, cv):
    if isinstance(cv, int):
        if cv >= 2:
            least_populated_class_members = ptm.Least_populated_class_calculation(Y_train)
            if least_populated_class_members != None: #Enough class
                if cv > least_populated_class_members:   
                    return f"The specified cv value is too big! The maximum value: {least_populated_class_members}"

                kf_score = cross_val_score(model, X_train, Y_train, cv = cv)
                kf_score_mean = kf_score.mean()
                return kf_score_mean
            else: #Not enough class
                return "Y_train contains only a single unique class, which is not suitable for classification."
        else:
            return "The specified cv value is too small! The minimum value: 2"
    elif cv == "LOOCV":
        loocv_score = cross_val_score(model, X_train, Y_train, cv = LeaveOneOut())
        loocv_score_mean = loocv_score.mean()
        return loocv_score_mean
    else:
        return "Invalid cross-validation value"
    
    
#Explaines the cross-validation
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
def Explained_cross_validation(model, X_train, Y_train, cv):
    X_train_temp = []
    Y_train_temp = []
    X_validation = []
    Y_validation = []
        
    if isinstance(cv, int):
        if cv >= 2:
            least_populated_class_members = ptm.Least_populated_class_calculation(Y_train)
            if least_populated_class_members != None: #Enough data
                if cv > least_populated_class_members:
                    return f"The specified cv value is too big! The maximum cv value: {least_populated_class_members}"

                skf = StratifiedKFold(n_splits=cv)

                for train_idx, validation_idx in skf.split(X_train, Y_train):
                    print(f"Train: {train_idx}  \t Validation: {validation_idx}")
                    X_train_temp.append(X_train[train_idx])
                    Y_train_temp.append(Y_train[train_idx])
                    X_validation.append(X_train[validation_idx])
                    Y_validation.append(Y_train[validation_idx])
            else: #Not enough data
                return "The dataset lacks sufficient data to perform cross-validation."
        else:
            return "The specified cv value is too small! The minimum value: 2"
    
    elif cv == "LOOCV":
        kf = KFold(n_splits=len(X_train))
        for train_idx, validation_idx in kf.split(X_train):
            print(f"Train: {train_idx}   Validation: {validation_idx}")
            X_train_temp.append(X_train[train_idx])
            Y_train_temp.append(Y_train[train_idx])
            X_validation.append(X_train[validation_idx])
            Y_validation.append(Y_train[validation_idx])
        
    else:
        return "Invalid cross-validation value"
    
    accuracy = []
    for i in range(len(X_train_temp)):
        model.fit(X_train_temp[i], Y_train_temp[i])
        accuracy.append(model.score(X_validation[i], Y_validation[i]))
            
    print()
    print(f"Accuracies: {accuracy}")
    print(f"Mean accuracy: {np.mean(accuracy)}")
    
    
##Calculates the scores of the NN
from tensorflow import keras
import os
def TS_NN_evaluate(folder_route, epoch, X_test, Y_test):
    model = keras.models.load_model(str(folder_route) + "/" + str(epoch) + ".h5")
    
    test_loss, test_score = model.evaluate(X_test, Y_test)
    
    return test_loss, test_score
    
    
#Evaluates the best models (param_tuning)
from sklearn.metrics import accuracy_score
from sklearn import metrics as m
def Best_model_evaluation(estimators, metric, X_test, Y_test): 
    scores = [] #Only one metric can be used at a time, so one list is enough
        
    metrics = {
            "accuracy": {
                "func": lambda: m.accuracy_score(Y_test, y_pred)
            },
            "recall": {
                "func": lambda: m.recall_score(Y_test, y_pred, average = metric_type, zero_division = 0)
            },
            "precision": { 
                "func": lambda: m.precision_score(Y_test, y_pred, average = metric_type, zero_division = 0)
            },
            "f1": {
                 "func": lambda: m.f1_score(Y_test, y_pred, average = metric_type, zero_division = 0)
            }
        }
    
    if metric != "accuracy":
        metric, metric_type = metric.split("_")

    current_metric = metrics[metric]
    
    for estimator in estimators:
        y_pred = estimator.predict(X_test)
        scores.append(current_metric["func"]())
  
    return scores

