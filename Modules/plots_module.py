#Visualizes the Decision Tree model
import matplotlib.pyplot as plt
from sklearn import tree
def Decision_tree_visualization(class_names, model, feature_names, fig_size):
    fig = plt.figure(figsize=fig_size)
    tree.plot_tree(model, class_names = class_names, feature_names = feature_names, filled = True)
    plt.show()


#Calculates the features imporantance of the Decision Tree model
import numpy as np
def Decision_tree_feature_importance(model, features, bar_color, fig_size, sort, only_not_null, vertical):
    fig = plt.figure(figsize=fig_size)
    features_importance = model.feature_importances_
    
    if sort == True: 
        sorted_indices = features_importance.argsort()[::-1] # Sorts the features importance from greatest to least using the sorted indices
        features = [features[i] for i in sorted_indices] 
        features_importance = features_importance[sorted_indices]
        
    if only_not_null == True:
        features_importance = [imp for imp in features_importance if imp != 0]
        features = [f for f, imp in zip(features, features_importance) if imp != 0]
                
    if vertical == True:
        plt.grid(which='major', linestyle='-', alpha = 0.7, axis = 'y')
        plt.grid(which='minor', linestyle='--', alpha = 0.4, axis = 'y')
        plt.minorticks_on()
        plt.tick_params('x', which='minor', bottom=False, top=False)
        plt.xticks(rotation ='vertical')
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.ylim(0, 1)
        plt.ylabel('Importance')
        plt.xlabel('Features')
        
        sn.barplot(features, features_importance, zorder = 2, color=bar_color)
        
    elif vertical == False:
        plt.grid(which='major', linestyle='-', alpha = 0.7, axis = 'x')
        plt.grid(which='minor', linestyle='--', alpha = 0.4, axis = 'x')
        plt.minorticks_on()
        plt.tick_params('y', which='minor', left=False)
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.xlim(0, 1)
        plt.ylabel('Features')
        plt.xlabel('Importance')

        sn.barplot(features_importance, features, zorder = 2, color=bar_color)


#Creates the confusion matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
def Confusion_matrix(title, model, X_train, X_test, Y_train, Y_test):
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    
    c_matrix = pd.crosstab(Y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

    h_map = sn.heatmap(c_matrix, annot=True)
    plt.title(title, fontsize = 15)
    

#Plots the scores of the kNN model with different neighbors    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from Modules import param_tuning_module as ptm
from sklearn import metrics as m
def kNN_k_values_plot(model, title, color, method, metric, metric_type, X_train, X_test, Y_train, Y_test, fig_size):
    return_errors = []
    maximum_n = 0
    
    #Checks the validity of the parameter values
    if method not in ["train/test", "LOOCV"] and isinstance(method, int) == False: #Type check
        return_errors.append("Invalid method value!")
    
    if metric not in ["accuracy", "precision", "recall", "f1-score"]: #metric check
        return_errors.append("Invalid metric value!")
    if metric_type not in ["macro", "weighted", None]: #metric_type check
        return_errors.append("Invalid metric_type value!")
    if metric in ["precision", "recall", "f1-score"] and metric_type == None:
        return_errors.append("Invalid metric-metric_type pair values!")
    if metric == "accuracy" and metric_type != None:
        return_errors.append("Invalid metric-metric_type pair values!")
        
    if isinstance(method, int): #For cv       
        #Invalid method
        if method < 2:
            return_errors.append("Invalid method value! The specified value is too small! The minimum method value: 2")
        elif method > ptm.Least_populated_class_calculation(Y_train):
            return_errors.append(f"Invalid method value! The specified value is too big! The maximum method value: {ptm.Least_populated_class_calculation(Y_train)}")
        #Valid method
        elif method >= 2:
            length = len(Y_train)
            max_neighbors_temp = length / method #This is float even if the two number in the division is int
            max_neighbors = 0
            
            if((max_neighbors_temp - int(max_neighbors_temp)) == 0.0):
                max_neighbors = int(length - max_neighbors_temp)
            else:
                max_neighbors = length - (int(max_neighbors_temp) + 1)
                
            maximum_n = max_neighbors
            #Invalid n_neighbors
            if model.get_params()['n_neighbors'] > max_neighbors:
                return_errors.append(f"Invalid n_neighbors value! The specified value is too big! The maximum n_neighbors value: {max_neighbors}")
                
    if method == "LOOCV": #for LOOCV
        max_neighbors = len(Y_train) - 1
        maximum_n = max_neighbors
        #Invalid n_neighbors
        if model.get_params()['n_neighbors'] > max_neighbors:
                return_errors.append(f"Invalid n_neighbors value! The specified value is too big! The maximum n_neighbors value: {max_neighbors}")
                
    if model.get_params()['n_neighbors'] > len(Y_train) and method == "train/test": #for train/test split
        return_errors.append(f"Invalid n_neighbors value! The specified value is too big! The maximum n_neighbors value: {len(Y_train)}")

    if len(return_errors) == 0:
        fig = plt.figure(figsize=fig_size)
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        cvs = {
            int: method,
            str: LeaveOneOut()
        }
        
        metrics = {
            "accuracy": {
                "list": accuracies,
                "funcs": [
                    lambda: knn_model.score(X_test, Y_test), #Train/test
                    lambda: cross_val_score(knn_model, X_train, Y_train, cv = cvs[type(method)], scoring = "accuracy", n_jobs=-1).mean(), #cv and LOOCV
                ]
            },
            "precision": {
                "list": precisions, 
                "funcs": [
                    lambda: m.precision_score(Y_test, y_pred, average=metric_type, zero_division=0),
                    lambda: cross_val_score(knn_model, X_train, Y_train, cv = cvs[type(method)], scoring = m.make_scorer(m.precision_score, average = metric_type, zero_division = 0), n_jobs=-1).mean()
                ]
            },
            "recall": {
                "list": recalls, 
                "funcs": [
                    lambda: m.recall_score(Y_test, y_pred, average=metric_type, zero_division=0),
                    lambda: cross_val_score(knn_model, X_train, Y_train, cv = cvs[type(method)], scoring = m.make_scorer(m.recall_score, average = metric_type, zero_division = 0), n_jobs=-1).mean()
                ]
            },
            "f1-score": {
                "list": f1_scores, 
                "funcs": [
                    lambda: m.f1_score(Y_test, y_pred, average=metric_type, zero_division=0),
                    lambda: cross_val_score(knn_model, X_train, Y_train, cv = cvs[type(method)], scoring = m.make_scorer(m.f1_score, average = metric_type, zero_division = 0), n_jobs=-1).mean()
                ]
            }
        }

        current_metric = metrics[metric]
        
        if method == "train/test":
            maximum_n = len(Y_train)          
            for i in range(1, maximum_n+1):
                knn_model = model.set_params(n_neighbors=i)
                knn_model.fit(X_train, Y_train)
                y_pred = knn_model.predict(X_test)
                current_metric["list"].append(current_metric["funcs"][0]())
                
        else: #CV and LOOCV
            for i in range(1,maximum_n+1):
                knn_model = model.set_params(n_neighbors=i)
                current_metric["list"].append(current_metric["funcs"][1]())
                    
        #Plot
        metrics_value = {
            "accuracy": accuracies,
            "precision": precisions,
            "recall": recalls,
            "f1-score": f1_scores
        }
        
        plt.grid(which='major', linestyle='-', alpha = 0.7, axis = 'both')
        plt.grid(which='minor', linestyle='--', alpha = 0.4, axis = 'y')
        plt.minorticks_on()
        plt.tick_params('x', which='minor', bottom=False, top=False)
        
        plt.plot(range(1,maximum_n+1), metrics_value[metric], color = color)
        
        plt.xticks(range(1,maximum_n+1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('Number of neighbors')
        if metric == "accuracy":
            plt.ylabel(metric.capitalize())
        else:
            plt.ylabel(metric.capitalize() + f" ({metric_type})")
        plt.title(title, fontsize = 15)

        plt.show()
    else:
        return return_errors

    
#Creates a bar plot
def Bar_plot(scores, title, x_axis, x_axis_label, y_axis, y_axis_label, legend_labels, colors, width, fig_size): 
    return_errors = []
    
    #if(len(scores) != len(legend_labels)):
    #    return_errors.append("The lengths of the scores and legend labels are not equal.")
    
    if len(return_errors) == 0:
        #Copying the lists to avoid overwriting the original ones
        not_valid_score_idxs = []
        score = copy.deepcopy(scores)
        x_ax = copy.deepcopy(x_axis)

        for sc in score:
            not_valid_score_idx = []
            for i in range(len(sc)):
                if not isinstance(sc[i], (float, int)):
                    not_valid_score_idx.append(i)

            for idx in reversed(not_valid_score_idx): #Reverse to not modify the indexes
                del sc[idx] # Deleting the not number values

            not_valid_score_idxs.append(not_valid_score_idx)

        for idx in reversed(not_valid_score_idxs[0]): #Reverse to not modify the indexes
            print(f"{x_ax[idx]} has no measurable values.")
            del x_ax[idx] # Deleting the not used x_axis
        
        #Plot
        fig = plt.figure(figsize=fig_size)
        x = np.arange(len(x_ax))

        plt.grid(which='major', linestyle='-', alpha = 0.9, axis = 'y')
        plt.grid(which='minor', linestyle='--', alpha = 0.6, axis = 'y')
        plt.minorticks_on()
        #minor turn off on x
        plt.tick_params('x', which='minor', bottom=False, top=False)
        
        if len(score) == 1:
            plt.bar(x, score[0], width, color = colors[0], edgecolor = 'black',  zorder = 2)
        if len(score) == 2:
            plt.bar(x - width/2, score[0], width, color = colors[0], edgecolor = 'black',  zorder = 2)
            plt.bar(x + width/2, score[1], width, color = colors[1], edgecolor = 'black',  zorder = 2)
        if len(score) == 3:
            plt.bar(x - width, score[0], width, color = colors[0], edgecolor = 'black',  zorder = 2)
            plt.bar(x, score[1], width, color = colors[1], edgecolor = 'black',  zorder = 2)
            plt.bar(x + width, score[2], width, color = colors[2], edgecolor = 'black',  zorder = 2)
        if len(score) == 4:
            plt.bar(x - width /2 - width, score[0], width, color = colors[0], edgecolor = 'black',  zorder = 2)
            plt.bar(x - width /2, score[1], width, color = colors[1], edgecolor = 'black',  zorder = 2)
            plt.bar(x + width /2, score[2], width, color = colors[2], edgecolor = 'black',  zorder = 2)
            plt.bar(x + width /2 + width, score[3], width, color = colors[3], edgecolor = 'black',  zorder = 2)

        plt.xticks(x, x_ax)
        plt.yticks(y_axis)
        plt.ylabel(y_axis_label)
        plt.xlabel(x_axis_label)
        plt.title(title, fontsize = 15)
        
        plt.legend(legend_labels, bbox_to_anchor = (1,1))
        plt.show()   
    else:
        print(return_errors)

        
#Creates a box plot for the non-time series param_tuning 
import copy
def Multiple_cv_hyperparameter_tuning_box_plot(accuracy, title, xticks, c, ylabel, yticks):
    return_errors = []
    
    acc_len = len(accuracy[0][0])
    for i in range(len(accuracy)):
        if len(accuracy[i][0])!= acc_len:
            return ("Some of the accuracy values are missing!")
        
    if len(accuracy[0]) != len(xticks):
        return_errors.append("Invalid number of labels!")
    
    if len(return_errors) == 0:
        deep_copy_accuracy = copy.deepcopy(accuracy)
        combined_acc = []
        num_rows = len(deep_copy_accuracy[0])

        for i in range(num_rows):
            combined_row = []
            for j in range(len(deep_copy_accuracy)):
                # Combine the corresponding rows from both lists within the nested list
                combined_row += deep_copy_accuracy[j][i]
            combined_acc.append(combined_row)
        
        plt.grid(which='major', linestyle='-', alpha = 0.7, axis = 'y')
        plt.grid(which='minor', linestyle='--', alpha = 0.4, axis = 'y')
        plt.minorticks_on()
        plt.tick_params('x', which='minor', bottom=False, top=False)

        box = plt.boxplot(combined_acc, patch_artist=True, boxprops=dict(facecolor='#E4E4E6'), medianprops=dict(color = c, linewidth=2))
            
        x = np.arange(1, len(xticks) + 1)
        plt.xticks(x, xticks)
        plt.yticks(yticks)
        plt.ylabel(ylabel)
        plt.title(title, fontsize = 14)

        plt.show()
    else:
        print(return_errors)

        
import pickle
import os
def NN_plot(folder_route, num_class, epoch, fig_size):
    history_path = str(folder_route) + "/" + str(epoch) + "_history.pkl"
    if os.path.exists(history_path):
        with open(history_path, "rb") as f:
            history = pickle.load(f)
    else:
        print("No history data")
        return

    plt.figure(figsize=fig_size)
    
    #print("History kulcsok:", history.keys())

    # Loss
    plt.subplot(1, 2, 1)
        
    plt.grid(which='major', linestyle='-', alpha = 0.9, axis = 'y')
    plt.grid(which='minor', linestyle='--', alpha = 0.6, axis = 'y')
    plt.minorticks_on()
    #minor turn off on x
    plt.tick_params('x', which='minor', bottom=False, top=False)
    
    plt.plot(history.get('loss', []), label='Train Loss')
    plt.plot(history.get('val_loss', []), label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
        
    plt.grid(which='major', linestyle='-', alpha = 0.9, axis = 'y')
    plt.grid(which='minor', linestyle='--', alpha = 0.6, axis = 'y')
    plt.minorticks_on()
    #minor turn off on x
    plt.tick_params('x', which='minor', bottom=False, top=False)
    
    if num_class == 2:
        plt.plot(history.get('binary_accuracy', []), label='Train Accuracy')
        plt.plot(history.get('val_binary_accuracy', []), label='Validation Accuracy')
    else:
        plt.plot(history.get('sparse_categorical_accuracy', []), label='Train Accuracy')
        plt.plot(history.get('val_sparse_categorical_accuracy', []), label='Validation Accuracy')
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.legend()

    plt.show()


def Line_chart(scores1, scores2, k_values, classifier_index, fig_size, title, plot_labels, colors, ylabel):
    return_errors = []
    
    if len(scores1) != len(scores2):
        return_errors.append("The scores have inconsistent lengths!")
    
    if len(return_errors) == 0:
        selected_classifier_scores1, selected_classifier_scores2 = [], []

        for i in range(len(scores1)):
            selected_classifier_scores1.append(scores1[i][classifier_index][0])
            selected_classifier_scores2.append(scores2[i][classifier_index][0])
        
        plt.figure(figsize=fig_size)
        
        plt.grid(which='major', linestyle='-', alpha = 0.7, axis = 'both')
        plt.grid(which='minor', linestyle='--', alpha = 0.4, axis = 'both')
        plt.minorticks_on()
        plt.tick_params('x', which='minor', bottom=False, top=False)
        
        plt.plot(k_values, selected_classifier_scores1, label = plot_labels[0], color = colors[0])
        plt.plot(k_values, selected_classifier_scores2, label = plot_labels[1], color = colors[1])
        
        plt.title(title)
        plt.xlabel("k-fold cross-validation folds")
        plt.xticks(k_values)
        plt.ylabel(ylabel)
        plt.ylim(0.1, 1.0, 0.1)
        plt.legend()
        plt.show()
    else:
        print(return_errors)