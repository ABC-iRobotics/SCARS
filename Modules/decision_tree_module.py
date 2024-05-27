#Creates the Decision Tree model
from sklearn.tree import DecisionTreeClassifier
def Decision_tree(criterion, splitter, max_depth, max_features):
    return_errors = []
    if(criterion not in ["gini", "entropy", "log_loss"]):
        return_errors.append("Invalid criterion value!")
    if(splitter not in ["best", "random"]):
        return_errors.append("Invalid splitter value!")
    if(max_depth != None and isinstance(max_depth, int) == False):
        return_errors.append("Invalid max_depth value!")
    if(max_features != None and max_features not in ["sqrt", "log2"] and (isinstance(max_features, int) == False and isinstance(max_features, float) == False)):
        return_errors.append("Invalid max_features value!")
        
    if(len(return_errors) == 0):
        return DecisionTreeClassifier(random_state=0, criterion = criterion, splitter = splitter, max_depth = max_depth, max_features = max_features)
    else:
        return return_errors


#Visualizes the Decision Tree model
import matplotlib.pyplot as plt
from sklearn import tree
def Decision_tree_visualization(model, header):
    fig = plt.figure(figsize=(15,10))
    tree.plot_tree(model, class_names = ["Novice", "Experienced"], feature_names = header, filled = True)
    plt.show()
    
    
#Calculates the features imporantance of the Decision Tree model
import numpy as np
def Decision_tree_feature_importance(model, color, header):
    features_importance = model.feature_importances_
    
    fig = plt.figure(figsize=(20,5))
    
    plt.grid(which='major', linestyle='-', alpha = 0.7, axis = 'y')
    plt.grid(which='minor', linestyle='--', alpha = 0.4, axis = 'y')
    plt.minorticks_on()
    plt.tick_params('x', which='minor', bottom=False, top=False)
    plt.bar(x = [h for h in header], height = features_importance, color = color, zorder = 2)
    
    #Including all features on the x-axis.
    plt.xticks([h for h in header], rotation ='vertical')
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylabel('Importance')
    plt.xlabel('Features')
    
    plt.show()