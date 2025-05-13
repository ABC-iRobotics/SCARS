#Creates the Decision Tree model
from sklearn.tree import DecisionTreeClassifier
def Decision_tree(criterion, splitter, max_depth, max_features):
    return_errors = []
    #Checks the validity of the parameter values
    if criterion not in ["gini", "entropy", "log_loss"]:
        return_errors.append("Invalid criterion value!")
    if splitter not in ["best", "random"]:
        return_errors.append("Invalid splitter value!")
    if max_depth != None and isinstance(max_depth, int) == False:
        return_errors.append("Invalid max_depth value!")
    if max_features != None and max_features not in ["sqrt", "log2"] and (isinstance(max_features, int) == False and isinstance(max_features, float) == False):
        return_errors.append("Invalid max_features value!")
        
    if len(return_errors) == 0:
        return DecisionTreeClassifier(random_state=0, criterion = criterion, splitter = splitter, max_depth = max_depth, max_features = max_features)
    else:
        print(return_errors)
        return None
    
    
##Creates the Logistic Regression model
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
def Logistic_regression(penalty, C, solver, max_iter, binary_class):
    return_errors = []
    #Checks the validity of the parameter values
    if isinstance(C, float) == False:
        return_errors.append("Invalid C value!")
    if solver not in ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]:
        return_errors.append("Invalid solver value!")
    if isinstance(max_iter, int) == False:
        return_errors.append("Invalid max_iter value!")
    if penalty not in ["l1", "l2", None]:
        return_errors.append("Invalid penalty value!")
    else:
        #Checks the validity of solver-penalty parameter values
        if solver == "lbfgs" and penalty not in ["l2", None]:
            return_errors.append("Invalid solver-penalty pair values!")
        elif solver == "liblinear" and penalty not in ["l1", "l2"]:
             return_errors.append("Invalid solver-penalty pair values!")
        elif (solver == "newton-cg" or solver == "sag") and penalty not in ["l2", None]:
            return_errors.append("Invalid solver-penalty pair values!")
        elif solver == "saga" and penalty not in ["l1", "l2", None]:
            return_errors.append("Invalid solver-penalty pair values!")
        
    if len(return_errors) == 0:
        LR_model = LogisticRegression(random_state = 0, penalty = penalty, C = C, solver = solver, max_iter = max_iter)
        
        if binary_class:
            return LR_model
        else:
            return OneVsRestClassifier(LR_model)
    else:
        print(return_errors)
        return None
    

#Creates the k-NN model
from sklearn.neighbors import KNeighborsClassifier
def kNN(n_neighbors, weights, metric):
    return_errors = []
    #Checks the validity of the parameter values
    if isinstance(n_neighbors, int) == False:
        return_errors.append("Invalid n_neighbors value!")
    if weights not in ["uniform", "distance"]:
        return_errors.append("Invalid weights value!")
    if metric not in ["euclidean", "manhattan", "minkowski"]:
        return_errors.append("Invalid metric value!")
        
    if len(return_errors) == 0:
        return KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights, metric = metric)
    else:
        print(return_errors)
        return None
    

#Creates the SVM model
from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier
def SVM(C, kernel, gamma, binary_class):
    return_errors = []
    #Checks the validity of the parameter values
    if isinstance(C, float) == False:
        return_errors.append("Invalid C value!")
    if kernel not in ["linear", "poly", "rbf", "sigmoid"]:
        return_errors.append("Invalid kernel value!")
    if gamma not in ["scale", "auto"] and isinstance(gamma, float) == False:
        return_errors.append("Invalid gamma value!")
        
    if len(return_errors) == 0:
        svm_model = svm.SVC(random_state = 0, C = C, kernel = kernel, gamma = gamma)
    
        if binary_class:
            return svm_model
        else:
            return OneVsOneClassifier(svm_model)
    else:
        print(return_errors)
        return None