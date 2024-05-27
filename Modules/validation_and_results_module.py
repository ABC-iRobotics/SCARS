#Validates with three validation method
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut, cross_val_score
def Validation(model, y_pred, X, Y, Y_test):
    # TODO LeT check
    #accuracy
    accuracy = metrics.accuracy_score(Y_test, y_pred)
    
    #5-fold cross-valdation
    kf_score = cross_val_score(model, X, Y, cv = 5)
    kf_score_mean = kf_score.mean()
    
    #LOOCV (Leave-One-Out cross-validation)
    loocv_score = cross_val_score(model, X, Y, cv = LeaveOneOut())
    loocv_score_mean = loocv_score.mean()
    
    return accuracy, kf_score_mean, loocv_score_mean