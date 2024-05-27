#Calculates the accuracy for the model
from sklearn.metrics import classification_report
def Model_classification_report(model, X_train, X_test, Y_train, Y_test):
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    
    print('                 Classification Report')
    print('-------------------------------------------------------')
    print(classification_report(Y_test, y_pred))
    
#Counting the zeros and ones in the target variable. (0, 1 = classes)
def Least_populated_class_calculation(Y):
    zero_count = 0
    one_count = 0
    
    for i in Y:
        if(i == 0):
            zero_count = zero_count + 1
        else:
            one_count = one_count + 1
            
    if(zero_count < one_count):
        return zero_count
    else:
        return one_count
    

#Cross-validation (K-fold, or Leave-One-Out)
from sklearn.model_selection import LeaveOneOut, cross_val_score
def Cross_validation(model, X, Y, cv):
    if(isinstance(cv, int) and cv >= 2):
        least_populated_class_members = Least_populated_class_calculation(Y)
        if(cv > least_populated_class_members):
            return "The specified cv value is too big! The maximum cv value: " + str(least_populated_class_members)
        
        kf_score = cross_val_score(model, X, Y, cv = cv)
        kf_score_mean = kf_score.mean()
        return kf_score_mean
    elif(cv == "LOOCV"):
        loocv_score = cross_val_score(model, X, Y, cv = LeaveOneOut())
        loocv_score_mean = loocv_score.mean()
        return loocv_score_mean
    else:
        return "Invalid cross-validation value"
    