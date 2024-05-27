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