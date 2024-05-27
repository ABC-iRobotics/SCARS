#Creates the decision tree model
from sklearn.tree import DecisionTreeClassifier
def Decision_tree(criterion, splitter, max_depth, max_features, X_train, Y_train):
    DT_model = DecisionTreeClassifier(random_state=0, criterion = criterion)
    DT_model.fit(X_train, Y_train)
    return DT_model


