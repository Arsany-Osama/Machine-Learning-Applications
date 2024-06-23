'''
Decision Tree Model
'''
from sklearn import tree
from sklearn.model_selection import cross_val_score
from data import load_and_preprocess_data

def run_decision_tree():
    
    X_train, y_train, X_test, y_test = load_and_preprocess_data()


    dt = tree.DecisionTreeClassifier().fit(X_train , y_train)
    
    '''
     Cross Validation to evaluate the performance
    Step-by-step explanation of how the data is split into 5 folds:
        a)The entire dataset is divided into 5 equal-sized subsets.
        b)In the first iteration, the first fold is used as the validation set, and the remaining 4 folds are used for training.
        c)In the second iteration, the second fold is used as the validation set, and the other 4 folds are used for training.
        d)This process continues for the remaining folds until all 5 folds have been used as the validation set once.
        e)The performance of the model is evaluated on each fold, and the results are typically averaged to obtain an overall performance measure.
    '''
    dt_cv = cross_val_score(dt, X_train, y_train, cv=5)  # cv=5 specifies 5 fold cross validation
    print("Decision Tree Cross-Validation Accuracy: {:.4f}%".format(dt_cv.mean() * 100))

    '''
    Reporting Decision Tree accuracy, precision, recall, F-score & confusion matrix
    '''
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    dt_predictions = dt.predict(X_test)  # Predict labels of testing features

    # Accuracy on testing features and label
    accuracy = accuracy_score(y_test, dt_predictions)
    print("Decision Tree Testing Accuracy: {:.4f}%".format(accuracy * 100))

    # Calculate confusion matrix
    dt_conf_matrix = confusion_matrix(y_test, dt_predictions)
    print("Decision Tree Confusion Matrix:")
    print(dt_conf_matrix)

    # Calculate precision, recall and F1-score
    dt_report = classification_report(y_test, dt_predictions)
    print("Decision Tree Classification Report:")
    print(dt_report)
