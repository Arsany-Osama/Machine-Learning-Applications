'''
Naive Bias Model
'''    
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from data import load_and_preprocess_data
def run_naive_bayes():
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    
    #fit on training set using features and label
    '''
    The Gaussian Naive Bayes algorithm is a probabilistic classification algorithm
    that is based on Bayes' theorem with the assumption of independence between 
    features (as this we call the term "naive"). It is often used for classification tasks
    when the features are continuous and assumed to have a Gaussian (normal) distribution.
    '''
    nb = GaussianNB()
    nb.fit(X_train, y_train) 
    
    #Cross Validation to evaluate the performance
    '''
     Cross Validation to evaluate the performance
    Step-by-step explanation of how the data is split into 5 folds:
        a)The entire dataset is divided into 5 equal-sized subsets.
        b)In the first iteration, the first fold is used as the validation set, and the remaining 4 folds are used for training.
        c)In the second iteration, the second fold is used as the validation set, and the other 4 folds are used for training.
        d)This process continues for the remaining folds until all 5 folds have been used as the validation set once.
        e)The performance of the model is evaluated on each fold, and the results are typically averaged to obtain an overall performance measure.
    '''
    nb_cv = cross_val_score(nb, X_train, y_train, cv=5) # cv=5 specifies 5 fold cross validation
    print("Naive Bayes Cross-Validation Accuracy: {:.4f}%".format(nb_cv.mean() * 100))
    
    '''
    Reporting Naive Bias accuracy, precision, recall, F-score & confusion matrix
    '''
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    #predict labels of testing features
    nb_predictions = nb.predict(X_test) 
    
    #Accuracy on testing features and label
    accuracy = accuracy_score(y_test, nb_predictions)
    print("Naive Bayes Testing Accuracy: {:.4f}%".format(accuracy * 100))
    
    #Calculate confusion matrix
    nb_conf_matrix = confusion_matrix(y_test, nb_predictions)
    print("Naive Bayes Confusion Matrix:")
    print(nb_conf_matrix)
    
    #Calculate precision, recall and F1-score 
    nb_report = classification_report(y_test, nb_predictions)
    print("Naive Bayes Classification Report:")
    print(nb_report)
